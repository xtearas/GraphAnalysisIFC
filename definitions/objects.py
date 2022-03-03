from OCC.Core.ShapeAnalysis import ShapeAnalysis_Curve
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
from OCC.Core.Geom import Geom_BSplineCurve
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Extend.ShapeFactory import make_edge
from collections import OrderedDict

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.placement

from definitions.utilities import get_wall_axis, get_wall_body, get_wall_width, get_project_length_unit, \
    get_project_tolerance

import numpy as np
import networkx as nx

# Initiate OCC Display
from OCC.Display.SimpleGui import init_display

display, start_display, add_menu, add_function_to_menu = init_display()


class WallSegment:
    def __init__(self, index: int, segment_curve: Geom_BSplineCurve, host_length: float, host_width: float, label=str):
        self.Curve = segment_curve
        self.Edge = make_edge(segment_curve)
        self.Index = index
        self.Label = label
        self.HostWallLength = host_length
        self.HostWallWidth = host_width
        self.HasDoor = False
        self.Mid = gp_Pnt(segment_curve.StartPoint().X() / 2 + segment_curve.EndPoint().X() / 2,
                          segment_curve.StartPoint().Y() / 2 + segment_curve.EndPoint().Y() / 2,
                          segment_curve.StartPoint().Z() / 2 + segment_curve.EndPoint().Z() / 2)


class WallUnit:
    def __init__(self, guid, axis: Geom_BSplineCurve, body: Geom_BSplineCurve, width, ifc_wall, level_id, elevation):
        self.ifcWall = ifc_wall
        self.GlobalId = guid
        self.Axis = axis
        self.Body = body
        self.AxisEdge = make_edge(axis)
        self.Tangent = gp_Vec(axis.StartPoint(), axis.EndPoint())
        self.Length = self.Tangent.Magnitude()
        self.Width = width
        self.Segments = []
        self.ConnectedWallsIds = []
        self.ConnectedWallUnits = []
        self.WallGuidsAtStart = 0
        self.WallGuidsAtEnd = 0
        self.WallUnitAtStart = None
        self.WallUnitAtEnd = None
        self.ConnectionPoints = OrderedDict()
        self.LevelId = level_id
        self.Elevation = elevation
        self.DoorPoints = []


class IFCProject:
    def __init__(self, file_path: str, building_storey_index: int):
        self.IfcFile = ifcopenshell.open(file_path)
        self.LengthUnits = get_project_length_unit(self.IfcFile)
        self.ProjectTolerance = get_project_tolerance(self.IfcFile)
        self.BuildingStoreys = self.IfcFile.by_type('IfcBuildingStorey')
        self.BuildingStorey = self.BuildingStoreys[building_storey_index]
        self.WallUnits = {}
        self.WallSegments = []
        self.WallConnectivityGraph = nx.Graph()
        self.WallConnectivitySubGraphs = []
        self.WallConnectivityMaxSubGraph = nx.Graph()
        self.RoomConnectivityGraph = nx.Graph()

    def get_wall_units(self):
        walls = self.IfcFile.by_type("IfcWall")
        storey = self.BuildingStorey

        wall_unit_ids = []
        wall_units = []
        wall_units_dict = {}

        # Collect all doors in project
        doors = self.IfcFile.by_type('IfcDoor')
        door_ids = []
        for door in doors:
            if len(door.FillsVoids) > 0:
                door_ids.append(door.FillsVoids[0].RelatingOpeningElement.GlobalId)

        for wall in walls:
            level_id_ = 'None'
            level_z = 0

            for structure in wall.ContainedInStructure:
                if structure.RelatingStructure.get_info().get('type') == 'IfcBuildingStorey':
                    level_id_ = structure.RelatingStructure.GlobalId
                    level_z = structure.RelatingStructure.Elevation

            if storey is not None and level_id_ != storey.GlobalId:
                continue

            # See if wall has a geometric representation of its axis
            axis_ = get_wall_axis(wall, z_coordinate=level_z)
            if axis_ is None:
                continue
            body_ = get_wall_body(wall, z_coordinate=level_z)
            width_ = get_wall_width(self.IfcFile, wall)
            guid_ = wall.GlobalId

            # Get wall transformation matrix from file
            translation_matrix = ifcopenshell.util.placement.get_local_placement(wall.ObjectPlacement)
            transformation_matrix = ifcopenshell.util.placement.get_axis2placement(
                wall.ObjectPlacement.RelativePlacement)
            wall_z = float(translation_matrix[2, 3])

            wall_guids_at_end = []
            wall_guids_at_start = []

            # Check wall relations
            related_door_points = []
            connected_wall_ids = []
            wall_relations = self.IfcFile.get_inverse(wall)
            for wall_relation in wall_relations:
                if wall_relation.is_a('IfcRelVoidsElement'):
                    if wall_relation.RelatedOpeningElement.is_a('IfcOpeningElement'):
                        door = wall_relation.RelatedOpeningElement.get_info()
                        global_id = door.get('GlobalId')
                        if global_id in door_ids:
                            coordinates = door.get("ObjectPlacement").RelativePlacement.Location.Coordinates
                            # Get Local point
                            loc_point = np.array([coordinates[0], coordinates[1], 0, 1])

                            # Get Relative point
                            rel_point = np.matmul(transformation_matrix, loc_point.T)
                            door_point = gp_Pnt(rel_point[0], rel_point[1], level_z)
                            related_door_points.append(door_point)
                elif wall_relation.is_a('IfcRelConnectsPathElements'):
                    related_wall_guid = wall_relation.get_info().get('RelatedElement').get_info().get('GlobalId')
                    relating_wall_guid = wall_relation.get_info().get('RelatingElement').get_info().get('GlobalId')
                    if related_wall_guid != guid_:
                        if wall_relation.RelatingConnectionType == 'ATEND':
                            wall_guid_at_end = related_wall_guid
                            wall_guids_at_end.append(wall_guid_at_end)
                        elif wall_relation.RelatingConnectionType == 'ATSTART':
                            wall_guid_at_start = related_wall_guid
                            wall_guids_at_start.append(wall_guid_at_start)
                        connected_wall_ids.append(related_wall_guid)
                    elif relating_wall_guid != guid_:
                        if wall_relation.RelatedConnectionType == 'ATEND':
                            wall_guid_at_end = relating_wall_guid
                            wall_guids_at_end.append(wall_guid_at_end)
                        elif wall_relation.RelatedConnectionType == 'ATSTART':
                            wall_guid_at_start = relating_wall_guid
                            wall_guids_at_start.append(wall_guid_at_start)
                        connected_wall_ids.append(relating_wall_guid)

            wall_unit = WallUnit(guid_, axis_, body_, width_, wall, level_id_, wall_z)
            wall_unit.DoorPoints = related_door_points
            wall_unit.ConnectedWallsIds = connected_wall_ids
            wall_unit.WallGuidsAtStart = wall_guids_at_start
            wall_unit.WallGuidsAtEnd = wall_guids_at_end
            wall_unit.ConnectionPoints[0.0] = wall_unit.Axis.StartPoint()
            wall_unit.ConnectionPoints[1.0] = wall_unit.Axis.EndPoint()

            wall_units.append(wall_unit)
            wall_unit_ids.append(guid_)
            wall_units_dict[guid_] = wall_unit

        self.WallUnits = wall_units_dict

    def get_wall_units_neighbors(self):

        wall_units = list(self.WallUnits.values())

        for i in range(len(wall_units) - 1):
            for j in range(i + 1, len(wall_units)):
                if i != j:
                    # Find the closest point on each of the the two related walls
                    dss = BRepExtrema_DistShapeShape(wall_units[i].AxisEdge, wall_units[j].AxisEdge)

                    point_on_this_wall = dss.PointOnShape1(1)
                    point_on_other_wall = dss.PointOnShape2(1)

                    # Find the distance between the closest point on the connected wall
                    # and the closest point on the current wall
                    dist = point_on_this_wall.Distance(point_on_other_wall)

                    if dist < self.ProjectTolerance:
                        wall_units[i].ConnectedWallsIds.append(wall_units[j].GlobalId)
                        wall_units[j].ConnectedWallsIds.append(wall_units[i].GlobalId)

    def get_wall_units_neighbors_ifc(self):

        for wall_unit in self.WallUnits.values():
            for connected_wall_unit_guid in wall_unit.ConnectedWallsIds:
                if self.WallUnits.get(connected_wall_unit_guid) is None:
                    continue
                connected_wall_unit = self.WallUnits[connected_wall_unit_guid]
                # Find the closest point on each of the the two related walls
                dss = BRepExtrema_DistShapeShape(wall_unit.AxisEdge, connected_wall_unit.AxisEdge)

                point_on_wall = dss.PointOnShape1(1)
                point_on_connected_wall = dss.PointOnShape2(1)

                # Find the parameter of the closest point on the current wall by projecting it
                # Instantiate projection point
                projection_point_on_wall = gp_Pnt()
                projection_point_on_connected_wall = gp_Pnt()

                distance1, projection_parameter_on_wall = ShapeAnalysis_Curve().Project(wall_unit.Axis, point_on_wall,
                                                                                        self.ProjectTolerance,
                                                                                        projection_point_on_wall)

                distance2, projection_parameter_on_connected_wall = ShapeAnalysis_Curve().Project(
                    connected_wall_unit.Axis,
                    point_on_connected_wall,
                    self.ProjectTolerance, projection_point_on_connected_wall)

                # Find the distance between the closest point on the connected wall
                # and the closest point on the current wall
                dist = point_on_wall.Distance(point_on_connected_wall)

                max_width = max([wall_unit.Width / 2, connected_wall_unit.Width / 2])

                if dist < max_width + self.ProjectTolerance:
                    if projection_parameter_on_wall < self.ProjectTolerance:
                        wall_unit.WallUnitAtStart = connected_wall_unit
                        wall_unit.ConnectionPoints[0.0] = point_on_connected_wall

                        if point_on_connected_wall.Distance(
                                connected_wall_unit.Axis.StartPoint()) < max_width + self.ProjectTolerance:
                            connected_wall_unit.ConnectionPoints[0.0] = point_on_connected_wall
                        elif point_on_connected_wall.Distance(
                                connected_wall_unit.Axis.EndPoint()) < max_width + self.ProjectTolerance:
                            connected_wall_unit.ConnectionPoints[1.0] = point_on_connected_wall

                    elif projection_parameter_on_wall > 1 - self.ProjectTolerance:
                        wall_unit.WallUnitAtEnd = connected_wall_unit
                        wall_unit.ConnectionPoints[1.0] = point_on_connected_wall

                        if point_on_connected_wall.Distance(
                                connected_wall_unit.Axis.StartPoint()) < max_width + self.ProjectTolerance:
                            connected_wall_unit.ConnectionPoints[0.0] = point_on_connected_wall
                        elif point_on_connected_wall.Distance(
                                connected_wall_unit.Axis.EndPoint()) < max_width + self.ProjectTolerance:
                            connected_wall_unit.ConnectionPoints[1.0] = point_on_connected_wall

                    elif projection_parameter_on_connected_wall < self.ProjectTolerance or projection_parameter_on_connected_wall > 1 - self.ProjectTolerance:
                        wall_unit.ConnectionPoints[projection_parameter_on_wall] = point_on_wall

                    if projection_parameter_on_connected_wall < self.ProjectTolerance:
                        connected_wall_unit.WallUnitAtStart = wall_unit
                        connected_wall_unit.ConnectionPoints[0.0] = point_on_wall

                        if point_on_wall.Distance(wall_unit.Axis.StartPoint()) < max_width + self.ProjectTolerance:
                            wall_unit.ConnectionPoints[0.0] = point_on_wall
                        elif point_on_wall.Distance(wall_unit.Axis.EndPoint()) < max_width + self.ProjectTolerance:
                            wall_unit.ConnectionPoints[1.0] = point_on_wall

                    elif projection_parameter_on_connected_wall > 1 - self.ProjectTolerance:
                        connected_wall_unit.WallUnitAtEnd = wall_unit
                        connected_wall_unit.ConnectionPoints[1.0] = point_on_wall

                        if point_on_wall.Distance(wall_unit.Axis.StartPoint()) < max_width + self.ProjectTolerance:
                            wall_unit.ConnectionPoints[0.0] = point_on_wall
                        elif point_on_wall.Distance(wall_unit.Axis.EndPoint()) < max_width + self.ProjectTolerance:
                            wall_unit.ConnectionPoints[1.0] = point_on_wall

                    elif projection_parameter_on_wall < self.ProjectTolerance or projection_parameter_on_wall > 1 - self.ProjectTolerance:
                        connected_wall_unit.ConnectionPoints[
                            projection_parameter_on_connected_wall] = point_on_connected_wall

                    if connected_wall_unit.GlobalId not in wall_unit.ConnectedWallsIds:
                        wall_unit.ConnectedWallsIds.append(connected_wall_unit.GlobalId)
                    wall_unit.ConnectedWallUnits.append(connected_wall_unit)

                    if wall_unit.GlobalId not in connected_wall_unit.ConnectedWallsIds:
                        connected_wall_unit.ConnectedWallsIds.append(wall_unit.GlobalId)
                    connected_wall_unit.ConnectedWallUnits.append(wall_unit)

    def get_geometric_wall_connectivity(self):
        wall_units = list(self.WallUnits.values())

        for i in range(len(wall_units)):

            if wall_units[i].Body is not None:
                display.DisplayShape(wall_units[i].Body, update=False, color="BLUE")

            points = []
            parameters = []

            # Sort points by their corresponding parameter
            zip_data = sorted(zip(wall_units[i].ConnectionPoints.keys(), wall_units[i].ConnectionPoints.values()),
                              key=lambda pair: pair[0])

            for parameter, point in zip_data:
                points.append(point)
                parameters.append(parameter)

            # Create wall segments as consecutive lines from the sorted points
            for p in range(len(points) - 1):
                array = TColgp_Array1OfPnt(1, 2)
                array.SetValue(1, points[p])
                array.SetValue(2, points[p + 1])
                segment = GeomAPI_PointsToBSpline(array, 1, 1).Curve()
                dist = segment.StartPoint().Distance(segment.EndPoint())

                # Avoid short segments by checking the distance between start and end point
                if dist > self.ProjectTolerance:
                    # Add segments as a WallSegment Class to the current WallUnit
                    wall_segment = WallSegment(i + p + 1, segment, wall_units[i].Tangent.Magnitude(),
                                               wall_units[i].Width,
                                               wall_units[i].GlobalId)
                    # Check if segment is near a door
                    if len(wall_units[i].DoorPoints) > 0:
                        cp = np.infty
                        for door_point in wall_units[i].DoorPoints:
                            # Find the parameter of the closest point on the current segment by projecting it
                            # Instantiate projection point
                            proj_point = gp_Pnt()
                            distance, proj_param = ShapeAnalysis_Curve().Project(segment, door_point,
                                                                                 self.ProjectTolerance, proj_point)
                            if distance < cp:
                                cp = distance

                        if cp < wall_units[i].Width / 2 + self.ProjectTolerance:
                            wall_segment.HasDoor = True

                    wall_units[i].Segments.append(wall_segment)

                    # Store all wall segments in a list
                    self.WallSegments.append(wall_segment)

                    display.DisplayShape(wall_segment.Curve, update=False, color="GREEN")
                    display.DisplayShape(wall_segment.Curve.EndPoint(), update=False, color="RED")
                    display.DisplayShape(wall_segment.Curve.StartPoint(), update=False, color="RED")

        start_display()

        for i in range(len(self.WallSegments) - 1):
            for j in range(i + 1, len(self.WallSegments)):
                # Find the closest point on each of the the two related walls
                dss = BRepExtrema_DistShapeShape(self.WallSegments[i].Edge, self.WallSegments[j].Edge)

                point_on_this_segment = dss.PointOnShape1(1)
                point_on_other_segment = dss.PointOnShape2(1)

                distance = point_on_this_segment.Distance(point_on_other_segment)

                if distance < self.ProjectTolerance:
                    weight = max([(self.WallSegments[i].HostWallLength + self.WallSegments[j].HostWallLength) / 2,
                                  self.WallSegments[i].Mid.Distance(self.WallSegments[j].Mid)])
                    self.WallConnectivityGraph.add_edge(i, j, weight=int(weight))

                    self.WallConnectivityGraph.nodes[i]['label'] = self.WallSegments[i].Label
                    self.WallConnectivityGraph.nodes[i]['pos'] = np.array(
                        [self.WallSegments[i].Mid.X(), self.WallSegments[i].Mid.Y()])
                    self.WallConnectivityGraph.nodes[i]['start'] = np.array(
                        [self.WallSegments[i].Curve.StartPoint().X(), self.WallSegments[i].Curve.StartPoint().Y()])
                    self.WallConnectivityGraph.nodes[i]['end'] = np.array(
                        [self.WallSegments[i].Curve.EndPoint().X(), self.WallSegments[i].Curve.EndPoint().Y()])
                    self.WallConnectivityGraph.nodes[i]['length'] = self.WallSegments[i].HostWallLength
                    self.WallConnectivityGraph.nodes[i]['door'] = self.WallSegments[i].HasDoor
                    self.WallConnectivityGraph.nodes[i]['type'] = 'wall'
                    self.WallConnectivityGraph.nodes[i]['exterior'] = False

                    self.WallConnectivityGraph.nodes[j]['label'] = self.WallSegments[j].Label
                    self.WallConnectivityGraph.nodes[j]['pos'] = np.array(
                        [self.WallSegments[j].Mid.X(), self.WallSegments[j].Mid.Y()])
                    self.WallConnectivityGraph.nodes[j]['start'] = np.array(
                        [self.WallSegments[j].Curve.StartPoint().X(), self.WallSegments[j].Curve.StartPoint().Y()])
                    self.WallConnectivityGraph.nodes[j]['end'] = np.array(
                        [self.WallSegments[j].Curve.EndPoint().X(), self.WallSegments[j].Curve.EndPoint().Y()])
                    self.WallConnectivityGraph.nodes[j]['length'] = self.WallSegments[j].HostWallLength
                    self.WallConnectivityGraph.nodes[j]['door'] = self.WallSegments[j].HasDoor
                    self.WallConnectivityGraph.nodes[j]['type'] = 'wall'
                    self.WallConnectivityGraph.nodes[j]['exterior'] = False

        self.WallConnectivitySubGraphs = [self.WallConnectivityGraph.subgraph(c).copy() for c in
                                          nx.connected_components(self.WallConnectivityGraph)]
        max_size = 0
        maxG = self.WallConnectivitySubGraphs[0]

        for subG in self.WallConnectivitySubGraphs:
            if subG.size() > max_size:
                max_size = subG.size()
                maxG = subG

        self.WallConnectivityMaxSubGraph = maxG

    def get_room_connectivity(self):

        # Compute minimum weight cycle basis
        cycles = [sorted(c) for c in
                  nx.algorithms.minimum_cycle_basis(self.WallConnectivityMaxSubGraph, weight='weight')]

        src_index = max(list(self.WallConnectivityMaxSubGraph.nodes)) + 1

        self.RoomConnectivityGraph = self.WallConnectivityMaxSubGraph.copy()

        for cycle in cycles:
            # Rooms are the faces of the graph that have more than 3 vertices
            if len(cycle) > 3:
                positions = []

                for key in cycle:
                    self.RoomConnectivityGraph.add_edge(src_index, key)
                    positions.append(self.WallConnectivityMaxSubGraph.nodes[key]['pos'])

                self.RoomConnectivityGraph.nodes[src_index]['label'] = src_index
                self.RoomConnectivityGraph.nodes[src_index]['pos'] = np.mean(np.array(positions), axis=0)
                self.RoomConnectivityGraph.nodes[src_index]['type'] = 'room'

                src_index = src_index + 1
