import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.placement

import numpy as np
from scipy import sparse
import networkx as nx
import matplotlib.pyplot as plt
from functools import reduce
from collections import OrderedDict

from OCC.Core.ShapeAnalysis import ShapeAnalysis_Curve
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Lin
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnCurve, GeomAPI_PointsToBSpline
from OCC.Core.Geom import Geom_BSplineCurve, Geom_OffsetCurve
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Extend.ShapeFactory import make_edge, make_vertex
from OCC.Display.SimpleGui import init_display
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCC.Core.BOPAlgo import BOPAlgo_Splitter
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.TopExp import TopExp_Explorer

# Viewport Customisation
from OCC.Core.AIS import AIS_Shape

from OCC.Extend.ShapeFactory import translate_shp

from settings import Tolerance, ProjectTolerance
from definitions.objects import WallSegment, WallUnit

# Initiate OCC Display
display, start_display, add_menu, add_function_to_menu = init_display()


def get_wall_elevations(file):
    walls = file.by_type("IfcWall")
    elevations = []
    for wall in walls:
        translation_matrix = ifcopenshell.util.placement.get_local_placement(wall.ObjectPlacement)
        elevation = float(translation_matrix[2, 3])
        if elevation not in elevations:
            elevations.append(elevation)

    return elevations


def get_wall_axis(ifc_wall, z_coordinate=None):
    wall_points = []

    # Check if wall has representation
    if ifc_wall.Representation is None:
        return None

    # Query all wall representations
    wall_representations = ifc_wall.Representation.get_info().get('Representations')

    # Check if wall has Axis representation
    has_axis = False
    for wall_representation in wall_representations:
        if wall_representation.get_info().get('RepresentationIdentifier') == 'Axis':
            first_item = wall_representation.get_info().get('Items')[0]
            if first_item.is_a('IfcTrimmedCurve'):
                wall_points.append(first_item.Trim1[0])
                wall_points.append(first_item.Trim2[0])
                has_axis = True
            elif first_item.is_a('IfcPolyline'):
                wall_points = wall_representation.get_info().get('Items')[0].get_info().get('Points')
                has_axis = True

    if not has_axis:
        return None

    translation_matrix = ifcopenshell.util.placement.get_local_placement(ifc_wall.ObjectPlacement)
    transformation_matrix = ifcopenshell.util.placement.get_axis2placement(ifc_wall.ObjectPlacement.RelativePlacement)

    wall_z = float(translation_matrix[2, 3])
    if z_coordinate is not None:
        wall_z = z_coordinate

    wall_array = TColgp_Array1OfPnt(1, len(wall_points))

    for w in range(len(wall_points)):
        # Get Local point
        loc_point = np.array([(wall_points[w].get_info().get('Coordinates')[0]),
                              wall_points[w].get_info().get('Coordinates')[1], 0, 1])

        # Get Relative point
        rel_pt = np.matmul(transformation_matrix, loc_point.T)

        # Get polyline points
        pl_point = gp_Pnt(rel_pt[0], rel_pt[1], wall_z)
        wall_array.SetValue(w + 1, pl_point)

    # Represent wall polyline as a BSpline with degree 1 and number of points = 2
    wall_axis = GeomAPI_PointsToBSpline(wall_array, 1, 1).Curve()

    return wall_axis


def get_width(file, ifc_wall):
    thickness = 0
    if ifc_wall.Representation is not None:
        wall_representations = ifc_wall.Representation.Representations
        for wall_representation in wall_representations:
            if wall_representation.RepresentationIdentifier == 'Body':
                area = wall_representation.Items[0].get_info().get('SweptArea')
                if area is not None:
                    if 'YDim' in area.get_info():
                        thickness = min([area.YDim, area.XDim])
                        if thickness == 0:
                            ok = 1
                        return thickness

        if len(ifc_wall.HasAssociations) > 0:
            for wall_assoc in ifc_wall.HasAssociations:
                if wall_assoc.is_a('IfcRelAssociatesMaterial'):
                    material_info = wall_assoc.RelatingMaterial.get_info()
                    if material_info.get('ForLayerSet') is not None:
                        thickness = 0
                        materials = wall_assoc.RelatingMaterial.ForLayerSet.MaterialLayers
                        for material in materials:
                            thickness = thickness + material.LayerThickness
                        if thickness == 0:
                            ok = 1
                        return thickness
    if thickness == 0:
        wall_relations = file.get_inverse(ifc_wall)
        for wall_relation in wall_relations:
            if wall_relation.is_a('IfcRelDefinesByType'):
                if wall_relation.RelatingType.HasAssociations:
                    for wall_assoc in wall_relation.RelatingType.HasAssociations:
                        if wall_assoc.is_a('IfcRelAssociatesMaterial'):
                            material_info = wall_assoc.RelatingMaterial.get_info()
                            if material_info.get('MaterialLayers') is not None:
                                thickness = 0
                                materials = wall_assoc.RelatingMaterial.MaterialLayers
                                for material in materials:
                                    thickness = thickness + material.LayerThickness
                                if thickness == 0:
                                    ok = 1
                                return thickness

        ok = 1

    return thickness


"""
def get_wall_units(file, storey=None):
    walls = file.by_type("IfcWall")
    wall_units = []
    doors = file.by_type('IfcDoor')
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
        axis_ = get_wall_axis(wall, z_coordinate=level_z)
        if axis_ is None:
            continue
        width_ = get_width(wall)
        guid_ = wall.GlobalId

        # Get wall transformation matrix from file
        translation_matrix = ifcopenshell.util.placement.get_local_placement(wall.ObjectPlacement)
        transformation_matrix = ifcopenshell.util.placement.get_axis2placement(wall.ObjectPlacement.RelativePlacement)
        wall_z = float(translation_matrix[2, 3])
        door_points = []
        wall_relations = file.get_inverse(wall)
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
                        door_points.append(door_point)

        wall_unit = WallUnit(guid_, axis_, width_, wall, level_id_, wall_z)
        wall_unit.DoorPoints = door_points
        wall_units.append(wall_unit)
        
    return wall_units
"""


def update_axis(wall_unit: WallUnit, new_point: gp_Pnt):

    control_points_array = TColgp_Array1OfPnt(1, wall_unit.Axis.NbPoles() + 1)
    control_points = list(wall_unit.Axis.Poles())
    for w in range(len(control_points)):
        control_points_array.SetValue(w + 1, control_points[w])
    control_points_array.SetValue(wall_unit.Axis.NbPoles() + 1, new_point)

    wall_unit.Axis = GeomAPI_PointsToBSpline(control_points_array, 1, 1).Curve()


def get_wall_units_neighbors(wall_units_dict):

    wall_units = list(wall_units_dict.values())

    for i in range(len(wall_units) - 1):
        for j in range(i + 1, len(wall_units)):
            if i != j:
                # Find the closest point on each of the the two related walls
                dss = BRepExtrema_DistShapeShape(wall_units[i].AxisEdge, wall_units[j].AxisEdge)

                point_on_this_wall = dss.PointOnShape1(1)
                point_on_other_wall = dss.PointOnShape2(1)

                # Find the parameter of the closest point on the current wall by projecting it
                # Instantiate projection point
                proj_point_on_this_wall = gp_Pnt()
                proj_point_on_other_wall = gp_Pnt()
                distance1, proj_param_on_this_wall = ShapeAnalysis_Curve().Project(wall_units[i].Axis, point_on_this_wall,
                                                                                   Tolerance, proj_point_on_this_wall)

                distance2, proj_param_on_other_wall = ShapeAnalysis_Curve().Project(wall_units[j].Axis, point_on_other_wall,
                                                                                    Tolerance, proj_point_on_other_wall)

                # Find the distance between the closest point on the connected wall
                # and the closest point on the current wall
                dist = point_on_this_wall.Distance(point_on_other_wall)

                #min_width = wall_units[i].Width / 2
                #
                #if proj_param_on_other_wall > 0.000001 and proj_param_on_other_wall < 0.999999:
                #    min_width = wall_units[j].Width / 2

                if dist < Tolerance:
                    """ 
                    if proj_param_on_this_wall == 0.0:
                        wall_units[i].WallUnitAtStart = wall_units[j]
                        wall_units[i].ConnectionPoints[0.0] = point_on_other_wall

                        if point_on_other_wall.Distance(wall_units[j].Axis.StartPoint()) < min_width + tolerance:
                            wall_units[j].ConnectionPoints[0.0] = point_on_other_wall
                        elif point_on_other_wall.Distance(wall_units[j].Axis.EndPoint()) < min_width + tolerance:
                            wall_units[j].ConnectionPoints[1.0] = point_on_other_wall

                        #update_axis(wall_units[i], point_on_other_wall)

                    elif proj_param_on_this_wall == 1.0:
                        wall_units[i].WallUnitAtEnd = wall_units[j]
                        wall_units[i].ConnectionPoints[1.0] = point_on_other_wall

                        if point_on_other_wall.Distance(wall_units[j].Axis.StartPoint()) < min_width + tolerance:
                            wall_units[j].ConnectionPoints[0.0] = point_on_other_wall
                        elif point_on_other_wall.Distance(wall_units[j].Axis.EndPoint()) < min_width + tolerance:
                            wall_units[j].ConnectionPoints[1.0] = point_on_other_wall

                        #update_axis(wall_units[i], point_on_other_wall)

                    else:
                        wall_units[i].ConnectionPoints[proj_param_on_this_wall] = point_on_this_wall
                        #update_axis(wall_units[i], point_on_this_wall)

                    if proj_param_on_other_wall == 0.0:
                        wall_units[j].WallUnitAtStart = wall_units[i]
                        wall_units[j].ConnectionPoints[0.0] = point_on_this_wall

                        if point_on_this_wall.Distance(wall_units[i].Axis.StartPoint()) < min_width + tolerance:
                            wall_units[i].ConnectionPoints[0.0] = point_on_this_wall
                        elif point_on_this_wall.Distance(wall_units[i].Axis.EndPoint()) < min_width + tolerance:
                            wall_units[i].ConnectionPoints[1.0] = point_on_this_wall

                        #update_axis(wall_units[j], point_on_this_wall)

                    elif proj_param_on_other_wall == 1.0:
                        wall_units[j].WallUnitAtEnd = wall_units[i]
                        wall_units[j].ConnectionPoints[1.0] = point_on_this_wall

                        if point_on_this_wall.Distance(wall_units[i].Axis.StartPoint()) < min_width + tolerance:
                            wall_units[i].ConnectionPoints[0.0] = point_on_this_wall
                        elif point_on_this_wall.Distance(wall_units[i].Axis.EndPoint()) < min_width + tolerance:
                            wall_units[i].ConnectionPoints[1.0] = point_on_this_wall

                        #update_axis(wall_units[j], point_on_this_wall)

                    else:
                        wall_units[j].ConnectionPoints[proj_param_on_other_wall] = point_on_other_wall
                        #update_axis(wall_units[j], point_on_other_wall)
                    """
                    wall_units[i].ConnectedWallsIds.append(wall_units[j].GlobalId)
                    wall_units[j].ConnectedWallsIds.append(wall_units[i].GlobalId)


def get_wall_units_neighbors_ifc(wall_units_dict):
    wall_units = wall_units_dict.values()
    wall_unit_ids = wall_units_dict.keys()

    for wall_unit in wall_units:
        for con_wall_unit_guid in wall_unit.ConnectedWallsIds:
            if wall_units_dict.get(con_wall_unit_guid) is None:
                continue
            con_wall_unit = wall_units_dict[con_wall_unit_guid]
            # Find the closest point on each of the the two related walls
            dss = BRepExtrema_DistShapeShape(wall_unit.AxisEdge, con_wall_unit.AxisEdge)

            point_on_this_wall = dss.PointOnShape1(1)
            point_on_other_wall = dss.PointOnShape2(1)

            # Find the parameter of the closest point on the current wall by projecting it
            # Instantiate projection point
            proj_point_on_this_wall = gp_Pnt()
            proj_point_on_other_wall = gp_Pnt()
            distance1, proj_param_on_this_wall = ShapeAnalysis_Curve().Project(wall_unit.Axis, point_on_this_wall,
                                                                               Tolerance, proj_point_on_this_wall)

            distance2, proj_param_on_other_wall = ShapeAnalysis_Curve().Project(con_wall_unit.Axis, point_on_other_wall,
                                                                                Tolerance, proj_point_on_other_wall)

            # Find the distance between the closest point on the connected wall
            # and the closest point on the current wall
            dist = point_on_this_wall.Distance(point_on_other_wall)

            #min_width = wall_unit.Width / 2
            #
            #if 0.000001 < proj_param_on_other_wall < 0.999999:
            #    min_width = con_wall_unit.Width / 2

            min_width = max([wall_unit.Width / 2, con_wall_unit.Width / 2])

            if dist < min_width + Tolerance:
                if proj_param_on_this_wall < 0.0001:
                    wall_unit.WallUnitAtStart = con_wall_unit
                    wall_unit.ConnectionPoints[0.0] = point_on_other_wall

                    if point_on_other_wall.Distance(con_wall_unit.Axis.StartPoint()) < min_width + Tolerance:
                        con_wall_unit.ConnectionPoints[0.0] = point_on_other_wall
                    elif point_on_other_wall.Distance(con_wall_unit.Axis.EndPoint()) < min_width + Tolerance:
                        con_wall_unit.ConnectionPoints[1.0] = point_on_other_wall

                elif proj_param_on_this_wall > 0.9999:
                    wall_unit.WallUnitAtEnd = con_wall_unit
                    wall_unit.ConnectionPoints[1.0] = point_on_other_wall

                    if point_on_other_wall.Distance(con_wall_unit.Axis.StartPoint()) < min_width + Tolerance:
                        con_wall_unit.ConnectionPoints[0.0] = point_on_other_wall
                    elif point_on_other_wall.Distance(con_wall_unit.Axis.EndPoint()) < min_width + Tolerance:
                        con_wall_unit.ConnectionPoints[1.0] = point_on_other_wall

                elif proj_param_on_other_wall < 0.0001 or proj_param_on_other_wall > 0.9999:
                    wall_unit.ConnectionPoints[proj_param_on_this_wall] = point_on_this_wall

                if proj_param_on_other_wall < 0.0001:
                    con_wall_unit.WallUnitAtStart = wall_unit
                    con_wall_unit.ConnectionPoints[0.0] = point_on_this_wall

                    if point_on_this_wall.Distance(wall_unit.Axis.StartPoint()) < min_width + Tolerance:
                        wall_unit.ConnectionPoints[0.0] = point_on_this_wall
                    elif point_on_this_wall.Distance(wall_unit.Axis.EndPoint()) < min_width + Tolerance:
                        wall_unit.ConnectionPoints[1.0] = point_on_this_wall

                elif proj_param_on_other_wall > 0.9999:
                    con_wall_unit.WallUnitAtEnd = wall_unit
                    con_wall_unit.ConnectionPoints[1.0] = point_on_this_wall

                    if point_on_this_wall.Distance(wall_unit.Axis.StartPoint()) < min_width + Tolerance:
                        wall_unit.ConnectionPoints[0.0] = point_on_this_wall
                    elif point_on_this_wall.Distance(wall_unit.Axis.EndPoint()) < min_width + Tolerance:
                        wall_unit.ConnectionPoints[1.0] = point_on_this_wall

                elif proj_param_on_this_wall < 0.0001 or proj_param_on_this_wall > 0.9999:
                    con_wall_unit.ConnectionPoints[proj_param_on_other_wall] = point_on_other_wall

                wall_unit.ConnectedWallUnits.append(con_wall_unit)
                con_wall_unit.ConnectedWallUnits.append(wall_unit)


def get_wall_units(file, storey=None):
    walls = file.by_type("IfcWall")
    wall_unit_ids = []
    wall_units = []
    wall_units_dict = {}

    # Collect all doors in project
    doors = file.by_type('IfcDoor')
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

        width_ = get_width(file, wall)
        guid_ = wall.GlobalId

        # Get wall transformation matrix from file
        translation_matrix = ifcopenshell.util.placement.get_local_placement(wall.ObjectPlacement)
        transformation_matrix = ifcopenshell.util.placement.get_axis2placement(wall.ObjectPlacement.RelativePlacement)
        wall_z = float(translation_matrix[2, 3])

        wall_guid_at_end = 0
        wall_guid_at_start = 0

        # Check wall relations
        related_door_points = []
        connected_wall_ids = []
        wall_relations = file.get_inverse(wall)
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
                    if wall_relation.RelatingConnectionType == 'ATEND':  # and wall_relation.RelatedConnectionType != 'ATPATH':
                        wall_guid_at_end = related_wall_guid
                    elif wall_relation.RelatingConnectionType == 'ATSTART':  # and wall_relation.RelatedConnectionType != 'ATPATH':
                        wall_guid_at_start = related_wall_guid
                    connected_wall_ids.append(related_wall_guid)
                elif relating_wall_guid != guid_:
                    if wall_relation.RelatedConnectionType == 'ATEND':  # and wall_relation.RelatingConnectionType != 'ATPATH':
                        wall_guid_at_end = relating_wall_guid
                    elif wall_relation.RelatedConnectionType == 'ATSTART':  # and wall_relation.RelatingConnectionType != 'ATPATH':
                        wall_guid_at_start = relating_wall_guid
                    connected_wall_ids.append(relating_wall_guid)

        wall_unit = WallUnit(guid_, axis_, width_, wall, level_id_, wall_z)
        wall_unit.DoorPoints = related_door_points
        wall_unit.ConnectedWallsIds = connected_wall_ids
        wall_unit.WallGuidsAtStart = wall_guid_at_start
        wall_unit.WallGuidsAtEnd = wall_guid_at_end
        wall_unit.ConnectionPoints[0.0] = wall_unit.Axis.StartPoint()
        wall_unit.ConnectionPoints[1.0] = wall_unit.Axis.EndPoint()

        wall_units.append(wall_unit)
        wall_unit_ids.append(guid_)
        wall_units_dict[guid_] = wall_unit


    return wall_units_dict  # wall_units, wall_unit_ids


def get_wall_at_end(file, ifc_wall):
    # w = ifc_wall.ConnectedTo[0].RelatedElement
    # return w
    wall_relations = file.get_inverse(ifc_wall)

    t = 0

    for wall_relation in wall_relations:
        if wall_relation.is_a('IfcRelConnectsPathElements') and wall_relation.RelatedConnectionType == 'ATEND':
            t = t + 1

    for wall_relation in wall_relations:
        if wall_relation.is_a('IfcRelConnectsPathElements') and wall_relation.RelatingConnectionType == 'ATEND':
            connected_wall_guid = wall_relation.get_info().get('RelatedElement').get_info().get('GlobalId')
            connected_wall = file.by_guid(connected_wall_guid)
            return connected_wall
    w = ifc_wall.ConnectedTo[0].RelatedElement
    return None


def get_wall_at_start(file, ifc_wall):
    wall_relations = file.get_inverse(ifc_wall)
    for wall_relation in wall_relations:
        if wall_relation.is_a('IfcRelConnectsPathElements') and wall_relation.RelatingConnectionType == 'ATSTART':
            connected_wall_guid = wall_relation.get_info().get('RelatedElement').get_info().get('GlobalId')
            connected_wall = file.by_guid(connected_wall_guid)
            return connected_wall
    return None


def display_wall_axes(filepath):
    file = ifcopenshell.open(filepath)
    wall_units_dict = get_wall_units(file)
    wall_units = wall_units_dict.values()
    wall_unit_ids = wall_units_dict.keys()

    zet = wall_units[0].Axis.StartPoint().Z()
    for wall_unit in wall_units:
        if wall_unit.Axis.StartPoint().Z() == zet:
            if wall_unit.Width != 0:
                start_pt = wall_unit.Axis.StartPoint()
                end_pt = wall_unit.Axis.EndPoint()
                transl_end_vec = gp_Vec(wall_unit.Tangent.X(), wall_unit.Tangent.Y(), wall_unit.Tangent.Z())
                transl_start_vec = gp_Vec(wall_unit.Tangent.X(), wall_unit.Tangent.Y(), wall_unit.Tangent.Z())
                wall_length = wall_unit.Tangent.Magnitude()

                end_width = 0
                start_width = 0

                wall_at_end = get_wall_at_end(file, wall_unit.ifcWall)
                wall_at_start = get_wall_at_start(file, wall_unit.ifcWall)
                if wall_at_end is not None:
                    end_width = get_width(file, wall_at_end)
                if wall_at_start is not None:
                    start_width = get_width(file, wall_at_start)

                # Make wall tangent vector unit length
                transl_end_vec.Scale(end_width / (2 * wall_length))
                # Translate wall end point
                transl_end_pt = gp_Pnt(end_pt.X(), end_pt.Y(), end_pt.Z())
                transl_end_pt.Translate(transl_end_vec)

                array1 = TColgp_Array1OfPnt(1, 2)
                array1.SetValue(1, transl_end_pt)
                array1.SetValue(2, end_pt)
                end_segment = GeomAPI_PointsToBSpline(array1, 1, 1).Curve()
                display.DisplayShape(end_segment, update=False, color="RED")

                # Make wall tangent vector unit length
                transl_start_vec.Scale(start_width / (2 * wall_length))
                # Translate wall end point
                transl_start_pt = gp_Pnt(start_pt.X(), start_pt.Y(), start_pt.Z())
                transl_start_pt.Translate(transl_start_vec)

                array2 = TColgp_Array1OfPnt(1, 2)
                array2.SetValue(1, transl_start_pt)
                array2.SetValue(2, start_pt)
                start_segment = GeomAPI_PointsToBSpline(array2, 1, 1).Curve()
                display.DisplayShape(start_segment, update=False, color="BLUE")

            display.DisplayShape(wall_unit.Axis, update=False, color="GREEN")
    display.FitAll()
    start_display()


def get_related_walls_max_width(wall_unit: WallUnit, file):
    wall_relations = file.get_inverse(wall_unit.ifcWall)
    connected_wall_widths = []
    for wall_relation in wall_relations:
        if wall_relation.is_a('IfcRelConnectsPathElements'):
            connected_wall_guid = wall_relation.get_info().get('RelatedElement').get_info().get('GlobalId')
            connected_wall = file.by_guid(connected_wall_guid)
            if connected_wall.Representation is None:
                continue
            wall_unit.ConnectedWallsIds.append(connected_wall_guid)
            connected_wall_width = get_width(file, connected_wall)
            if connected_wall_width is None:
                continue
            # if wall_relation.RelatingConnectionType == 'ATEND':
            connected_wall_widths.append(connected_wall_width)
    if len(connected_wall_widths) > 0:
        connected_wall_widths.sort()
        a = connected_wall_widths[len(connected_wall_widths) - 1]
        return a
    else:
        return 0


def split_edge_with_edge(target_shapes, splitter_shapes):
    # Initialize splitter
    splitter = BOPAlgo_Splitter()
    # Add the edge1 as an argument and edge2 as a tool. This will split
    # edge1 with edge2.
    splitter.AddArgument(target_shapes)
    splitter.AddTool(splitter_shapes)
    splitter.Perform()

    edges = []
    exp = TopExp_Explorer(splitter.Shape(), TopAbs_EDGE)
    while exp.More():
        edges.append(exp.Current())
        exp.Next()

    return edges


def get_wall_connectivity(file, storey=None):
    wall_units_dict = get_wall_units(file, storey=storey)
    wall_units = list(wall_units_dict.values())
    wall_unit_ids = list(wall_units_dict.keys())

    get_wall_units_neighbors(wall_units)

    all_wall_segments = []

    for i in range(len(wall_units)):

        proj_points = []
        proj_params = []

        end_connected = False
        start_connected = False

        for connected_wall_guid in wall_units[i].ConnectedWallsIds:
            if connected_wall_guid in wall_unit_ids:
                connected_wall_unit = wall_units_dict.get(connected_wall_guid)

                # Find the closest point on each of the the two related walls
                dss = BRepExtrema_DistShapeShape(wall_units[i].AxisEdge, connected_wall_unit.AxisEdge)

                point_on_wall = dss.PointOnShape1(1)
                point_on_connected_wall = dss.PointOnShape2(1)

                # Find the parameter of the closest point on the current wall by projecting it
                # Instantiate projection point
                proj_point = gp_Pnt()
                proj_point2 = gp_Pnt()
                distance, proj_param = ShapeAnalysis_Curve().Project(wall_units[i].Axis, point_on_wall,
                                                                     Tolerance, proj_point)

                distance2, proj_param2 = ShapeAnalysis_Curve().Project(connected_wall_unit.Axis,
                                                                       point_on_connected_wall,
                                                                       Tolerance, proj_point2)
                # Find the distance between the closest point on the connected wall
                # and the closest point on the current wall
                dist = point_on_wall.Distance(point_on_connected_wall)

                min_width = wall_units[i].Width / 2

                if proj_param2 != 0.0 and proj_param2 != 1.0:
                    min_width = connected_wall_unit.Width / 2

                if connected_wall_guid == wall_units[i].WallGuidsAtEnd:
                    proj_points.append(proj_point2)
                    proj_params.append(1.1)
                    end_connected = True
                    continue
                if connected_wall_guid == wall_units[i].WallGuidsAtStart:
                    proj_points.append(proj_point)
                    proj_params.append(-0.1)
                    start_connected = True
                    continue

                # If the distance is less than the project tolerance and if the parameter is in (0,1)
                # then current wall must be split at this point tolerance dist < min_width + tolerance and
                if proj_param > 0.00001 and proj_param < 0.99999:
                    proj_points.append(proj_point)
                    proj_params.append(proj_param)
                # if dist < min_width + tolerance and proj_param > 0.99999:
                #    proj_points.append(proj_point2)
                #    proj_params.append(1.1)
                # if dist < min_width + tolerance and proj_param < 0.00001:
                #    proj_points.append(proj_point2)
                #    proj_params.append(-0.1)

        points = []
        parameters = []

        # Add start and end point of current wall
        if not start_connected:
            proj_points.append(wall_units[i].Axis.StartPoint())
            proj_params.append(0.0)
        if not end_connected:
            proj_points.append(wall_units[i].Axis.EndPoint())
            proj_params.append(1.0)

        # Sort points by their corresponding parameter
        zip_data = sorted(zip(proj_params, proj_points), key=lambda pair: pair[0])

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
            if dist > Tolerance:
                # Add segments as a WallSegment Class to the current WallUnit
                wall_segment = WallSegment(i + p + 1, segment, wall_units[i].Tangent.Magnitude(), wall_units[i].Width,
                                           wall_units[i].GlobalId)
                wall_units[i].Segments.append(wall_segment)

                # Store all wall segments in a list
                all_wall_segments.append(wall_segment)

                display.DisplayShape(wall_segment.Curve, update=False, color="GREEN")
                display.DisplayShape(wall_segment.Curve.EndPoint(), update=False, color="RED")

    int_graph = nx.Graph()

    for i in range(len(all_wall_segments) - 1):
        for j in range(i + 1, len(all_wall_segments)):
            if all_wall_segments[j].Curve.StartPoint().Z() == all_wall_segments[i].Curve.StartPoint().Z():
                proj_point1 = gp_Pnt()
                proj_point2 = gp_Pnt()
                proj_point3 = gp_Pnt()
                proj_point4 = gp_Pnt()

                dist_start_to_start, proj_param1 = ShapeAnalysis_Curve().Project(all_wall_segments[i].Curve,
                                                                                 all_wall_segments[
                                                                                     j].Curve.StartPoint(),
                                                                                 Tolerance,
                                                                                 proj_point1)
                dist_start_to_end, proj_param2 = ShapeAnalysis_Curve().Project(all_wall_segments[i].Curve,
                                                                               all_wall_segments[j].Curve.EndPoint(),
                                                                               Tolerance,
                                                                               proj_point2)
                dist_end_to_start, proj_param3 = ShapeAnalysis_Curve().Project(all_wall_segments[j].Curve,
                                                                               all_wall_segments[i].Curve.StartPoint(),
                                                                               Tolerance,
                                                                               proj_point3)
                dist_end_to_end, proj_param4 = ShapeAnalysis_Curve().Project(all_wall_segments[j].Curve,
                                                                             all_wall_segments[i].Curve.EndPoint(),
                                                                             Tolerance,
                                                                             proj_point4)

                min_dist = min([dist_start_to_start, dist_start_to_end, dist_end_to_end, dist_end_to_start])

                if min_dist < Tolerance:
                    weight = all_wall_segments[i].Mid.Distance(all_wall_segments[j].Mid)
                    # weight = (all_wall_segments[i].HostWallLength + all_wall_segments[j].HostWallLength) / 2
                    int_graph.add_edge(i, j, weight=weight)

                    int_graph.nodes[i]['label'] = all_wall_segments[i].Label
                    int_graph.nodes[i]['mid'] = all_wall_segments[i].Mid
                    int_graph.nodes[i]['pos'] = np.array([all_wall_segments[i].Mid.X(), all_wall_segments[i].Mid.Y()])
                    int_graph.nodes[i]['length'] = all_wall_segments[i].HostWallLength
                    int_graph.nodes[i]['type'] = 'wall'

                    int_graph.nodes[j]['label'] = all_wall_segments[j].Label
                    int_graph.nodes[j]['mid'] = all_wall_segments[j].Mid
                    int_graph.nodes[j]['pos'] = np.array([all_wall_segments[j].Mid.X(), all_wall_segments[j].Mid.Y()])
                    int_graph.nodes[j]['length'] = all_wall_segments[j].HostWallLength
                    int_graph.nodes[j]['type'] = 'wall'

    start_display()
    G = int_graph

    """
    G2 = G.copy()

    nodes_to_remove = []
    edges_to_add = []

    for node in G.nodes(data=True):
        if G.nodes[node[0]]['is_conn']:
            nodes_to_remove.append(node[0])
            neighbors1 = list(G.neighbors(node[0]))
            neighbors2 = neighbors1.copy()
            neighbors2.pop(0)
            neighbors2.append(neighbors1[0])
            for neighbor1, neighbor2 in zip(neighbors1, neighbors2):
                if G.nodes[neighbor1]['is_conn'] == False and G.nodes[neighbor2]['is_conn'] == False:
                    w = G.nodes[neighbor1]['mid'].Distance(G.nodes[neighbor2]['mid'])
                    G2.add_edge(neighbor1, neighbor2, weight=w)
                    #(G.nodes[neighbor1]['length'] + G.nodes[neighbor2]['length']) / 2)

    G2.remove_nodes_from(nodes_to_remove)
    G = G2
    """
    return G


def get_geometric_wall_connectivity(file, storey=None, elevation=None):
    wall_units_dict = get_wall_units(file, storey=storey)
    wall_units = list(wall_units_dict.values())
    wall_unit_ids = list(wall_units_dict.keys())

    get_wall_units_neighbors(wall_units_dict)
    get_wall_units_neighbors_ifc(wall_units_dict)

    all_wall_segments = []

    for i in range(len(wall_units)):

        points = []
        parameters = []

        #wall_units[i].ConnectionPoints[0.0] = wall_units[i].Axis.StartPoint()
        #wall_units[i].ConnectionPoints[1.0] = wall_units[i].Axis.EndPoint()

        """

        for connected_wall_unit in wall_unit.ConnectedWallUnits:

            # Find the closest point on each of the the two related walls
            wall_edge, connected_wall_edge = make_edge(wall_units[i].Axis), make_edge(wall_units[j].Axis)
            dss = BRepExtrema_DistShapeShape(wall_edge, connected_wall_edge)

            point_on_wall = dss.PointOnShape1(1)
            point_on_connected_wall = dss.PointOnShape2(1)

            # Find the parameter of the closest point on the current wall by projecting it
            # Instantiate projection point
            proj_point = gp_Pnt()
            distance, proj_param = ShapeAnalysis_Curve().Project(wall_units[i].Axis, point_on_wall,
                                                                 tolerance, proj_point)

            # Find the distance between the closest point on the connected wall
            # and the closest point on the current wall
            dist = point_on_wall.Distance(point_on_connected_wall)

            # If the distance is less than the project tolerance and if the parameter is in (0,1)
            # then current wall must be split at this point
            # max_width = max([wall_units[i].Width, wall_units[j].Width, mindist])
            max_width = mindist
            # use 0.501*max_width for unconnected axes
            if dist < 0.501 * max_width and proj_param != 0.0 and proj_param != 1.0:
                proj_points.append(proj_point)
                proj_params.append(proj_param)

        
        wall_relations = file.get_inverse(wall_units[i].ifcWall)
        for wall_relation in wall_relations:
            if wall_relation.is_a('IfcRelConnectsPathElements'):
                connected_wall_guid = wall_relation.get_info().get('RelatedElement').get_info().get('GlobalId')
                connected_wall = file.by_guid(connected_wall_guid)

                # Check if wall has representation
                if connected_wall.Representation is None:
                    # If wall has no representation continue the for loop
                    continue

                # Query all wall representations
                connected_wall_representations = connected_wall.Representation.get_info().get('Representations')

                # Check if wall has Axis representation
                has_axis = False
                for connected_wall_representation in connected_wall_representations:
                    if connected_wall_representation.get_info().get('RepresentationIdentifier') == 'Axis':
                        has_axis = True

                if not has_axis:
                    # If wall has no 2D Curve representation continue the for loop
                    continue

                # Append GuId of connected wall to current wall
                wall_units[i].RelatedWallsIds.append(connected_wall_guid)

                # Find the closest point on each of the the two related walls
                wall_edge, connected_wall_edge = make_edge(wall_units[i].Axis), make_edge(get_wall_axis(connected_wall))
                dss = BRepExtrema_DistShapeShape(wall_edge, connected_wall_edge)

                point_on_wall = dss.PointOnShape1(1)
                point_on_connected_wall = dss.PointOnShape2(1)

                # Find the parameter of the closest point on the current wall by projecting it
                # Instantiate projection point
                proj_point = gp_Pnt()
                distance, proj_param = ShapeAnalysis_Curve().Project(wall_units[i].Axis, point_on_wall,
                                                                     tolerance, proj_point)

                # Find the distance between the closest point on the connected wall
                # and the closest point on the current wall
                dist = point_on_wall.Distance(point_on_connected_wall)

                # If the distance is less than the project tolerance and if the parameter is in (0,1)
                # then current wall must be split at this point
                if dist < tolerance and proj_param != 0.0 and proj_param != 1.0:
                    proj_points.append(proj_point)
                    proj_params.append(proj_param)
        
        points = []
        parameters = []

        # Add start and end point of current wall

        proj_points.append(wall_units[i].Axis.EndPoint())
        proj_params.append(1.0)

        proj_points.append(wall_units[i].Axis.StartPoint())
        proj_params.append(0.0)
        """
        # Sort points by their corresponding parameter
        zip_data = sorted(zip(wall_units[i].ConnectionPoints.keys(),  wall_units[i].ConnectionPoints.values()), key=lambda pair: pair[0])

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
            if dist > Tolerance:
                # Check if segment is near a door
                cp = np.infty
                for door_point in wall_units[i].DoorPoints:
                    # Find the parameter of the closest point on the current segment by projecting it
                    # Instantiate projection point
                    proj_point = gp_Pnt()
                    distance, proj_param = ShapeAnalysis_Curve().Project(segment, door_point, Tolerance, proj_point)
                    if distance < cp:
                        cp = distance

                # Add segments as a WallSegment Class to the current WallUnit
                wall_segment = WallSegment(i + p + 1, segment, wall_units[i].Tangent.Magnitude(), wall_units[i].Width,
                                           wall_units[i].GlobalId)
                if cp < ProjectTolerance:
                    wall_segment.HasDoor = True

                wall_units[i].Segments.append(wall_segment)

                # Store all wall segments in a list
                all_wall_segments.append(wall_segment)

                display.DisplayShape(wall_segment.Curve, update=False, color="GREEN")
                #display.DisplayShape(wall_segment.Curve.EndPoint(), update=False, color="RED")
                display.DisplayShape(wall_segment.Curve.StartPoint(), update=False, color="RED")

    start_display()   

    int_graph = nx.Graph()

    for i in range(len(all_wall_segments)):
        for j in range(i + 1, len(all_wall_segments)):
            if j != i:  # all_wall_segments[j].Curve.StartPoint().Z() == all_wall_segments[i].Curve.StartPoint().Z() and
                proj_point1 = gp_Pnt()
                proj_point2 = gp_Pnt()
                proj_point3 = gp_Pnt()
                proj_point4 = gp_Pnt()

                dist_start_to_start, proj_param1 = ShapeAnalysis_Curve().Project(all_wall_segments[i].Curve,
                                                                                 all_wall_segments[
                                                                                     j].Curve.StartPoint(),
                                                                                 Tolerance,
                                                                                 proj_point1)
                dist_start_to_end, proj_param2 = ShapeAnalysis_Curve().Project(all_wall_segments[i].Curve,
                                                                               all_wall_segments[j].Curve.EndPoint(),
                                                                               Tolerance,
                                                                               proj_point2)
                dist_end_to_start, proj_param3 = ShapeAnalysis_Curve().Project(all_wall_segments[j].Curve,
                                                                               all_wall_segments[i].Curve.StartPoint(),
                                                                               Tolerance,
                                                                               proj_point3)
                dist_end_to_end, proj_param4 = ShapeAnalysis_Curve().Project(all_wall_segments[j].Curve,
                                                                             all_wall_segments[i].Curve.EndPoint(),
                                                                             Tolerance,
                                                                             proj_point4)
                dist_list = [dist_start_to_start, dist_start_to_end, dist_end_to_end, dist_end_to_start]
                dist_set = set(dist_list)
                min_dist = min([dist_start_to_start, dist_start_to_end, dist_end_to_end, dist_end_to_start])
                max_width = max([all_wall_segments[i].HostWallWidth, all_wall_segments[j].HostWallWidth, ProjectTolerance])
                #max_width = mindist
                if len(dist_set) == len(dist_list):
                    ok = 1

                # use 0.501*max_width for unconnected axes
                if min_dist < Tolerance:# and len(dist_set) == len(dist_list):
                    #weight = all_wall_segments[i].Mid.Distance(all_wall_segments[j].Mid)
                    weight = (all_wall_segments[i].HostWallLength + all_wall_segments[j].HostWallLength) / 2
                    int_graph.add_edge(i, j, weight=weight)

                    int_graph.nodes[i]['label'] = all_wall_segments[i].Label
                    int_graph.nodes[i]['pos'] = np.array([all_wall_segments[i].Mid.X(), all_wall_segments[i].Mid.Y()])
                    int_graph.nodes[i]['start'] = np.array(
                        [all_wall_segments[i].Curve.StartPoint().X(), all_wall_segments[i].Curve.StartPoint().Y()])
                    int_graph.nodes[i]['end'] = np.array(
                        [all_wall_segments[i].Curve.EndPoint().X(), all_wall_segments[i].Curve.EndPoint().Y()])
                    int_graph.nodes[i]['length'] = all_wall_segments[i].HostWallLength
                    int_graph.nodes[i]['door'] = all_wall_segments[i].HasDoor
                    int_graph.nodes[i]['type'] = 'wall'
                    int_graph.nodes[i]['exterior'] = False

                    int_graph.nodes[j]['label'] = all_wall_segments[j].Label
                    int_graph.nodes[j]['pos'] = np.array([all_wall_segments[j].Mid.X(), all_wall_segments[j].Mid.Y()])
                    int_graph.nodes[j]['start'] = np.array(
                        [all_wall_segments[j].Curve.StartPoint().X(), all_wall_segments[j].Curve.StartPoint().Y()])
                    int_graph.nodes[j]['end'] = np.array(
                        [all_wall_segments[j].Curve.EndPoint().X(), all_wall_segments[j].Curve.EndPoint().Y()])
                    int_graph.nodes[j]['length'] = all_wall_segments[j].HostWallLength
                    int_graph.nodes[j]['door'] = all_wall_segments[j].HasDoor
                    int_graph.nodes[j]['type'] = 'wall'
                    int_graph.nodes[j]['exterior'] = False

    
    G = int_graph

    return G


def get_room_connectivity(file, G: nx.Graph):
    cycles = [sorted(c) for c in nx.minimum_cycle_basis(G, weight='weight')]

    src_index = max(list(G.nodes)) + 1

    Gc = G.copy()

    for cycle in cycles:
        if len(cycle) > 3:
            positions = []

            for key in cycle:
                Gc.add_edge(src_index, key)
                positions.append(G.nodes[key]['pos'])

            Gc.nodes[src_index]['label'] = src_index
            Gc.nodes[src_index]['pos'] = np.mean(np.array(positions), axis=0)
            Gc.nodes[src_index]['type'] = 'room'

            src_index = src_index + 1

    return Gc


def find_ext_walls(G):
    for node in G.nodes:
        if G.nodes[node]['type'] == 'wall':
            neighbors = [G.nodes[n]['type'] for n in G.neighbors(node)]
            a = neighbors.count('room')
            if a == 1:
                G.nodes[node]['exterior'] = True
            else:
                G.nodes[node]['exterior'] = False

    return 0


def plot_graph(G: nx.Graph, plot_name):
    fig = plt.figure("Degree of a random graph", figsize=(15, 15))

    # fig.gca().set_aspect('equal', adjustable='box')
    # Set figure size and aspect ratio to 1.0

    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(5, 5)
    ax0 = fig.add_subplot(axgrid[0:5, :])
    ax0.set_aspect('equal', adjustable='box')

    room_nodes = []
    room_labels = []
    room_positions = []

    int_wall_nodes = []
    int_wall_labels = []
    int_wall_positions = []

    ext_wall_nodes = []
    ext_wall_labels = []
    ext_wall_positions = []

    for node in G:
        if G.nodes[node]['type'] == 'room':
            # Get node type from type key
            room_nodes.append(node)
            # Get node label from label key
            room_labels.append((node, G.nodes[node]['label']))
            # Get positions for room nodes from position key
            room_positions.append((node, G.nodes[node]['pos']))
        elif G.nodes[node]['type'] == 'wall':
            if G.nodes[node]['exterior']:
                # Get node type from type key
                ext_wall_nodes.append(node)
                # Get node label from label key
                ext_wall_labels.append((node, G.nodes[node]['label']))
                # Get positions for room nodes from position key
                ext_wall_positions.append((node, G.nodes[node]['pos']))
                # Plot wall segment axis
                x_values = [G.nodes[node]['start'][0], G.nodes[node]['end'][0]]
                y_values = [G.nodes[node]['start'][1], G.nodes[node]['end'][1]]
                ax0.plot(x_values, y_values, label='exterior walls', linewidth=7.0, color='#8d9091')

            else:
                # Get node type from type key
                int_wall_nodes.append(node)
                # Get node label from label key
                int_wall_labels.append((node, G.nodes[node]['label']))
                # Get positions for room nodes from position key
                int_wall_positions.append((node, G.nodes[node]['pos']))
                # Plot wall segment axis
                x_values = [G.nodes[node]['start'][0], G.nodes[node]['end'][0]]
                y_values = [G.nodes[node]['start'][1], G.nodes[node]['end'][1]]
                ax0.plot(x_values, y_values, label='exterior walls', linewidth=3.0, color='#8d9091')

    room_door_edges = []
    room_wall_edges = []
    wall_edges = []

    for edge in G.edges:
        if G.nodes[edge[0]]['type'] == 'room' and G.nodes[edge[1]]['type'] == 'wall':
            if G.nodes[edge[1]]['door']:
                room_door_edges.append(edge)
            else:
                room_wall_edges.append(edge)
        elif G.nodes[edge[1]]['type'] == 'room' and G.nodes[edge[0]]['type'] == 'wall':
            if G.nodes[edge[0]]['door']:
                room_door_edges.append(edge)
            else:
                room_wall_edges.append(edge)
        elif G.nodes[edge[0]]['type'] == 'wall' and G.nodes[edge[1]]['type'] == 'wall':
            wall_edges.append(edge)

    room_options = {"edgecolors": "#000000", "node_size": 200}
    wall_options = {"edgecolors": "#000000", "node_size": 50}

    if len(int_wall_nodes) > 0:
        nx.draw_networkx_nodes(G, dict(int_wall_positions), ax=ax0, nodelist=int_wall_nodes, node_color="#f2e0e5",
                               **wall_options,
                               label='label')
        # nx.draw_networkx_labels(G, dict(int_wall_positions), dict(int_wall_labels), font_size=7, font_color="#000000")

    if len(ext_wall_nodes) > 0:
        nx.draw_networkx_nodes(G, dict(ext_wall_positions), ax=ax0, nodelist=ext_wall_nodes, node_color="#98c5e3",
                               **wall_options,
                               label='label')
        # nx.draw_networkx_labels(G, dict(ext_wall_positions), dict(ext_wall_labels), font_size=7, font_color="#000000")

    if len(room_nodes) > 0:
        nx.draw_networkx_nodes(G, dict(room_positions), ax=ax0, nodelist=room_nodes, node_color="#FFC300",
                               **room_options,
                               label='label')
        # nx.draw_networkx_labels(G, dict(room_positions), dict(room_labels), font_size=7, font_color="#000000")

    nx.draw_networkx_edges(G, G.nodes(data='pos'), ax=ax0, edgelist=room_door_edges, width=2, alpha=0.7,
                           edge_color="#000000")
    nx.draw_networkx_edges(G, G.nodes(data='pos'), ax=ax0, edgelist=room_wall_edges, style="dashed", width=1, alpha=0.2,
                           edge_color="#000000")
    nx.draw_networkx_edges(G, G.nodes(data='pos'), ax=ax0, edgelist=wall_edges, width=1, alpha=0.2, edge_color="#000000")

    ax0.set_title("Floor plan connectivity graph")
    ax0.set_axis_off()
    """
    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)

    ax1 = fig.add_subplot(axgrid[3:, :2])
    ax1.plot(degree_sequence, "b-", marker="o")
    ax1.set_title("Degree Rank Plot")
    ax1.set_ylabel("Degree")
    ax1.set_xlabel("Rank")

    ax2 = fig.add_subplot(axgrid[3:, 2:])
    ax2.bar(*np.unique(degree_sequence, return_counts=True))
    ax2.set_title("Degree histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("# of Nodes")
    """
    fig.tight_layout()
    plt.savefig(plot_name, format='png')
    plt.show()
