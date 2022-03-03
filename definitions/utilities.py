import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.placement

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from OCC.Core.ShapeAnalysis import ShapeAnalysis_Curve
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Lin
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnCurve, GeomAPI_PointsToBSpline
from OCC.Core.Geom import Geom_BSplineCurve, Geom_OffsetCurve
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Extend.ShapeFactory import make_edge, make_vertex
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCC.Core.BOPAlgo import BOPAlgo_Splitter
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.TopExp import TopExp_Explorer

# Viewport Customisation
from OCC.Core.AIS import AIS_Shape

from OCC.Extend.ShapeFactory import translate_shp


def get_project_length_unit(file):
    # Get project from file
    project = file.by_type('IfcProject')[0]

    # Get project length units
    units = project.UnitsInContext.Units
    project_length_units = 'none'
    for unit in units:
        unit_info = unit.get_info()
        if unit_info.get('UnitType') == 'LENGTHUNIT':
            if unit_info.get('Prefix') is not None:
                project_length_units = unit_info.get('Prefix') + unit_info.get('Name')
            else:
                project_length_units = unit_info.get('Name')

    return project_length_units


def get_project_tolerance(file):
    # Get project from file
    project = file.by_type('IfcProject')[0]

    # Get project representation contexts
    contexts = project.RepresentationContexts
    project_tolerance = 0.000001
    for context in contexts:
        context_info = context.get_info()
        if context_info.get('ContextType') == 'Plan':
            if context_info.get('Precision') is not None:
                project_tolerance = context_info.get('Precision')

    return project_tolerance


def get_wall_body(ifc_wall, z_coordinate=None):
    wall_boundary_points = []

    # Check if wall has representation
    if ifc_wall.Representation is None:
        return None

    # Query all wall representations
    wall_representations = ifc_wall.Representation.get_info().get('Representations')

    # Check if wall has Axis representation
    has_body = False
    for wall_representation in wall_representations:
        if wall_representation.get_info().get('RepresentationIdentifier') == 'Body':
            first_item = wall_representation.get_info().get('Items')[0]
            if first_item.is_a('IfcExtrudedAreaSolid'):
                boundary = first_item.get_info().get('SweptArea').get_info().get('OuterCurve')
                if boundary is not None:
                    points = boundary.get_info().get('Points')
                    if points is not None:
                        wall_boundary_points = points
                        has_body = True

    if not has_body:
        return None

    translation_matrix = ifcopenshell.util.placement.get_local_placement(ifc_wall.ObjectPlacement)
    transformation_matrix = ifcopenshell.util.placement.get_axis2placement(ifc_wall.ObjectPlacement.RelativePlacement)

    wall_z = float(translation_matrix[2, 3])
    if z_coordinate is not None:
        wall_z = z_coordinate

    wall_array = TColgp_Array1OfPnt(1, len(wall_boundary_points))

    for w in range(len(wall_boundary_points)):
        # Get Local point
        loc_point = np.array([(wall_boundary_points[w].get_info().get('Coordinates')[0]),
                              wall_boundary_points[w].get_info().get('Coordinates')[1], 0, 1])

        # Get Relative point
        rel_pt = np.matmul(transformation_matrix, loc_point.T)

        # Get polyline points
        pl_point = gp_Pnt(rel_pt[0], rel_pt[1], wall_z)
        wall_array.SetValue(w + 1, pl_point)

    # Represent wall polyline as a BSpline with degree 1 and number of points = 2
    wall_boundary = GeomAPI_PointsToBSpline(wall_array, 1, 1).Curve()

    return wall_boundary


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


def get_wall_width(file, ifc_wall):
    wall_relations = file.get_inverse(ifc_wall)
    thickness = 0

    if ifc_wall.Representation is not None:
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
    return thickness


def find_exterior_walls(G: nx.Graph):
    for node in G.nodes:
        if G.nodes[node]['type'] == 'wall':
            neighbors = [G.nodes[n]['type'] for n in G.neighbors(node)]
            a = neighbors.count('room')
            if a == 1:
                G.nodes[node]['exterior'] = True
            else:
                G.nodes[node]['exterior'] = False

    return 0


def plot_room_connectivity_graph(G: nx.Graph, plot_name: str):
    fig = plt.figure(figsize=(15, 15))

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
                ax0.plot(x_values, y_values, label='exterior walls', linewidth=8.0, color='#8d9091', zorder=0)

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
                ax0.plot(x_values, y_values, label='exterior walls', linewidth=4.0, color='#8d9091', zorder=1)

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

    room_options = {"edgecolors": "#000000", "node_size": 250, "linewidths": 3}
    wall_options = {"edgecolors": "#000000", "node_size": 80, "linewidths": 2}

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

    nx.draw_networkx_edges(G, G.nodes(data='pos'), ax=ax0, edgelist=room_door_edges, width=3, alpha=1.0,
                           edge_color="#000000")
    nx.draw_networkx_edges(G, G.nodes(data='pos'), ax=ax0, edgelist=room_wall_edges, style="dashed", width=3, alpha=0.2,
                           edge_color="#000000")

    ax0.set_title("Floor plan room connectivity graph")
    ax0.set_axis_off()

    fig.tight_layout()
    plt.savefig(plot_name, format='png')
    plt.show()


def plot_wall_connectivity_graph(G: nx.Graph, plot_name: str):
    fig = plt.figure(figsize=(15, 15))

    # fig.gca().set_aspect('equal', adjustable='box')
    # Set figure size and aspect ratio to 1.0

    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(5, 5)
    ax0 = fig.add_subplot(axgrid[0:5, :])
    ax0.set_aspect('equal', adjustable='box')

    int_wall_nodes = []
    int_wall_labels = []
    int_wall_positions = []

    for node in G:
        # Get node type from type key
        int_wall_nodes.append(node)
        # Get node label from label key
        int_wall_labels.append((node, G.nodes[node]['label']))
        # Get positions for room nodes from position key
        int_wall_positions.append((node, G.nodes[node]['pos']))
        # Plot wall segment axis
        x_values = [G.nodes[node]['start'][0], G.nodes[node]['end'][0]]
        y_values = [G.nodes[node]['start'][1], G.nodes[node]['end'][1]]
        ax0.plot(x_values, y_values, label='exterior walls', linewidth=3.0, color='#8d9091', zorder=0)

    wall_options = {"edgecolors": "#000000", "node_size": 160, "linewidths": 2}

    if len(int_wall_nodes) > 0:
        nx.draw_networkx_nodes(G, dict(int_wall_positions), ax=ax0, nodelist=int_wall_nodes, node_color="#f2e0e5",
                               **wall_options,
                               label='label')

    nx.draw_networkx_edges(G, G.nodes(data='pos'), ax=ax0, width=2, edge_color="#000000")

    ax0.set_title("Floor plan wall connectivity graph")
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


def plot_walls(G: nx.Graph, plot_name: str):
    fig = plt.figure("Degree of a random graph", figsize=(15, 15))

    # fig.gca().set_aspect('equal', adjustable='box')
    # Set figure size and aspect ratio to 1.0

    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(5, 5)
    ax0 = fig.add_subplot(axgrid[0:5, :])
    ax0.set_aspect('equal', adjustable='box')

    int_wall_nodes = []
    int_wall_labels = []
    int_wall_positions = []

    for node in G:
        # Get node type from type key
        int_wall_nodes.append(node)
        # Get node label from label key
        int_wall_labels.append((node, G.nodes[node]['label']))
        # Get positions for room nodes from position key
        int_wall_positions.append((node, G.nodes[node]['pos']))
        # Plot wall segment axis
        x_values = [G.nodes[node]['start'][0], G.nodes[node]['end'][0]]
        y_values = [G.nodes[node]['start'][1], G.nodes[node]['end'][1]]
        ax0.plot(x_values[0], y_values[0], linewidth=3.0, marker="x", markersize=8, markeredgecolor="red", zorder=1)
        ax0.plot(x_values[1], y_values[1], linewidth=3.0, marker="x", markersize=8, markeredgecolor="red", zorder=1)

        ax0.plot(x_values, y_values, label='exterior walls', linewidth=2.0, color='#000000', zorder=0)

    ax0.set_title("wall axes")
    ax0.set_axis_off()

    fig.tight_layout()
    plt.savefig(plot_name, format='png')
    plt.show()
