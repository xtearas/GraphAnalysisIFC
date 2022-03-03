import networkx as nx

from definitions.utilities import plot_room_connectivity_graph, find_exterior_walls, plot_wall_connectivity_graph, plot_walls
from settings import FilePath, SavePath
from definitions.objects import IFCProject

if __name__ == '__main__':

    # Set index of building storey
    storey_index = 0

    # Concatenate graph name for respective building
    graph_name = FilePath.split('/')[-1].split('.')[0] + '_storey_' + str(storey_index)

    # Initialize IFCProject class for storing graph based data
    ifcProject = IFCProject(FilePath, storey_index)

    # Extract all wall units
    ifcProject.get_wall_units()

    # Find wall neighbours based on geometric intersection
    ifcProject.get_wall_units_neighbors()

    # Find wall neighbours based on ifc relationships
    ifcProject.get_wall_units_neighbors_ifc()

    # Construct wall segment connectivity graphs
    ifcProject.get_geometric_wall_connectivity()

    # Write the maximal subgraph wall connectivity graph
    nx.write_gpickle(ifcProject.WallConnectivityMaxSubGraph, SavePath + graph_name + '_wall_connectivity' + '.gpickle')

    # Plot wall axes and their intersection points
    plot_walls(ifcProject.WallConnectivityMaxSubGraph, plot_name=SavePath + graph_name + '_walls_plot' + '.png')

    # Plot wall connectivity graph
    plot_wall_connectivity_graph(ifcProject.WallConnectivityMaxSubGraph, plot_name=SavePath + graph_name + '_wall_connectivity_plot' + '.png')

    # Find room connectivity graph from the maximal wall connectivity graph using minimum weight cycle basis decomposition
    ifcProject.get_room_connectivity()

    # Find exterior walls from the room connectivity graph
    find_exterior_walls(ifcProject.RoomConnectivityGraph)

    # Write room connectivity graph
    nx.write_gpickle(ifcProject.RoomConnectivityGraph, SavePath + graph_name + '_room_connectivity' + '.gpickle')

    # Plot room connectivity graph
    plot_room_connectivity_graph(ifcProject.RoomConnectivityGraph, plot_name=SavePath + graph_name + '_room_connectivity_plot' + '.png')
