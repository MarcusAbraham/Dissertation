import ast
import networkx as nx
import torch
from torch_geometric.data import Data


# Iterate over each compound in coords1
def create_coordinate_graphs(compound_list):
    # List to store the atom graphs for each compound
    atom_graphs = []

    # Iterate over each compound string in the list
    for compound in compound_list:
        # Create an empty graph for the current compound
        atom_graph = nx.Graph()

        # Parse the string into a list of tuples using ast.literal_eval
        atom_list = ast.literal_eval(compound)

        # Iterate over each atom in the compound
        for atom in atom_list:
            index, atom_type, x, y, z = atom
            # Add node for each atom to the current compound graph
            atom_graph.add_node(index, atom_type=atom_type, x=x, y=y, z=z)

        # Append the current compound graph to the list of atom graphs
        atom_graphs.append(atom_graph)

    return atom_graphs


def add_charges(atom_graphs, compound_charges):
    # Iterate over each pair of graph and charges
    for graph, charges in zip(atom_graphs, compound_charges):
        # Parse the string into a list of tuples using ast.literal_eval
        charges_list = ast.literal_eval(charges)

        # Iterate over each charge tuple (index, charge)
        for index, charge in charges_list:
            # Check if the node with given index exists in the graph
            if index in graph.nodes:
                # Add charge attribute to the node
                graph.nodes[index]['charge'] = charge

    return atom_graphs


def add_bonds(atom_graphs, compound_bonds):
    # Iterate over each pair of graph and bonds
    for graph, bonds in zip(atom_graphs, compound_bonds):
        # Parse the string into a list of tuples using ast.literal_eval
        bonds_list = ast.literal_eval(bonds)

        # Iterate over each bond tuple (source_index, target_index, bond_value)
        for source_index, target_index, bond_value in bonds_list:
            # Add edge between nodes with specified bond value
            graph.add_edge(source_index, target_index, bond=bond_value)
    return atom_graphs


def add_lengths(atom_graphs, compound_lengths):
    # Iterate over each pair of graph and lengths
    for graph, lengths in zip(atom_graphs, compound_lengths):
        lengths_list = ast.literal_eval(lengths)

        # Iterate over each length tuple (source_index, target_index, length_value)
        for source_index, target_index, length_value in lengths_list:
            # Check if the edge exists in the graph
            if graph.has_edge(source_index, target_index):
                # Add 'length' attribute to the edge
                graph[source_index][target_index]['length'] = length_value
    return atom_graphs


def graph_to_data(graph):
    # Extract node features and coordinates
    node_features = []
    for node_id, data in graph.nodes(data=True):
        node_features.append([data.get('x', 0), data.get('y', 0), data.get('z', 0), data.get('charge', 0)])

    # Assuming graph is a NetworkX graph
    adjacency_matrix = nx.to_numpy_array(graph)

    # Assuming graph is a NetworkX graph
    edge_index = []
    edge_attributes = []

    # Iterate over edges and extract weights, lengths, and edge indices
    for idx, (source, target, data) in enumerate(graph.edges(data=True)):
        # Extract length if available, otherwise default to 0
        # WE'RE ONLY ADDING THE LENGTH TO THE EDGE ATTRIBUTES BECAUSE GCNConv ONLY ACCEPTS A TENSOR WITH 1 DIMENSION
        length = data.get('length', 0)

        # Append a tuple containing both weight and length to edge_attributes
        edge_attributes.append(length)

        # Append edge indices to edge_index
        edge_index.append([source, target])

    # Convert lists to tensors
    node_features = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attributes = torch.tensor(edge_attributes, dtype=torch.float)

    return node_features, edge_index, edge_attributes


def concatenate_data(atom_graphs1, atom_graphs2, side_effects, side_effect_to_idx):

    # Create list to store graph data
    graphs_data = []
    # Create Data objects for each drug pair
    for idx, (graphs1, graphs2, side_effect) in enumerate(zip(atom_graphs1, atom_graphs2, side_effects)):

        # Convert graphs to data format
        features1, edge_index1, edge_attributes1 = graph_to_data(graphs1)
        features2, edge_index2, edge_attributes2 = graph_to_data(graphs2)

        if features1 is not None and features2 is not None:
            # Concatenate features and edge indexes
            concatenated_features = torch.cat((features1, features2), dim=0)
            # Concatenate the padded edge indices
            concatenated_edge_index = torch.cat([edge_index1, edge_index2], dim=1)
            # Concatenate edge attributes
            concatenated_edge_attributes = torch.cat([edge_attributes1, edge_attributes2], dim=0)

            # Make the side effect into a tensor label
            label = torch.tensor([side_effect_to_idx[side_effect]])

            # Create PyTorch Geometric Data object
            data = Data(x=concatenated_features,
                        edge_index=concatenated_edge_index,
                        edge_attr=concatenated_edge_attributes,
                        y=label)

            # Append to graphs_data list
            graphs_data.append(data)

    return graphs_data