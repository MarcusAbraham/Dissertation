import sys
import random
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox
from sklearn.metrics import accuracy_score, classification_report
from torch_geometric.graphgym import optim
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
import plotly.graph_objs as go
from ModelView import Ui_ModelView
from data_gathering import get_compound_info, calculate_distance
import graph_creation


# Class for the 3D-DDI-GCN Model
class GCNGraphClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNGraphClassifier, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, edge_attributes, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Apply convolutions and ReLu
        x = self.conv1(x, edge_index, edge_attributes)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attributes)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_attributes)
        x = F.relu(x)

        # Compute attention weights
        attention_weights = F.softmax(self.attention(x), dim=0)
        # Weighted sum of node embeddings
        x_weighted = torch.matmul(x.t(), attention_weights).squeeze()

        # Pool the node level embeddings learned from the hidden layers
        x = global_mean_pool(x, batch)
        x = self.fc(x)

        return F.log_softmax(x, dim=1), attention_weights, x_weighted


# Class to define the model view
class ModelView(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the user interface from PyQT Designer
        self.ui = Ui_ModelView()
        self.ui.setupUi(self)

        # Connect the clicked signal of predictButton to 'predict_side_effect'
        self.ui.predictButton.clicked.connect(self.predict_side_effect)

    # Function to predict the side effect for a given drug pair
    def predict_side_effect(self):
        # Retrieve input from Id1 and Id2 input fields
        Id1 = self.ui.Id1.text()
        Id2 = self.ui.Id2.text()

        # Check if either of the input fields is empty, if so send an error message
        if not Id1 or not Id2:
            msg = QMessageBox()
            msg.setText("Please fill in both ID fields.")
            msg.setStyleSheet("background-color: rgb(71, 52, 209);color:white;")
            msg.exec()
        else:
            # Otherwise, begin the prediction
            # Get the compound ID of varying length, remove trailing 0's
            try:
                Id1 = str(int(re.search(r'\d+', Id1).group()))
                Id2 = str(int(re.search(r'\d+', Id2).group()))
            except:
                print("User entered invalid Ids")

            # Gather data for both Ids
            result1 = get_compound_info(Id1)
            result2 = get_compound_info(Id2)
            smiles1, coords1, bonds1, charges1 = result1
            smiles2, coords2, bonds2, charges2 = result2

            # Check if 3d data is returned for both Id's
            if coords1 is None or coords2 is None:
                # If not, prompt an error
                if coords1 is None:
                    msg = QMessageBox()
                    msg.setText("First ID is invalid. Please enter a valid ID.")
                    msg.setStyleSheet("background-color: rgb(71, 52, 209);color:white;")
                    msg.exec()
                if coords2 is None:
                    msg = QMessageBox()
                    msg.setText("Second ID is invalid. Please enter a valid ID")
                    msg.setStyleSheet("background-color: rgb(71, 52, 209);color:white;")
                    msg.exec()
            else:
                # Calculate the first compounds bond lengths
                lengths1 = []
                for bond1 in bonds1:
                    atom1_index, atom2_index, _ = bond1
                    atom1_coords = coords1[atom1_index - 1][2:]
                    atom2_coords = coords1[atom2_index - 1][2:]
                    distance = calculate_distance(atom1_coords, atom2_coords)
                    lengths1.append((atom1_index, atom2_index, distance))

                # Calculate the second compounds bond lengths
                lengths2 = []
                for bond2 in bonds2:
                    atom1_index, atom2_index, _ = bond2
                    atom1_coords = coords2[atom1_index - 1][2:]
                    atom2_coords = coords2[atom2_index - 1][2:]
                    distance = calculate_distance(atom1_coords, atom2_coords)
                    lengths2.append((atom1_index, atom2_index, distance))

                # Construct atom graphs for compound 1, and add charges, bonds, length
                graph1 = graph_creation.create_coordinate_graphs([str(coords1)])
                graph1 = graph_creation.add_charges(graph1, [str(charges1)])
                graph1 = graph_creation.add_bonds(graph1, [str(bonds1)])
                graph1 = graph_creation.add_lengths(graph1, [str(lengths1)])[0]

                # Construct atom graphs for compound 2, and add charges, bonds, length
                graph2 = graph_creation.create_coordinate_graphs([str(coords2)])
                graph2 = graph_creation.add_charges(graph2, [str(charges2)])
                graph2 = graph_creation.add_bonds(graph2, [str(bonds2)])
                graph2 = graph_creation.add_lengths(graph2, [str(lengths2)])[0]

                # Convert the graphs to data objects
                features1, edge_index1, edge_attributes1 = graph_creation.graph_to_data(graph1)
                features2, edge_index2, edge_attributes2 = graph_creation.graph_to_data(graph2)

                input_data = None
                # Concatenate the data objects
                if features1 is not None and features2 is not None:
                    # Concatenate features and edge indexes
                    concatenated_features = torch.cat((features1, features2), dim=0)
                    # Concatenate the padded edge indices
                    concatenated_edge_index = torch.cat([edge_index1, edge_index2], dim=1)
                    # Concatenate edge attributes
                    concatenated_edge_attributes = torch.cat([edge_attributes1, edge_attributes2], dim=0)

                    # Create PyTorch Geometric Data object
                    input_data = Data(x=concatenated_features,
                                      edge_index=concatenated_edge_index,
                                      edge_attr=concatenated_edge_attributes)

                # Make a prediction using the trained model - retrieve the predicted label
                output, attention_weights, x_weighted = model(input_data)
                _, predicted_label = torch.max(output, 1)

                # Create a new dictionary with reversed key-value pairs
                idx_to_side_effect = {v: k for k, v in side_effect_to_idx.items()}

                # Display the predicted side effect in the UI
                self.ui.Prediction.setText(f"Prediction: {idx_to_side_effect.get(predicted_label.item())}")

                # Get the information for the node weights
                # Convert attention weights tensor to numpy array for easier manipulation
                attention_weights_np = attention_weights.detach().cpu().numpy()
                # Find the maximum attention weight
                max_attention_weight = np.max(attention_weights_np)
                # Identify nodes with maximum attention weight
                nodes_with_max_attention = np.where(attention_weights_np == max_attention_weight)[0]
                # Indexes for graphs start at 1 VS nodes_with_max_attention that starts at 0
                important_nodes = nodes_with_max_attention + 1

                # Open the windows which visualise the nodes
                visualise_3d(graph1, graph2, important_nodes)


# Function to train the 3D-DDI-GCN model
def train_model(train_data):
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        total_loss = 0

        # Set the model to training mode
        model.train()

        # Pass each data object in the training data to the model
        for graph_data in train_data:
            # Forward pass
            output = model(graph_data)[0]

            # Calculate the loss
            loss = criterion(output, graph_data.y)
            total_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print average loss
        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {total_loss / len(train_data)}")


# Function to evaluate the 3D-DDI-GCN model
def evaluate_model(test_data):
    # Evaluation loop
    with torch.no_grad():
        # Set the model to evaluation mode
        model.eval()
        true_labels = []
        predicted_labels = []

        # Pass each data object in the test data to the model, and get the predicted VS true labels
        for graph_data in test_data:
            output = model(graph_data)[0]
            _, predicted = torch.max(output, 1)
            true_labels.extend(graph_data.y.tolist())
            predicted_labels.extend(predicted.tolist())

        # Calculate and print the models accuracy
        accuracy = accuracy_score(true_labels, predicted_labels)
        print(f"Accuracy: {accuracy}")

        # Calculate precision, recall, and F1-score
        report = classification_report(true_labels, predicted_labels, zero_division=0)
        print(report)


# Function to visualise 3D-DDI-GCN's predictions
def visualise_3d(first_graph, second_graph, important_nodes):
    # Extract node positions, types and charges for first_graph
    node_type_first = [data['atom_type'] for _, data in first_graph.nodes(data=True)]
    node_x_first = [data['x'] for _, data in first_graph.nodes(data=True)]
    node_y_first = [data['y'] for _, data in first_graph.nodes(data=True)]
    node_z_first = [data['z'] for _, data in first_graph.nodes(data=True)]
    node_charge_first = [data.get('charge', '0') for _, data in first_graph.nodes(data=True)]

    # Define colors for nodes based on whether their 'index' property is in the important nodes list
    node_colors_first = ['yellow' if node_id in important_nodes else 'purple' for node_id, _ in
                         first_graph.nodes(data=True)]

    # Create node traces for first_graph with color based on node index
    node_trace_first = go.Scatter3d(
        x=node_x_first,
        y=node_y_first,
        z=node_z_first,
        mode='markers',
        marker=dict(
            size=8,
            color=node_colors_first,
            line=dict(color='rgb(0,0,0)', width=1)
        ),
        hoverinfo='text',
        hovertext=[f"Node ID: {node_id}<br>Type: {atom_type}<br>X: {x}<br>Y: {y}<br>Z: {z}<br>Charge: {charge}"
                   for node_id, atom_type, x, y, z, charge in
                   zip(first_graph.nodes(), node_type_first, node_x_first, node_y_first, node_z_first,
                       node_charge_first)]
    )

    # Create edge traces for first_graph
    edge_trace_first = []
    for edge in first_graph.edges(data=True):
        start = edge[0]
        end = edge[1]
        bond_value = edge[2].get('bond', 'N/A')

        # Adjust node IDs to start from 0-based indexing
        start_index = start - 1
        end_index = end - 1

        # Ensure node IDs are within the valid range
        if start_index in range(len(node_x_first)) and end_index in range(len(node_x_first)):
            edge_trace = go.Scatter3d(
                x=[node_x_first[start_index], node_x_first[end_index]],
                y=[node_y_first[start_index], node_y_first[end_index]],
                z=[node_z_first[start_index], node_z_first[end_index]],
                mode='lines',
                line=dict(color='rgb(125,125,125)', width=2),
                hoverinfo='text',
                hovertext=f"Start: {start}<br>End: {end}<br>Bond: {bond_value}<br>Length: {edge[2].get('length', 'N/A')}"
            )
            edge_trace_first.append(edge_trace)
        else:
            print(f"Issue with node IDs in edge: {start} -> {end}")

    # Extract node positions, types and charges for second_graph
    node_type_second = [data['atom_type'] for _, data in second_graph.nodes(data=True)]
    node_x_second = [data['x'] for _, data in second_graph.nodes(data=True)]
    node_y_second = [data['y'] for _, data in second_graph.nodes(data=True)]
    node_z_second = [data['z'] for _, data in second_graph.nodes(data=True)]
    node_charge_second = [data.get('charge', '0') for _, data in second_graph.nodes(data=True)]

    # Account for the length of the first graph (the important nodes assumes both graphs are concatenated
    important_nodes_second = [node - len(first_graph.nodes(data=True)) for node in important_nodes]

    # Define colors for nodes based on whether their 'index' property matches the number
    node_colors_second = ['yellow' if node_id in important_nodes_second else 'purple' for node_id, _ in
                          second_graph.nodes(data=True)]

    # Create node traces for second_graph with color based on node index
    node_trace_second = go.Scatter3d(
        x=node_x_second,
        y=node_y_second,
        z=node_z_second,
        mode='markers',
        marker=dict(
            size=8,
            color=node_colors_second,
            line=dict(color='rgb(0,0,0)', width=1)
        ),
        hoverinfo='text',
        hovertext=[f"Node ID: {node_id}<br>Type: {atom_type}<br>X: {x}<br>Y: {y}<br>Z: {z}<br>Charge: {charge}"
                   for node_id, atom_type, x, y, z, charge in
                   zip(second_graph.nodes(), node_type_second, node_x_second, node_y_second, node_z_second,
                       node_charge_second)]
    )

    # Create edge traces for second_graph
    edge_trace_second = []
    for edge in second_graph.edges(data=True):
        start = edge[0]
        end = edge[1]
        bond_value = edge[2].get('bond', 'N/A')

        # Adjust node IDs to start from 0-based indexing
        start_index = start - 1
        end_index = end - 1

        # Ensure node IDs are within the valid range
        if start_index in range(len(node_x_second)) and end_index in range(len(node_x_second)):
            edge_trace = go.Scatter3d(
                x=[node_x_second[start_index], node_x_second[end_index]],
                y=[node_y_second[start_index], node_y_second[end_index]],
                z=[node_z_second[start_index], node_z_second[end_index]],
                mode='lines',
                line=dict(color='rgb(125,125,125)', width=2),
                hoverinfo='text',
                hovertext=f"Start: {start}<br>End: {end}<br>Bond: {bond_value}<br>Length: {edge[2].get('length', 'N/A')}"
            )
            edge_trace_second.append(edge_trace)
        else:
            print(f"Issue with node IDs in edge: {start} -> {end}")

    # Create the visualisation layout
    layout = go.Layout(
        title='3D Graph Visualization',
        showlegend=False,
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
        ),
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        ),
        hovermode='closest'
    )

    # Combine traces and create figure for first_graph
    fig_first = go.Figure(data=[node_trace_first] + edge_trace_first, layout=layout)
    # Combine traces and create figure for second_graph
    fig_second = go.Figure(data=[node_trace_second] + edge_trace_second, layout=layout)

    # Show plots in two tabs
    fig_first.show()
    fig_second.show()


# Read the dataset
df = pd.read_csv("ChChSe-Decagon_polypharmacy/filteredData.csv")

# Extract 3d compound information and side effects
coords1 = df['C1 Coords']
charges1 = df['C1 Charges']
bonds1 = df['C1 Bonds']
lengths1 = df['C1 Computed Lengths']

coords2 = df['C2 Coords']
charges2 = df['C2 Charges']
bonds2 = df['C2 Bonds']
lengths2 = df['C2 Computed Lengths']

side_effects = df["Side Effect Name"]

# Create atom graphs for the first compound
atom_graphs1 = graph_creation.create_coordinate_graphs(coords1)
atom_graphs1 = graph_creation.add_charges(atom_graphs1, charges1)
atom_graphs1 = graph_creation.add_bonds(atom_graphs1, bonds1)
atom_graphs1 = graph_creation.add_lengths(atom_graphs1, lengths1)

# Create atom graphs for the second compound
atom_graphs2 = graph_creation.create_coordinate_graphs(coords2)
atom_graphs2 = graph_creation.add_charges(atom_graphs2, charges2)
atom_graphs2 = graph_creation.add_bonds(atom_graphs2, bonds2)
atom_graphs2 = graph_creation.add_lengths(atom_graphs2, lengths2)

# Map side effects to class indices
unique_side_effects = side_effects.unique()
side_effect_to_idx = {side_effect: idx for idx, side_effect in enumerate(unique_side_effects)}

# Concatenate the graph data
graphs_data = graph_creation.concatenate_data(atom_graphs1, atom_graphs2, side_effects, side_effect_to_idx)

# Assign random seeds
random.seed(42)
torch.manual_seed(42)

# Split data into train and test sets
random.shuffle(graphs_data)
split_index = int(0.8 * len(graphs_data))  # 80% train, 20% test
train_data = graphs_data[:split_index]
test_data = graphs_data[split_index:]

# Define the dimensions
input_dim = 4
hidden_dim = 32
output_dim = len(unique_side_effects)

# Create an instance of the model
model = GCNGraphClassifier(input_dim, hidden_dim, output_dim)
# Define the loss function
criterion = nn.CrossEntropyLoss()
# Define the optimizer
optimizer = optim.Adam(model.parameters())

# Test and train the model - WARNING: Train/test can take up to 20 minutes
train_model(train_data)
evaluate_model(test_data)

if __name__ == "__main__":
    # Load the pyqt view
    app = QApplication(sys.argv)
    main_window = ModelView()
    main_window.show()
    sys.exit(app.exec_())
