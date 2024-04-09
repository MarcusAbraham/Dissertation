import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from sklearn.metrics import accuracy_score, classification_report
from torch_geometric.graphgym import optim
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data


class GCNGraphClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNGraphClassifier, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)  # Use global mean pooling
        x = self.fc(x)

        return F.log_softmax(x, dim=1)


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        # Convert molecule to graph
        adj_matrix = torch.tensor(Chem.GetAdjacencyMatrix(mol))
        features = []
        for atom in mol.GetAtoms():
            # Extract features for each atom
            atom_features = []

            # Atomic number
            atom_features.append(atom.GetAtomicNum())
            # Formal charge
            atom_features.append(atom.GetFormalCharge())
            # Atomic mass
            atom_features.append(atom.GetMass())

            features.append(atom_features)

        features = torch.tensor(features, dtype=torch.float)

        return features, adj_matrix
    else:
        return None, None


# Read the dataset
df = pd.read_csv("ChChSe-Decagon_polypharmacy/filteredData.csv")

# Extract features and target variable
smiles1 = df['C1 SMILES']
smiles2 = df['C2 SMILES']
side_effects = df["Side Effect Name"]

# Map side effects to class indices
unique_side_effects = side_effects.unique()
side_effect_to_idx = {side_effect: idx for idx, side_effect in enumerate(unique_side_effects)}

# Create list to store graph data
graphs_data = []

# Create Data objects for each drug pair
for idx, (smiles1, smiles2, side_effect) in enumerate(zip(smiles1, smiles2, side_effects)):
    # Convert SMILES to graphs
    features1, adj_matrix1 = smiles_to_graph(smiles1)
    features2, adj_matrix2 = smiles_to_graph(smiles2)

    if features1 is not None and features2 is not None:
        # PAD THE ADJACENCY MATRICES TO MATCH THE SIZES
        max_num_atoms = max(adj_matrix1.size(0), adj_matrix2.size(0))
        pad1 = max_num_atoms - adj_matrix1.size(0)
        pad2 = max_num_atoms - adj_matrix2.size(0)
        adj_matrix1 = torch.nn.functional.pad(adj_matrix1, (0, pad1, 0, pad1))
        adj_matrix2 = torch.nn.functional.pad(adj_matrix2, (0, pad2, 0, pad2))

        # Calculate edge indexes
        edge_index1 = torch.nonzero(adj_matrix1, as_tuple=False).t().contiguous()
        edge_index2 = torch.nonzero(adj_matrix2, as_tuple=False).t().contiguous()
        # Update edge indexes for the second graph
        edge_index2 += features1.size(0)  # Offset the node indices for the second graph

        # Concatenate features and edge indexes
        concatenated_features = torch.cat((features1, features2), dim=0)
        concatenated_edge_index = torch.cat([edge_index1, edge_index2], dim=1)

        #Make the side effect into a tensor label
        label = torch.tensor([side_effect_to_idx[side_effect]])

        # Create PyTorch Geometric Data object
        data = Data(x=concatenated_features,
                    edge_index=concatenated_edge_index,
                    y=label)

        # Append to graphs_data list
        graphs_data.append(data)

# Split data into train and test sets
random.seed(42)  # For reproducibility

random.shuffle(graphs_data)

split_index = int(0.8 * len(graphs_data))  # 80% train, 20% test

train_data = graphs_data[:split_index]
test_data = graphs_data[split_index:]

# Define the dimensions
input_dim = 3
hidden_dim = 32  # Hidden dimension
output_dim = len(unique_side_effects)  # Number of classes (labels)

# Create an instance of the model
model = GCNGraphClassifier(input_dim, hidden_dim, output_dim)

# Define your loss function
criterion = nn.CrossEntropyLoss()

# Define your optimizer
optimizer = optim.Adam(model.parameters())

'''# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    
    # Set the model to training mode
    model.train()

    for graph_data in train_data:
        # Forward pass
        output = model(graph_data)

        # Calculate the loss
        loss = criterion(output, graph_data.y)
        total_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print average loss
    print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {total_loss / len(train_data)}")'''

# Evaluation after training
with torch.no_grad():
    model.eval()
    true_labels = []
    predicted_labels = []

    for graph_data in test_data:
        output = model(graph_data)
        _, predicted = torch.max(output, 1)
        true_labels.extend(graph_data.y.tolist())
        predicted_labels.extend(predicted.tolist())

    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Accuracy: {accuracy}")

    # Calculate precision, recall, and F1-score
    report = classification_report(true_labels, predicted_labels, zero_division=0)
    print(report)

