import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym import optim
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
import random


class GCNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNClassifier, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)  # Use global mean pooling
        x = self.fc(x)

        return F.log_softmax(x, dim=1)


graphs_data = [
    Data(x=torch.randn(4, 5), edge_index=torch.tensor([[0, 1, 2, 0], [1, 0, 2, 2]]), y=torch.tensor([0])),
    Data(x=torch.randn(3, 5), edge_index=torch.tensor([[0, 1, 2], [1, 0, 2]]), y=torch.tensor([1])),
    Data(x=torch.randn(5, 5), edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]), y=torch.tensor([2])),
    Data(x=torch.randn(6, 5), edge_index=torch.tensor([[0, 1, 2, 3, 4, 5], [1, 0, 2, 3, 4, 5]]), y=torch.tensor([0])),
    Data(x=torch.randn(4, 5), edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 2, 3]]), y=torch.tensor([0])),
    Data(x=torch.randn(5, 5), edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]), y=torch.tensor([1])),
    Data(x=torch.randn(2, 5), edge_index=torch.tensor([[0, 1, ], [1, 0, ]]), y=torch.tensor([2])),
    Data(x=torch.randn(7, 5), edge_index=torch.tensor([[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6]]), y=torch.tensor([0])),
    Data(x=torch.randn(5, 5), edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]), y=torch.tensor([1]))
]

# Split data into train and test sets
random.seed(42)  # For reproducibility
random.shuffle(graphs_data)
split_index = int(0.8 * len(graphs_data))  # 80% train, 20% test

train_data = graphs_data[:split_index]
test_data = graphs_data[split_index:]

# Define the dimensions
input_dim = 5  # Dimension of node features
hidden_dim = 16  # Hidden dimension
output_dim = 3  # Number of classes (labels)

# Create an instance of the model
model = GCNClassifier(input_dim, hidden_dim, output_dim)

# Define your loss function
criterion = nn.CrossEntropyLoss()

# Define your optimizer
optimizer = optim.Adam(model.parameters())

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
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
    print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {total_loss / len(train_data)}")

# Evaluation after training
with torch.no_grad():
    model.eval()
    correct = 0
    total = 0
    for graph_data in test_data:
        output = model(graph_data)
        _, predicted = torch.max(output, 1)
        total += graph_data.y.size(0)
        correct += (predicted == graph_data.y).sum().item()
    accuracy = correct / total
    print(f"Accuracy: {accuracy}")