import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(json_folder, adjacency_file):
    user_data = {}
    for file_name in os.listdir(json_folder):
        if file_name.endswith(".json"):
            file_path = os.path.join(json_folder, file_name)
            with open(file_path, "r") as file:
                user_data[os.path.splitext(file_name)[0]] = json.load(file)
    
    # Load adjacency matrix
    adjacency_matrix = pd.read_csv(adjacency_file, index_col=0)
    return user_data, adjacency_matrix

def generate_features(user_data):
    text_data = []
    user_ids = []
    
    for user_id, data in user_data.items():
        user_ids.append(user_id)
        combined_text = " ".join([str(value) for value in data.values() if isinstance(value, str)])
        text_data.append(combined_text)

    # Convert text data into TF-IDF features
    vectorizer = TfidfVectorizer(max_features=100)  # Adjust max_features as needed
    features = vectorizer.fit_transform(text_data).toarray()
    return features, user_ids

def split_data(adjacency_matrix, test_size=50):
    user_ids = list(adjacency_matrix.index)
    train_ids, test_ids = train_test_split(user_ids, test_size=test_size, random_state=42)

    # Create adjacency matrices
    train_adj = adjacency_matrix.loc[train_ids, train_ids]
    test_adj = adjacency_matrix.loc[test_ids, test_ids]

    # Disconnect train and test in the full adjacency matrix
    train_adj_full = adjacency_matrix.copy()
    for test_id in test_ids:
        train_adj_full.loc[test_id, :] = 0
        train_adj_full.loc[:, test_id] = 0

    return train_ids, test_ids, train_adj, train_adj_full, test_adj

def prepare_graph_data(features, adjacency_matrix, user_ids):
    edge_index = np.array(np.nonzero(adjacency_matrix.values))
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    x = torch.tensor(features, dtype=torch.float)
    return x, edge_index, user_ids

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x
    
def train_model(model, x, edge_index, labels, train_idx, epochs=100, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

def recommend_profiles(model, x, edge_index, test_idx):
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
    return out[test_idx]

def main():
    # File paths
    json_folder = "final_user_profiles"  # Replace with folder path containing 4000 JSON files
    adjacency_file = "adjacency_matrix.csv"

    # Load data
    user_data, adjacency_matrix = load_data(json_folder, adjacency_file)

    # Generate features
    features, user_ids = generate_features(user_data)

    # Split data
    train_ids, test_ids, train_adj, train_adj_full, test_adj = split_data(adjacency_matrix)

    # Prepare graph data
    train_features = features[[user_ids.index(uid) for uid in train_ids]]
    test_features = features[[user_ids.index(uid) for uid in test_ids]]

    x_train, edge_index_train, train_user_ids = prepare_graph_data(train_features, train_adj_full, train_ids)

    # Labels (dummy labels for illustration)
    labels = torch.tensor([i for i in range(len(train_user_ids))], dtype=torch.long)

    # Train-test split indices
    train_idx = torch.tensor(list(range(len(train_ids))), dtype=torch.long)

    # Initialize and train the GNN
    model = GNN(input_dim=train_features.shape[1], hidden_dim=16, output_dim=len(train_ids))
    train_model(model, x_train, edge_index_train, labels, train_idx)

    # Prepare graph data for testing
    x_test, edge_index_test, test_user_ids = prepare_graph_data(test_features, test_adj, test_ids)

    # Generate recommendations for test set
    recommendations = recommend_profiles(model, x_test, edge_index_test, test_user_ids)
    print("Recommendations for test set:")
    for i, rec in zip(test_user_ids, recommendations):
        print(f"User {test_user_ids[i]} recommendations: {rec}")

if __name__ == "__main__":
    main()