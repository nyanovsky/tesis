import sys
import torch
sys.path.append("..")
from gnn_interface.data import GNNData
from gnn_interface.model import LinkPredictor, train, test
import networkx as nx

# 1. Create a GNNData object from CSV files
print("--- Loading data ---")
gnn_data = GNNData(
    node_path='data/sample_nodes.csv',
    edge_path='data/sample_edges.csv',
    node_index_col='node_id',
    node_type_col='type',
    src_edge_col='source',
    dst_edge_col='target',
    edge_type_col='relation',
    src_node_type_col='source_type',
    dst_node_type_col='target_type'
)

print(f"Loaded {'heterogeneous' if gnn_data.is_hetero else 'homogeneous'} graph.")
print("Original PyG data object:")
print(gnn_data.data)
print("-" * 20)

# 2. Initialize features
print("--- Initializing features ---")
gnn_data.initialize_features(dim=16, inplace=True)
print("Features initialized.")
print("-" * 20)

# 3. Split data for Link Prediction
print("--- Splitting data for Link Prediction ---")
# This is a small graph, so we'll use a small validation and test set.
# disjoint_train_ratio ensures the message-passing graph in training doesn't contain validation/test edges
train_data, val_data, test_data = gnn_data.split_data(
    task='link_prediction',
    num_val=1, 
    num_test=1,
    disjoint_train_ratio=0.2,
    edge_types=('gene', 'interacts', 'drug'),
    rev_edge_types=('drug', 'rev_interacts', 'gene') # Assuming reverse edges are not in this sample data
)
print("Data split into training, validation, and test sets.")
print("Train data:", train_data)
print("-" * 20)

# 4. Create a LinkPredictor model
print("--- Creating model ---")
model = LinkPredictor(
    gnn_data=gnn_data,
    hidden_channels=32,
    model_type='sage',
    num_layers=2
)
print(model)
print("-" * 20)

# 5. Train the model
print("--- Training model ---")
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(10):
    loss = train(model, train_data, optimizer)
    val_auc = test(model, val_data)
    print(f"Epoch {epoch+1:02d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}")
print("-" * 20)

# 6. Evaluate on the test set
print("--- Evaluating on Test Set ---")
test_auc = test(model, test_data)
print(f"Test AUC: {test_auc:.4f}")
print("-" * 20)


# 7. Demonstrate NetworkX integration (as before)
print("--- NetworkX Integration ---")
nx_graph = gnn_data.nx_graph
print("Original graph (from NetworkX):")
print(f"Edges: {nx_graph.edges(data=True)}")

print("\nManipulating graph: removing edge ('A', 'B')...")
if nx_graph.has_edge('A', 'B'):
    nx_graph.remove_edge('A', 'B')

gnn_data.update_from_nx(nx_graph)
print("\nPyG data after update:")
print(gnn_data.data)
print("Edges for ('gene', 'similar', 'gene'):")
print(gnn_data.data.edge_index_dict.get(('gene', 'similar', 'gene')))
print("-" * 20)

print("Interface demonstration complete.") 