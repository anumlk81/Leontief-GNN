"""
train_gnn.py (v3)
Adds richer node features (row-sum, col-sum from A matrix = how much an
industry supplies vs. consumes) and reduces regularization so the model
can actually differentiate industries instead of collapsing to a constant.
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRAPH_DIR = PROJECT_ROOT / "datasets" / "graphs"
MATRIX_DIR = PROJECT_ROOT / "datasets" / "matrices"

torch.manual_seed(42)


def add_structural_features(data, A_year):
    """Attach row-sum (supplier-ness) and col-sum (input-intensity) from A matrix."""
    A = pd.read_csv(MATRIX_DIR / f"A_{A_year}.csv", index_col=0)
    A = A.reindex(index=data.industry_codes, columns=data.industry_codes).fillna(0.0)

    row_sum = torch.tensor(A.sum(axis=1).values, dtype=torch.float).unsqueeze(1)  # supplies to others
    col_sum = torch.tensor(A.sum(axis=0).values, dtype=torch.float).unsqueeze(1)  # consumes from others

    data.x = torch.cat([data.x, row_sum, col_sum], dim=1)  # now 3 features per node
    return data


class IOGNN(torch.nn.Module):
    def __init__(self, in_dim, hidden=16):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, 1)
        self.skip = torch.nn.Linear(in_dim, 1)  # lets model retain per-node identity

    def forward(self, x, edge_index, edge_weight):
        h = F.relu(self.conv1(x, edge_index, edge_weight))
        h = F.dropout(h, p=0.1, training=self.training)
        delta = self.conv2(h, edge_index, edge_weight) + self.skip(x)
        return delta.squeeze()


def rmse(pred, target):
    return torch.sqrt(F.mse_loss(pred, target)).item()


def main():
    train_data = torch.load(GRAPH_DIR / "train_graph.pt", weights_only=False)
    val_data = torch.load(GRAPH_DIR / "val_graph.pt", weights_only=False)

    train_data = add_structural_features(train_data, "2007")
    val_data = add_structural_features(val_data, "2012")

    # standardize using TRAIN stats only
    x_mean, x_std = train_data.x.mean(dim=0), train_data.x.std(dim=0)
    train_x_norm = (train_data.x - x_mean) / x_std
    val_x_norm = (val_data.x - x_mean) / x_std

    train_residual = train_data.y - train_data.x[:, 0]
    val_residual = val_data.y - val_data.x[:, 0]

    model = IOGNN(in_dim=3, hidden=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

    print("Training (v3: richer features, lighter regularization)...")
    for epoch in range(1, 301):
        model.train()
        optimizer.zero_grad()
        pred_delta = model(train_x_norm, train_data.edge_index, train_data.edge_attr)
        loss = F.mse_loss(pred_delta, train_residual)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                val_delta = model(val_x_norm, val_data.edge_index, val_data.edge_attr)
                val_pred_level = val_data.x[:, 0] + val_delta
                val_rmse = rmse(val_pred_level, val_data.y)
            print(f"Epoch {epoch:3d} | train MSE: {loss.item():.4f} | val RMSE (level): {val_rmse:.4f}")

    model.eval()
    with torch.no_grad():
        val_delta = model(val_x_norm, val_data.edge_index, val_data.edge_attr)
        gnn_pred = val_data.x[:, 0] + val_delta

    gnn_rmse = rmse(gnn_pred, val_data.y)
    naive_rmse = rmse(val_data.x[:, 0], val_data.y)
    corr = torch.corrcoef(torch.stack([val_delta, val_residual]))[0, 1]

    print(f"\n--- Final comparison on 2012 -> 2017 (log-scale RMSE) ---")
    print(f"GNN (residual) RMSE: {gnn_rmse:.4f}")
    print(f"Naive RMSE:          {naive_rmse:.4f}")
    print(f"Predicted delta std: {val_delta.std():.4f}  (actual delta std: {val_residual.std():.4f})")
    print(f"Correlation(pred delta, actual delta): {corr:.4f}")

    torch.save(model.state_dict(), GRAPH_DIR.parent / "gnn_model.pt")


if __name__ == "__main__":
    main()