"""
diagnose.py
Checks whether the GNN is learning genuine industry-specific corrections,
or just shrinking predictions toward zero (which would trivially beat naive
without actually using graph structure).
"""
import torch
import torch.nn.functional as F
from pathlib import Path
from train_gnn import IOGNN  # reuse the model class

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRAPH_DIR = PROJECT_ROOT / "datasets" / "graphs"

train_data = torch.load(GRAPH_DIR / "train_graph.pt", weights_only=False)
val_data = torch.load(GRAPH_DIR / "val_graph.pt", weights_only=False)

x_mean, x_std = train_data.x.mean(), train_data.x.std()
val_x_norm = (val_data.x - x_mean) / x_std

model = IOGNN(in_dim=1, hidden=16)
model.load_state_dict(torch.load(GRAPH_DIR.parent / "gnn_model.pt", weights_only=True))
model.eval()

with torch.no_grad():
    pred_delta = model(val_x_norm, val_data.edge_index, val_data.edge_attr)

actual_delta = val_data.y - val_data.x.squeeze()

print(f"Predicted delta -> mean: {pred_delta.mean():.4f}, std: {pred_delta.std():.4f}")
print(f"Actual delta    -> mean: {actual_delta.mean():.4f}, std: {actual_delta.std():.4f}")

corr = torch.corrcoef(torch.stack([pred_delta, actual_delta]))[0, 1]
print(f"Correlation(predicted delta, actual delta): {corr:.4f}")

print("\nSample industry comparisons (code, actual output change, predicted change):")
for i in range(10):
    code = val_data.industry_codes[i]
    print(f"{code:8s} actual={actual_delta[i]:+.4f}  predicted={pred_delta[i]:+.4f}")