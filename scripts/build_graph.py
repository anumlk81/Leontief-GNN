"""
build_graph.py
Builds PyTorch Geometric graph objects from the A matrices and output vectors.

Train graph: edges from A_2007, x = 2007 output, y = 2012 output
Val graph:   edges from A_2012, x = 2012 output, y = 2017 output
"""
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MATRIX_DIR = PROJECT_ROOT / "datasets" / "matrices"
CLEAN_DIR = PROJECT_ROOT / "datasets" / "clean_datasets"
OUT_DIR = PROJECT_ROOT / "datasets" / "graphs"

EDGE_THRESHOLD = 0.01  # drop tiny/noisy input-output flows to keep graph sparse


def load_A(year: str) -> pd.DataFrame:
    return pd.read_csv(MATRIX_DIR / f"A_{year}.csv", index_col=0)


def load_output(year: str) -> pd.Series:
    return pd.read_csv(CLEAN_DIR / f"clean_output_{year}.csv", index_col=0).iloc[:, 0]


def build_edges(A: pd.DataFrame, threshold: float = EDGE_THRESHOLD):
    """Convert A matrix into (edge_index, edge_weight) for PyG, dropping weak edges."""
    idx = np.argwhere(A.values > threshold)
    edge_index = torch.tensor(idx.T, dtype=torch.long)
    edge_weight = torch.tensor(A.values[A.values > threshold], dtype=torch.float)
    return edge_index, edge_weight


def build_graph(feature_year: str, target_year: str, A_year: str) -> Data:
    A = load_A(A_year)
    x_series = load_output(feature_year)
    y_series = load_output(target_year)

    # align industries in the same order as the A matrix
    x_series = x_series.reindex(A.index).fillna(0.0)
    y_series = y_series.reindex(A.index).fillna(0.0)

    edge_index, edge_weight = build_edges(A)

    # log-scale + normalize outputs (raw $ values span orders of magnitude,
    # which destabilizes GNN training)
    x = torch.tensor(np.log1p(x_series.values), dtype=torch.float).unsqueeze(1)
    y = torch.tensor(np.log1p(y_series.values), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)
    data.industry_codes = list(A.index)
    return data


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_data = build_graph(feature_year="2007", target_year="2012", A_year="2007")
    val_data = build_graph(feature_year="2012", target_year="2017", A_year="2012")

    print("Train graph:", train_data)
    print("Val graph:  ", val_data)

    torch.save(train_data, OUT_DIR / "train_graph.pt")
    torch.save(val_data, OUT_DIR / "val_graph.pt")
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()