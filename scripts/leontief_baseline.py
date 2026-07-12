"""
leontief_baseline.py
Classical Leontief input-output forecast, compared against naive and GNN.

Classical method (no ML):
  1. Recover implied final demand: f = (I - A) x   for 2007 and 2012
  2. Extrapolate the demand trend forward to 2017: f_2017_est = f_2012 + (f_2012 - f_2007)
  3. Solve the Leontief system forward: x_2017_pred = (I - A_2012)^-1 * f_2017_est
"""
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path

from train_gnn import IOGNN, add_structural_features  # reuse trained model + feature builder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MATRIX_DIR = PROJECT_ROOT / "datasets" / "matrices"
CLEAN_DIR = PROJECT_ROOT / "datasets" / "clean_datasets"
GRAPH_DIR = PROJECT_ROOT / "datasets" / "graphs"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_output(year):
    return pd.read_csv(CLEAN_DIR / f"clean_output_{year}.csv", index_col=0).iloc[:, 0]


def load_A(year):
    return pd.read_csv(MATRIX_DIR / f"A_{year}.csv", index_col=0)


def classical_leontief_forecast():
    x_2007 = load_output("2007")
    x_2012 = load_output("2012")
    A_2007 = load_A("2007")
    A_2012 = load_A("2012")

    codes = A_2012.index
    x_2007 = x_2007.reindex(codes).fillna(0.0).values
    x_2012 = x_2012.reindex(codes).fillna(0.0).values
    A_2007 = A_2007.reindex(index=codes, columns=codes).fillna(0.0).values
    A_2012 = A_2012.reindex(index=codes, columns=codes).fillna(0.0).values

    I = np.eye(len(codes))

    # recover implied final demand at each known year
    f_2007 = (I - A_2007) @ x_2007
    f_2012 = (I - A_2012) @ x_2012

    # extrapolate demand trend forward one step (2012 -> 2017, same gap as 2007 -> 2012)
    f_2017_est = f_2012 + (f_2012 - f_2007)

    # solve the Leontief system forward using 2012's production structure
    x_2017_pred = np.linalg.solve(I - A_2012, f_2017_est)
    x_2017_pred = np.clip(x_2017_pred, a_min=0, a_max=None)  # output can't be negative

    return pd.Series(x_2017_pred, index=codes), list(codes)


def gnn_forecast():
    val_data = torch.load(GRAPH_DIR / "val_graph.pt", weights_only=False)
    val_data = add_structural_features(val_data, "2012")

    train_data = torch.load(GRAPH_DIR / "train_graph.pt", weights_only=False)
    train_data = add_structural_features(train_data, "2007")
    x_mean, x_std = train_data.x.mean(dim=0), train_data.x.std(dim=0)
    val_x_norm = (val_data.x - x_mean) / x_std

    model = IOGNN(in_dim=3, hidden=16)
    model.load_state_dict(torch.load(PROJECT_ROOT / "datasets" / "gnn_model.pt", weights_only=True))
    model.eval()

    with torch.no_grad():
        delta = model(val_x_norm, val_data.edge_index, val_data.edge_attr)
        pred_log = val_data.x[:, 0] + delta

    pred = np.expm1(pred_log.numpy())  # undo log1p back to dollar scale
    return pd.Series(pred, index=val_data.industry_codes)


def rmse_log(pred: pd.Series, actual: pd.Series):
    pred_log = np.log1p(pred.clip(lower=0))
    actual_log = np.log1p(actual)
    return np.sqrt(np.mean((pred_log - actual_log) ** 2))


def main():
    x_2012 = load_output("2012")
    x_2017 = load_output("2017")

    leontief_pred, codes = classical_leontief_forecast()
    gnn_pred = gnn_forecast()
    naive_pred = x_2012.reindex(codes)
    actual = x_2017.reindex(codes)

    results = {
        "Naive\n(no change)": rmse_log(naive_pred, actual),
        "Classical\nLeontief": rmse_log(leontief_pred, actual),
        "GNN": rmse_log(gnn_pred, actual),
    }

    print("\n--- RMSE comparison (log-scale) predicting 2017 output ---")
    for k, v in results.items():
        print(f"{k.replace(chr(10), ' '):20s}: {v:.4f}")

    # ---- Chart 1: RMSE bar chart ----
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(results.keys(), results.values(), color=["#888888", "#c0603c", "#3c7fc0"])
    ax.set_ylabel("RMSE (log scale)")
    ax.set_title("Forecast Accuracy: Naive vs Classical Leontief vs GNN\n(predicting 2017 industry output)")
    for bar, val in zip(bars, results.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.003, f"{val:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "rmse_comparison.png", dpi=150)
    print(f"\nSaved {RESULTS_DIR / 'rmse_comparison.png'}")

    # ---- Chart 2: predicted vs actual scatter, all three methods ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    methods = [("Naive", naive_pred), ("Classical Leontief", leontief_pred), ("GNN", gnn_pred)]

    for ax, (name, pred) in zip(axes, methods):
        actual_log = np.log1p(actual)
        pred_log = np.log1p(pred.clip(lower=0))
        ax.scatter(actual_log, pred_log, alpha=0.6)
        lims = [min(actual_log.min(), pred_log.min()), max(actual_log.max(), pred_log.max())]
        ax.plot(lims, lims, 'k--', alpha=0.5, label="perfect prediction")
        ax.set_title(name)
        ax.set_xlabel("Actual 2017 output (log)")
        ax.set_ylabel("Predicted 2017 output (log)")
        ax.legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "predicted_vs_actual.png", dpi=150)
    print(f"Saved {RESULTS_DIR / 'predicted_vs_actual.png'}")

    plt.show()


if __name__ == "__main__":
    main()