import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MATRIX_DIR = PROJECT_ROOT / "datasets" / "matrices"
CLEAN_DIR = PROJECT_ROOT / "datasets" / "clean_datasets"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

YEAR = "2017"       
EDGE_THRESHOLD = 0.02 


def load_data(year):
    A = pd.read_csv(MATRIX_DIR / f"A_{year}.csv", index_col=0)
    names = pd.read_csv(CLEAN_DIR / f"clean_names_{year}.csv", index_col=0).iloc[:, 0]
    output = pd.read_csv(CLEAN_DIR / f"clean_output_{year}.csv", index_col=0).iloc[:, 0]
    return A, names, output


def build_networkx_graph(A: pd.DataFrame, threshold: float) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(A.index)
    for i in A.index:
        for j in A.columns:
            w = A.loc[i, j]
            if w > threshold:
                G.add_edge(i, j, weight=w)
    return G


def main():
    A, names, output = load_data(YEAR)
    G = build_networkx_graph(A, EDGE_THRESHOLD)

    G.remove_nodes_from(list(nx.isolates(G)))
    print(f"Graph: {G.number_of_nodes()} industries, {G.number_of_edges()} edges "
          f"(threshold={EDGE_THRESHOLD})")

    G_undirected = G.to_undirected()
    communities = nx.community.greedy_modularity_communities(G_undirected, weight="weight")
    print(f"Detected {len(communities)} clusters")

    cluster_of = {}
    for cluster_id, members in enumerate(communities):
        for node in members:
            cluster_of[node] = cluster_id

    print("\n--- Cluster membership (by industry name) ---")
    for cluster_id, members in enumerate(communities):
        member_names = [names.get(m, m) for m in members]
        print(f"\nCluster {cluster_id} ({len(members)} industries):")
        print(", ".join(member_names[:8]) + (", ..." if len(member_names) > 8 else ""))

    pos = nx.spring_layout(G, k=0.6, weight="weight", seed=42, iterations=100)

    log_output = {n: np.log1p(output.get(n, 0)) for n in G.nodes()}
    max_log = max(log_output.values())
    node_sizes = [200 + 1800 * (log_output[n] / max_log) for n in G.nodes()]

    node_colors = [cluster_of.get(n, -1) for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(16, 12))

    nx.draw_networkx_edges(G, pos, alpha=0.15, arrows=True, arrowsize=6,
                            connectionstyle="arc3,rad=0.05", ax=ax)
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                                    node_color=node_colors, cmap=cm.tab20, ax=ax)

    degree = dict(G.degree())
    top_nodes = sorted(degree, key=degree.get, reverse=True)[:20]
    labels = {n: names.get(n, n) for n in top_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)

    ax.set_title(f"US Input-Output Network ({YEAR}) — {len(communities)} Detected Clusters\n"
                 f"Node size = output, color = cluster, edges = input-output flows",
                 fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "industry_network.png", dpi=150)
    print(f"\nSaved {RESULTS_DIR / 'industry_network.png'}")
    plt.show()


if __name__ == "__main__":
    main()