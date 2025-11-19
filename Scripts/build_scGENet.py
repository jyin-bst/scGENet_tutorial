import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import umap
import igraph as ig
import leidenalg

def read_data(file_path):
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path, index_col=0)
    else:
        return pd.read_csv(file_path, sep="\t", index_col=0)

def run_tsne(data, perplexity=200):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=123)
    coords = tsne.fit_transform(data)
    return pd.DataFrame(coords, index=data.index, columns=["x", "y"])

def run_umap(data):
    reducer = umap.UMAP(n_components=2,random_state=123)
    coords = reducer.fit_transform(data)
    return pd.DataFrame(coords, index=data.index, columns=["x", "y"])

def build_knn_graph(data, k=5, metric="euclidean"):
    if metric == "cosine":
        sim = cosine_similarity(data)
        np.fill_diagonal(sim, 0)
        indices = np.argsort(-sim, axis=1)[:, 1:k+1]
    else:
        nbrs = NearestNeighbors(n_neighbors=k+1, metric=metric).fit(data)
        indices = nbrs.kneighbors(data, return_distance=False)[:, 1:]

    edges = []
    for i, neighbors in enumerate(indices):
        for j in neighbors:
            edges.append((i, j))
    return edges

def cluster_graph(edges, num_nodes, method="louvain", resolution=2):
    g = ig.Graph()
    g.add_vertices(num_nodes)
    g.add_edges(edges)

    if method == "leiden":
        part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=resolution)
    else:
        part = g.community_multilevel(resolution=resolution)

    return np.array(part.membership)

def save_result(df, out_file):
    df_out = df.copy()
    df_out.insert(0, "Gene", df.index)
    df_out.to_csv(out_file, sep="\t", index=False)

def plot_embedding(df, out_prefix):
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x="x", y="y", hue="cluster", data=df, palette="tab20", s=10, linewidth=0)
    plt.title("Embedding with Clusters")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_plot.png", dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Build gene co-expression network using gene embedding and perform clustering analysis.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input file, gene embedding matrix")
    parser.add_argument("--output", "-o", type=str, required=True, help="Out file for network")
    parser.add_argument("--perp", "-p", type=int, default=200, help="TSNE perplexity")
    parser.add_argument("-k", type=int, default=5, help="k for kNN")
    parser.add_argument("--res", "-r", type=float, default=2.0, help="Resolution for clustering")
    parser.add_argument("--method", type=str, choices=["tsne", "umap"], default="tsne", help="Dimensionality reduction method")
    parser.add_argument("--metric", type=str, choices=["euclidean", "cosine"], default="euclidean", help="Distance metric for kNN")
    parser.add_argument("--cluster", type=str, choices=["louvain", "leiden"], default="louvain", help="Clustering algorithm")

    args = parser.parse_args()

    data = read_data(args.input)
    print(f"{args.input} has {data.shape[0]} genes, {data.shape[1]} dimensions.")

    emb = run_tsne(data, args.perp) if args.method == "tsne" else run_umap(data)

    edges = build_knn_graph(data.values, k=args.k, metric=args.metric)
    clusters = cluster_graph(edges, num_nodes=data.shape[0], method=args.cluster, resolution=args.res)

    emb["cluster"] = clusters
    save_result(emb, args.output)
    plot_embedding(emb, args.output.replace(".txt", ""))

if __name__ == "__main__":
    main()
