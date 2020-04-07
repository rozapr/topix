from sklearn.cluster import KMeans
from sklearn import decomposition
import matplotlib.pyplot as plt
from kneed import KneeLocator
import plotly.express as px
import plotly.offline as of
from typing import List
import pandas as pd
import numpy as np
import os

BASE_OUTPUT_FOLDER = f'{os.path.dirname(os.path.abspath("program.py"))}/outputs'
DISTORTIONS_FILE_PATH = f'{BASE_OUTPUT_FOLDER}/distortions_graph.png'
CLUSTERS_FILE_PATH = f'{BASE_OUTPUT_FOLDER}/clusters.html'


def k_means(vectors: List[np.ndarray], num_clusters: int = None) -> List[int]:
    if not num_clusters:
        num_clusters = get_clusters_num(vectors)
    print(f'clustering docs to {num_clusters} clusters.')
    clustering_model = KMeans(n_clusters=num_clusters)
    clustering_model.fit(vectors)
    cluster_assignment = clustering_model.labels_
    save_3d_embedding_figure(vectors, cluster_assignment)
    return cluster_assignment


def get_clusters_num(vectors: List[np.ndarray]) -> int:
    distortions = []
    cluster_num_series = range(1, 10)
    for clusters_num in cluster_num_series:
        km = KMeans(
            n_clusters=clusters_num, init='random',
            n_init=10, max_iter=300,
            tol=1e-04, random_state=0
        )
        km.fit(vectors)
        distortions.append(km.inertia_)

    kn = KneeLocator(cluster_num_series, distortions, curve='convex', direction='decreasing')
    save_distortions_graph(cluster_num_series, distortions)
    return kn.knee


def save_distortions_graph(clusters_num_series: List[int], distortions: List[float]):
    plt.plot(clusters_num_series, distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.savefig(DISTORTIONS_FILE_PATH)


def save_3d_embedding_figure(vectors: List[np.ndarray], clusters: List[int]):
    pca = decomposition.PCA(n_components=3)
    pca.fit(vectors)
    embeddings_3d = pca.transform(vectors)
    df = pd.DataFrame(embeddings_3d, columns=['x', 'y', 'z'])
    df['cluster'] = clusters
    fig = px.scatter_3d(df, x='x', y='y', z='z', color='cluster')
    of.plot(fig, filename=CLUSTERS_FILE_PATH, auto_open=False)


