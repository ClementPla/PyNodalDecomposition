from functools import lru_cache

import numpy as np
import torch
from networkx import Graph
from networkx.algorithms.clique import enumerate_all_cliques
from networkx.algorithms.community import (
    label_propagation_communities,
    louvain_communities,
)
from skimage.measure import perimeter
from skimage.morphology import skeletonize
from skimage.segmentation import relabel_sequential

from pynodal_decomposition.adjacency import SpatialAdjacency


def get_spatial_adjacency_matrix(segment):
    torch_segment = torch.from_numpy(segment).unsqueeze(0).unsqueeze(0).long()

    spatial_adj_module = _get_spatial_adj_module()
    adj = spatial_adj_module(torch_segment).squeeze().numpy()
    return adj


def build_neighbor_over_perimeter_adjancency_matrix(segment):
    adj = get_spatial_adjacency_matrix(segment)
    adj_ratio = np.zeros_like(adj)

    for i in range(0, adj.shape[0]):
        mask_seg = segment == i
        skeleton = skeletonize(mask_seg)
        p = perimeter(skeleton, neighborhood=8)
        p = p - adj[i, 0]  # We only want the perimeter of the segment that does not touch the border
        neighbors = adj[i] / (
            p + 0.1
        )  # We find the ratio of the perimeter of the segment that is shared with the neighbors
        adj_ratio[i] = neighbors
        adj_ratio[i, i] = 1.0  # We set the ratio of the segment with itself to 1

    adj_ratio[0] = 0  #  The background is not a segment
    adj_ratio[:, 0] = 0  # Segments do not have the background as a neighbor
    adj_ratio[0, 0] = 0  # The background does not have itself as a neighbor

    return adj_ratio


def merge_cliques(segment, adj_ratio, threshold=0.5):
    if threshold:
        g = Graph(adj_ratio > threshold)
    else:
        g = Graph(adj_ratio)
    cliques = list(enumerate_all_cliques(g))
    return merge_communities(segment, cliques)


def merge_communities(segment, communities):
    area = np.unique(segment, return_counts=True)[1]
    results = segment.copy()
    for com in communities:
        if len(com) == 1:
            continue

        com = list(com)
        area_com = area[com]
        idx = np.argmax(area_com)
        for i in com:
            if i != com[idx]:
                results[results == i] = com[idx]
    return reindex_labelmap(results)


def merge_louvain_communities(segment, adj_ratio, resolution=1.0):
    G = Graph(adj_ratio)
    communities = louvain_communities(G, resolution=resolution)
    return merge_communities(segment, communities)


def label_propagation(segment, adj_ratio, threshold=None):
    if threshold:
        G = Graph(adj_ratio > threshold)
    else:
        G = Graph(adj_ratio)
    communities = label_propagation_communities(G)
    merge_communities(segment, communities)


@lru_cache(maxsize=1)
def _get_spatial_adj_module():
    return SpatialAdjacency(keep_self_loops=True)


def reindex_labelmap(segment):
    return relabel_sequential(segment)[0]
