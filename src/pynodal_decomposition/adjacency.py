from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, to_dense_adj


class SpatialAdjacency(nn.Module):
    def __init__(self, keep_self_loops: bool = False):
        super().__init__()

        self.keep_self_loops = keep_self_loops

        kernel_x = torch.zeros(2, 1, 1, 2, dtype=torch.float)
        kernel_x[0, :, 0, 0] = 1.0
        kernel_x[1, :, 0, 1] = 1.0
        kernel_y = kernel_x.permute(0, 1, 2, 3)

        self.register_buffer("kernel_x", kernel_x)
        self.register_buffer("kernel_y", kernel_y)

    @torch.no_grad()
    def forward(self, segments):
        if segments.ndim == 3:
            segments = segments.unsqueeze(1)

        n_segments = torch.amax(segments) + 1

        b = segments.shape[0]

        indexed_segments, batch_index = reindex_all_segments_across_batch(segments)

        xx = F.conv2d(indexed_segments.float(), self.kernel_x).permute(1, 0, 2, 3).reshape(2, -1).long()
        yy = F.conv2d(indexed_segments.float(), self.kernel_y).permute(1, 0, 2, 3).reshape(2, -1).long()
        edge_index = torch.cat([xx, yy], 1)
        if not self.keep_self_loops:
            edge_index, _ = remove_self_loops(edge_index=edge_index)

        adj = to_dense_adj(edge_index, batch=batch_index, batch_size=b, max_num_nodes=int(n_segments))
        adj = (adj + adj.permute(0, 2, 1)) / 2

        if self.keep_self_loops:
            diag = torch.diagonal(adj, dim1=-2, dim2=-1)
            diag /= 2

        return adj


def reindex_all_segments_across_batch(segments: torch.Tensor, with_padding=False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reindex the segments in the input tensor in increasing order, in consecutive order (with_padding=False).
    If with_padding=True, the segments are reindexed in increasing order with padding between each batch element
    so that each segment on the i-th batch element is indexed by i*num_nodes + segment_index.
    segments: tensor (type int64) of shape BxWxH
    e.g: with padding False:
    segments = [[0, 1],
                [0, 3],
                [2, 3]]
    reindex_segments_from_batch(segments, with_padding=False):
    tensor([[0, 1],
            [2, 5],
            [8, 9]])
    e.g: with padding True:
    reindex_segments_from_batch(segments, with_padding=True):
    tensor([[0, 1], # Missing 2, 3 (NMax = 4)
            [4, 7], # Missing 5, 6
            [10, 11]]) # Missing 8, 9

    """
    segments = segments.clone()
    b = segments.shape[0]
    segments_per_image = (torch.amax(segments, dim=(1, 2, 3)) + 1).long()
    # We reindex each segment in increasing order along the batch dimension
    if with_padding:
        max_nodes = segments.amax() + 1
        padding = torch.arange(0, b, device=segments.device) * max_nodes
        padding[-1] = padding[-1] - 1
        segments += padding.view(-1, 1, 1, 1)
        batch_index = torch.arange(0, b, device=segments.device).repeat_interleave(segments_per_image)
    else:
        cum_sum = torch.cumsum(segments_per_image, -1)
        cum_sum = torch.roll(cum_sum, 1)
        cum_sum[0] = 0
        segments += cum_sum.view(-1, 1, 1, 1)
        batch_index = torch.cat([segments.new_ones(s) * i for i, s in enumerate(segments_per_image)], 0)

    return segments, batch_index
