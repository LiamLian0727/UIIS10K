import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GAT


def build_knn_edges(h, k=7):
    """
    This function is used to calculate the adjacency matrix in GAT's node attention mechanism.
    Args:
        h: the hidden node feature, shape is (B, H*W, feature_dim)
        k: number of neighbors, the nearest k nodes will be connected
    Returns:
        edge_index: the edge index, shape is (B, 2, num_edges)
    """
    B, node_num, _ = h.size()
    assert k <= node_num, (
        "The number of neighbors should be less than the number of nodes"
    )
    device = h.device
    # Normalization plus matrix multiplication gives cosine similarity.
    with torch.no_grad():
        normalized = F.normalize(h, p=2, dim=-1, eps=1e-8)
        dist = torch.bmm(normalized, normalized.transpose(1, 2))

    connection_index = torch.topk(dist, k, dim=2, largest=True).indices
    score_index = (
        torch.arange(0, node_num)
        .repeat_interleave(k)
        .view(node_num, k)
        .repeat(B, 1, 1)
        .to(device)
    )
    edge_index = (
        torch.stack((score_index, connection_index), dim=3)
        .view(B, -1, 2)
        .transpose(1, 2)
        .contiguous()
    )

    return edge_index


class ImageGraphAttention(nn.Module):
    """
    A batch version of GAT for information aggregation by image patching
    """

    def __init__(
        self, in_channels, hidden_channels, num_layers, out_channels, v2=False, **kwargs
    ):
        super().__init__()
        heads = kwargs.pop("heads", 1)
        self.gat = GAT(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            v2=v2,
            heads=heads,
            **kwargs,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: img feature, shape is (B, H*W, feature_dim)
        edge_index: the edge index, shape is (B, 2, num_edges)
        """
        B, node_num, _ = x.size()
        data_list = [
            Data(x=node, edge_index=index) for node, index in zip(x, edge_index)
        ]
        batch = Batch.from_data_list(data_list)
        assert batch.batch_size == B, (
            "The batch size out from GAT is not equal to the input batch size"
        )
        out = self.gat(batch.x, batch.edge_index)
        out = out.view(B, node_num, -1)
        return out
