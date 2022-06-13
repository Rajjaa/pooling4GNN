import torch.nn as nn

from torch_geometric.graphgym.models.layer import (
    GeneralLayer,
    LayerConfig
)
from torch_geometric.graphgym.register import register_pooling
from torch_geometric.nn import dense_mincut_pool


@register_pooling('mincutpool')
class DenseMincutPool(nn.Module):
    def __init__(self, dim_in, n_clusters):
        super().__init__()
        mlp_layer_config = LayerConfig(
            dim_in = dim_in,
            dim_out = n_clusters,
            num_layers=1,
            has_act=False,
            has_bias=True,
            dropout=0,
            has_l2norm=False
            )
        self.mlp = GeneralLayer('linear', mlp_layer_config)


    def forward(self, batch):
        x, adj, mask = batch.x, batch.adj, batch.mask
        # compute the assignment matrix
        s = self.mlp(batch.x)
        x, adj, mc, o = dense_mincut_pool(x, adj, s, mask)
        batch.x = x
        batch.adj = adj
        batch.s = s
        batch.aux_loss['o'] += o
        batch.aux_loss['mc'] += mc
        return batch