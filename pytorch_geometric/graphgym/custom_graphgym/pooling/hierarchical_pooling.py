from turtle import forward
import torch.nn as nn

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import (
    GeneralLayer,
    GeneralMultiLayer,
    LayerConfig,
    MLP
)
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.register import register_pooling
from torch_geometric.nn import dense_diff_pool, dense_mincut_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj


@register_pooling('diffpool_block')
class DiffPoolBlock(nn.Module):
    def __init__(self, dim_in, dim_out, no_new_nodes, **kwargs):
        super().__init__()
        GNNBlock = register.layer_dict['gnn_dense_block']
        self.gnn_embed = GNNBlock(dim_in, dim_out,
         num_layers=cfg.diffpool.num_block_mp_layers, dense=True)
        self.gnn_pool = GNNBlock(dim_out, no_new_nodes,
         num_layers=cfg.diffpool.num_block_pooling_layers, final_act=False, dense=True
         )

    def forward(self, batch, to_dense=False):
        if to_dense:
            x, mask = to_dense_batch(batch.x, batch.batch)
            adj = to_dense_adj(batch.edge_index, batch.batch)
            batch.x = x
            batch.adj = adj
            batch.mask = mask
        else:
            x = batch.x
            adj = batch.adj
            mask = None
            batch.mask = mask
        # compute the assignment matrix
        batch = self.gnn_pool(batch)
        s = batch.x
        batch.x = x
        # compute the nodes embeddings
        batch = self.gnn_embed(batch)
        x = batch.x
        x_pool, adj_pool, l, e = dense_diff_pool(x, adj, s, mask)
        batch.x = x_pool
        batch.adj = adj_pool
        batch.mask = None
        batch.aux_loss['l'] += l
        batch.aux_loss['e'] += e
        return batch


# @register_pooling('mincutpool')
# class DenseMincutPool(nn.Module):
#     def __init__(self, dim_in, n_clusters):
#         super().__init__()
#         mlp_layer_config = LayerConfig(
#             dim_in = dim_in,
#             dim_out = n_clusters,
#             num_layers=1,
#             has_act=False,
#             has_bias=True,
#             dropout=0,
#             has_l2norm=False
#             )
#         self.mlp = GeneralLayer('linear', mlp_layer_config)


#     def forward(self, batch):
#         x, adj, mask = batch.x, batch.adj, batch.mask
#         # compute the assignment matrix
#         s = self.mlp(batch.x)
#         x, adj, mc, o = dense_mincut_pool(x, adj, s, mask)
#         batch.x = x
#         batch.adj = adj
#         batch.aux_loss['o'] += o
#         batch.aux_loss['mc'] += mc
#         return batch


@register_pooling('mincutpool_block')
class MincutPoolBlock(nn.Module):
    def __init__(self, dim_in, dim_out, no_new_nodes, **kwargs):
        super().__init__()
        GNNBlock = register.layer_dict['gnn_dense_block']
        dense = kwargs.get('dense', True)
        self.gnn_embed = GNNBlock(dim_in, dim_out,
         num_layers=cfg.mincutpool.num_block_mp_layers , dense=dense)

        # mlp for the pooling block
        mlp_layer_config = LayerConfig(
            dim_in=dim_in,
            dim_out=no_new_nodes,
            num_layers=cfg.mincutpool.num_block_pooling_layers,
            has_act=True,
            final_act=False, # the softmax activation on the assignment matrix is applied by dense_mincut_pool
            has_bias=True,
            dropout=cfg.gnn.dropout,
            has_l2norm=False
            )
        self.mlp_pool = MLP(mlp_layer_config)

    def forward(self, batch, to_dense=False):
        # compute the nodes embeddings
        batch = self.gnn_embed(batch)
        # compute the assignment matrix
        if to_dense:
            x, mask = to_dense_batch(batch.x, batch.batch)
            adj = to_dense_adj(batch.edge_index, batch.batch)
        else:
            x, adj = batch.x, batch.adj
            mask = None
        s = self.mlp_pool(x)
        x_pool, adj_pool, mc, o = dense_mincut_pool(x, adj, s, mask)
        batch.x = x_pool
        batch.adj = adj_pool
        batch.mask = None
        batch.aux_loss['o'] += o
        batch.aux_loss['mc'] += mc
        return batch