from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.init import init_weights
from torch_geometric.graphgym.models.layer import (
    BatchNorm1dNode,
    GeneralLayer,
    GeneralMultiLayer,
    new_layer_config,
)
from torch_geometric.graphgym.register import register_stage
from torch_geometric.utils import to_dense_batch, to_dense_adj


@register_stage('hierarchical_pooling')
class HieraPoolStage(nn.Module):
    """
    Stage that stack GNN layers and pooling layers
    and a flat pooling layer after the last GNN layer

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        num_layers (int): Number of GNN layers
    """
    def __init__(self, dim_in, dim_out, num_layers):
        super().__init__()
        GNNLayer = register.layer_dict['gnn_layer']
        HieraPoolLayer = register.pooling_dict[cfg.hierarchical_pooling.type]

        self.num_layers = num_layers

        max_num_nodes = cfg.dataset.max_num_nodes
        pool_ratio = cfg.hierarchical_pooling.pool_ratio
        no_new_clusters = ceil(pool_ratio * max_num_nodes)

        for i in range(num_layers):
            d_in = dim_in if i == 0 else dim_out
            # use dense layers for GNN layers after the first pooling layer
            dense = i != 0
            gnn_layer = GNNLayer(d_in, dim_out, dense=dense)
            self.add_module('gnn_layer{}'.format(i), gnn_layer)
            if i != self.num_layers-1:
                pool_layer = HieraPoolLayer(dim_out, no_new_clusters)
                self.add_module('pool_layer{}'.format(i), pool_layer)
                no_new_clusters = ceil(pool_ratio * no_new_clusters)
        

    def forward(self, batch):
        """"""
        for i in range(self.num_layers):
            if cfg.hierarchical_pooling.type == 'mincutpool':
                batch.aux_loss = {
                    'o': 0,
                    'mc': 0
                }
            gnn_layer = self.get_submodule(f'gnn_layer{i}'.format(i))
            if i == 0:
                # forward through the message passing layers
                batch = gnn_layer(batch)
                x, edge_index = batch.x, batch.edge_index
                x, mask = to_dense_batch(x, batch.batch)
                adj = to_dense_adj(edge_index, batch.batch)
                batch.x = x
                batch.mask = mask
                batch.adj = adj
                # forward through the pooling layer
                pool_layer = self.get_submodule(f'pool_layer{i}'.format(i))
                batch = pool_layer(batch)
            elif i != self.num_layers-1:
                batch = gnn_layer(batch)
                pool_layer = self.get_submodule(f'pool_layer{i}'.format(i))
                batch = pool_layer(batch)
            else:
                batch = gnn_layer(batch)
        if cfg.gnn.l2norm:
            batch.x = F.normalize(batch.x, p=2, dim=-1)
        return batch