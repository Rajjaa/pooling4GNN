from turtle import forward
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric as pyg
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer

@register_layer('dense_gcnconv')
class DenseGCNConv(nn.Module):
    """
    Dense Graph Convolutional Network (GCN) layer
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = pyg.nn.dense.DenseGCNConv(
            layer_config.dim_in, layer_config.dim_out,
            bias=layer_config.has_bias
            )

    def forward(self, batch, add_loop=True):
        if hasattr(batch, 'mask'):
            mask = batch.mask
        else:
            mask = None
        batch.x = self.model(batch.x, batch.adj, mask=mask, add_loop=add_loop)
        return batch


@register_layer('dense_sageconv')
class DenseSAGEConv(nn.Module):
    """
    Dense GraphSAGE Conv layer
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = pyg.nn.dense.DenseSAGEConv(
            layer_config.dim_in, layer_config.dim_out,
            bias=layer_config.has_bias,
            normalize=cfg.gnn.l2norm)

    def forward(self, batch):
        if hasattr(batch, 'mask'):
            mask = batch.mask
        else:
            mask = None
        batch.x = self.model(batch.x, batch.adj, mask)
        return batch


@register_layer('dense_graphconv')
class DenseGraphConv(nn.Module):
    """
    Dense GraphConv layer
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = pyg.nn.dense.DenseGraphConv(
            layer_config.dim_in, layer_config.dim_out,
            bias=layer_config.has_bias, aggr=cfg.gnn.agg, **kwargs
        )

    def forward(self, batch):
        if hasattr(batch, 'mask'):
            mask = batch.mask
        else:
            mask = None
        batch.x = self.model(batch.x, batch.adj, mask)
        return batch
