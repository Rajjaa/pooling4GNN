import torch.nn as nn
import torch_geometric as pyg
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import (
    GeneralLayer,
    LayerConfig,
    new_layer_config
)
from torch_geometric.graphgym.register import register_layer

@register_layer('gnn_layer')
def GNNLayer(dim_in, dim_out, has_act=True, dense=False):
    """
    Wrapper for a GNN layer
    It extends the wrapper of GraphGym by adding the dense parameter

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        has_act (bool): Whether has activation function after the layer
        dense (bool): Whether to use the dense version of the layer
    """
    layer_type = f'dense_{cfg.gnn.layer_type}' if dense else cfg.gnn.layer_type
    return GeneralLayer(
        layer_type,
        layer_config=new_layer_config(dim_in, dim_out, 1, has_act=has_act,
                                      has_bias=False, cfg=cfg))


@register_layer('gnn_dense_block')
class GNNBlock(nn.Module):
    def __init__(self, dim_in, dim_out, num_layers, final_act=True, dense=False):
        super().__init__()
        self.num_layers = num_layers

        for i in range(num_layers):
            d_in = dim_in if i == 0 else dim_out
            has_act = i!=num_layers-1 or final_act
            gnn_layer = GNNLayer(d_in, dim_out, dense=dense, has_act=has_act)
            self.add_module(f'gnn_layer{i}', gnn_layer)

    
    def forward(self, batch, mask=None):
        """"""
        for i, module in enumerate(self.children()):
            batch = module(batch)
        return batch


@register_layer('graphconv')
class GraphConv(nn.Module):
    """
    The graph neural network operator from the “Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks” paper
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = pyg.nn.GraphConv(
            layer_config.dim_in, layer_config.dim_out,
            bias=layer_config.has_bias, aggr=cfg.gnn.agg, **kwargs
        )

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch

