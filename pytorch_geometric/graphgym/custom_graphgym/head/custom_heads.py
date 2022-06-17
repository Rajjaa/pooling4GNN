import torch.nn as nn
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import (
    MLP,
    GeneralLayer,
    LayerConfig
)
from torch_geometric.graphgym.register import register_head


@register_head('mlp_head')
class MLPHead(nn.Module):
    """Head for graph classification applied after the last final step"""
    def __init__(self, dim_in, dim_out):
        super().__init__()

        mlp_layer_config = LayerConfig(
            dim_in=dim_in,
            dim_out=dim_out,
            has_l2norm=False,
            dropout=cfg.mlp_head.dropout,
            has_act=cfg.mlp_head.has_act,
            final_act=False,
            act=cfg.gnn.act,
            has_bias=True,
            dim_inner=cfg.gnn.dim_inner,
            num_layers=cfg.mlp_head.num_layers ,
        )

        self.mlp_head = MLP(
            mlp_layer_config
                 )


    def _apply_index(self, batch):
        return batch.graph_feature, batch.y

    
    def forward(self, batch):
        batch.graph_feature = self.mlp_head(batch.graph_feature)
        batch.pred = batch.graph_feature
        batch.true = batch.y
        return batch


@register_head('hpool_head')
class HPoolHead(nn.Module):
    """Head for graph classification applied after the last final step for hierarchical pooling architectures"""
    def __init__(self, dim_in, dim_out):
        super().__init__()

        pooling_func_name = f'dense_{cfg.hierarchical_pooling.flat_pooling}'
        self.pooling_fun = register.pooling_dict[pooling_func_name]

        mlp_layer_config = LayerConfig(
            dim_in=dim_in,
            dim_out=dim_out,
            has_l2norm=False,
            dropout=cfg.mlp_head.dropout,
            has_act=cfg.mlp_head.has_act,
            final_act=False,
            act=cfg.gnn.act,
            has_bias=True,
            dim_inner=cfg.gnn.dim_inner,
            num_layers=cfg.mlp_head.num_layers ,
        )

        self.mlp_head = MLP(
            mlp_layer_config
                 )

    
    def forward(self, batch):
        # apply flat pooling method
        x_pool = self.pooling_fun(batch.x)
        batch.graph_feature = self.mlp_head(x_pool)
        batch.pred = batch.graph_feature
        batch.true = batch.y
        return batch