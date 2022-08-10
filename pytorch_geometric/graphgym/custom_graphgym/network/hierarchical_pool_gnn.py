import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.init import init_weights
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network

@register_network('hpoolgnn')
class HPoolGNN(nn.Module):
    """
    GNN module for graph classification: encoder + stage + global pooling + head

    Args:
        dim_in (int): Input dimension
        dim_out (int): Ouptut dimension
        **kwargs (optional): Optional additional args
    """

    def __init__(self, dim_in, dim_out, **kwargs):
        super().__init__()

        # Encoder
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        # Layers before message passing
        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner,
                                   cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        # Message passing layers interweaved with pooling layers
        GNNStage = register.stage_dict[cfg.gnn.stage_type]
        self.mp = GNNStage(dim_in=dim_in, dim_out=cfg.gnn.dim_inner,
                            num_blocks=cfg.hierarchical_pooling.num_blocks)

        # Classification head (MLP)
        GNNHead = register.head_dict['hpool_head']
        self.head = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

        self.apply(init_weights)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
