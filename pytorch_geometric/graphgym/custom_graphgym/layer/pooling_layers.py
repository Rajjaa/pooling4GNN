import torch
import torch.nn as nn
from torch.nn import Parameter

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import (
    GeneralLayer,
    LayerConfig,
)
from torch_geometric.graphgym.register import register_layer
from torch_geometric.nn import dense_mincut_pool

@register_layer('post_pool_linear')
class PostPool(nn.Module):
    def __init__(self, layer_config):
        super().__init__()
        self.model = GeneralLayer('linear', layer_config)

    
    def forward(self, batch):
        batch.graph_feature = self.model(batch.graph_feature)
        return batch

