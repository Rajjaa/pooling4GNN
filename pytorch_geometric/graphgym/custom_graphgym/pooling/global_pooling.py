import torch
import torch_geometric.graphgym.register as register
from torch.nn import Conv1d, Linear
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import (
    GeneralLayer,
    LayerConfig,
    new_layer_config
)
from torch_geometric.graphgym.register import register_pooling
from torch_geometric.nn import (
    global_sort_pool,
    GlobalAttention,
    Set2Set
)

@register_pooling('set2set')
class Set2SetPool(torch.nn.Module):
    def __init__(self, dim_in, dim_out, cfg):
        super().__init__()

        self.set2set = Set2Set(dim_in, processing_steps=cfg.set2set.processing_steps)

        PostPool = register.layer_dict[cfg.global_pooling.post_pool_layer]

        self.post_pool = PostPool(
            LayerConfig(
                dim_in = 2 * dim_in, dim_out = dim_out, 
                has_l2norm=False, dropout=cfg.global_pooling.dropout, has_act=True,
                has_bias=True
            )
        )
    
        
    def forward(self, batch):
        batch.graph_feature = self.set2set(batch.x, batch.batch)
        batch = self.post_pool(batch)
        return batch


@register_pooling('globalattention')
class GlobalAttPool(torch.nn.Module):
    def __init__(self):
        super().__init__(dim_in, dim_out, cfg)

        self.att = GlobalAttention(Linear(dim_in, 1))


    def forward(self, batch):
        batch.graph_feature = self.att(batch.x, batch.batch)
        return batch


@register_pooling('sortpool')
class SortPool(torch.nn.Module):
    def __init__(self, dim_in, dim_out, cfg):
        super().__init__()

        self.conv1d = Conv1d(dim_in, cfg.sort_pooling.conv_dim, cfg.sort_pooling.conv_kernel_size)
        self.act = register.act_dict[cfg.gnn.act]

        PostPool = register.layer_dict[cfg.global_pooling.post_pool_layer]
        self.post_pool = PostPool(
            LayerConfig(
                dim_in = cfg.sort_pooling.conv_dim * (cfg.sort_pooling.k - cfg.sort_pooling.conv_kernel_size + 1),
                dim_out = dim_out, 
                has_l2norm=False, dropout=cfg.global_pooling.post_pool_dropout, has_act=True,
                has_bias=True
            )
        )

    def forward(self, batch):
        batch.graph_feature = global_sort_pool(batch.x, batch.batch, cfg.sort_pooling.k)

        batch.graph_feature = batch.graph_feature.view(len(batch.graph_feature), cfg.sort_pooling.k, -1).permute(0, 2, 1)
        batch.graph_feature = self.act(self.conv1d(batch.graph_feature))
        batch.graph_feature = batch.graph_feature.view(len(batch.graph_feature), -1)

        batch = self.post_pool(batch)
        return batch
