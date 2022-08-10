from collections import defaultdict
from math import ceil

import torch.nn as nn

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_stage


@register_stage('hierarchical_pooling')
class HieraPoolStage(nn.Module):
    """
    Stage that stacks GNN Layers interweaved with pooling layers and after pooling layers

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        num_layers (int): Number of GNN layers
    """
    def __init__(self, dim_in, dim_out, num_blocks):
        super().__init__()
        PoolBlock = register.pooling_dict[f'{cfg.hierarchical_pooling.type}_block']
        self.num_blocks = num_blocks
        self.post_pool = cfg.hierarchical_pooling.num_post_pool_layers != 0
        PostPoolBlock = register.layer_dict['gnn_dense_block']

        max_num_nodes = cfg.dataset.max_num_nodes
        pool_ratio = cfg.hierarchical_pooling.pool_ratio
        no_new_nodes = ceil(pool_ratio * max_num_nodes)
                

        for i in range(self.num_blocks):
            d_in = dim_in if i == 0 else dim_out
            dense = i != 0
            pool_block = PoolBlock(d_in, dim_out, no_new_nodes, dense=dense)
            self.add_module(f'pool_block{i}', pool_block)
            no_new_nodes = ceil(pool_ratio * no_new_nodes)
            if self.post_pool:
                post_pool_block = PostPoolBlock(dim_out, dim_out,
                 cfg.hierarchical_pooling.num_post_pool_layers, final_act=False, dense=True)
                self.add_module(f'post_pool_block{i}', post_pool_block)


    def forward(self, batch):
        """"""
        if cfg.model.aux_loss:
            batch.aux_loss = defaultdict(int)

        for i in range(self.num_blocks):
            pool_block = self.get_submodule(f'pool_block{i}')
            to_dense = i == 0
            batch = pool_block(batch, to_dense)
            if self.post_pool:
                post_pool_block = self.get_submodule(f'post_pool_block{i}')
                batch = post_pool_block(batch)
        return batch