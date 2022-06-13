from yacs.config import CfgNode as CN

from torch_geometric.graphgym.register import register_config


@register_config('pooling')
def set_cfg_pooling(cfg):
    """
    This function sets the default values for customized pooling layers
    """
    # ----------------------------------------------------------------------- #
    # global pooling options
    # ----------------------------------------------------------------------- #

    cfg.global_pooling = CN()
    cfg.global_pooling.post_pool_layer = 'post_pool_linear'
    cfg.global_pooling.post_pool_dropout = 0.5


    # ----------------------------------------------------------------------- #
    # sort pooling options
    # ----------------------------------------------------------------------- #

    cfg.sort_pooling  = CN()

    cfg.sort_pooling.k = 30
    cfg.sort_pooling.conv_dim = 32
    cfg.sort_pooling.conv_kernel_size = 5


    # ----------------------------------------------------------------------- #
    # set2set pooling options
    # ----------------------------------------------------------------------- #

    cfg.set2set = CN()
    cfg.set2set.processing_steps = 4

    # ----------------------------------------------------------------------- #
    # Hierarchical pooling options
    # ----------------------------------------------------------------------- #

    cfg.hierarchical_pooling = CN()
    cfg.hierarchical_pooling.pool_ratio = 0.5
    cfg.hierarchical_pooling.type = 'mincut_pool'
    cfg.hierarchical_pooling.flat_pooling = 'mean'
