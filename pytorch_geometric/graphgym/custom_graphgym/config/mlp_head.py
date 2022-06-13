from yacs.config import CfgNode as CN

from torch_geometric.graphgym.register import register_config


@register_config('mlp_head')
def set_cfg_mlp_head(cfg):
    """
    This function sets the default values for customized pooling layers
    """

    # ----------------------------------------------------------------------- #
    # Customized options
    # ----------------------------------------------------------------------- #

    cfg.mlp_head  = CN()

    cfg.mlp_head.has_act = True

    cfg.mlp_head.num_layers = 1

    cfg.mlp_head.dropout = 0.0
