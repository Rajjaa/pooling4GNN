from yacs.config import CfgNode as CN

from torch_geometric.graphgym.register import register_config


@register_config('custom')
def set_cfg_pooling(cfg):
    """
    This function sets the default values for customized config values
    """
    # ----------------------------------------------------------------------- #
    # dataset options
    # ----------------------------------------------------------------------- #

    cfg.dataset.subset = False

    # ----------------------------------------------------------------------- #
    # train options
    # ----------------------------------------------------------------------- #

    # whether there is an auxiliary loss computed by the model forward
    cfg.model.aux_loss = False

    cfg.model.gradient_clip_val = 0.0


    # ----------------------------------------------------------------------- #
    # gnn options
    # ----------------------------------------------------------------------- #