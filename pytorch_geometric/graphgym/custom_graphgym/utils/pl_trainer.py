
from torch_geometric.graphgym.config import cfg

def config_trainer(cfg):
    trainer_config = {
        'gradient_clip_val': cfg.model.gradient_clip_val,
    }
    return trainer_config