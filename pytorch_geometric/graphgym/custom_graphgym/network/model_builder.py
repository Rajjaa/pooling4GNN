import time
from typing import Any, Dict, Tuple

import torch

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.imports import LightningModule
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.models.gnn import GNN
from torch_geometric.graphgym.optim import create_optimizer, create_scheduler
from torch_geometric.graphgym.register import network_dict, register_network


compute_loss = register.loss_dict['compute_loss']

@register_network('graphGymModule')
class GraphGymModule(LightningModule):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.cfg = cfg
        self.model = network_dict[cfg.model.type](dim_in=dim_in,
                                                  dim_out=dim_out)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self) -> Tuple[Any, Any]:
        optimizer = create_optimizer(self.model.parameters(), self.cfg.optim)
        scheduler = create_scheduler(optimizer, self.cfg.optim)
        return [optimizer], [scheduler]

    def _shared_step(self, batch, split: str) -> Dict:
        batch.split = split
        batch = self(batch)
        loss, pred_score = compute_loss(batch)
        true = batch.true
        aux_loss = batch.aux_loss if cfg.model.aux_loss else {}
        step_end_time = time.time()
        return dict(loss=loss, aux_loss=aux_loss, true=true, pred_score=pred_score,
                    step_end_time=step_end_time)

    def training_step(self, batch, *args, **kwargs):
        return self._shared_step(batch, split="train")

    def validation_step(self, batch, *args, **kwargs):
        return self._shared_step(batch, split="val")

    def test_step(self, batch, *args, **kwargs):
        return self._shared_step(batch, split="test")

    @property
    def encoder(self) -> torch.nn.Module:
        return self.model.encoder

    @property
    def mp(self) -> torch.nn.Module:
        return self.model.mp

    @property
    def post_mp(self) -> torch.nn.Module:
        return self.model.post_mp

    @property
    def pre_mp(self) -> torch.nn.Module:
        return self.model.pre_mp

@register_network('create_model')
def create_model(to_device=True, dim_in=None, dim_out=None) -> GraphGymModule:
    r"""Create model for graph machine learning.

    Args:
        to_device (string): The devide that the model will be transferred to
        dim_in (int, optional): Input dimension to the model
        dim_out (int, optional): Output dimension to the model
    """
    dim_in = cfg.share.dim_in if dim_in is None else dim_in
    dim_out = cfg.share.dim_out if dim_out is None else dim_out
    # binary classification, output dim = 1
    if 'classification' in cfg.dataset.task_type and dim_out == 2:
        dim_out = 1

    model = GraphGymModule(dim_in, dim_out)
    if to_device:
        model.to(torch.device(cfg.device))
    return model