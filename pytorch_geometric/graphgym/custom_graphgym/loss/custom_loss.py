import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss

@register_loss('compute_loss')
def compute_loss(batch):
    """
    Compute loss and prediction score

    Returns: Loss, normalized prediction score

    """
    bce_loss = nn.BCEWithLogitsLoss(reduction=cfg.model.size_average)
    mse_loss = nn.MSELoss(reduction=cfg.model.size_average)

    # default manipulation for pred and true
    # can be skipped if special loss computation is needed
    pred, true = batch.pred, batch.true
    pred = pred.squeeze(-1) if pred.ndim > 1 else pred
    true = true.squeeze(-1) if true.ndim > 1 else true


    if cfg.model.loss_fun == 'cross_entropy':
        # multiclass
        if pred.ndim > 1 and true.ndim == 1:
            pred = F.log_softmax(pred, dim=-1)
            loss = F.nll_loss(pred, true)
        # binary or multilabel
        else:
            true = true.float()
            loss = bce_loss(pred, true)
            pred = torch.sigmoid(pred)
    elif cfg.model.loss_fun == 'mse':
        true = true.float()
        loss = mse_loss(pred, true)
    else:
        raise ValueError('Loss func {} not supported'.format(
            cfg.model.loss_fun))

    if cfg.model.aux_loss:
        aux_loss = sum(batch.aux_loss.values())
    else:
        aux_loss = 0
    
    return loss + aux_loss, pred
