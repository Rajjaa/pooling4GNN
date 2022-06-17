from torch_geometric.graphgym.register import register_pooling


@register_pooling('dense_add')
def dense_global_add_pool(x, mask=None):
    r"""Returns batch-wise graph-level-outputs by adding node features
    across the node dimension, so that for a single graph
    :math:`\mathcal{G}_i` its output is computed by

    .. math::
        \mathbf{r}_i = \sum_{n=1}^{N_i} \mathbf{x}_n

    x (Tensor): Node feature matrix
    :math:`\mathbf{X} \in \mathbb{R}^{B \times N_{\max} \times F}` (with
    :math:`N_{\max} = \max_i^B N_i`).

    """
    if mask is not None:
        batch_size = x.size(0)
        num_nodes = x.size(1)
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x = x * mask
    
    return x.sum(dim=1)

@register_pooling('dense_mean')
def dense_global_mean_pool(x, mask=None):
    r"""Returns batch-wise graph-level-outputs by adding node features
    across the node dimension, so that for a single graph
    :math:`\mathcal{G}_i` its output is computed by

    .. math::
        \mathbf{r}_i = \sum_{n=1}^{N_i} \mathbf{x}_n

    x (Tensor): Node feature matrix
    :math:`\mathbf{X} \in \mathbb{R}^{B \times N_{\max} \times F}` (with
    :math:`N_{\max} = \max_i^B N_i`).

    """
    if mask is not None:
        batch_size = x.size(0)
        num_nodes = x.size(1)
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x = x * mask
    
    return x.mean(dim=1)

@register_pooling('dense_max')
def dense_global_max_pool(x, mask=None):
    r"""Returns batch-wise graph-level-outputs by adding node features
    across the node dimension, so that for a single graph
    :math:`\mathcal{G}_i` its output is computed by

    .. math::
        \mathbf{r}_i = \sum_{n=1}^{N_i} \mathbf{x}_n

    x (Tensor): Node feature matrix
    :math:`\mathbf{X} \in \mathbb{R}^{B \times N_{\max} \times F}` (with
    :math:`N_{\max} = \max_i^B N_i`).

    """
    if mask is not None:
        batch_size = x.size(0)
        num_nodes = x.size(1)
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x = x * mask
    x, _ = x.max(dim=1)
    return x