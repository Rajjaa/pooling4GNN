import torch_geometric.graphgym.register as register
from typing import Optional

from torch.utils.data import DataLoader

from torch_geometric.data.lightning_datamodule import LightningDataModule
# from torch_geometric.graphgym import create_loader
from torch_geometric.graphgym.checkpoint import get_ckpt_dir
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.imports import pl
from torch_geometric.graphgym.logger import LoggerCallback
from torch_geometric.graphgym.model_builder import GraphGymModule
from torch_geometric.graphgym.register import register_train

create_loader = register.loader_dict['custom_create_loader']


@register_train('datamodule')
class GraphGymDataModule(LightningDataModule):
    def __init__(self):
        self.loaders = create_loader()
        super().__init__(has_val=True, has_test=True)

    def train_dataloader(self) -> DataLoader:
        return self.loaders[0]

    def val_dataloader(self) -> DataLoader:
        # better way would be to test after fit.
        # First call trainer.fit(...) then trainer.test(...)
        return self.loaders[1]

    def test_dataloader(self) -> DataLoader:
        return self.loaders[2]

