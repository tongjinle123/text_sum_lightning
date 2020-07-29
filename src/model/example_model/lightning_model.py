import pytorch_lightning as pl
from argparse import Namespace


def dict_config2namespace(dict_config):
    items = {}
    for key, value in dict_config.items():
        items[key] = value
    name_space = Namespace(**items)
    return name_space


class BaseLightning(pl.LightningModule):
    def __init__(self, hparams, model, **kwargs):
        self.hparams = hparams
        self.model = model
        self.hparams = dict_config2namespace(self.hparams)

