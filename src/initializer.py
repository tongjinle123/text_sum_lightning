from omegaconf import DictConfig
import torch as t
from torch.utils.data.dataset import ConcatDataset
from hydra.utils import instantiate
from hydra._internal.utils import _strict_mode_strategy, split_config_path, create_automatic_config_search_path
from hydra._internal.hydra import Hydra
from hydra.utils import get_class
from warnings import filterwarnings

filterwarnings('ignore')
import sys
import os

sys.path.append(os.getcwd())


class Initializer:
    """
    init all modules just from config.yaml in anywhere
    """

    def __init__(self, config: DictConfig):
        self.config = config

    def load_from_yaml(self, config_path, strict=True):
        config_dir, config_file = split_config_path(config_path)
        strict = _strict_mode_strategy(strict, config_file)
        search_path = create_automatic_config_search_path(
            config_file, None, config_dir
        )
        hydra = Hydra.create_main_hydra2(
            task_name='sdfs', config_search_path=search_path, strict=strict
        )
        config = hydra.compose_config(config_file, [])
        config.pop('hydra')
        self.config = config
    #
    # def get_audio_line(self):
    #     audio_line = instantiate(self.config.audio_line)
    #     return audio_line
    #
    # def get_vocab(self):
    #     vocab = instantiate(self.config.vocab)
    #     return vocab
    #
    # def get_train_dataset(self):
    #     vocab = self.get_vocab()
    #     audio_line = self.get_audio_line()
    #     clazz = get_class(self.config.dataset['class'])
    #     audio_sets = [
    #         clazz(**self.config.dataset.params, audio_line=audio_line, vocab=vocab,
    #               manifest=file,
    #               ) for file in self.config.data.manifestlist
    #     ]
    #     dataset = ConcatDataset(audio_sets)
    #     # file = 'data/filterd_manifest/libri_100.csv'
    #     # dataset = clazz(**self.config.dataset.params, audio_line=audio_line, vocab=vocab, manifest=file)
    #     return dataset
    #
    # def get_dataset(self, given_rate, manifestlist):
    #     vocab = self.get_vocab()
    #     audio_line = self.get_audio_line()
    #     clazz = get_class(self.config.dataset['class'])
    #     audio_sets = [
    #         clazz(**self.config.dataset.params, given_rate=given_rate, audio_line=audio_line, vocab=vocab,
    #               manifest=file,
    #               ) for file in manifestlist
    #     ]
    #     dataset = ConcatDataset(audio_sets)
    #     # file = 'data/filterd_manifest/libri_100.csv'
    #     # dataset = clazz(**self.config.dataset.params, audio_line=audio_line, vocab=vocab, manifest=file)
    #     return dataset
    #
    # def get_train_dataloader(self):
    #     dataset = self.get_dataset(self.config.data.train_given_rate, self.config.data.train_mlist)
    #     clazz = get_class(self.config.dataloader['class'])
    #     dataloader = clazz(**self.config.dataloader.params, dataset=dataset, shuffle=True)
    #     return dataloader
    #
    # def get_dev_dataloader(self):
    #     dataset = self.get_dataset(self.config.data.test_given_rate, self.config.data.dev_mlist)
    #     clazz = get_class(self.config.dataloader['class'])
    #     dataloader = clazz(**self.config.dataloader.params, dataset=dataset, shuffle=False)
    #     return dataloader
    #
    # def get_test_dataloader(self):
    #     dataset = self.get_dataset(self.config.data.test_given_rate, self.config.data.test_mlist)
    #     clazz = get_class(self.config.dataloader['class'])
    #     dataloader = clazz(**self.config.dataloader.params, dataset=dataset, shuffle=False)
    #     return dataloader
    #
    # def get_model(self):
    #     vocab = self.get_vocab()
    #     clazz = get_class(self.config.model['class'])
    #     model = clazz(**self.config.model.params, vocab=vocab)
    #     return model
    #
    # def get_lightning_model(self):
    #     model = self.get_model()
    #     clazz = get_class(self.config.model.params.lightning_model_class)
    #     lightning_model = clazz(hparams=self.config.model.params, model=model)
    #     return lightning_model
    #
    # def get_trained_model(self, ckpt, to_cuda=False):
    #     model = self.get_lightning_model()
    #     state = t.load(ckpt, map_location='cpu')
    #     model.load_state_dict(state['state_dict'])
    #     model = model.model
    #     model.eval()
    #     if to_cuda:
    #         model.cuda()
    #     del state
    #     return model
    #
    # def get_lm_dataset(self, manifestlist):
    #     vocab = self.get_vocab()
    #     clazz = get_class(self.config.dataset['class'])
    #     audio_sets = [
    #         clazz(**self.config.dataset.params, vocab=vocab,
    #               manifest=file,
    #               ) for file in manifestlist
    #     ]
    #     dataset = ConcatDataset(audio_sets)
    #     return dataset
    #
    # def get_lm_train_dataloader(self):
    #     dataset = self.get_lm_dataset(self.config.data.train_mlist)
    #     clazz = get_class(self.config.dataloader['class'])
    #     dataloader = clazz(**self.config.dataloader.params, dataset=dataset, shuffle=True)
    #     return dataloader
    #
    # def get_lm_dev_dataloader(self):
    #     dataset = self.get_lm_dataset(self.config.data.dev_mlist)
    #     clazz = get_class(self.config.dataloader['class'])
    #     dataloader = clazz(**self.config.dataloader.params, dataset=dataset, shuffle=False)
    #     return dataloader


if __name__ == '__main__':
    ini = Initializer(None)
    ini.load_from_yaml('src/configs/rnn_lm_full.yaml')
    # audioline = ini.get_audio_line()
    # audioline('data/extracted/libri_100/LibriSpeech/train-clean-100/103/1241/103-1241-0000.flac')
    vocab = ini.get_vocab()
    print(vocab.str2id('a'))
    # dataset = ini.get_dataset()
    # dataset.load_wav('data/extracted/libri_100/LibriSpeech/train-clean-100/103/1241/103-1241-0000.flac')
    dataloader = ini.get_lm_train_dataloader()
    for i in dataloader:
        print(i)
        break
    # trans = ini.get_model()
    # lig = ini.get_lightning_model()




