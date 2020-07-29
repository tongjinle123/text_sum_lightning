import fire
from src.initializer import Initializer
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from warnings import filterwarnings
filterwarnings('ignore')


def main(config_path):
    seed_everything(42)
    initializer = Initializer(None)
    initializer.load_from_yaml(config_path)
    config = initializer.config
    train_loader = initializer.get_train_dataloader()
    val_loader = initializer.get_dev_dataloader()
    model = initializer.get_lightning_model()
    model_name = config.model['class'].split('.')[-1]
    logger = TensorBoardLogger(**config.logger_ckpt, name=model_name)
    file_path = f'{logger.save_dir}/{model_name}/version_{logger.version}/' + '{epoch}-{val_loss: .4f}-{val_mer: .4f}'
    model_checkpoint = ModelCheckpoint(
        filepath=file_path,
        monitor='val_loss', verbose=True, save_top_k=2)
    trainer = Trainer(
        **config.trainer, checkpoint_callback=model_checkpoint, logger=logger, profiler=True,
    )
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    fire.Fire(main)

# python main_fire.py src/configs/config.yaml