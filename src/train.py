import os
import pyrallis
import pytorch_lightning as pl

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from definitions import CONFIGS_DIR
from src.dataset import Languages, MultiLanguageDataModule
from src.model import BertLangNER
from src.configs.config_classes import TrainConfig


@pyrallis.wrap(config_path=os.path.join(CONFIGS_DIR, "bert_config.yaml"))
def train(cfg: TrainConfig):
    pl.seed_everything(cfg.seed)

    n_classes = len(Languages)
    model = BertLangNER(cfg, n_classes)
    datamodule = MultiLanguageDataModule(cfg)

    logger = WandbLogger(project="text-lang-predictor", log_model=True)
    lr_logger = LearningRateMonitor()
    checkpoint_callback = ModelCheckpoint(
        filename="{step}-{val_loss:.3f}-{val_f1:.3f}",
        dirpath=logger.experiment.dir,
        save_top_k=1,
        save_last=True,
        verbose=True,
        monitor="val_f1",
        mode="max",
    )

    trainer = pl.Trainer(
        val_check_interval=cfg.training.val_interval,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        gpus=int(cfg.training.use_gpu),
        progress_bar_refresh_rate=1,
        log_every_n_steps=10,
        logger=logger,
        callbacks=[lr_logger, checkpoint_callback],
        deterministic=True,
        max_steps=cfg.training.n_steps,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    train()
