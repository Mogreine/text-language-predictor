import os.path
from dataclasses import dataclass, field
from typing import Union

import pyrallis

from definitions import CONFIGS_DIR
from src.dataset import Languages, MultiLanguageDataModule
from src.model import BertLangNER


@dataclass
class DataConfig:
    n_sentences: int = field(default=10)
    max_seq_length: int = field(default=512)
    val_dataset_size: int = field(default=1e5)
    num_workers: int = field(default=0)


@dataclass
class TrainingConfig:
    batch_size: int = field(default=8)
    n_steps: int = field(default=10)
    n_warmup_steps: Union[int, float] = field(default=0.1)
    lr: float = field(default=2e-5)
    w_decay: float = field(default=0)
    scheduler: str = field(default="cosine")
    # ratio for slanted scheduler
    ratio: int = field(default=100)
    # lr decay for bert layers
    lr_decay: float = field(default=0.95)


@dataclass
class TrainConfig:
    """ Training config for Machine Learning """

    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 57


@pyrallis.wrap(config_path=os.path.join(CONFIGS_DIR, "bert_config.yaml"))
def train(cfg: TrainConfig):
    n_classes = len(Languages)
    model = BertLangNER(cfg, n_classes)
    datamodule = MultiLanguageDataModule(cfg)
    ...


if __name__ == "__main__":
    train()
