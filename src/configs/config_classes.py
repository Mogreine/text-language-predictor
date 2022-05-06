from dataclasses import dataclass, field
from typing import Union


@dataclass
class DataConfig:
    n_sentences: int = field(default=10)
    max_seq_length: int = field(default=512)
    val_dataset_size: int = field(default=10)
    num_workers: int = field(default=0)


@dataclass
class OptConfig:
    use_gpu: bool = field(default=True)
    batch_size: int = field(default=8)
    n_steps: int = field(default=10)
    accumulate_grad_batches: int = field(default=1)
    val_interval: int = field(default=5)

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
    training: OptConfig = field(default_factory=OptConfig)
    seed: int = 57
