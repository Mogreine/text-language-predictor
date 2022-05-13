from dataclasses import dataclass, field
from typing import Union, List, Optional


@dataclass
class DataConfig:
    # Max number of sentences if a sample
    n_sentences: int = field(default=10)
    # Max sequence length of a sample
    max_seq_length: int = field(default=512)
    # Number of samples in validation dataset
    val_dataset_size: int = field(default=10)
    # Number of workers for training dataloader
    num_workers: int = field(default=0)
    # Probability of permutation of all words in a sample
    word_perm_prob: float = field(default=0.5)
    # Supports "constant", "uniform" and "custom"
    language_sampling_strategy: str = field(default="constant")
    lengths: Optional[List[int]] = field(default=None)
    length_probs: Optional[List[float]] = field(default=None)

    def __post_init__(self):
        assert self.language_sampling_strategy in [
            "constant",
            "uniform",
            "custom",
        ], f"Unknown sampling strategy: {self.language_sampling_strategy}. Must be 'constant', 'uniform' or 'custom'."


@dataclass
class OptConfig:
    # Whether to use gpu for training or not
    use_gpu: bool = field(default=True)
    # Batch size for training and validation
    batch_size: int = field(default=8)
    # Max number of steps for training
    n_steps: int = field(default=10)
    # Number of steps to accumulate gradient for
    accumulate_grad_batches: int = field(default=1)
    # Run validation every val_interval steps
    val_interval: int = field(default=5)

    # Number of warmup steps for training. May be either int or float
    n_warmup_steps: Union[int, float] = field(default=0.1)
    # Learning rate for AdamW optimizer
    lr: float = field(default=2e-5)
    # Weight decay for AdamW optimizer
    w_decay: float = field(default=0)
    # Scheduler type. Supports "cosine" and "slanted"
    scheduler: str = field(default="cosine")
    # Ratio for slanted scheduler
    ratio: int = field(default=100)
    # Lr decay for bert layers
    lr_decay: float = field(default=0.95)

    def __post_init__(self):
        assert self.scheduler in [
            "cosine",
            "slanted",
            "constant",
        ], f"Unknown scheduler: {self.scheduler}. Must be 'cosine', 'slanted' or 'constant'."


@dataclass
class TrainConfig:
    """ Config for training. """

    data: DataConfig = field(default_factory=DataConfig)
    training: OptConfig = field(default_factory=OptConfig)
    seed: int = 57


@dataclass
class InferenceConfig:
    """ Config for inference. """

    # Whether to run the model using gpu or not
    use_gpu: bool = field(default=False)
    # Max sequence length in tokens. Used for parsing long texts
    max_seq_length: int = field(default=256)
    # The text is split in chunks. Batch size if number of simultaneously processed chunks by the model
    batch_size: int = field(default=4)
    # Run id from W&B
    run_id: str = field(default="130npnhb")
