import itertools
import os.path
import re
import torch
import numpy as np
import pytorch_lightning as pl

from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer, AutoTokenizer
from datasets import load_dataset
from enum import Enum
from typing import Tuple, Dict, Optional
from torch.utils.data import IterableDataset, get_worker_info, DataLoader
from tqdm import trange

from definitions import ROOT_DIR
from src.configs.config_classes import TrainConfig


class Languages(Enum):
    RUSSIAN = "ru"
    UKRAINIAN = "uk"
    Belarusian = "be"
    Kazakh = "kk"
    Azerbaijani = "az"
    Armenian = "hy"
    Georgian = "ka"
    Hebrew = "he"
    English = "en"
    German = "de"


class MultiLanguageDataset(IterableDataset):
    _symbols_to_replace_regex = re.compile("( +[0-9]+|\*)")
    _langs_to_ids = {lang.value: idx for idx, lang in enumerate(Languages)}
    _ids_to_langs = {idx: lang for lang, idx in _langs_to_ids.items()}

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        n_sentences: int = 10,
        max_seq_length: int = 512,
        val_dataset_size: int = int(1e5),
        is_train: bool = True,
        seed: int = 57,
    ):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        np.random.seed(seed + worker_id)

        self.is_train = is_train
        self.n_sentences = n_sentences
        self.max_seq_length = max_seq_length
        self.val_dataset_size = val_dataset_size

        self.tokenizer = tokenizer

        datasets_raw = [
            load_dataset("wikiann", lang.value, cache_dir=os.path.join(ROOT_DIR, "data")) for lang in Languages
        ]
        datasets_raw = [ds["train"] if is_train else ds["validation"] for ds in datasets_raw]
        self.datasets = {
            lang.value: self._prepare_sentences(datasets_raw[lang_id])
            for lang_id, lang in zip(self._langs_to_ids.values(), Languages)
        }

        if not is_train:
            self.val_samples = [
                self._generate_sample(n_sentences=self.n_sentences, max_seq_length=self.max_seq_length)
                for _ in trange(self.val_dataset_size, desc="Creating validation dataset...")
            ]

            sample_lens = [s[0].shape[0] for s in self.val_samples]
            print(f"Average sample length: {np.mean(sample_lens)}")

    def _prepare_sentences(self, dataset):
        """
        Merges all the tokens in dataset and deletes separate numbers and other symbols.
        :param datasets:
        :return:
        Returns a list of sentences
        """
        sentences_tokenized = dataset["tokens"]
        sentences = [" ".join(sent_tokenized) for sent_tokenized in sentences_tokenized]
        sentences = [re.sub(self._symbols_to_replace_regex, "", s) for s in sentences]

        return sentences

    def __iter__(self):
        if self.is_train:
            while True:
                yield self._generate_sample(n_sentences=self.n_sentences, max_seq_length=self.max_seq_length)
        else:
            for sample in self.val_samples:
                yield sample

    def _generate_sample(
        self, n_languages: int = 5, n_sentences_per_language: int = 1, n_sentences: int = 10, max_seq_length: int = 512
    ):
        languages = np.random.choice([lang.value for lang in Languages], n_sentences, replace=True)
        sentences = np.array([np.random.choice(self.datasets[lang], 1)[0] for lang in languages])

        permutation = np.random.permutation(len(languages))

        languages = languages[permutation]
        sentences = sentences[permutation]

        labels = [
            [self._langs_to_ids[lang]] * len(self.tokenizer.tokenize(sent)) for sent, lang in zip(sentences, languages)
        ]

        sample = " ".join(sentences)
        input_ids, token_type_ids, attention_mask = self.tokenizer(
            sample, max_length=max_seq_length, truncation=True, padding=False, return_tensors="pt"
        ).values()

        # -100 labels is for tokenizer special tokens not to calculate loss on them
        labels = [-100] + list(itertools.chain.from_iterable(labels))[:max_seq_length-2] + [-100]
        labels = torch.tensor(labels)

        return input_ids.flatten(), token_type_ids.flatten(), attention_mask.flatten(), labels


class MultiLangCollate:
    def __call__(self, batch: Tuple[torch.Tensor, ...]) -> Dict[str, torch.Tensor]:
        input_ids, token_type_ids, attention_mask, labels = list(zip(*batch))
        input_ids, token_type_ids, attention_mask = [
            pad_sequence(t, batch_first=True) for t in [input_ids, token_type_ids, attention_mask]
        ]
        # -100 not to calculate loss on padding tokens
        labels = pad_sequence(labels, padding_value=-100, batch_first=True)

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class MultiLanguageDataModule(pl.LightningDataModule):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.n_sentences = cfg.data.n_sentences
        self.max_seq_length = cfg.data.max_seq_length
        self.val_dataset_size = cfg.data.val_dataset_size
        self.batch_size = cfg.training.batch_size
        self.num_workers = cfg.data.num_workers
        self.seed = cfg.seed

        self.collate_fn = MultiLangCollate()

    def setup(self, stage: Optional[str] = None) -> None:
        self.ds_train = MultiLanguageDataset(
            self.tokenizer, self.n_sentences, self.max_seq_length, self.val_dataset_size, True, seed=self.seed
        )

        self.ds_val = MultiLanguageDataset(
            self.tokenizer, self.n_sentences, self.max_seq_length, self.val_dataset_size, False, seed=self.seed
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
        )
