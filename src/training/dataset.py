import itertools
import os.path
import re
from string import punctuation

import torch
import numpy as np
import pytorch_lightning as pl

from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer, AutoTokenizer
from datasets import load_dataset
from enum import Enum
from typing import Tuple, Dict, Optional, List
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
    _symbols_to_replace_regex = re.compile(f"[\*0-9{punctuation}]")
    _punctuation_regex = re.compile(f"[{punctuation}]")
    _langs_to_ids = {lang.value: idx for idx, lang in enumerate(Languages)}
    _ids_to_langs = {idx: lang for lang, idx in _langs_to_ids.items()}

    @classmethod
    def preprocess_text(cls, text: str, delete_punctuation: bool = True) -> str:
        text = re.sub(cls._symbols_to_replace_regex, "", text)
        if delete_punctuation:
            text = re.sub(cls._punctuation_regex, "", text)

        return text

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        cfg: TrainConfig,
        is_train: bool,
    ):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        np.random.seed(cfg.seed + worker_id)

        self.is_train = is_train
        self.n_sentences = cfg.data.n_sentences
        self.max_seq_length = cfg.data.max_seq_length
        self.val_dataset_size = cfg.data.val_dataset_size
        self.word_perm_prob = cfg.data.word_perm_prob
        self.language_sampling_strategy = cfg.data.language_sampling_strategy
        self.lengths = cfg.data.lengths
        self.length_probs = cfg.data.length_probs
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
        sentences = [self.preprocess_text(s) for s in sentences]

        return sentences

    def __iter__(self):
        if self.is_train:
            while True:
                yield self._generate_sample(n_sentences=self.n_sentences, max_seq_length=self.max_seq_length)
        else:
            for sample in self.val_samples:
                yield sample

    def _generate_sample(self, n_sentences: int = 10, max_seq_length: int = 512):
        if self.language_sampling_strategy == "constant":
            n_sentences = self.n_sentences
        elif self.language_sampling_strategy == "uniform":
            n_sentences = np.random.randint(1, n_sentences + 1)
        else:
            [n_sentences] = np.random.choice(self.lengths, 1, p=self.length_probs)

        languages = np.random.choice([lang.value for lang in Languages], n_sentences, replace=True)
        sentences = np.array([self.datasets[lang][np.random.randint(len(self.datasets[lang]))] for lang in languages])

        if np.random.rand() <= self.word_perm_prob:
            tokens, labels = self._word_permutation(sentences, languages)
        else:
            tokens, labels = self._sentence_permutation(sentences, languages)

        sample = " ".join(tokens)
        input_ids, token_type_ids, attention_mask = self.tokenizer(
            sample, max_length=max_seq_length, truncation=True, padding=False, return_tensors="pt"
        ).values()

        # -100 labels is for tokenizer special tokens not to calculate loss on them
        labels = [-100] + list(itertools.chain.from_iterable(labels))[: max_seq_length - 2] + [-100]
        labels = torch.tensor(labels)

        assert (
            input_ids.shape[-1] == labels.shape[0]
        ), f"Token ids shape does not match labels shape -- {input_ids.shape[-1]} != {labels.shape[0]}"

        return input_ids.flatten(), token_type_ids.flatten(), attention_mask.flatten(), labels

    def _sentence_permutation(self, sentences: np.ndarray, languages: np.ndarray) -> Tuple[List[str], List[List[int]]]:
        permutation = np.random.permutation(len(languages))
        languages = languages[permutation]
        sentences = sentences[permutation]
        labels = [
            [self._langs_to_ids[lang]] * len(self.tokenizer.tokenize(sent)) for sent, lang in zip(sentences, languages)
        ]

        return sentences, labels

    def _word_permutation(self, sentences: np.ndarray, languages: np.ndarray) -> Tuple[List[str], List[List[int]]]:
        # split sentences into words
        wrd_lang_pairs = [(word, languages[i]) for i in range(len(sentences)) for word in sentences[i].split()]
        np.random.shuffle(wrd_lang_pairs)
        words, _ = zip(*wrd_lang_pairs)
        labels = [[self._langs_to_ids[lang]] * len(self.tokenizer.tokenize(word)) for word, lang in wrd_lang_pairs]

        return words, labels


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
        self.batch_size = cfg.training.batch_size
        self.num_workers = cfg.data.num_workers

        self.cfg = cfg

        self.collate_fn = MultiLangCollate()

    def setup(self, stage: Optional[str] = None) -> None:
        self.ds_train = MultiLanguageDataset(
            self.tokenizer,
            self.cfg,
            True,
        )

        self.ds_val = MultiLanguageDataset(
            self.tokenizer,
            self.cfg,
            False,
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
