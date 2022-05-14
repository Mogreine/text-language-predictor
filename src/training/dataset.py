import itertools
import json
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
from tqdm import trange, tqdm

from definitions import ROOT_DIR
from src.configs.config_classes import TrainConfig


class Languages(Enum):
    RUSSIAN = "ru"
    UKRAINIAN = "uk"
    BELARUSIAN = "be"
    KAZAKH = "kk"
    AZERBAIJANI = "az"
    ARMENIAN = "hy"
    GEORGIAN = "ka"
    HEBREW = "he"
    ENGLISH = "en"
    GERMAN = "de"


class MultiLanguageDataset(IterableDataset):
    """
    Dataset for text languages prediction detection task.

    Supports two datasets:
        1. Wikiann
        2. OpenSubtitles + mc4 (be and az languages)

    When working in train mode it generates random samples by mixing texts of different languages.
    When working in eval mode it generates a fixed evaluation dataset at initialization.

    """

    _symbols_to_replace_regex = re.compile(f"[\*0-9{punctuation}]")
    _punctuation_regex = re.compile(f"[{punctuation}]")
    _langs_to_ids = {lang.value: idx for idx, lang in enumerate(Languages)}
    _ids_to_langs = {idx: lang for lang, idx in _langs_to_ids.items()}

    @classmethod
    def preprocess_text(cls, text: str, delete_punctuation: bool = True) -> str:
        text = re.sub(cls._symbols_to_replace_regex, "", text)
        if delete_punctuation:
            text = re.sub(cls._punctuation_regex, "", text)

        return text.strip()

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

        if cfg.data.dataset_name == "wikiann":
            self.datasets = self._prepare_wikiann()
        else:
            self.datasets = self._prepare_opensub()

        if not is_train:
            self.val_samples = [
                self._generate_sample(n_sentences=self.n_sentences, max_seq_length=self.max_seq_length)
                for _ in trange(self.val_dataset_size, desc="Creating validation dataset...")
            ]

            sample_lens = [s[0].shape[0] for s in self.val_samples]
            print(f"Average sample length: {np.mean(sample_lens)}")

    def _prepare_wikiann(self) -> Dict[str, List[str]]:
        datasets_raw = [
            load_dataset("wikiann", lang.value, cache_dir=os.path.join(ROOT_DIR, "data")) for lang in Languages
        ]
        datasets_raw = [ds["train"] if self.is_train else ds["validation"] for ds in datasets_raw]

        def prepare_sentences(dataset):
            """
            Merges all the tokens in dataset and deletes separate numbers and other symbols.
            :param datasets:
            :return:
            Returns a list of sentences
            """
            sentences_tokenized = dataset["tokens"]
            sentences = [" ".join(sent_tokenized) for sent_tokenized in sentences_tokenized]
            sentences = [MultiLanguageDataset.preprocess_text(s) for s in sentences]

            return sentences

        return {
            lang.value: prepare_sentences(datasets_raw[lang_id])
            for lang_id, lang in zip(self._langs_to_ids.values(), Languages)
        }

    def _prepare_opensub(self) -> Dict[str, List[str]]:
        dataset_type = "train" if self.is_train else "val"
        dataset_path = os.path.join(ROOT_DIR, f"data/open_subtitles/{dataset_type}.json")

        with open(dataset_path, "r") as f:
            datasets = json.load(f)

        for lang, texts in datasets.items():
            texts_preprocessed = []
            for text in tqdm(texts, desc=f"Preprocessing {lang}..."):
                text = self.preprocess_text(text)
                if len(text.split()) != 0:
                    texts_preprocessed.append(text)
            datasets[lang] = texts_preprocessed

        return datasets

    def __iter__(self):
        if self.is_train:
            while True:
                yield self._generate_sample(n_sentences=self.n_sentences, max_seq_length=self.max_seq_length)
        else:
            for sample in self.val_samples:
                yield sample

    def _generate_sample(self, n_sentences: int = 10, max_seq_length: int = 512) -> Tuple[torch.Tensor, ...]:
        """
        Generates samples combining texts from different languages.

        Generation may be divided into 2 parts.
        Part 1. Defining number of sentences (texts) in a sample. There are 3 strategies:
            1. Constant -- every sample consists of fixed number of sentences
            2. Uniform -- every sample consists of `n` sentences. Where `n` is uniformly distributed between 1 and `n_sentences`
            3. Custom -- the number of sentences is sampled from the defined number of `lengths` with defined probabilities `length_probs`

        Part 2. Sentences (texts) sampling. Once number of sentences `n_sentences` is defined then
            1. Sample `n` languages (each with equal probability)
            2. Sample random texts
            3. Shuffle texts or split texts into words and shuffle them
            4. Concatenate texts/words into sample

        :param n_sentences: number of sentences (texts) in a sample
        :param max_seq_length: max sequence length of a sample in tokens
        :return: tensor inputs for bert model (input_ids, token_type_ids, attention_mask and labels)
        """
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

        # -100 labels is to mask tokenizer's special tokens not to calculate loss on them
        labels = [-100] + list(itertools.chain.from_iterable(labels))[: max_seq_length - 2] + [-100]
        labels = torch.tensor(labels)

        assert (
            input_ids.shape[-1] == labels.shape[0]
        ), f"Token ids shape does not match labels shape -- {input_ids.shape[-1]} != {labels.shape[0]}"

        return input_ids.flatten(), token_type_ids.flatten(), attention_mask.flatten(), labels

    def _sentence_permutation(self, sentences: np.ndarray, languages: np.ndarray) -> Tuple[List[str], List[List[int]]]:
        """
        Shuffles all the sentences and creates labels for tokens.
        :param sentences: sentences in a sample
        :param languages: languages corresponding to the sentences
        :return: ndarray of sentences and labels for tokens
        """
        permutation = np.random.permutation(len(languages))
        languages = languages[permutation]
        sentences = sentences[permutation]
        labels = [
            [self._langs_to_ids[lang]] * len(self.tokenizer.tokenize(sent)) for sent, lang in zip(sentences, languages)
        ]

        return sentences, labels

    def _word_permutation(self, sentences: np.ndarray, languages: np.ndarray) -> Tuple[List[str], List[List[int]]]:
        """
        Splits sentences into words, shuffles words and creates labels for tokens.
        :param sentences: sentences in a sample
        :param languages: languages corresponding to the sentences
        :return: ndarray of words and labels for tokens
        """
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
