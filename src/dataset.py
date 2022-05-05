import itertools
import re
import numpy as np

from transformers import PreTrainedTokenizer
from datasets import load_dataset
from enum import Enum
from torch.utils.data import Dataset, IterableDataset, get_worker_info


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
    _lang_ids = {lang.value: idx for idx, lang in enumerate(Languages)}

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        is_train: bool = True,
        n_sentences: int = 10,
        max_seq_length: int = 512,
        val_dataset_size: int = int(1e5),
        seed: int = 57
    ):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        np.random.seed(seed + worker_id)

        self.is_train = is_train
        self.n_sentences = n_sentences
        self.max_seq_length = max_seq_length
        self.val_dataset_size = val_dataset_size

        self.tokenizer = tokenizer
        self.bos_token_id = tokenizer.cls_token_id
        self.eos_token_id = tokenizer.sep_token_id

        datasets_raw = [load_dataset("wikiann", lang.value) for lang in Languages]
        datasets_raw = [ds["train"] if is_train else ds["validation"] for ds in datasets_raw]
        self.datasets = {
            lang.value: self._prepare_sentences(datasets_raw[lang_id])
            for lang_id, lang in zip(self._lang_ids.values(), Languages)
        }

        if not is_train:
            self.val_samples = [
                self._generate_sample(n_sentences=self.n_sentences, max_seq_length=self.max_seq_length)
                for _ in range(self.val_dataset_size)
            ]

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

        labels = [[self._lang_ids[lang]] * len(self.tokenizer.tokenize(sent)) for sent, lang in zip(sentences, languages)]

        sample = " ".join(sentences)
        sample = self.tokenizer.encode(sample, add_special_tokens=False)[: max_seq_length - 2]
        sample = [self.bos_token_id] + sample + [self.eos_token_id]

        labels = list(itertools.chain.from_iterable(labels))

        return sample, labels
