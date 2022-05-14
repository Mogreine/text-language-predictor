import itertools
import os.path
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np
import torch
import wandb
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, BertForTokenClassification

from definitions import ROOT_DIR, CHECKPOINTS_DIR
from src.configs.config_classes import InferenceConfig
from src.training.dataset import MultiLanguageDataset
from src.training.model import BertLangNER


class TextLangPredictor:
    """
    Class for `BertLangNER` model inference.
    """

    def __init__(self, cfg: InferenceConfig):
        self._device = "cuda" if cfg.use_gpu else "cpu"
        self._run_id = cfg.run_id
        self._batch_size = cfg.batch_size
        self._max_seq_length = cfg.max_seq_length

        self._model = self._load_model()
        self._tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    def parse_text(self, text: str) -> Dict[str, str]:
        """
        Parses `text`. First it is processed and split into chunks so `text` might be of an arbitrary length.
        :param text: text to be pared
        :return: dict where keys are detected languages and values are corresponding text spans
        """
        text = MultiLanguageDataset.preprocess_text(text)
        chunks = self._chunk_text(text)
        chunks_encoded = self._tokenizer.batch_encode_plus(
            chunks, add_special_tokens=False, is_split_into_words=False, return_tensors="pt", padding=True
        )
        langs_ids = self._get_token_predictions(chunks_encoded)
        langs = self._lang_ids_to_names(langs_ids)

        return self._get_spans(text, langs)

    def _chunk_text(self, text: str) -> List[str]:
        """
        Chunks text and adds special tokens.
        :param text: a text to be chunked
        :return: chunks of initial text
        """
        sentences_tokenized = self._tokenizer.tokenize(text)
        n_chunks = int(np.ceil(len(sentences_tokenized) / self._max_seq_length))
        chunks = np.array_split(sentences_tokenized, n_chunks)
        chunks = [[self._tokenizer.cls_token] + chunk.tolist() + [self._tokenizer.sep_token] for chunk in chunks]
        chunks = [self._tokenizer.convert_tokens_to_string(tokens) for tokens in chunks]
        return chunks

    @torch.inference_mode()
    def _get_token_predictions(self, chunks: Dict[str, torch.Tensor]):
        """
        Calculates language prediction for every token in chunks.
        :param chunks: encoded by tokenizer chunks
        :return: predictions for every token.
        """
        preds = []
        dl = self._create_dataloader(chunks)
        for batch in tqdm(dl, desc="Making predictions..."):
            input_ids, token_type_ids, attention_mask = batch
            logits = self._model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).logits

            attention_mask[
                (input_ids == self._tokenizer.cls_token_id).logical_or(input_ids == self._tokenizer.sep_token_id)
            ] = 0
            token_preds = logits.argmax(-1)
            mask = attention_mask.to(device=self._device, dtype=torch.bool)

            lang_ids = torch.masked_select(token_preds, mask)

            preds.append(lang_ids.cpu().numpy().tolist())

        return list(itertools.chain.from_iterable(preds))

    def _create_dataloader(self, chunks: Dict[str, torch.Tensor]):
        ds = TensorDataset(*[t.to(self._device) for t in chunks.values()])
        return DataLoader(ds, batch_size=self._batch_size, shuffle=False)

    def _lang_ids_to_names(self, langs_ids: List[int]) -> List[str]:
        """
        Maps language ids to language codes.
        :param langs_ids: language ids
        :return: language codes
        """
        return [MultiLanguageDataset._ids_to_langs[idx] for idx in langs_ids]

    def _get_spans(self, text: str, langs: List[str]) -> Dict[str, str]:
        """
        Tokenizes input text and uses predicted languages for every oken to construct spans. After tokenization
        number of tokens is equal to number of prediction.
        :param text: initial text
        :param langs: predicted languages for tokens
        :return: dict where keys are detected languages and values are corresponding text spans
        """
        tokens = self._tokenizer.tokenize(text, add_special_tokens=False)
        res = defaultdict(list)
        for token, lang in zip(tokens, langs):
            res[lang].append(token)
        res = {lang: self._tokenizer.convert_tokens_to_string(lang_tokens) for lang, lang_tokens in res.items()}
        return res

    def _load_model(self) -> BertForTokenClassification:
        """
        Downloads a checkpoint from a specified W&B run.
        :return: fine-tuned `BertForTokenClassification` model
        """
        api = wandb.Api()
        run = api.run(f"falca/text-lang-predictor/{self._run_id}")

        run_dir = os.path.join(CHECKPOINTS_DIR, run.name)
        ckpt_path = os.path.join(run_dir, "best.ckpt")

        if not os.path.exists(ckpt_path):
            print(f"Downloading checkpoint to {run_dir}...")
            run.file("best.ckpt").download(run_dir)

        print(f"Loading checkpoint {ckpt_path}")

        model = BertLangNER.load_from_checkpoint(ckpt_path, map_location=self._device, log_val_metrics=False).model
        model.eval()

        return model
