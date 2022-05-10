import itertools
import os.path
from collections import defaultdict
from typing import List, Dict

import numpy as np
import torch
import wandb
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from definitions import ROOT_DIR, CHECKPOINTS_DIR
from src.training.dataset import MultiLanguageDataset
from src.training.model import BertLangNER


class TextLangPredictor:
    def __init__(self):
        self.max_seq_length = 512
        self.batch_size = 2
        self.device = "cpu"
        self.run_id = "31wx3mxu"
        self.model = self._load_model()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    def parse_text(self, text: str):
        text = MultiLanguageDataset.preprocess_text(text)
        chunks = self._chunk_text(text)
        # TODO: move it into chunking function
        chunks = [self.tokenizer.convert_tokens_to_string(tok) for tok in chunks]

        chunks_encoded = self.tokenizer.batch_encode_plus(
            chunks, add_special_tokens=False, is_split_into_words=False, return_tensors="pt", padding=True
        )
        langs_ids = self._get_token_predictions(chunks_encoded)
        langs = self._lang_ids_to_names(langs_ids)

        return self._get_spans(text, langs)

    def _get_spans(self, text: str, langs: List[str]) -> Dict[str, str]:
        tokens = self.tokenizer.tokenize(text, add_special_tokens=False)
        res = defaultdict(list)
        for token, lang in zip(tokens, langs):
            res[lang].append(token)
        res = {lang: self.tokenizer.convert_tokens_to_string(lang_tokens) for lang, lang_tokens in res.items()}
        return res

    def _lang_ids_to_names(self, langs_ids: List[int]) -> List[str]:
        return [MultiLanguageDataset._ids_to_langs[idx] for idx in langs_ids]

    @torch.inference_mode()
    def _get_token_predictions(self, chunks: Dict[str, torch.Tensor]):
        preds = []
        dl = self._create_dataloader(chunks)
        for batch in tqdm(dl, desc="Making predictions..."):
            input_ids, token_type_ids, attention_mask = batch
            logits = self.model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).logits

            attention_mask[
                (input_ids == self.tokenizer.cls_token_id).logical_or(input_ids == self.tokenizer.sep_token_id)
            ] = 0

            token_preds = logits.argmax(-1)
            mask = attention_mask.to(device=self.device, dtype=torch.bool)

            lang_ids = torch.masked_select(token_preds, mask)

            preds.append(lang_ids.cpu().numpy().tolist())

        return list(itertools.chain.from_iterable(preds))

    def _create_dataloader(self, chunks: Dict[str, torch.Tensor]):
        ds = TensorDataset(*[t.to(self.device) for t in chunks.values()])
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False)

    def _chunk_text(self, text: str) -> List[List[int]]:
        sentences_tokenized = self.tokenizer.tokenize(text)
        n_chunks = int(np.ceil(len(sentences_tokenized) / self.max_seq_length))
        chunks = np.array_split(sentences_tokenized, n_chunks)
        chunks = [[self.tokenizer.cls_token] + chunk.tolist() + [self.tokenizer.sep_token] for chunk in chunks]
        return chunks

    def _load_model(self):
        api = wandb.Api()
        run = api.run(f"falca/text-lang-predictor/{self.run_id}")

        run_dir = os.path.join(CHECKPOINTS_DIR, run.name)
        ckpt_path = os.path.join(run_dir, "last.ckpt")

        if not os.path.exists(ckpt_path):
            print(f"Downloading checkpoint to {run_dir}...")
            run.file("last.ckpt").download(run_dir)

        print(f"Loading checkpoint {ckpt_path}")

        model = BertLangNER.load_from_checkpoint(ckpt_path, map_location=self.device, log_val_metrics=False).model
        model.eval()

        return model
