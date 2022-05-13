import numpy as np
import torch
import pytorch_lightning as pl

from torch import nn
from torchmetrics import F1Score, Accuracy, Recall, Precision
from transformers import get_cosine_schedule_with_warmup, AutoModelForTokenClassification

from src.configs.config_classes import TrainConfig


class BertLangNER(pl.LightningModule):
    def __init__(
        self,
        cfg: TrainConfig,
        n_classes: int,
    ):
        super().__init__()

        self.model = AutoModelForTokenClassification.from_pretrained(
            "bert-base-multilingual-cased", num_labels=n_classes
        )

        self.metrics = nn.ModuleDict(
            {
                "acc": Accuracy(num_classes=n_classes),
                "precision": Precision(num_classes=n_classes, average="macro", mdmc_average="global"),
                "recall": Recall(num_classes=n_classes, average="macro", mdmc_average="global"),
                "f1": F1Score(num_classes=n_classes, average="macro", mdmc_average="global"),
            }
        )

        self.lr = cfg.training.lr
        self.lr_decay = cfg.training.lr_decay
        self.w_decay = cfg.training.w_decay
        self.ratio = cfg.training.ratio
        self.scheduler = cfg.training.scheduler
        self.warmup_steps = cfg.training.n_warmup_steps
        self.num_training_steps = cfg.training.n_steps

        self.save_hyperparameters()

    def forward(self, batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        model_out = self(batch)

        # logging
        self.log("train_loss", model_out.loss.item(), on_step=True, prog_bar=True, logger=True)

        return model_out.loss

    def validation_step(self, batch, batch_idx):
        model_out = self(batch)

        # calculating metrics
        mask = batch["labels"] != -100
        preds = model_out.logits.argmax(-1)[mask]
        target = batch["labels"][mask]

        for metric in self.metrics.values():
            metric(preds, target)

        output = {"loss": model_out.loss.item()}

        return output

    def validation_epoch_end(self, validation_step_outputs):
        loss = np.mean([l["loss"] for l in validation_step_outputs])
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        for metric_name, metric in self.metrics.items():
            self.log(f"val_{metric_name}", metric, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = self._create_optimizer()
        if isinstance(self.warmup_steps, float):
            warmup_steps = self.num_training_steps * self.warmup_steps
        else:
            warmup_steps = self.warmup_steps

        if self.scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=self.num_training_steps,
            )
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

            return [optimizer], [scheduler]
        elif self.scheduler == "slanted":
            scheduler = self.get_slanted_triangular_scheduler(
                optimizer,
                ratio=self.ratio,
                num_warmup_steps=warmup_steps,
                num_training_steps=self.num_training_steps,
            )
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

            return [optimizer], [scheduler]
        else:
            return optimizer

    def _create_optimizer(self):
        lr = self.lr
        parameters = [{"params": self.model.classifier.parameters(), "lr": lr}]
        for l in self.model.bert.encoder.layer:
            parameters += [{"params": l.parameters(), "lr": lr}]
            lr *= self.lr_decay
        parameters += [{"params": self.model.bert.embeddings.parameters(), "lr": lr}]
        optimizer = torch.optim.AdamW(parameters, lr=self.lr, weight_decay=self.w_decay)

        return optimizer

    @classmethod
    def get_slanted_triangular_scheduler(cls, optimizer, num_warmup_steps, num_training_steps, ratio=32, last_epoch=-1):
        cut = num_warmup_steps
        cut_frac = num_warmup_steps / num_training_steps

        def lr_lambda(current_step: int):
            if current_step < cut:
                p = current_step / cut
            else:
                p = 1 - (current_step - cut) / (cut * (1 / cut_frac - 1))

            return (1 + p * (ratio - 1)) / ratio

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
