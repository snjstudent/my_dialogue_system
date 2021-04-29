import torch.nn as nn
from torch import optim
import torch
from torch.nn.modules.module import Module
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch.nn.functional as F
import pytorch_lightning as pl
import dataloader


class T5DialogModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            "sonoisa/t5-base-japanese")

    def forward(self, x):
        return self.t5_model(x)

    def cal_loss(self, pred, gt):
        loss = nn.CrossEntropyLoss(pred, gt)
        return loss


class T5DialoguePlModel(pl.LightningModule):
    def __init__(self, cfg, writer):
        super().__init__()
        self.cfg = cfg
        self.writer = writer
        self.model = T5DialogModel()
        self.tokenizer = T5Tokenizer.from_pretrained(
            "sonoisa/t5-base-japanese")
        self.writer.log_params_from_omegaconf_dict(cfg)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        target, responce = batch
        outputs = self.model(batch)
        loss_output = self.model.cal_loss(outputs[0, :, :], responce[0, :])
        self.log("training_loss", loss_output)
        return {"loss": loss_output}

    def validation_step(self, batch, batch_idx):
        target, responce = batch
        outputs = self.model(batch)
        loss_output = self.model.cal_loss(outputs[0, :, :], responce[0, :])
        self.log("validation_loss", loss_output)
        return {"loss": loss_output}

    def configure_optimizers(self):
        return optim.__dict__[self.cfg.optimizer.algorizum](self.model.parameters(), lr=self.cfg.lr)

    def train_dataloader(self):
        return dataloader.get_dataloader(loader_category="tweet_reply", model="train", batch_size=self.cfg.batch_size)

    def val_dataloader(self):
        return dataloader.get_dataloader(loader_category="tweet_reply", model="val", batch_size=self.cfg.batch_size)
