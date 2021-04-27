import torch.nn as nn
from torch import optim
import torch
from torch.nn.modules.module import Module
from transformers import BlenderbotModel, T5ForConditionalGeneration, T5Tokenizer
import torch.nn.functional as F
import pytorch_lightning as pl


class T5DialogModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            "sonoisa/t5-base-japanese")

    def forward(self, x):
        return self.t5_model(x)


class T5DialoguePlModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = T5DialogModel()
        self.tokenizer = T5Tokenizer.from_pretrained(
            "sonoisa/t5-base-japanese")
        self.loss = NotImplemented

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        target, responce = batch
        outputs = self.model(batch)
        loss_output = self.loss(outputs[0, :, :], responce[0::])
        self.log("training_loss", loss_output)
        return {"loss": loss_output}

    def validation_step(self, batch, batch_idx):
        target, responce = batch
        outputs = self.model(batch)
        loss_output = self.loss(outputs[0, :, :], responce[0::])
        self.log("validation_loss", loss_output)
        return {"loss": loss_output}

    def configure_optimizers(self):
        return super().configure_optimizers()

    def train_dataloader(self):
        return super().train_dataloader()

    def val_dataloader(self):
        return super().val_dataloader()
