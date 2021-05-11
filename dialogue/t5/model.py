import torch.nn as nn
from torch import optim
import torch
from torch.nn.modules.module import Module
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch.nn.functional as F
import pytorch_lightning as pl
import dataloader


class T5DialogModel(nn.Module):
    def __init__(self, pretrained_path) -> None:
        super().__init__()
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            "sonoisa/t5-base-japanese")

    def forward(self, speak, responce):
        labels = responce["input_ids"]
        labels[labels[:, :] == 0] = -100
        return self.t5_model(input_ids=speak["input_ids"],
                             decoder_input_ids=None,
                             attention_mask=speak["attention_mask"],
                             decoder_attention_mask=responce['attention_mask'],
                             labels=labels)

    def cal_loss(self, pred, gt):
        loss = nn.CrossEntropyLoss(pred, gt)
        return loss

    def generate(self, question):
        return self.t5_model.generate(input_ids=question["input_ids"],
                                      attention_mask=question["attention_mask"])


class T5DialoguePlModel(pl.LightningModule):
    def __init__(self, cfg, writer, pretrained_path=None):
        super().__init__()
        self.cfg = cfg
        self.writer = writer
        self.model = T5DialogModel(pretrained_path)

        self.tokenizer = T5Tokenizer.from_pretrained(
            "sonoisa/t5-base-japanese")
        self.writer.log_params_from_omegaconf_dict(cfg)

    def forward(self, target, responce):
        return self.model(target, responce)

    def training_step(self, batch, batch_idx):
        target, responce = batch
        outputs = self.model(target, responce)
        loss_output = outputs[0]
        self.log("training_loss", loss_output)
        return {"loss": loss_output}

    def validation_step(self, batch, batch_idx):
        target, responce = batch
        outputs = self.model(target, responce)
        loss_output = outputs[0]
        self.log("validation_loss", loss_output)
        return {"loss": loss_output}

    def test(self, question):
        self.model.eval()
        print("Q : " + question)
        question = self.tokenizer(question, max_length=200, truncation=True,
                                  padding="max_length", return_tensors="pt")
        responce = self.model.generate(question)
        print(self.tokenizer.decode(responce[0]))

    def configure_optimizers(self):
        return optim.__dict__[self.cfg.train.optimizer.algorithm](self.model.parameters(), lr=self.cfg.train.optimizer.lr)

    def train_dataloader(self):
        return dataloader.get_dataloader(loader_category=self.cfg.train.data_category, mode="train", batch_size=self.cfg.train.batch_size)

    def val_dataloader(self):
        return dataloader.get_dataloader(loader_category=self.cfg.train.data_category, mode="val", batch_size=self.cfg.train.batch_size)
