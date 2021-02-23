import torch.nn as nn
from torch import optim
import torch
from torch.nn.modules.module import Module
from transformers import BartModel, BartConfig
from transformers.tokenization_bert_japanese import BertJapaneseTokenizer
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super(Model, self).__init__()
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(
            'cl-tohoku/bert-base-japanese-whole-word-masking')
        configuration = BartConfig(
            max_position_embeddings=1920, encoder_layers=2, decoder_layers=24)
        self.encoder_decoder = BartModel(configuration)
        self.linear = nn.Linear(
            self.encoder_decoder.get_output_embeddings(), vocab_size)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.tokenizer(x)
        x = self.encoder_decoder(x)
        return F.log_softmax(self.linear(x))
