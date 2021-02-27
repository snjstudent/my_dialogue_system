import torch.nn as nn
from torch import optim
import torch
from torch.nn.modules.module import Module
from transformers import BartModel, BartConfig  # ,PreTrainedTokenizer
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super(Model, self).__init__()
        configuration = BartConfig(
            max_position_embeddings=1920, encoder_layers=2, decoder_layers=24)
        self.encoder_decoder = BartModel(configuration)
        self.linear = nn.Linear(
            1024, vocab_size, bias=True)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.encoder_decoder(x)
        """
        x_max, x_indexes = torch.max(
            F.softmax(self.linear(x.last_hidden_state)), 2)
        """
        return F.softmax(self.linear(x.last_hidden_state))
