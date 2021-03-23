import torch.nn as nn
from torch import optim
import torch
from torch.nn.modules.module import Module
from transformers import BlenderbotModel, BlenderbotConfig  # ,PreTrainedTokenizer
import torch.nn.functional as F

import pdb


class Model(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super(Model, self).__init__()
        configuration = BlenderbotConfig(
            encoder_layers=2, decoder_layers=8, vocab_size=vocab_size, max_position_embeddings=1920)
        self.encoder_decoder = BlenderbotModel(configuration)
        self.linear = nn.Linear(
            2560, vocab_size, bias=True)

    def forward(self, x: torch.FloatTensor, y: torch.FloatTensor) -> torch.FloatTensor:
        x = self.encoder_decoder(input_ids=x, decoder_input_ids=y)
        """
        x_max, x_indexes = torch.max(
            F.softmax(self.linear(x.last_hidden_state)), 2)
        """
        return F.softmax(self.linear(x.last_hidden_state))
