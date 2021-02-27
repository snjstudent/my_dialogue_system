import torch
import torch.nn as nn
import torch.optim
import deepspeed
import argparse
import dataloader
import pdb


class Trainer:
    def __init__(self, model: object, dataloader: torch.utils.data.DataLoader, loss: object, optimizer_name: str, lr: float) -> None:
        self.dataloader = dataloader
        self.loss = loss
        self.lr = lr
        self.device = "cuda:0"
        model = model.to(self.device)
        parser = argparse.ArgumentParser(description='My training script.')
        parser.add_argument('--local_rank', type=int, default=-1)
        parser = deepspeed.add_config_arguments(parser)
        cmd_args = parser.parse_args()
        cmd_args.deepspeed_config = '../../../deepspeed_config.json'
        self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
            args=cmd_args, model=model, model_parameters=model.parameters())
        deepspeed.init_distributed()

    def train(self, epoch: int) -> None:
        for epoch_num in range(epoch):
            for speak, responce in self.dataloader:
                speak, responce = speak.to(
                    self.device), responce.to(self.device)
                # self.optimizer.zero_grad()
                output = self.model_engine(speak)
                loss_output = 0
                for i in range(responce.shape[1]):
                    loss_output += self.loss(output[:, i, :], responce[:, i])
                loss_output /= responce.shape[1]
                # pdb.set_trace()
                self.model_engine.backward(loss_output)
                self.model_engine.step()

        print("epoch{}：終了\n".format(epoch_num+1))
