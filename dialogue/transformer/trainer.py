import torch
import torch.nn as nn
import torch.optim
import deepspeed
import argparse

class Trainer:
    def __init__(self, model: object, dataloader: torch.utils.data.DataLoader, loss: object, optimizer_name: str,lr:float) -> None:
        self.dataloader = dataloader
        self.loss = loss
        self.lr=lr
        #self.optimizer = self._select_optimizer(optimizer_name,lr,model)
        import pdb
        parser = argparse.ArgumentParser(description='My training script.')
        parser.add_argument('--local_rank', type=int, default=-1)
        parser = deepspeed.add_config_arguments(parser)
        cmd_args = parser.parse_args()
        cmd_args.deepspeed_config='../../../deepspeed_config.json'
        self.model,self.optimizer,_ = deepspeed.initialize(args=cmd_args, model=model, model_parameters=model.parameters())

    @classmethod
    def _select_optimizer(self, optimizer_name: str,lr:float,model:object) -> None:
        if optimizer_name == "Adam":
            return torch.optim.Adam(model.parameters(),lr=lr)
        else:
            return

    def train(self, epoch: int) -> None:
        assert self.optimizer, print("You have to set Optimizer")
        # 訓練ループ
        self.model.train()
        for epoch_num in range(epoch):
            for speak, responce in self.dataloader:
                data, target = Variable(speak), Variable(responce)  # 微分可能に変換
                self.optimizer.zero_grad()  # 一度計算された勾配結果を0にリセット

                output = self.model(speak)  # 入力dataをinputし、出力を求める
                loss = loss(output, responce)  # 出力と訓練データの正解との誤差を求める
                model.backward(loss)  # 誤差のバックプロパゲーションを求める
                model.step()  # バックプロパゲーションの値で重みを更新する

                print("epoch{}：終了\n".format(epoch+1))
