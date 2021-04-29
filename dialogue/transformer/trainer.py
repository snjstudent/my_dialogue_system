import torch
import torch.nn as nn
import torch.optim
import argparse
import dataloader as dl
import pdb
import transformers


class Trainer:
    def __init__(self, model: object, dataloader: torch.utils.data.DataLoader, loss: object, optimizer_name: str, lr: float) -> None:
        self.dataloader = dataloader
        self.loss = loss
        self.lr = lr
        self.device = "cuda:0"
        self.tokenizer = dl.load_custom_tokenizer(
            "tokenizer.json")
        self.model = model.cuda()
        # self.optimizer = transformers.Adafactor(
        #     self.model.parameters(), relative_step=True, warmup_init=True)
        self.optimizer = torch.optim.AdamW(self.model.parameters())
        # parser = argparse.ArgumentParser(description='My training script.')
        # parser.add_argument('--local_rank', type=int, default=-1)
        # parser = deepspeed.add_config_arguments(parser)
        # cmd_args = parser.parse_args()
        # cmd_args.deepspeed_config = '../../../deepspeed_config.json'
        # self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
        #     args=cmd_args, model=self.model, model_parameters=self.model.parameters())
        # deepspeed.init_distributed()

    def train(self, epoch: int) -> None:
        self.model.train()
        self.model.zero_grad()
        for epoch_num in range(epoch):
            # self.model_engine.save_checkpoint(save_dir="./checkpoint/")
            torch.save(self.model.state_dict(), f"model_{epoch_num}.pth")
            gradient_auumuration_count = 128
            for speak, responce in self.dataloader:

                speak, responce = speak.cuda(), responce.cuda()
                # self.optimizer.zero_grad()

                output = self.model(speak, responce)
                speak_decode = self.tokenizer.decode(
                    speak[0, :].detach().cpu().numpy())
                responce_decode_model = self.tokenizer.decode(
                    torch.argmax(output[0, :, :], dim=1).detach().cpu().numpy())
                responce_decode = self.tokenizer.decode(
                    responce[0, :].detach().cpu().numpy())
                print("Question : ", speak_decode)
                print("Correct Responce : ", responce_decode)
                print("Model Responce : ", responce_decode_model)
                loss_output = 0
                # for i in range(responce.shape[0]):
                #     loss_output += self.loss(output[i, :, :], responce[i, :])
                loss_output = self.loss(output[0, :, :], responce[0, :])/128
                #loss_output /= responce.shape[0]
                print(f"batch loss : {loss_output}")
                loss_output.backward()
                del loss_output
                del output
                del speak
                del responce

                gradient_auumuration_count -= 1
                if gradient_auumuration_count == 0:
                    gradient_auumuration_count = 128
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                # pdb.set_trace()
            #     self.model_engine.backward(loss_output)
            #     self.model_engine.step()
            # self.model_engine.save_checkpoint(save_dir="./checkpoint/")

        print("epoch{}：終了\n".format(epoch_num+1))
