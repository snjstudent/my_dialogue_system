import trainer
import dataloader
import model
import hydra
import torch.nn as nn


@hydra.main(config_path='config.yaml')
def main(config) -> None:
    talk_model = model.Model(config.model.vocab_size)
    talk_dataloader = dataloader.get_dataloader(
        config.data.data_category, config.train_conf.batch_size)
    loss = nn.CrossEntropyLoss()
    talk_trainer = trainer.Trainer(talk_model, talk_dataloader, loss,
                              config.train_conf.optimizer,config.train_conf.optimizer_lr)
    talk_trainer.train(epoch=config.train_conf.epoch)


if __name__ == "__main__":
    main()
