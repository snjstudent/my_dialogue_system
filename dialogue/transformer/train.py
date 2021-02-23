import trainer
import dataloader
import model


@hydra.main(config_path='config.yaml')
def main(cfg) -> None:
    model = model.Model(config.model.vocab_size)
    dataloader = dataloader.get_dataloader(
        config.data.data_category, config.train_conf.batch_size)
    loss = NotImplemented
    trainer = trainer.Trainer(model, dataloader, loss,
                              config.train_conf.optimizer)
    trainer.train(epoch=config.train_conf.epoch)


if __name__ == "__main__":
    main()
