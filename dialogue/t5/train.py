import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from model import T5DialoguePlModel
import hydra
from mlflow_utils import MlflowWriter
import os

EXPERIMENT_NAME = "t5_dialog"

# tweet '../../../outputs/2021-05-07/21-24-28/lightning_logs/version_0/checkpoints/epoch=2-step=127676.ckpt'
# meidai '../../../outputs/2021-05-13/09-14-39/lightning_logs/version_0/checkpoints/epoch=4-step=1189.ckpt'
# j-tocc '../../../outputs/2021-05-13/10-37-44/lightning_logs/version_0/checkpoints/epoch=13-step=8063.ckpt'
# meidai -> j-tocc '../../../outputs/2021-05-13/15-52-33/lightning_logs/version_0/checkpoints/epoch=4-step=2879.ckpt'
# meidai(105) outputs/2021-05-14/15-47-46/lightning_logs/version_0/checkpoints/epoch=105-step=25227.ckpt
# meidai->j-tocc(41) '../../../outputs/2021-05-13/18-00-32/lightning_logs/version_0/checkpoints/epoch=41-step=24191.ckpt'


@hydra.main(config_path='config.yaml')
def train(cfg):
    early_stop_callback = EarlyStopping(
        monitor=cfg.train.early_stop.loss, mode=cfg.train.early_stop.mode, patience=cfg.train.early_stop.patience)
    writer = MlflowWriter(EXPERIMENT_NAME)
    t5_dialogue_model = T5DialoguePlModel.load_from_checkpoint(
        '../../../outputs/2021-05-14/15-47-46/lightning_logs/version_0/checkpoints/epoch=105-step=25227.ckpt', max_epochs=1000, strict=False, cfg=cfg, writer=writer)
    trainer = pl.Trainer(gpus=1, accumulate_grad_batches=8,
                         callbacks=[early_stop_callback])
    trainer.fit(t5_dialogue_model)
    writer.log_artifact(os.path.join(os.getcwd(), '.hydra/config.yaml'))
    writer.log_artifact(os.path.join(os.getcwd(), '.hydra/hydra.yaml'))
    writer.log_artifact(os.path.join(os.getcwd(), '.hydra/overrides.yaml'))
    writer.log_artifact(os.path.join(os.getcwd(), 'main.log'))
    writer.set_terminated()


train()
