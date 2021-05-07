import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import trainer
from model import T5DialoguePlModel
import hydra
from mlflow_utils import MlflowWriter
import os

EXPERIMENT_NAME = "t5_dialog"


@hydra.main(config_path='config.yaml')
def train(cfg):
    writer = MlflowWriter(EXPERIMENT_NAME)
    t5_dialogue_model = T5DialoguePlModel.load_from_checkpoint(
        '../../../outputs/2021-05-06/14-14-24/lightning_logs/version_0/checkpoints/epoch=0-step=42558.ckpt', strict=False, cfg=cfg, writer=writer)
    trainer = pl.Trainer(gpus=1, accumulate_grad_batches=8)
    trainer.fit(t5_dialogue_model)
    writer.log_artifact(os.path.join(os.getcwd(), '.hydra/config.yaml'))
    writer.log_artifact(os.path.join(os.getcwd(), '.hydra/hydra.yaml'))
    writer.log_artifact(os.path.join(os.getcwd(), '.hydra/overrides.yaml'))
    writer.log_artifact(os.path.join(os.getcwd(), 'main.log'))
    writer.set_terminated()


train()
