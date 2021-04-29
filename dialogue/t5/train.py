import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import trainer
from model import T5DialoguePlModel
import hydra
from mlflow_util import MlflowWriter
import os

EXPERIMENT_NAME = "t5_dialog"


@hydra.main(config_path='config.yaml')
def train(cfg):
    writer = MlflowWriter(EXPERIMENT_NAME)
    t5_dialogue_model = T5DialoguePlModel(cfg)
    trainer = pl.Trainer()
    trainer.fit(t5_dialogue_model)
    writer.log_artifact(os.path.join(os.getcwd(), '.hydra/config.yaml'))
    writer.log_artifact(os.path.join(os.getcwd(), '.hydra/hydra.yaml'))
    writer.log_artifact(os.path.join(os.getcwd(), '.hydra/overrides.yaml'))
    writer.log_artifact(os.path.join(os.getcwd(), 'main.log'))
    writer.set_terminated()
