import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import trainer
from model import T5DialoguePlModel
import hydra
from mlflow_utils import MlflowWriter
import os

EXPERIMENT_NAME = "t5_dialog_test"


@hydra.main(config_path='config.yaml')
def test(cfg):
    writer = MlflowWriter(EXPERIMENT_NAME)
    t5_dialogue_model = T5DialoguePlModel.load_from_checkpoint(
        '../../../outputs/2021-05-07/21-24-28/lightning_logs/version_0/checkpoints/epoch=2-step=127676.ckpt', strict=False, cfg=cfg, writer=writer)
    trainer = pl.Trainer(gpus=1, accumulate_grad_batches=8)
    t5_dialogue_model.test("コンパイラって難しいよね")


test()
