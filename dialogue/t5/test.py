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
        '../../../outputs/2021-05-13/18-00-32/lightning_logs/version_0/checkpoints/epoch=41-step=24191.ckpt', strict=False, cfg=cfg, writer=writer)
    trainer = pl.Trainer(gpus=1, accumulate_grad_batches=8)
    t5_dialogue_model.test("休みの日にどこか行きたいんだけど、いいところある？")


test()
