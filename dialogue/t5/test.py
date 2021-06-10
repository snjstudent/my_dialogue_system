import argparse
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
    writer = None
    #writer = MlflowWriter(EXPERIMENT_NAME)
    t5_dialogue_model = T5DialoguePlModel.load_from_checkpoint(
        '../../../outputs/2021-05-13/18-00-32/lightning_logs/version_0/checkpoints/epoch=41-step=24191.ckpt', strict=False, cfg=cfg, writer=writer, do_train=False)
    trainer = pl.Trainer(gpus=1, accumulate_grad_batches=8)
    with open('../../../question/question.txt') as f:
        input_txt = f.readlines()[0]
    t5_dialogue_model.test(input_txt)


test()
# if __name__ == "__main__":
#     # refered from https://qiita.com/kenichi-hamaguchi/items/dda5532f3b218142e7c9
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-i', '--func_args',
#                         nargs='*',
#                         help='args in function',
#                         default=[])
#     args = parser.parse_args()
#     input_txt = str(args.func_args[0])
#     # 関数実行
#     test()
#     # ret = func_dict[args.function_name](*func_args)
