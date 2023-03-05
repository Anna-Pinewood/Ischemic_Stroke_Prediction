"""Get metrics."""
import os
from typing import Dict, List
import click
import pytorch_lightning as pl
from src.neural_network import DeepSymNet
from src.CTDataModule import CTDataModule
from src.utils import seed_everything

seed_everything(42)


# @click.option('--num-workers', type=int, default=6,
#               help="Num of workers in a datamodule.")

# def main(**params):

def test_get_merics(test_dir,
                    checkpoint_path) -> Dict[str, float]:
    model = DeepSymNet.load_from_checkpoint(checkpoint_path)
    dm = CTDataModule(data_dir=test_dir, batch_size=32, num_workers=6)
    trainer = pl.Trainer(deterministic=True, logger=False)
    return trainer.test(model=model, datamodule=dm)


def get_all_checkpoints(checkpoints_path) -> List[str]:
    path_checkpoints = os.path.join(checkpoints_path, "checkpoints")
    ckpt_files = os.listdir(path_checkpoints)
    full_pathes = [os.path.join(path_checkpoints, ckpt_file)
                   for ckpt_file in ckpt_files]
    return full_pathes


if __name__ == "__main__":
    checkpoint_path = "/home/martinumer/Ischemic_Stroke_Prediction/lightning_logs/version_64"
    test_dir = "/home/martinumer/IschemicData/test"

    checkpoints = get_all_checkpoints(checkpoint_path)
    for checkpoint in checkpoints:
        print()
        print(os.path.split(checkpoint)[1])
        metric_dict = test_get_merics(test_dir, checkpoint)
        print(metric_dict)
