"""Get metrics."""
import os
from typing import Dict, List
import click
import pytorch_lightning as pl
from src.neural_network import DeepSymNet
from src.CTDataModule import CTDataModule
from src.utils import seed_everything

seed_everything(42)


def test_get_merics(test_dir,
                    checkpoint_path,
                    batch_size: int = 32,
                    num_workers: int = 6,
                    gpu: bool = False) -> Dict[str, float]:
    model = DeepSymNet.load_from_checkpoint(checkpoint_path)
    dm = CTDataModule(data_dir=test_dir,
                      batch_size=batch_size, num_workers=num_workers)
    trainer = pl.Trainer(deterministic=True,
                         logger=False,
                         accelerator='gpu' if gpu == True else 'cpu',
                         devices=-1 if gpu == True else None)
    return trainer.test(model=model, datamodule=dm)


def get_all_checkpoints(checkpoints_path) -> List[str]:
    path_checkpoints = os.path.join(checkpoints_path, "checkpoints")
    ckpt_files = os.listdir(path_checkpoints)
    full_pathes = [os.path.join(path_checkpoints, ckpt_file)
                   for ckpt_file in ckpt_files]
    return full_pathes


@click.command()
@click.argument('dataset_path', type=click.Path(exists=True))
@click.argument('version_path', type=click.Path(exists=True))
@click.option('--num-workers', type=int, default=6,
              help="Num of workers in a datamodule.")
@click.option('--batch-size', type=int, default=32,
              help="Batch size in a datamodule.")
@click.option('--gpu', type=bool, default=False,
              help="GPU is gpu when it is used.")
def main(**params):
    """
    Test all saved model in study version folder.
    dataset_path path to dataset with images for testing.
    version_path is path to version study folder,
    for example './lightning_logs/version_5' .
    """
    dataset_path = params["dataset_path"]
    version_path = params["version_path"]
    batch_size = params["batch_size"]
    num_workers = params["batch_size"]
    gpu = params["gpu"]

    checkpoints = get_all_checkpoints(version_path)
    for checkpoint in checkpoints:
        print()
        print(f"Checkpoint {os.path.split(checkpoint)[1]}")
        metric_dict = test_get_merics(dataset_path,
                                      checkpoint,
                                      num_workers=num_workers,
                                      batch_size=batch_size,
                                      gpu=gpu)
        print(f"Metrics are:\n {metric_dict}")


if __name__ == "__main__":
    # checkpoint_path = "/home/high_fly_bird/workspace/Ischemic_Stroke_Prediction/lightning_logs/demo"
    # test_dir = "/home/martinumer/HemorrData/hemorr_test"
    main()  # pylint: disable=no-value-for-parameter
