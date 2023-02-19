"""Training runner from terminal."""
import logging
import os

import click
from matplotlib import pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src.CTDataModule import CTDataModule
from src.neural_network import DeepSymNet

logger = logging.getLogger(__name__)


@click.command()
@click.argument('dataset_path', type=click.Path(exists=True))
@click.option('--checkpoints-path', type=click.Path(), default=".",
              help=("Path where trained model will be saved."
                    "It selected path a folder /lightning_logs/"
                    "will be created and models will be inside."))
@click.option('--batch-size', type=int, default=32,
              help="Batch size in a datamodule.")
@click.option('--num-workers', type=int, default=6,
              help="Num of workers in a datamodule.")
@click.option('--callback-patience', type=int, default=50,
              help=("Number of epochs with no improvement"
                    "after which training will be stopped"))
@click.option('--save-top-k', type=int, default=2,
              help="How many best models will be saved.")
@click.option('--max-epochs', type=int, default=100,
              help="Max training epochs to run.")
@click.option('--min-epochs', type=int, default=None,
              help="Min training epochs to run.")
@click.option('--optimizer-name', type=str, default='adam',
              help="'adam' or 'adamax'")
@click.option('--auto-tune-learning-rate', type=bool, default=False,
              help="Use True if you want your learning_rate to be auto tuned ")
@click.option('--learning-rate', type=float, default=None,
              help=("If you do not use auto tune learning rate set your own lr in a model."
                    "Else it would be set to default model value."))
@click.option('--version-name', type=str, default=None,
              help=("Name of study folder in 'checkpoints_path/lightning_logs/'"
                    "By default it is version_{$num}."))
@click.option('--logging-level', type=int, default=logging.WARNING,
              help="Logging level, 30 for WARNING , 20 for INFO, 10 for DEBUG")
@click.option('--gpu', type=bool, default=False,
              help="GPU is gpu when it is used.")
def main(**params):
    """Run model training.
    dataset_path is path to dataset with images for training.
    """
    logging.basicConfig(level=params["logging_level"])
    script_params = {'dataset_path', 'checkpoints_path', 'logging_level'}
    trainig_params = sorted(list(set(params.keys()) - script_params))

    logger.info("Script params are:")
    for param_name in list(script_params):
        logger.info("|  %s=%s", param_name, str(params[param_name]))

    logger.info("Training params are:")
    for param_name in trainig_params:
        logger.info("|  %s=%s", param_name, str(params[param_name]))

    dataset_path = params["dataset_path"]
    batch_size = params["batch_size"]
    num_workers = params["batch_size"]
    save_top_k = params["save_top_k"]
    patience = params["callback_patience"]
    max_epochs = params["max_epochs"]
    min_epochs = params["min_epochs"]
    optimizer_name = params["optimizer_name"]
    checkpoints_path = params["checkpoints_path"]
    gpu = params["gpu"]
    learning_rate = params["learning_rate"]
    auto_lr_find = params["auto_tune_learning_rate"]
    version_name = params["version_name"]

    abs_path_checkpoints = os.path.abspath(checkpoints_path)
    path_model = os.path.join(abs_path_checkpoints, "lightning_logs")

    if version_name is None:
        version_dirs = [f for f in os.listdir(
            path_model) if f.startswith('version_')]
        last_version = max([int(file.split('_')[1]) for file in version_dirs])
        version_name = f"version_{last_version+1}"

    logger.info("Model will be saved in %s",
                os.path.join(path_model, version_name))

    dm = CTDataModule(data_dir=dataset_path,
                      batch_size=batch_size, num_workers=num_workers)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        save_on_train_epoch_end=True,
        save_last=True,
        save_top_k=save_top_k
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",  min_delta=0.03, patience=patience, mode="min")

    tb_logger = pl.loggers.TensorBoardLogger(save_dir=checkpoints_path,
                                             version=version_name)

    model = DeepSymNet(optimizer_name=optimizer_name)
    trainer = pl.Trainer(default_root_dir=checkpoints_path,
                         logger=tb_logger,
                         min_epochs=min_epochs,
                         max_epochs=max_epochs,
                         callbacks=[early_stop_callback, checkpoint_callback],
                         log_every_n_steps=20,
                         accelerator='gpu' if gpu == True else 'cpu',
                         devices=-1 if gpu == True else None,
                         auto_lr_find=auto_lr_find)

    if trainer.auto_lr_find:
        lr_finder = trainer.tuner.lr_find(model, dm, early_stop_threshold=None)

        fig = lr_finder.plot(suggest=True)
        plt.savefig('LR_FINDER.png')

        # trainer.tune(model, dm)
        model.learning_rate = lr_finder.suggestion()

    elif learning_rate is not None:
        model.learning_rate = learning_rate

    logger.info('Learning rate: %s', str(model.learning_rate))

    trainer.fit(model, dm)

    logger.info("End training.")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
