"""Training runner from terminal."""
import logging

import click
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src.CTDataModule import CTDataModule
from src.neural_network import DeepSymNet

logger = logging.getLogger(__name__)


@click.command()
@click.argument('dataset_path', type=click.Path(exists=True))
@click.argument('checkpoint', type=click.Path(exists=True))
@click.option('--checkpoints_path', type=click.Path(), default="./lightning_logs/",
              help="Path where trained model will be saved.")  # default_root_dir
@click.option('--batch_size', type=int, default=32,
              help="Batch size in a datamodule.")
@click.option('--num_workers', type=int, default=6,
              help="Num of workers in a datamodule.")
@click.option('--callback_patience', type=int, default=30,
              help=("Number of epochs with no improvement"
                    "after which training will be stopped"))
@click.option('--save_top_k', type=int, default=2,
              help="How many best models will be saved.")
@click.option('--max_epochs', type=int, default=100,
              help="Max training epochs to run.")
@click.option('--logging_level', type=int, default=logging.WARNING,
              help="Logging level, 30 for WARNING , 20 for INFO, 10 for DEBUG")
@click.option('--gpu', type=bool, default=False,
              help="GPU is gpu when it is used.")
@click.option('--auto_tune_learning_rate', type=bool, default=True,
              help="Use True if you want your learning_rate to be auto tuned ")

def main(**params):
    """Take already trained model and
    continue its training.
    dataset_path is path to dataset with images for training.
    checkpoint is path to model checkpoint to train it futher.
    """
    logging.basicConfig(level=params["logging_level"])
    logger.info("Script params are:")
    for param_name, value in params.items():
        logger.info("%s=%s", param_name, str(value))

    dataset_path = params["dataset_path"]
    batch_size = params["batch_size"]
    num_workers = params["batch_size"]
    save_top_k = params["save_top_k"]
    patience = params["callback_patience"]
    max_epochs = params["max_epochs"]
    checkpoints_path = params["checkpoints_path"]
    checkpoint = params["checkpoint"]
    gpu = params["gpu"]
    learning_rate = params["auto_tune_learning_rate"]


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

    model = DeepSymNet.load_from_checkpoint(checkpoint)
    trainer = pl.Trainer(default_root_dir=checkpoints_path,
                         max_epochs=max_epochs,
                         callbacks=[early_stop_callback, checkpoint_callback],
                         log_every_n_steps=20,
                         accelerator='gpu' if gpu == True else 'cpu',
                         devices=-1 if gpu == True else None,
                         auto_lr_find=learning_rate) 

    lr_finder = trainer.tuner.lr_find(model)
    
    #trainer.tune(model)
    model.hparams.learning_rate = lr_finder.suggestion()
    model.learning_rate

    trainer.fit(model, dm)

    logger.info("End futher training.")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
