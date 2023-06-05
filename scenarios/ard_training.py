from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl

from src.CTDataModule import CTDataModule
from src.neural_network_ard import DeepSymNetArd

if __name__ == "__main__":
    dataset_path = "/home/martinumer/NoBoneData/train"
    batch_size = 10
    num_workers = 16
    throw_out_random = 0
    patience = 50
    min_epochs = 10
    max_epochs = 100
    gpu = True
    version_name = "june_fourth"

    dm = CTDataModule(data_dir=dataset_path,
                      batch_size=batch_size,
                      num_workers=num_workers,
                      throw_out_random=throw_out_random)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        save_on_train_epoch_end=True,
        save_last=True,
        save_top_k=2)
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.03, patience=patience, mode="min")
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir="/home/high_fly_bird/workspace/Ischemic_Stroke_Prediction/lightning_logs_ard",
        version=version_name)
    model = DeepSymNetArd()
    trainer = pl.Trainer(logger=tb_logger,
                         min_epochs=min_epochs,
                         max_epochs=max_epochs,
                         callbacks=[early_stop_callback, checkpoint_callback],
                         log_every_n_steps=20,
                         accelerator='gpu' if gpu == True else 'cpu',
                         devices=-1 if gpu == True else None)
    trainer.fit(model, dm)
