import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from src.CTDataModule import CTDataModule
from neural_network import DeepSymNet

if __name__ == "__main__":

    data_dir_new = '/home/high_fly_bird/Data/brain_dataset'
    dm = CTDataModule(data_dir=data_dir_new, batch_size=64, num_workers=16)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss', #
        save_on_train_epoch_end=True,
        save_last=True,
        save_top_k=2
    )

    model = DeepSymNet()
    trainer = pl.Trainer(max_epochs=30, callbacks=[checkpoint_callback])  # log_every_n_steps=2
    trainer.fit(model, dm)
