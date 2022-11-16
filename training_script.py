import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from CTDataModule import CTDataModule
from neural_network import DeepSymNet

if __name__ == "__main__":

    data_dir_new = './circles_dataset'
    dm = CTDataModule(data_dir=data_dir_new, batch_size=64)
    dm.setup()
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        save_on_train_epoch_end=True
    )

    model = DeepSymNet()
    trainer = pl.Trainer(max_epochs=30, callbacks=[checkpoint_callback])  # log_every_n_steps=2
    trainer.fit(model, dm)
