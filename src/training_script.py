import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


from CTDataModule import CTDataModule
from neural_network import DeepSymNet

if __name__ == "__main__":

    data_dir_new = '/home/martinumer/BigData/BigData/train_big'
    dm = CTDataModule(data_dir=data_dir_new, batch_size=32, num_workers=6)
    # checkpoint_callback = ModelCheckpoint(
    #     monitor='val_loss', #
    #     save_on_train_epoch_end=True,
    #     save_last=True,
    #     save_top_k=2
    # )
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss', #
        save_on_train_epoch_end=True,
        save_last=True,
        save_top_k=2
    )

    early_stop_callback = EarlyStopping(monitor="val_loss",  min_delta=0.03, patience=30, mode="min")

    # model = DeepSymNet()
    ckpt_path = '/home/high_fly_bird/Workspace/Ischemic_Stroke_Prediction/lightning_logs/version_9/checkpoints/last.ckpt'
    model = DeepSymNet.load_from_checkpoint(ckpt_path)

    trainer = pl.Trainer(max_epochs=35, callbacks=[early_stop_callback, checkpoint_callback], log_every_n_steps=20)
                            # accelerator='gpu', devices=[0])  # log_every_n_steps=2, track_grad_norm=2
    trainer.fit(model, dm)
