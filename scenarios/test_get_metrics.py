"""Get metrics."""
from src.neural_network import DeepSymNet
from src.CTDataModule import CTDataModule

import pytorch_lightning as pl

if __name__ == "__main__":

    test_dir = "/home/martinumer/BigData/test_big"
    checkpoint_path = "/home/martinumer/Ischemic_Stroke_Prediction/lightning_logs/lightning_logs/version_8/checkpoints/epoch=7-step=224.ckpt"

    model = DeepSymNet.load_from_checkpoint(checkpoint_path)
    dm = CTDataModule(data_dir=test_dir, batch_size=32, num_workers=6)
    trainer = pl.Trainer()
    print(trainer.test(model=model, datamodule=dm))
