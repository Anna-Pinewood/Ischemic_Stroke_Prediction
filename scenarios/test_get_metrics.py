"""Get metrics."""
import pytorch_lightning as pl
from src.neural_network import DeepSymNet
from src.CTDataModule import CTDataModule
from src.utils import seed_everything

seed_everything(42)


if __name__ == "__main__":

    test_dir = "/home/martinumer/HemorrData/hemorr_test"
    checkpoint_path = "/home/martinumer/Ischemic_Stroke_Prediction/lightning_logs/version_16/checkpoints/epoch=54-step=1155.ckpt"

    model = DeepSymNet.load_from_checkpoint(checkpoint_path)

    dm = CTDataModule(data_dir=test_dir, batch_size=32, num_workers=6)
    trainer = pl.Trainer(deterministic=True)
    print(trainer.test(model=model, datamodule=dm))
