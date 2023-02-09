"""Get predictions, compute threshold, build ROC-curve and confusion matrix. """
from src.neural_network import DeepSymNet
from src.CTDataModule import CTDataModule

from src.predictions import get_test_predictions, plot_conf_matrix, plot_roc_curve


if __name__ == "__main__":

    test_dir = "/home/martinumer/BigData/test_big"
    train_dir = "/home/martinumer/BigData/train_big"
    checkpoint_path = "/home/high_fly_bird/workspace/Ischemic_Stroke_Prediction/models/version_0/checkpoints/epoch=74-step=1500.ckpt"

    model = DeepSymNet.load_from_checkpoint(checkpoint_path)
    dm_train = CTDataModule(data_dir=test_dir, batch_size=32, num_workers=6)

    labels_preds = get_test_predictions(dm_predict=dm_train,
                                        model=model)
    threshold_best = model.find_threshold(labels_preds.y_pred,
                                          labels_preds.y_true)
    labels_preds['pred_bin'] = (
        labels_preds.y_pred > threshold_best).astype(int)

    plot_conf_matrix(labels_preds['y_true'], labels_preds['pred_bin'])

    plot_roc_curve(labels_preds['y_pred'], labels_preds['y_true'])

