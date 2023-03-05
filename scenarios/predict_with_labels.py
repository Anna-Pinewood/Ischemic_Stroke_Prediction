# %%
"""Get predictions, compute threshold, build ROC-curve and confusion matrix. """
from sklearn.metrics import f1_score, balanced_accuracy_score

from src.neural_network import DeepSymNet
from src.CTDataModule import CTDataModule

from src.predictions import get_test_predictions, plot_conf_matrix, plot_roc_curve
from src.utils import seed_everything

seed_everything(42)

if __name__ == "__main__":
    # test_dir = "/home/martinumer/IschemicData/test/"
    test_dir = "/home/martinumer/HemorrData/test"
    checkpoint_path = "/home/high_fly_bird/workspace/Ischemic_Stroke_Prediction/lightning_logs/hem_lr_best/checkpoints/epoch=21-step=616.ckpt"

    model = DeepSymNet.load_from_checkpoint(checkpoint_path)
    model.eval()

    dm_predict = CTDataModule(data_dir=test_dir,
                              batch_size=32,
                              num_workers=6,
                              test_shuffle=False)

    labels_preds = get_test_predictions(dm_predict=dm_predict,
                                        model=model)
    threshold_best = model.find_threshold(labels_preds.y_pred_proba,
                                          labels_preds.y_true,
                                          metric=balanced_accuracy_score)
    # threshold_best = model.find_threshold(labels_preds.y_pred_proba,
    #                                       labels_preds.y_true,
    #                                       metric=f1_score,
    #                                       average='weighted')

    print(f'best threshold is {threshold_best}')
    labels_preds['pred_bin'] = (
        labels_preds.y_pred_proba > threshold_best).astype(int)

    plot_conf_matrix(labels_preds['y_true'], labels_preds['pred_bin'])
    plot_roc_curve(labels_preds['y_pred_proba'], labels_preds['y_true'])

# %%
