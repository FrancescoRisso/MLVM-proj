import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.dataset import DataSet
from dataloader.Song import Song
from model.model import HarmonicCNN
from model_rnn.model import HarmonicRNN
from settings import Model
from settings import Settings as s
from train.losses import harmoniccnn_loss
from train.rnn_losses import np_midi_loss
from train.utils import (
    midi_to_label_matrices,
    to_tensor,
    soft_continuous_accuracy,
    binary_classification_metrics,
)


def evaluate(model_path, dataset):

    device = s.device
    print(f"Evaluating on {device}")

    model = (HarmonicCNN() if s.model == Model.CNN else HarmonicRNN()).to(device)

    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded pre-trained model from {model_path}")
    else:
        print("No model found insert a valid model")
        return

    model.eval()

    test_dataset = DataSet(dataset, s.seconds)
    test_loader = DataLoader(test_dataset, batch_size=s.batch_size, shuffle=False)

    running_loss = 0.0
    running_acc_yp = 0.0
    total_batches = len(test_loader)
    yp_metrics_accumulator = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(test_loader), total=total_batches):
            (midis_np, tempos, ticks_per_beats, nums_messages), audios = batch

            if s.model == Model.CNN:

                yo_true_batch = []
                yn_true_batch = []
                yp_true_batch = []

                audio_input_batch = []

                for i in range(midis_np.shape[0]):
                    midi = Song.from_np(
                        midis_np[i], tempos[i], ticks_per_beats[i], nums_messages[i]
                    ).get_midi()
                    yo, yp = midi_to_label_matrices(
                        midi, s.sample_rate, s.hop_length, n_bins=88
                    )
                    yn = yp  # In this case, yp is the same as yn

                    yo_true_batch.append(to_tensor(yo).to(device))
                    yp_true_batch.append(to_tensor(yp).to(device))
                    if not s.remove_yn:
                        yn_true_batch.append(to_tensor(yn).to(device))

                    audio_input_batch.append(audios[i].to(device))

                yo_true_batch = torch.stack(yo_true_batch)
                yp_true_batch = torch.stack(yp_true_batch)
                if not s.remove_yn:
                    yn_true_batch = torch.stack(yn_true_batch)

                audio_input_batch = torch.stack(audio_input_batch)

                yo_pred, yp_pred, yn_pred = model(audio_input_batch)

                yp_pred_sig = torch.sigmoid(yp_pred).squeeze(1)

                yo_pred = yo_pred.squeeze(1)
                yp_pred = yp_pred.squeeze(1)
                if not s.remove_yn:
                    yn_pred = yn_pred.squeeze(1)

                acc_yp = soft_continuous_accuracy(yp_pred, yp_true_batch)
                running_acc_yp += acc_yp

                batch_metrics = binary_classification_metrics(
                    yp_pred_sig, yp_true_batch
                )
                for key in ["TP", "FP", "FN", "TN"]:
                    yp_metrics_accumulator[key] += batch_metrics[key]

                loss = harmoniccnn_loss(
                    yo_pred,  # yo_logits
                    yp_pred,  # yp_logits
                    yo_true_batch,  # yo_true
                    yp_true_batch,  # yp_true
                    yn_pred,  # yn_logits (opzionale)
                    yn_true_batch,  # yn_true (opzionale)
                    label_smoothing=s.label_smoothing,
                    weighted=s.weighted,
                )

                running_loss += sum(loss.values())

            else:  # RNN TODO
                audios = audios.reshape(
                    (
                        audios.shape[0],  # leave batch items untouched
                        -1,  # all the seconds
                        s.sample_rate,  # samples per secon
                    )
                )

                pred_midi, pred_len = model(audios)

                running_loss += np_midi_loss(
                    pred_midi, pred_len, midis_np, nums_messages
                )

    avg_loss = running_loss / total_batches

    if s.model == Model.CNN:

        # Calcolo metriche globali per CNN
        tp = yp_metrics_accumulator["TP"]
        fp = yp_metrics_accumulator["FP"]
        fn = yp_metrics_accumulator["FN"]
        tn = yp_metrics_accumulator["TN"]
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        avg_acc_yp = running_acc_yp / total_batches

        print(f"[YP] Validation Epoch Metrics")
        print(f"TP: {tp:.0f}, FP: {fp:.0f}, FN: {fn:.0f}, TN: {tn:.0f}")
        print(f"Average accuracy: {avg_acc_yp:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        return (
            avg_loss,
            {
                "yp_TP": tp,
                "yp_FP": fp,
                "yp_FN": fn,
                "yp_TN": tn,
                "yp_accuracy": avg_acc_yp,
                "yp_precision": precision,
                "yp_recall": recall,
                "yp_f1": f1,
            },
        )

    else:  # RNN TODO
        print(f"[Evaluation] Loss: {avg_loss:.4f}")

    return avg_loss


def soft_continuous_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Accuracy continua: 1 - |pred - true| mediato.
    Valori predetti vicini al target sono premiati di più.
    """
    with torch.no_grad():
        error = torch.abs(y_pred - y_true)
        score = 1.0 - error
        return score.mean().item()
