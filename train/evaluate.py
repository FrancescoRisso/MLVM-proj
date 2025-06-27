import numpy as np
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
    plot_prediction_vs_ground_truth,
    to_tensor,
    weighted_soft_accuracy,
)


def evaluate(model_path, dataset):
    """
    Evaluate the model on the test set.
    Args:
        model_path (str): Path to the pre-trained model.
        dataset (str): Dataset to evaluate on. Options: "DataSet(Split.TRAIN, s.seconds)", "DataSet(Split.TRAIN, s.seconds)".
    """

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
    total_batches = len(test_loader)

    all_acc_yo = []
    all_acc_yn = []

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
            yo_pred = yo_pred.squeeze(1)
            yp_pred = yp_pred.squeeze(1)
            yp_pred = yp_pred.squeeze(1)
            if not s.remove_yn:
                yn_pred = yn_pred.squeeze(1)

            if not s.remove_yn:
                loss = harmoniccnn_loss(
                    yo_pred,       # yo_logits
                    yp_pred,       # yp_logits
                    yo_true_batch,  # yo_true
                    yp_true_batch,  # yp_true
                    yn_pred,        # yn_logits (opzionale)
                    yn_true_batch,  # yn_true (opzionale)
                    label_smoothing=s.label_smoothing,
                    weighted=s.weighted,
                    positive_weight=s.positive_weight,
                )
            else:
                loss = harmoniccnn_loss(
                    yo_pred,        # yo_logits
                    yp_pred,        # yp_logits
                    yo_true_batch,   # yo_true
                    yp_true_batch,   # yp_true
                    # yn_logits e yn_true omessi
                    label_smoothing=s.label_smoothing,
                    weighted=s.weighted,
                    positive_weight=s.positive_weight,
                )

            # TODO MIGLIORARE IL CALCOLO DELLA SOFT ACCURACY
            # acc_yo = weighted_soft_accuracy(yo_pred, yo_true_batch)
            # acc_yn = weighted_soft_accuracy(yn_pred, yn_true_batch)

            running_loss += sum(loss.values())
            # all_acc_yo.append(acc_yo)
            # all_acc_yn.append(acc_yn)

            #modello deve essere cnn
            if s.model == Model.CNN:
                print("Plotting predictions vs ground truth (first batch)...")
                # plot_prediction_vs_ground_truth(
                #     yo_pred[0], yp_pred[0], yo_true_batch[0], yn_true_batch[0]
                # )

            else:  # Using RNN
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
        avg_acc_yo = np.mean(all_acc_yo)
        avg_acc_yn = np.mean(all_acc_yn)

        print(
            f"[Evaluation] Loss: {avg_loss:.4f} | YO Acc: {avg_acc_yo:.4f} | YN Acc: {avg_acc_yn:.4f}"
        )
    else:
        print(f"[Evaluation] Loss: {avg_loss:.4f}")

    return avg_loss