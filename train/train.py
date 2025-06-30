import os
import random
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader

import wandb

from dataloader.dataset import DataSet
from dataloader.Song import Song
from dataloader.split import Split
from model.model import HarmonicCNN
from model_rnn.model import HarmonicRNN
from settings import Model
from settings import Settings as s
from train.losses import harmoniccnn_loss
from train.rnn_losses import np_midi_loss
from train.evaluate import evaluate
from train.utils import (
    midi_to_label_matrices,
    plot_prediction_vs_ground_truth,
    to_tensor,
    binary_classification_metrics,
    soft_continuous_accuracy,
    should_log_image,
)


def train_one_epoch(
    model: HarmonicCNN | HarmonicRNN, dataloader, optimizer, device, epoch, session_dir
):

    model.train()

    running_loss = 0.0
    total_accuracy = 0.0
    total_batches = len(dataloader)
    example_outputs = None

    # Accumulatore globale per le metriche di YP
    yp_metrics_accumulator = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}

    for batch_idx, batch in tqdm.tqdm(
        enumerate(dataloader), total=total_batches, desc=f"Epoch {epoch+1}/{s.epochs}"
    ):
        (midis_np, tempos, ticks_per_beats, nums_messages), audios = batch
        optimizer.zero_grad()

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
                yn = yp

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

            (yo_pred, yp_pred, yn_pred) = model(audio_input_batch)

            yo_pred_sig = torch.sigmoid(yo_pred).squeeze(1)
            yp_pred_sig = torch.sigmoid(yp_pred).squeeze(1)
            if not s.remove_yn:
                yn_pred_sig = torch.sigmoid(yn_pred).squeeze(1)

            yo_pred = yo_pred.squeeze(1)
            yp_pred = yp_pred.squeeze(1)
            if not s.remove_yn:
                yn_pred = yn_pred.squeeze(1)

            batch_metrics = binary_classification_metrics(yp_pred_sig, yp_true_batch)
            for key in ["TP", "FP", "FN", "TN"]:
                yp_metrics_accumulator[key] += batch_metrics[key]

            accuracy = soft_continuous_accuracy(yp_pred_sig, yp_true_batch)
            total_accuracy += accuracy
            print(f"Soft Continuous Accuracy: {accuracy:.4f}")

            loss = harmoniccnn_loss(
                yo_pred,
                yp_pred,
                yo_true_batch,
                yp_true_batch,
                yn_pred,
                yn_true_batch,
                label_smoothing=s.label_smoothing,
                weighted=s.weighted,
            )

            total_loss = sum(loss.values())

            if batch_idx == total_batches - 1:
                example_outputs = {
                    "yo_pred": yo_pred_sig.detach().cpu(),
                    "yp_pred": yp_pred_sig.detach().cpu(),
                    "yn_pred": yn_pred_sig.detach().cpu() if not s.remove_yn else None,
                    "yo_true": yo_true_batch.detach().cpu(),
                    "yp_true": yp_true_batch.detach().cpu(),
                    "yn_true": (
                        yn_true_batch.detach().cpu() if not s.remove_yn else None
                    ),
                }

        else:
            assert isinstance(model, HarmonicRNN)
            audios = audios.reshape((audios.shape[0], -1, s.sample_rate))
            pred_midi, pred_len = model(audios)
            total_loss = np_midi_loss(pred_midi, pred_len, midis_np, nums_messages)
            if batch_idx == total_batches - 1:
                example_outputs = None

        total_loss.backward()
        optimizer.step()
        running_loss += total_loss.item()

    if s.save_model:
        os.makedirs(session_dir, exist_ok=True)
        path = os.path.join(
            session_dir,
            "harmoniccnn.pth" if s.model == Model.CNN else "harmonicrnn.pth",
        )
        torch.save(model.state_dict(), path)
        print(f"Model saved as '{path}'")

    if s.model == Model.RNN:
        return running_loss / total_batches, None, None

    # Calcolo metriche globali per CNN
    tp = yp_metrics_accumulator["TP"]
    fp = yp_metrics_accumulator["FP"]
    fn = yp_metrics_accumulator["FN"]
    tn = yp_metrics_accumulator["TN"]
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    average_soft_accuracy = total_accuracy / total_batches

    print(f"[YP] Epoch Metrics")
    print(f"TP: {tp:.0f}, FP: {fp:.0f}, FN: {fn:.0f}, TN: {tn:.0f}")
    print(f"Average accuracy: {average_soft_accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Return loss + example + metriche
    return (
        running_loss / total_batches,
        example_outputs,
        {
            "yp_TP": tp,
            "yp_FP": fp,
            "yp_FN": fn,
            "yp_TN": tn,
            "yp_accuracy": accuracy,
            "yp_precision": precision,
            "yp_recall": recall,
            "yp_f1": f1,
        },
    )


def train():
    seed = random.randrange(sys.maxsize)
    random.seed(seed)
    print("Seed was:", seed)

    device = s.device
    print(f"Training on {device}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = os.path.join("model_saves")

    wandb.init(
        project="MLVM-Project",
        name=f"harmonic_training_{timestamp}",
        config={
            "epochs": s.epochs,
            "batch_size": s.batch_size,
            "learning_rate": s.learning_rate,
            "model": s.model.name,
            "seed": seed,
            "pre_trained": True if s.pre_trained_model_path else False,
            "pre_trained_model_path": s.pre_trained_model_path if s.pre_trained_model_path else None,
        },
    )

    model = (HarmonicCNN() if s.model == Model.CNN else HarmonicRNN()).to(device)
    if s.pre_trained_model_path is not None:
        model.load_state_dict(torch.load(s.pre_trained_model_path, map_location=device))
        print(f"Loaded pre-trained model from {s.pre_trained_model_path}")

    optimizer = optim.Adam(model.parameters(), lr=s.learning_rate)

    train_dataset = DataSet(Split.TRAIN, s.seconds)
    train_loader = DataLoader(train_dataset, batch_size=s.batch_size, shuffle=True)

    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    for epoch in range(s.epochs):

        avg_train_loss, example_outputs, train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, epoch, session_dir
        )
        print(f"[Epoch {epoch+1}/{s.epochs}] Train Loss: {avg_train_loss:.4f}")

        if s.model == Model.CNN:
            model_path = os.path.join(session_dir, "harmoniccnn.pth")
        else:
            model_path = os.path.join(session_dir, "harmonicrnn.pth")

        print("Evaluating on validation set...")
        avg_val_loss, example_outputs_val, val_metrics  = evaluate(model_path, Split.VALIDATION)
        print(f"[Epoch {epoch+1}/{s.epochs}] Validation Loss: {avg_val_loss:.4f}")

        if s.model == Model.CNN:
            wandb.log(
                {
                    "loss/train": avg_train_loss,
                    "loss/val": avg_val_loss,

                    "metrics_TRAIN/average accuracy/train_yp": train_metrics["yp_accuracy"],
                    "metrics_TRAIN/precision/train_yp": train_metrics["yp_precision"],
                    "metrics_TRAIN/recall/train_yp": train_metrics["yp_recall"],
                    "metrics_TRAIN/f1/train_yp": train_metrics["yp_f1"],
                    "metrics_TRAIN/TP/train_yp": train_metrics["yp_TP"],
                    "metrics_TRAIN/FP/train_yp": train_metrics["yp_FP"],
                    "metrics_TRAIN/FN/train_yp": train_metrics["yp_FN"],
                    "metrics_TRAIN/TN/train_yp": train_metrics["yp_TN"],

                    "metrics_VAL/average accuracy/train_yp": train_metrics["yp_accuracy"],
                    "metrics_VAL/precision/train_yp": train_metrics["yp_precision"],
                    "metrics_VAL/recall/train_yp": train_metrics["yp_recall"],
                    "metrics_VAL/f1/train_yp": train_metrics["yp_f1"],
                    "metrics_VAL/TP/train_yp": train_metrics["yp_TP"],
                    "metrics_VAL/FP/train_yp": train_metrics["yp_FP"],
                    "metrics_VAL/FN/train_yp": train_metrics["yp_FN"],
                    "metrics_VAL/TN/train_yp": train_metrics["yp_TN"],
                },
                step=epoch + 1,
            )

            if should_log_image(epoch) and example_outputs is not None:
                fig = plot_prediction_vs_ground_truth(
                    example_outputs["yo_pred"][0],
                    example_outputs["yp_pred"][0],
                    (
                        example_outputs["yn_pred"][0]
                        if example_outputs["yn_pred"] is not None
                        else None
                    ),
                    example_outputs["yo_true"][0],
                    example_outputs["yp_true"][0],
                    (
                        example_outputs["yn_true"][0]
                        if example_outputs["yn_true"] is not None
                        else None
                    ),
                )
                wandb.log(
                    {"prediction_vs_gt": wandb.Image(fig, caption=f"Epoch {epoch+1}")},
                    step=epoch + 1,
                )
                plt.close(fig)
                
        else: #RNN TODO
            pass

        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            patience_counter = 0
            print(f"Validation loss improved to {best_val_loss:.4f}")

            best_model_path = os.path.join(session_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(
                    f"Early stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.4f}"
                )
                break
