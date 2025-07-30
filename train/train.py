import os
import random
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
import torch.optim as optim
from tqdm.auto import tqdm
import wandb
from torch.utils.data import DataLoader

from dataloader.dataset import DataSet
from dataloader.Song import Song
from dataloader.split import Split
from model.model import HarmonicCNN
from model_rnn.model import HarmonicRNN
from settings import Model
from settings import Settings as s
from train.evaluate import evaluate
from train.losses import harmoniccnn_loss
from train.rnn_losses import np_midi_loss
from train.utils import (
    binary_classification_metrics,
    midi_to_label_matrices,
    plot_fixed_sample,
    plot_harmoniccnn_outputs,
    should_log_image,
    soft_continuous_accuracy,
    to_tensor,
)


from tqdm import tqdm  # import corretto della funzione tqdm


def save_plot(sample, name, output_dir):
    f1, idx, audio_input, yo_pred, yp_pred, yn_pred, yo_true, yp_true, yn_true = sample
    
    # Gestione dimensioni tensori
    yo_pred = yo_pred.squeeze(0).squeeze(0) if len(yo_pred.shape) == 4 else yo_pred.squeeze(0)
    yp_pred = yp_pred.squeeze(0).squeeze(0) if len(yp_pred.shape) == 4 else yp_pred.squeeze(0)
    yn_pred = yn_pred.squeeze(0).squeeze(0) if yn_pred is not None and len(yn_pred.shape) == 4 else yn_pred.squeeze(0) if yn_pred is not None else None
    
    yo_true = yo_true.squeeze(0).squeeze(0) if len(yo_true.shape) == 4 else yo_true.squeeze(0)
    yp_true = yp_true.squeeze(0).squeeze(0) if len(yp_true.shape) == 4 else yp_true.squeeze(0)
    yn_true = yn_true.squeeze(0).squeeze(0) if yn_true is not None and len(yn_true.shape) == 4 else yn_true.squeeze(0) if yn_true is not None else None

    fig_pred = plot_harmoniccnn_outputs(
        yo_pred,
        yp_pred,
        yn_pred,
        title_prefix=f"Prediction idx {idx}",
    )
    fig_gt = plot_harmoniccnn_outputs(
        yo_true,
        yp_true,
        yn_true,
        title_prefix=f"Ground Truth idx {idx}",
    )
    fig_pred.savefig(os.path.join(output_dir, f"{name}_pred_{idx}.png"))
    fig_gt.savefig(os.path.join(output_dir, f"{name}_gt_{idx}.png"))
    plt.close(fig_pred)
    plt.close(fig_gt)

def evaluate_and_plot_extremes(
    model_path: str, dataset: Split, output_dir: str = "eval_plots", top_k: int = 5
) -> None:
    device = s.device
    print(f"Evaluating model {model_path} for top/bottom F1 samples on {device}")

    # Carica modello
    model = (
        HarmonicCNN().to(device) if s.model == Model.CNN else HarmonicRNN().to(device)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Dataset e loader
    test_dataset = DataSet(dataset, s.seconds)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    os.makedirs(output_dir, exist_ok=True)

    scores = []

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, desc="Computing F1 scores")):
            (midis_np, tempos, ticks_per_beats, nums_messages), audios = batch

            # Estrai label true da MIDI
            midi = Song.from_np(
                midis_np[0], tempos[0], ticks_per_beats[0], nums_messages[0]
            ).get_midi()
            yo_true, yp_true = midi_to_label_matrices(
                midi, s.sample_rate, s.hop_length, n_bins=88
            )
            yn_true = yp_true if not s.remove_yn else None

            yo_true_t = to_tensor(yo_true).to(device).unsqueeze(0)
            yp_true_t = to_tensor(yp_true).to(device).unsqueeze(0)
            yn_true_t = to_tensor(yn_true).to(device).unsqueeze(0) if yn_true is not None else None

            audio_input = audios.to(device)

            # Predizione - MODIFICATO QUI
            outputs = model(audio_input)
            if s.remove_yn:
                yo_pred = outputs[0]
                yp_pred = outputs[1]
                yn_pred = None
            else:
                yo_pred, yp_pred, yn_pred = outputs

            yp_pred_sig = torch.sigmoid(yp_pred).squeeze(1)

            # Calcolo metriche (F1 sample-wise)
            metrics = binary_classification_metrics(yp_pred_sig, yp_true_t)
            tp, fp, fn = metrics["TP"], metrics["FP"], metrics["FN"]
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)

            scores.append(
                (
                    f1,
                    idx,
                    audio_input.cpu(),
                    yo_pred.cpu(),
                    yp_pred.cpu(),
                    yn_pred.cpu() if yn_pred is not None else None,
                    yo_true_t.cpu(),
                    yp_true_t.cpu(),
                    yn_true_t.cpu() if yn_true_t is not None else None,
                )
            )

    # Ordina per F1 score (decrescente)
    scores_sorted = sorted(scores, key=lambda x: x[0], reverse=True)

    # Prendi top e bottom k
    top_samples = scores_sorted[:top_k]
    bottom_samples = scores_sorted[-top_k:]

    # Salva le immagini
    for sample in top_samples:
        save_plot(sample, "best", output_dir)
    for sample in bottom_samples:
        save_plot(sample, "worst", output_dir)

    print(f"Saved {top_k} best and {top_k} worst plots to '{output_dir}'")

def train_one_epoch(
    model: HarmonicCNN | HarmonicRNN,
    dataloader: DataLoader[
        tuple[
            tuple[npt.NDArray[np.uint16], int, int, int],
            npt.NDArray[np.float32] | torch.Tensor,
        ]
    ],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    session_dir: str,
) -> tuple[float, dict[str, float]]:

    model.train()

    running_loss = 0.0
    total_accuracy = 0.0
    total_batches = len(dataloader)

    # Accumulatore globale per le metriche di YP
    yp_metrics_accumulator = {"TP": 0.0, "FP": 0.0, "FN": 0.0}

    for _, batch in tqdm.tqdm(
        enumerate(dataloader), total=total_batches, desc=f"Epoch {epoch+1}/{s.epochs}"
    ):
        (midis_np, tempos, ticks_per_beats, nums_messages), audios = batch
        optimizer.zero_grad()

        if s.model == Model.CNN:

            yo_true_batch: list[torch.Tensor] = []
            yn_true_batch: list[torch.Tensor] = []
            yp_true_batch: list[torch.Tensor] = []

            audio_input_batch: list[torch.Tensor] = []

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

            yo_true_batch_stacked = torch.stack(yo_true_batch)
            yp_true_batch_stacked = torch.stack(yp_true_batch)

            if s.remove_yn:
                yn_true_batch_stacked = None
            else:
                yn_true_batch_stacked = torch.stack(yn_true_batch)

            audio_input_batch_stacked = torch.stack(audio_input_batch)

            (yo_pred, yp_pred, yn_pred) = model(audio_input_batch_stacked)

            yp_pred_sig = torch.sigmoid(yp_pred).squeeze(1)

            yo_pred = yo_pred.squeeze(1)
            yp_pred = yp_pred.squeeze(1)
            if not s.remove_yn:
                yn_pred = yn_pred.squeeze(1)

            batch_metrics = binary_classification_metrics(
                yp_pred_sig, yp_true_batch_stacked
            )
            for key in ["TP", "FP", "FN"]:
                yp_metrics_accumulator[key] += batch_metrics[key]

            accuracy = soft_continuous_accuracy(yp_pred_sig, yp_true_batch_stacked)
            total_accuracy += accuracy

            loss = harmoniccnn_loss(
                yo_pred,
                yp_pred,
                yo_true_batch_stacked,
                yp_true_batch_stacked,
                yn_pred,
                yn_true_batch_stacked,
                label_smoothing=s.label_smoothing,
                weighted=s.weighted,
            )

            total_loss = sum(loss.values())

        else:  # RNN
            assert isinstance(model, HarmonicRNN)
            audios = audios.reshape((audios.shape[0], -1, s.sample_rate))
            pred_midi, pred_len, pred_tpb = model(audios)
            total_loss = np_midi_loss(
                pred_midi, pred_len, pred_tpb, midis_np, nums_messages, ticks_per_beats
            )

        total_loss.backward()  # type: ignore
        optimizer.step()
        cur_loss = total_loss.item()  # type: ignore
        assert isinstance(cur_loss, float)
        running_loss += cur_loss

    if s.save_model:
        os.makedirs(session_dir, exist_ok=True)
        path = os.path.join(
            session_dir,
            "harmoniccnn.pth" if s.model == Model.CNN else "harmonicrnn.pth",
        )
        torch.save(model.state_dict(), path)
        print(f"Model saved as '{path}'")

    if s.model == Model.RNN:
        return running_loss / total_batches, {}

    # Calcolo metriche globali per CNN
    tp = yp_metrics_accumulator["TP"]
    fp = yp_metrics_accumulator["FP"]
    fn = yp_metrics_accumulator["FN"]
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    average_soft_accuracy = total_accuracy / total_batches

    return (
        running_loss / total_batches,
        {
            "yp_accuracy": average_soft_accuracy,
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

    project_name = (
        ("CNN" if s.model == Model.CNN else "RNN")
        + ("_single" if s.single_element_training else "")
        + f"_{timestamp}"
    )

    wandb.init(
        project="MLVM-Project",
        name=project_name,
        config={
            "epochs": s.epochs,
            "batch_size": s.batch_size,
            "learning_rate": s.learning_rate,
            "model": s.model.name,
            "seed": seed,
            "pre_trained": True if s.pre_trained_model_path else False,
            "pre_trained_model_path": (
                s.pre_trained_model_path if s.pre_trained_model_path else None
            ),
            "single_element_training": s.single_element_training,
        },
    )

    model = (HarmonicCNN() if s.model == Model.CNN else HarmonicRNN()).to(device)
    if s.pre_trained_model_path is not None:
        model.load_state_dict(torch.load(s.pre_trained_model_path, map_location=device))
        print(f"Loaded pre-trained model from {s.pre_trained_model_path}")

    optimizer = optim.Adam(model.parameters(), lr=s.learning_rate)

    train_dataset = (
        DataSet(Split.TRAIN, s.seconds)
        if not s.single_element_training
        else DataSet(Split.SINGLE_AUDIO, s.seconds)
    )
    train_loader = DataLoader(train_dataset, batch_size=s.batch_size, shuffle=True)

    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    for epoch in range(s.epochs):

        avg_train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, epoch, session_dir
        )
        print(f"[Epoch {epoch+1}/{s.epochs}] Train Loss: {avg_train_loss:.4f}")

        if s.model == Model.CNN:
            model_path = os.path.join(session_dir, "harmoniccnn.pth")
        else:
            model_path = os.path.join(session_dir, "harmonicrnn.pth")

        if s.single_element_training:
            avg_val_loss = 0
            val_metrics = {}
        else:
            print("Evaluating on validation set...")
            avg_val_loss, val_metrics = evaluate(model_path, Split.VALIDATION)
            print(f"[Epoch {epoch+1}/{s.epochs}] Validation Loss: {avg_val_loss:.4f}")

        if s.model == Model.CNN:
            if s.single_element_training:
                wandb.log(
                    {
                        "loss/train": avg_train_loss,
                        "metrics_TRAIN/average accuracy/train_yp": train_metrics[
                            "yp_accuracy"
                        ],
                        "metrics_TRAIN/precision/train_yp": train_metrics[
                            "yp_precision"
                        ],
                        "metrics_TRAIN/recall/train_yp": train_metrics["yp_recall"],
                        "metrics_TRAIN/f1/train_yp": train_metrics["yp_f1"],
                    }
                )
            else:
                wandb.log(
                    {
                        "loss/train": avg_train_loss,
                        "loss/val": avg_val_loss,
                        "metrics_TRAIN/average accuracy/train_yp": train_metrics[
                            "yp_accuracy"
                        ],
                        "metrics_TRAIN/precision/train_yp": train_metrics[
                            "yp_precision"
                        ],
                        "metrics_TRAIN/recall/train_yp": train_metrics["yp_recall"],
                        "metrics_TRAIN/f1/train_yp": train_metrics["yp_f1"],
                        "metrics_VAL/average accuracy/train_yp": val_metrics[
                            "yp_accuracy"
                        ],
                        "metrics_VAL/precision/train_yp": val_metrics["yp_precision"],
                        "metrics_VAL/recall/train_yp": val_metrics["yp_recall"],
                        "metrics_VAL/f1/train_yp": val_metrics["yp_f1"],
                    },
                    step=epoch + 1,
                )

        else:  # RNN
            wandb.log(
                {
                    "loss/train": avg_train_loss,
                    "loss/val": avg_val_loss,
                }
            )

        if should_log_image(epoch):
            d = DataSet(Split.SINGLE_AUDIO, s.seconds)
            fixed_sample = d[0]

            (midis_np, tempos, ticks_per_beats, nums_messages), audio = fixed_sample
            midi = Song.from_np(
                midis_np.astype(np.uint16), tempos, ticks_per_beats, nums_messages
            ).get_midi()

            if s.model == Model.CNN:
                fig = plot_fixed_sample(model, fixed_sample, device)

                yo_true, yp_true = midi_to_label_matrices(
                    midi, s.sample_rate, s.hop_length, n_bins=88
                )
                yn_true = yp_true

                title_prefix = "Ground Truth"
                gt_fig = plot_harmoniccnn_outputs(
                    torch.Tensor(yo_true),
                    torch.Tensor(yp_true),
                    torch.Tensor(yn_true),
                    title_prefix,
                )

                # Log both
                wandb.log(
                    {
                        "prediction_vs_gt": wandb.Image(
                            fig, caption=f"Prediction Epoch {epoch+1}"
                        ),
                        "ground_truth": (
                            wandb.Image(gt_fig, caption=f"Ground Truth Epoch {epoch+1}")
                            if epoch == 0 or epoch == 2
                            else None
                        ),
                    },
                    step=epoch + 1,
                )
                plt.close(fig)
                plt.close(gt_fig)

            else:  # RNN
                audio_input = audio.reshape((1, -1, s.sample_rate))
                pred_midi, pred_len, pred_tpb = model(torch.Tensor(audio_input))
                out = Song.from_np(
                    pred_midi[0].to(torch.uint16),
                    None,
                    int(pred_tpb[0]),
                    int(pred_len[0]),
                )

                wandb.log(
                    {
                        "out_audio": wandb.Audio(out.to_wav(), s.sample_rate),
                        "ground_truth_audio": (
                            wandb.Audio(audio, s.sample_rate) if epoch == 0 else None
                        ),
                    },
                    step=epoch + 1,
                )

        if not s.single_element_training:
            if avg_val_loss < best_val_loss - 1e-4:
                best_val_loss = avg_val_loss
                patience_counter = 0

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
