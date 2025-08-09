import os
import random
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
import torch.optim as optim
from pretty_midi import PrettyMIDI  # type: ignore
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import wandb
from dataloader.dataset import DataSet
from dataloader.Song import Song
from dataloader.split import Split
from model.HarmonicNet import Statistics
from model.model import HarmonicCNN
from model_rnn.model import HarmonicRNN
from settings import Model
from settings import Settings as s
from train.evaluate import evaluate
from train.utils import (
    midi_to_label_matrices,
    plot_fixed_sample,
    plot_harmoniccnn_outputs,
    should_log_image,
    to_numpy,
)


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
    total_batches = len(dataloader)

    stats = Statistics(0, 0, 0)

    for _, batch in tqdm(
        enumerate(dataloader), total=total_batches, desc=f"Epoch {epoch+1}/{s.epochs}"
    ):
        optimizer.zero_grad()

        model.set_input(batch, device)
        model.exec_forward()
        total_loss = model.get_loss()
        stats += model.get_batch_statistics()

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

    return running_loss / total_batches, {
        "f1": stats.f1,
        "precision": stats.precision,
        "recall": stats.recall,
    }


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
        project="MLVM",
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

        if s.single_element_training:
            wandb.log(
                {
                    "loss/train": avg_train_loss,
                    "metrics_TRAIN/precision/train_yp": train_metrics["precision"],
                    "metrics_TRAIN/recall/train_yp": train_metrics["recall"],
                    "metrics_TRAIN/f1/train_yp": train_metrics["f1"],
                }
            )
        else:
            wandb.log(
                {
                    "loss/train": avg_train_loss,
                    "loss/val": avg_val_loss,
                    "metrics_TRAIN/precision/train_yp": train_metrics["precision"],
                    "metrics_TRAIN/recall/train_yp": train_metrics["recall"],
                    "metrics_TRAIN/f1/train_yp": train_metrics["f1"],
                    "metrics_VAL/precision/train_yp": val_metrics["precision"],
                    "metrics_VAL/recall/train_yp": val_metrics["recall"],
                    "metrics_VAL/f1/train_yp": val_metrics["f1"],
                },
                step=epoch + 1,
            )

        if should_log_image(epoch):
            d = DataSet(Split.SINGLE_AUDIO, s.seconds)

            fixed_batch = DataLoader(d).__iter__().__next__()
            fixed_sample = (
                tuple([val[0] for val in fixed_batch[0]]),
                fixed_batch[1][0],
            )

            model.set_input(fixed_batch, device)
            model.exec_forward()
            out = model.get_network_output()[0]

            (midis_np, tempos, ticks_per_beats, nums_messages), _ = fixed_sample
            midi = Song.from_np(
                to_numpy(midis_np).astype(np.uint16), tempos, ticks_per_beats, nums_messages  # type: ignore
            ).get_midi()

            fig = plot_fixed_sample(fixed_sample, device, out.yo, out.yp, out.yn)[0]

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

            # Log midi to wandb
            artifact = wandb.Artifact(f"{wandb.run.name}_midi", type="midi")  # type: ignore
            run_tmp_dir = Path(tempfile.gettempdir()) / wandb.run.name  # type: ignore
            run_tmp_dir.mkdir(exist_ok=True)  # type: ignore
            midi_filename = f"midi_epoch_{epoch+1}.mid"
            full_path = run_tmp_dir / midi_filename  # type: ignore

            if isinstance(out.midi, PrettyMIDI):
                out.midi.write(str(full_path))  # type: ignore
            else:
                out.midi.save(str(full_path))  # type: ignore

            artifact.add_file(str(full_path), name=midi_filename)  # type: ignore

            wav_data = Song.from_path(str(full_path)).to_wav()  # type: ignore

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
                    "audio": wandb.Audio(wav_data, sample_rate=s.sample_rate),
                },
                step=epoch + 1,
            )
            wandb.log_artifact(artifact)

            # Elimino il wav temporaneo (? Lo stiamo facendo?)
            plt.close(fig)  # type: ignore
            plt.close(gt_fig)

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
