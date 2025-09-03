import os

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataloader.dataset import DataSet
from dataloader.split import Split
from model.model import HarmonicCNN
from model_rnn.model import HarmonicRNN
from settings import Model
from settings import Settings as s
from train.utils import save_plot


def evaluate_and_plot_extremes(
    model_path: str, dataset: Split, output_dir: str = "eval_plots", top_k: int = 5
) -> None:
    device = torch.device(s.device)
    print(f"Evaluating model {model_path} for top/bottom F1 samples on {device}")

    model = (HarmonicCNN() if s.model == Model.CNN else HarmonicRNN()).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))  # type: ignore
    model.eval()

    val_dataset = DataSet(dataset, s.seconds)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    os.makedirs(output_dir, exist_ok=True)

    scores: list[
        tuple[
            float,
            int,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
        ]
    ] = []
    skipped_samples = 0

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader, desc="Computing F1 scores")):
            _, audios = batch

            model.set_input(batch, device)
            model.exec_forward()
            out_batch = model.get_network_output()
            gt_batch = model.get_network_input()

            for gt, pred, audio_input, stat in zip(
                gt_batch, out_batch, audios, model.get_separate_statistics()
            ):
                assert isinstance(audio_input, torch.Tensor)

                scores.append(
                    (
                        stat.f1,
                        idx,
                        audio_input.cpu(),
                        pred.yo.cpu(),
                        pred.yp.cpu(),
                        pred.yn.cpu() if pred.yn is not None else None,
                        gt.yo.cpu(),
                        gt.yp.cpu(),
                        gt.yn.cpu() if gt.yn is not None else None,
                    )
                )

    scores_sorted = sorted(scores, key=lambda x: x[0], reverse=True)
    top_samples = scores_sorted[:top_k]
    bottom_samples = scores_sorted[-top_k:]

    for sample in top_samples:
        save_plot(sample, "best", output_dir)
    for sample in bottom_samples:
        save_plot(sample, "worst", output_dir)

    print(f"Saved {top_k} best and {top_k} worst plots to '{output_dir}'")
    print(
        f"Skipped {skipped_samples} samples where both ground truth and prediction were all zeros"
    )
