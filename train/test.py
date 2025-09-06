import os
import random
import shutil
import sys

import soundfile as sf  # type: ignore
import torch
from matplotlib import pyplot as plt
from pretty_midi import PrettyMIDI  # type: ignore
from torch.utils.data import DataLoader

from dataloader.dataset import DataSet
from dataloader.Song import Song
from dataloader.split import Split
from model.HarmonicNet import HarmonicNet, Statistics
from model.model import HarmonicCNN
from model_rnn.model import HarmonicRNN
from settings import Model
from settings import Settings as s
from train.utils import plot_harmoniccnn_outputs


def test(num_tests: int | None, save_dir: str | None, verbose: bool):
    seed = random.randrange(sys.maxsize)
    random.seed(seed)
    print("Seed was:", seed)

    device = torch.device(s.device)
    print(f"Testing on {device}")

    if save_dir is not None:
        shutil.rmtree(save_dir)
        os.mkdir(save_dir)

    model: HarmonicNet = (HarmonicCNN() if s.model == Model.CNN else HarmonicRNN()).to(device)  # type: ignore
    if s.pre_trained_model_path is not None:
        model.load_state_dict(torch.load(s.pre_trained_model_path, map_location=device))  # type: ignore
        print(f"Loaded pre-trained model from {s.pre_trained_model_path}")
    else:
        print("\n\tWARN: you are testing an untrained model!\n")

    test_dataset = DataSet(Split.TEST, s.seconds, num_tests)
    test_loader = DataLoader(test_dataset, batch_size=s.batch_size, shuffle=True)

    model.eval()  # type: ignore

    # fmt: off
    print(" ________________________________________________________________________________")
    print("|        |                                   |                                   |")
    print("|        |          NOTE STATISTICS          |           Yp STATISTICS           |")
    print("| TEST # |___________________________________|___________________________________|")
    print("|        |           |           |           |           |           |           |")
    print("|        | PRECISION | RECALL    | F1 SCORE  | PRECISION | RECALL    | F1 SCORE  |")
    print("|________|___________|___________|___________|___________|___________|___________|")
    print("|        |           |           |           |           |           |           |")
    # fmt: on

    cumul_stats = Statistics(0, 0, 0, 0, 0, 0)
    test_no = 1

    for _, batch in enumerate(test_loader):
        model.set_input(batch, device)
        model.exec_forward()

        if save_dir is not None:
            test_no_copy = test_no
            all_outputs = model.get_network_output()
            all_inputs = model.get_network_input()

            for input, output in zip(all_inputs, all_outputs):
                test_path = os.path.join(save_dir, f"test_{test_no_copy:03}")
                os.mkdir(test_path)

                gt_midi = os.path.join(test_path, "gt_midi.midi")
                gt_audio = os.path.join(test_path, "gt_audio.wav")
                gt_images = os.path.join(test_path, "gt_posteriograms.png")
                out_midi = os.path.join(test_path, "out_midi.midi")
                out_audio = os.path.join(test_path, "out_audio.wav")
                out_images = os.path.join(test_path, "out_posteriograms.png")
                out_images_thresh = os.path.join(
                    test_path, "out_posteriograms_tresholded.png"
                )

                if isinstance(input.midi, PrettyMIDI):
                    input.midi.write(gt_midi)  # type:ignore
                else:
                    input.midi.save(gt_midi)  # type:ignore

                plot_harmoniccnn_outputs(input.yo, input.yp, input.yn, "Ground Truth")
                plt.savefig(gt_images)  # type: ignore
                sf.write(gt_audio, Song.from_path(gt_midi).to_wav(), s.sample_rate)  # type: ignore

                if isinstance(output.midi, PrettyMIDI):
                    output.midi.write(out_midi)  # type:ignore
                else:
                    output.midi.save(out_midi)  # type:ignore

                plot_harmoniccnn_outputs(output.yo, output.yp, output.yn, "Output")
                plt.savefig(out_images)  # type: ignore

                plot_harmoniccnn_outputs(
                    output.yo > s.threshold,
                    output.yp > s.threshold,
                    None if output.yn is None else output.yn > s.threshold,
                    "Thresholded output",
                )
                plt.savefig(out_images_thresh)  # type: ignore
                sf.write(out_audio, Song.from_path(out_midi).to_wav(), s.sample_rate)  # type: ignore

                test_no_copy += 1

        for stat in model.get_separate_statistics():
            cumul_stats += stat
            if verbose:
                print(
                    f"| {test_no:6}",
                    f"| {stat.precision:9.4f} | {stat.recall:9.4f} | {stat.f1:9.4f}",
                    f"| {stat.precision_bins:9.4f} | {stat.recall_bins:9.4f} | {stat.f1_bins:9.4f} |",
                )

            test_no += 1

    # fmt: off
    if verbose:
        print("|________|___________|___________|___________|___________|___________|___________|")
        print("|        |           |           |           |           |           |           |")
    # fmt: on

    print(
        f"| TOTAL ",
        f"| {cumul_stats.precision:9.4f} | {cumul_stats.recall:9.4f} | {cumul_stats.f1:9.4f}",
        f"| {cumul_stats.precision_bins:9.4f} | {cumul_stats.recall_bins:9.4f} | {cumul_stats.f1_bins:9.4f} |",
    )

    # fmt: off
    print("|________|___________|___________|___________|___________|___________|___________|")
    # fmt: on
