import argparse
import os
from argparse import Namespace

from dataloader.split import Split
from settings import Model, Settings
from train.extremes import evaluate_and_plot_extremes
from train.inference import inference
from train.train import train


def train_cmd(args: Namespace):
    train()
    evaluate_and_plot_extremes(
        (
            "model_saves/harmoniccnn.pth"
            if Settings.model == Model.CNN
            else "model_saves/harmonicrnn.pth"
        ),
        Split.SINGLE_AUDIO,
    )


def process_cmd(args: Namespace):
    input_file = args.input
    output_file = args.output or os.path.splitext(input_file)[0] + ".midi"

    if args.model is not None:
        assert args.model in ["RNN", "CNN"], f"Invalid model {args.model}"
        Settings.model = Model.CNN if args.model == "CNN" else Model.RNN

    if args.model_path is not None:
        Settings.pre_trained_model_path = args.model_path

    inference(input_file, write_to_file=True, output_path=output_file)

    print(f'Midi saved as "{output_file}"')


def main():
    parser = argparse.ArgumentParser(description="Neural Network Audio-to-MIDI CLI")
    subparsers = parser.add_subparsers(title="commands", dest="command")

    # train command
    train_parser = subparsers.add_parser("train", help="Train the neural network")
    train_parser.set_defaults(func=train_cmd)

    # process command
    process_parser = subparsers.add_parser("process", help="Transform audio into MIDI")
    process_parser.add_argument(
        "-i", "--input", required=True, help="Input audio file path"
    )
    process_parser.add_argument("-o", "--output", help="Output MIDI path")
    process_parser.add_argument(
        "-m", "--model", choices=["RNN", "CNN"], help="Model type to use"
    )
    process_parser.add_argument("-p", "--model-path", help="Path to trained model")
    process_parser.set_defaults(func=process_cmd)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
