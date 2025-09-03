import argparse
import os
from argparse import Namespace

from settings import Model, Settings


def train_cmd(args: Namespace):
    from dataloader.split import Split
    from train.extremes import evaluate_and_plot_extremes
    from train.train import train

    train()
    evaluate_and_plot_extremes(
        (
            "model_saves/best_model.pth"
            if Settings.model == Model.CNN
            else "model_saves/harmonicrnn.pth"
        ),
        dataset=(
            Split.SINGLE_AUDIO if Settings.single_element_training else Split.VALIDATION
        ),
    )


def process_cmd(args: Namespace):
    from train.inference import inference

    input_file = args.input
    output_file = args.output or os.path.splitext(input_file)[0] + ".midi"

    if args.model is not None:
        assert args.model in ["RNN", "CNN"], f"Invalid model {args.model}"
        Settings.model = Model.CNN if args.model == "CNN" else Model.RNN

    if args.model_path is not None:
        Settings.pre_trained_model_path = args.model_path

    inference(input_file, write_to_file=True, output_path=output_file)

    print(f'Midi saved as "{output_file}"')


def convert_cmd(args: Namespace):
    import soundfile as sf  # type: ignore

    from dataloader.Song import Song

    midi_file = args.midi_file
    output_file = args.output or os.path.splitext(midi_file)[0] + ".wav"
    Settings.audio_font_path = args.sound_font or Settings.audio_font_path
    Settings.seconds = None  # type: ignore

    sf.write(output_file, Song.from_path(midi_file).to_wav(), Settings.sample_rate)  # type: ignore

    print(f'Audio saved as "{output_file}"')


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

    # convert command
    convert_parser = subparsers.add_parser(
        "convert", help="Render an existing MIDI file into audio"
    )
    convert_parser.add_argument("midi_file", help="Path to the existing MIDI file")
    convert_parser.add_argument(
        "-o", "--output", help="Output audio path (defaults to .wav alongside input)"
    )
    convert_parser.add_argument(
        "-s",
        "--sound-font",
        help="Path to the sound font to use (defaults to settings.py)",
    )
    convert_parser.set_defaults(func=convert_cmd)


    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
