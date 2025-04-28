import os
import torch

# automatic stuff, the actual settings are in the class Settings below
cur_dir = os.path.dirname(__file__)
audio_font_folder = os.path.join(cur_dir, "audio-fonts")


class Settings:
    # folder names
    dataset_folder = "data"

    train_folder = "train"
    validation_folder = "validation"
    test_folder = "test"

    # dataset downloading settings
    metadata_files_to_keep = [
        "LICENSE",
        "README",
        "maestro-v3.0.0.csv",
        "maestro-v3.0.0.json",
    ]

    # audio file generation
    sample_rate = 44100
    audio_font_path = os.path.join(audio_font_folder, "Piano.sf2")
    tmp_midi_file = "tmp.midi"
    tmp_audio_file = "tmp.wav"
    seconds = 2
    max_midi_messages = 200
    
    # model settings
    hop_length = 256
    n_bins = 84
    harmonic_shifts = [-12, 0, 12]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # training settings
    epochs = 1
    batch_size = 2
    learning_rate = 1e-4
    label_smoothing = 0.1
    weighted = True
    positive_weight = 0.95
    weighted = True
