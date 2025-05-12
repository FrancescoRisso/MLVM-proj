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
    generate_audio_on_download = False  # see note below
    metadata_files_to_keep = [
        "LICENSE",
        "README",
        "maestro-v3.0.0.csv",
        "maestro-v3.0.0.json",
    ]

    # audio file generation
    sample_rate = 22050
    audio_font_path = os.path.join(audio_font_folder, "Piano.sf2")
    tmp_midi_file = "tmp.midi"
    tmp_audio_file = "tmp.wav"
    seconds = 2
    max_midi_messages = 300

    # model settings
    hop_length = 512
    n_bins = 88
    harmonic_shifts = [-12, 0, 12]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # training settings
    epochs = 5
    batch_size = 30
    learning_rate = 1e-4
    label_smoothing = 0.1
    weighted = True
    positive_weight = 0.95
    weighted = True
    save_model = True

    # RNN settings
    hidden_size = 10000
    encoder_num_layers = 1
    decoder_num_layers = 1


"""
NOTE ABOUT OPTION "generate_audio_on_download"
---------------------------------------------------------------------

PREMISE

The original Maestro dataset is composed of midi files and the
relative audio file.
We however discard the audio files, since they do not only contain
what is described in the midi (the piano), but also other
instruments.

For this reason, we only keep the midi from the dataset, and we
synthesyze the audio file from that (which is a trivial task).

---------------------------------------------------------------------

OPTIONS

To generate the audio files, we have two options:

1)	ON THE FLY (generate_audio_on_download = False)
    When we download the dataset, we just store the midis.
    Then, when we access a dataset item, the midi is cut according to
    the settings, and then the cut midi is synthesized to audio.
    
2)	ON DOWNLOAD (generate_audio_on_download = True)
    After downloading the dataset, all the complete midis are
    synthesized.
    Then, on dataset access, it will be enough to cut the midi, then
    load the full corresponding audio, and cut it the same way.

---------------------------------------------------------------------

TIME REQUIRED

1)	ON THE FLY
    Downloading & preparing the dataset: ~15 s
    Accessing 1 data point (on average): ~1.2 s
    Accessing the full training set: ~20 min
    
2)	ON DOWNLOAD
    Downloading & preparing the dataset: ~4.5 h
    Accessing 1 data point (on average): ~0.9 s
    Accessing the full training set: ~15 min

---------------------------------------------------------------------

EFFICIENCY NOTE

This settings only affects the dataset downloading, not the data
access.
This means that, if the audio files are present, the program will
still use them even if the option is False.
This option is just about downloading the set.
"""
