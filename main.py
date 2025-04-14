from dataloader.dataset import DataSet
from dataloader.split import Split
from model.preprocessing import constant_q_transform, harmonic_stacking
from model import model

# WiP file


def main():
    dataset = DataSet(Split.TRAIN, (15, 20))
    midi, (sample_rate, audio) = dataset[-1]

    """
    # Hyperparameters
    wav_path = ""       # TODO: Pick paths from dataset
    sr = 16000
    hop_length = 256
    n_bins = 84
    harmonic_shifts = [-12, 0, 12]

    # Preprocessing
    cqt = constant_q_transform(wav_path, sr, hop_length, n_bins)
    stacked = harmonic_stacking(cqt, harmonic_shifts)

    # Modello
    output = model(stacked)
    """


if __name__ == "__main__":
    main()
