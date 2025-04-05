from dataloader.dataset import DataSet
from dataloader.split import Split

# WiP file


def main():
    dataset = DataSet(Split.TRAIN, (15, 20))
    midi, (sample_rate, audio) = dataset[-1]


if __name__ == "__main__":
    main()
