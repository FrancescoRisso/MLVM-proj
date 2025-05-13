from dataloader.dataset import DataSet
from dataloader.split import Split
from model.model import HarmonicCNN
from settings import Settings


# WiP file


def main():
    dataset = DataSet(Split.TRAIN, Settings.seconds)
    midi, audio = dataset[-1]

    net = HarmonicCNN().to(Settings.device)
    (yo, yp, yn), out_midi = net(audio)


if __name__ == "__main__":
    main()
