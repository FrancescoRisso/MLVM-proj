from dataloader.dataset import DataSet
from dataloader.split import Split
from model.postprocessing import postprocess

# WiP file


def main():
    dataset = DataSet(Split.TRAIN, (15, 20))
    midi, (sample_rate, audio) = dataset[-1]

    out_midi = postprocess()    


if __name__ == "__main__":
    main()
