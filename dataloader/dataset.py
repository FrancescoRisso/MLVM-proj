import os

from mido import MidiFile  # type: ignore

from dataloader.dataset_folder_management import download_dataset, is_dataset_ok
from dataloader.Song import Song
from dataloader.split import Split
from settings import Settings


class DataSet:
    def __init__(self, split: Split, duration: None | float | tuple[float, float]):
        """
        Creates a new dataset for a specific split.

        No transformation is available, except for a random time crop

        ---------------------------------------------------------------------
        PARAMETERS
        ----------
        - split: which split the dataset should work on
        - duration: how long the crops should be, in one of the following
            formats:
            - None: don't crop the songs, return them fully every time
            - num: create random crops of exactly num seconds (equivalent to
                the previous one if num > song length)
            - (min, max): select every time a random duration between min and
                max, then select a random crop of that duration (max is
                cropped to the song length, and the option is equivalent to
                None if also min is greater than the song length)
        """

        if not is_dataset_ok():
            download_dataset()

        self.__duration = duration

        folder_path = os.path.join(Settings.dataset_folder, split.value)
        self.__data = [
            os.path.join(folder_path, file) for file in os.listdir(folder_path)
        ]

    def __len__(self) -> int:
        """
        Returns the number of songs in the dataset

        ---------------------------------------------------------------------
        OUTPUT
        ------
        The number of songs
        """
        return len(self.__data)

    def __getitem__(self, index: int) -> MidiFile:
        """
        Returns the index-th song in the dataset

        ---------------------------------------------------------------------
        PARAMETERS
        ----------
        - index: the index to fetch

        ---------------------------------------------------------------------
        OUTPUT
        ------
        The song, as midi pattern
        """
        song = Song.from_path(self.__data[index])

        crop_region = song.choose_cut_boundary(self.__duration)
        if crop_region is not None:
            song = song.cut(crop_region[0], crop_region[1])

        return song.get_midi()
