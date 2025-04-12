import os
import platform
import subprocess

import numpy as np
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
        self.__check_sw_dependencies()

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

    def __getitem__(self, index: int) -> tuple[MidiFile, tuple[int, np.ndarray]]:
        """
        Returns the index-th song in the dataset

        ---------------------------------------------------------------------
        PARAMETERS
        ----------
        - index: the index to fetch

        ---------------------------------------------------------------------
        OUTPUT
        ------
        The song, as tuple composed of:
        - midi pattern
        - audio file, represented as a tuple with:
            - the sample rate
            - the actual data, as a np array
        """
        song = Song.from_path(self.__data[index])

        crop_region = song.choose_cut_boundary(self.__duration)
        if crop_region is not None:
            song = song.cut(crop_region[0], crop_region[1])

        return song.get_midi(), song.to_wav()

    def __check_sw_dependencies(self) -> None:
        """
        Checks the various dependencies based on the operating system.

        ---
        For both Linux and Windows, it checks that the fluidsynth command is
        present. If not, the program crashes, instructing the user to install
        the required software, by prompting the correct command for the
        installation (sudo apt/choco install fluidsynth).

        Moreover, on Windows it also checks for the command to work, since it
        sometimes does need an extra library. If the command is present but
        not working, the program crashes, explaining how to solve that.

        On MacOS, the program just prints a warning that the program is not
        tested there
        """
        if platform.system() == "Windows":
            exit_code, _ = subprocess.getstatusoutput("where fluidsynth")
            if exit_code != 0:
                print("ERROR: the synthesizer is not installed")
                print('Please install it via "sudo choco install fluidsynth"')
                exit(-1)

            exit_code, _ = subprocess.getstatusoutput("fluidsynth -h")
            if exit_code != 0:
                print('ERROR: the synthesizer is likely missing the "SDL3.dll" library')
                print(
                    'Please download it from here: "https://github.com/libsdl-org/SDL/releases/tag/release-3.2.10"'
                )
                print(
                    "And copy it to the fluidsynth folder (by default, C:\\ProgramData\\chocolatey\\bin\\fluidsynth.exe)"
                )
                exit(-1)

            return

        if platform.system() == "Darwin":
            print("WARNING: the synthesizer was not tested on MacOS")
            return

        if platform.system() == "Linux":
            exit_code, _ = subprocess.getstatusoutput("fluidsynth -h")
            if exit_code != 0:
                print("ERROR: the synthesizer is not installed")
                print('Please install it via "sudo apt install fluidsynth"')
                exit(-1)

            return
