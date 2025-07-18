import os
import platform
import subprocess

import numpy as np
import numpy.typing as npt
from torch.utils.data.dataset import Dataset as TorchDataset

from dataloader.dataset_folder_management import download_dataset, is_dataset_ok
from dataloader.Song import Song
from dataloader.split import Split
from settings import Settings


class DataSet(
    TorchDataset[
        tuple[
            tuple[npt.NDArray[np.uint16], int, int, int],
            npt.NDArray[np.float32],
        ]
    ]
):
    def __init__(self, split: Split, duration: None | int | tuple[int, int]):
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

        self.__single_audio = split == Split.SINGLE_AUDIO
        if self.__single_audio:
            split = Split.VALIDATION

        folder_path = os.path.join(Settings.dataset_folder, str(split.value), "midi")
        self.__data = [
            os.path.join(folder_path, file) for file in os.listdir(folder_path)
        ]

        if Settings.always_same_portion and self.__duration is not None:
            self.__cuts: list[tuple[float, float] | None] | None = []

            for path in self.__data:
                song = Song.from_path(path)
                self.__cuts.append(song.choose_cut_boundary(self.__duration))

        else:
            self.__cuts = None

        if self.__single_audio:
            self.__data = [self.__data[0]]

    def __len__(self) -> int:
        """
        Returns the number of songs in the dataset

        ---------------------------------------------------------------------
        OUTPUT
        ------
        The number of songs
        """
        return len(self.__data)

    def __getitem__(
        self, index: int
    ) -> tuple[tuple[npt.NDArray[np.uint16], int, int, int], npt.NDArray[np.float32]]:
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
        - the midi pattern, described as a tuple of:
            - the messages, in a custom numpy format (recreate the real midi
                via Song.from_np(...))
            - the tempo of the song
            - the number of ticks per beat
            - the number of messages in the song
        - audio file, as loaded from librosa
        """
        wav_path = (
            self.__data[index].replace("midi", "wav")
            if Settings.generate_audio_on_download
            else None
        )
        song = Song.from_path(self.__data[index], wav_path=wav_path)

        if self.__duration is not None:

            if self.__cuts is not None:
                crop_region = self.__cuts[index]
            else:
                crop_region = song.choose_cut_boundary(self.__duration)

            if crop_region is not None:
                if self.__single_audio:
                    crop_region = (0, crop_region[1] - crop_region[0])

                song = song.cut(crop_region[0], crop_region[1])

            if Settings.generate_audio_on_download and crop_region is not None:
                return song.to_np(), song.load_cut_wav(*crop_region)

        return song.to_np(), song.to_wav()

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
