import csv
import io
import os
import re
from shutil import rmtree
from zipfile import ZipFile

import requests

from settings import Settings


def download_dataset():
    """!
    @brief Downloads and sets up the dataset
    """
    _clean_folder()
    zip_folder = _download_zip()
    _extract_zip(zip_folder)
    _reorganize_dataset()


def _clean_folder() -> None:
    """
    Removes everyhing from the dataset folder
    """
    print(f"Cleaning folder {Settings.dataset_folder}... ", end="")

    if os.path.isdir(Settings.dataset_folder):
        rmtree(Settings.dataset_folder)

    print("Done!")


def _download_zip() -> ZipFile:
    """
    Downloads the dataset as a zipped file

    ---------------------------------------------------------------------
    OUTPUT
    ------
    The downloaded zip folder
    """
    print("Downloading dataset... ", end="")

    resp = requests.get(
        "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
    )

    print("Done!")

    return ZipFile(io.BytesIO(resp.content))


def _extract_zip(zip_folder: ZipFile) -> None:
    """
    Extracts the zipped folder to the dataset folder (in the original
    folder structure as downloaded)

    ---------------------------------------------------------------------
    PARAMETERS
    ----------
    - zip_folder: the downloaded zipped folder
    """
    print("Extracting zip folder... ", end="")

    zip_folder.extractall(Settings.dataset_folder)

    print("Done!")


def _normalize_str(s: str) -> str:
    """
    Replaces various characters in a string, to make it both
    "path-friendly" and in line with the rest of the file name.

    ---------------------------------------------------------------------
    In particular:
    - spaces after a dot are removed
    - one or more consecutive characters among the following are replaced
        by a single underscore:
        - \-
        - (space)
        - ,
        - ;
        - :
        - \\
        - /

    ---------------------------------------------------------------------
    PARAMETERS
    ----------
    - s: the string to convert

    ---------------------------------------------------------------------
    OUTPUT
    ------
    The converted string
    """
    # rm spaces after dot
    s = re.sub(r"\. *", ".", s)

    # (punctuation/spaces/dash/apostrophe/slashes) ==> (underscore)
    s = re.sub(r"[,;: -'\\/]*", "_", s)

    # rm quotes
    s = re.sub(r"[\"“”]", "", s)
    return s


def _move_song(
    song: dict[str, str], download_base_path: str, num_versions: dict[str, int]
) -> None:
    """
    Moves a song to the correct split folder, while also making the file
    name more meaningful

    ---------------------------------------------------------------------
    PARAMETERS
    ----------
    - song: the song details, formatted as a dict from the csv
    - download_base_path: the path where all songs have been downloaded
    - num_versions: a counter of how many instances were found for a
        specific (composer, title, year)
    """

    fname = _normalize_str(song["canonical_composer"])
    fname = f"{fname}-{_normalize_str(song["canonical_title"])}"
    fname = f"{fname}-{_normalize_str(song["year"])}"
    lowercase_fname = fname.lower()

    if lowercase_fname not in num_versions.keys():
        num_versions[lowercase_fname] = 1
    else:
        num_versions[lowercase_fname] += 1

    fname = f"{fname}-v{num_versions[lowercase_fname]}.midi"

    os.rename(
        os.path.join(download_base_path, song["midi_filename"]),
        os.path.join(Settings.dataset_folder, song["split"], fname),
    )


def _reorganize_dataset() -> None:
    """
    Transforms the dataset folder from the downloaded folder structure to
    the desired one.

    ---------------------------------------------------------------------
    In particular:
    - creates folders `./train`, `./validation` and `./test` inside the
        dataset folder
    - moves all the midi files to the corresponding folder, based on the
        split suggested by the dataset.
        Each file is also renamed in a more meaningful way:
        _\<composer\>-\<title\>-\<year\>-v<instance #>.midi_
    - moves all the metadata files to the dataset folder
    - removes the leftover folders from the downloaded structure
    """
    os.mkdir(os.path.join(Settings.dataset_folder, "train"))
    os.mkdir(os.path.join(Settings.dataset_folder, "validation"))
    os.mkdir(os.path.join(Settings.dataset_folder, "test"))

    download_base_path = os.path.join(Settings.dataset_folder, "maestro-v3.0.0")
    csv_path = os.path.join(download_base_path, "maestro-v3.0.0.csv")
    num_versions: dict[str, int] = {}

    with open(csv_path, "r", newline="", encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=",", quotechar='"')
        for song in csv_reader:
            _move_song(song, download_base_path, num_versions)

    for file in Settings.metadata_files_to_keep:
        os.rename(
            os.path.join(download_base_path, file),
            os.path.join(Settings.dataset_folder, file),
        )

    rmtree(download_base_path)
