from enum import Enum

from settings import Settings


class Split(Enum):
    TRAIN = Settings.train_folder
    VALIDATION = Settings.validation_folder
    TEST = Settings.test_folder
    SINGLE_AUDIO = 0

    @classmethod
    def list(cls):
        return [val.value for val in cls if val != Split.SINGLE_AUDIO]
