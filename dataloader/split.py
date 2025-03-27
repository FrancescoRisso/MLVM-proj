from enum import Enum

from settings import Settings


class Split(Enum):
    TRAIN = Settings.train_folder
    VALIDATION = Settings.validation_folder
    TEST = Settings.test_folder

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))
