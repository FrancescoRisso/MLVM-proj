import dataloader.dataset as dataset
from dataloader.split import Split


class Dataloader:
    def __init__(self, split: Split):
        """
        Creates a new dataloader for a specific dataset split

        ---------------------------------------------------------------------
        PARAMETERS
        ----------
        - split: which split the dataloader should work on
        """

        if not dataset.is_dataset_ok():
            dataset.download_dataset()

