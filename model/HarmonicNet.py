from __future__ import annotations

from abc import ABC as Abstract
from abc import abstractmethod
from typing import NamedTuple, Optional

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from mido import MidiFile  # type: ignore
from pretty_midi import PrettyMIDI  # type: ignore

from train.quality_index import evaluate_note_prediction


class HarmonicNetOutput(NamedTuple):
    midi: MidiFile | PrettyMIDI
    yo: torch.Tensor
    yp: torch.Tensor
    yn: torch.Tensor | None


class Statistics(NamedTuple):
    TP: int
    FP: int
    FN: int

    @property
    def precision(self) -> float:
        return self.TP / (self.TP + self.FP + 1e-9)

    @property
    def recall(self) -> float:
        return self.TP / (self.TP + self.FN + 1e-9)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r + 1e-9)

    def __add__(self, other: object) -> Statistics:
        assert isinstance(other, Statistics)
        return Statistics(self.TP + other.TP, self.FP + other.FP, self.FN + other.FN)


class HarmonicNet(Abstract, nn.Module):
    def __init__(self, *args, **kwargs) -> None:  # type: ignore
        super().__init__(*args, **kwargs)  # type: ignore
        self.__cached_input: Optional[list[HarmonicNetOutput]] = None
        self.__cached_output: Optional[list[HarmonicNetOutput]] = None

    def set_input(
        self,
        batch: tuple[
            tuple[
                npt.NDArray[np.uint16],  # Midis in np format
                npt.NDArray[np.int32],  # Tempos
                npt.NDArray[np.int32],  # Ticks per beat
                npt.NDArray[np.int32],  # Number of messages
            ],
            npt.NDArray[np.float32],  # Audios
        ],
        device: torch.device,
    ) -> None:
        self.__cached_input = self.__cached_output = None
        self.set_input_internal(batch, device)

    @abstractmethod
    def set_input_internal(
        self,
        batch: tuple[
            tuple[
                npt.NDArray[np.uint16],  # Midis in np format
                npt.NDArray[np.int32],  # Tempos
                npt.NDArray[np.int32],  # Ticks per beat
                npt.NDArray[np.int32],  # Number of messages
            ],
            npt.NDArray[np.float32],  # Audios
        ],
        device: torch.device,
    ) -> None: ...

    @abstractmethod
    def exec_forward(self) -> None: ...

    @abstractmethod
    def get_loss(self) -> torch.Tensor: ...

    def get_network_output(self) -> list[HarmonicNetOutput]:
        if self.__cached_output is None:
            self.__cached_output = self.get_network_output_internal()
        return self.__cached_output

    def get_network_input(self) -> list[HarmonicNetOutput]:
        if self.__cached_input is None:
            self.__cached_input = self.get_network_input_internal()
        return self.__cached_input

    @abstractmethod
    def get_network_output_internal(self) -> list[HarmonicNetOutput]: ...

    @abstractmethod
    def get_network_input_internal(self) -> list[HarmonicNetOutput]: ...

    def get_separate_statistics(self) -> list[Statistics]:
        stats: list[Statistics] = []

        out_batch = self.get_network_output()
        gt_batch = self.get_network_input()

        for gt, pred in zip(gt_batch, out_batch):
            gt_yo = gt.yo
            gt_yp = gt.yp
            gt_yn = (gt.yn if gt.yn is not None else gt.yp)

            gt_yo_squeezed = torch.zeros_like(pred.yo)
            gt_yp_squeezed = torch.zeros_like(pred.yp)
            gt_yn_squeezed = torch.zeros_like(pred.yn) if pred.yn is not None else torch.zeros_like(pred.yp)

            for i in range(gt_yo.shape[0]):
                gt_yo_squeezed += gt_yo[i,:,:]
                gt_yp_squeezed += gt_yp[i,:,:]
                gt_yn_squeezed += gt_yn[i,:,:]

            eval = evaluate_note_prediction(
                gt_yo_squeezed.unsqueeze(0),
                gt_yp_squeezed.unsqueeze(0),
                gt_yn_squeezed.unsqueeze(0),
                pred.yo.unsqueeze(0),
                pred.yp.unsqueeze(0),
                (pred.yn if pred.yn is not None else pred.yp).unsqueeze(0),
            )

            stats.append(Statistics(int(eval["TP"]), int(eval["FP"]), int(eval["FN"])))

        return stats

    def get_batch_statistics(self) -> Statistics:
        cumul = Statistics(0, 0, 0)
        for stat in self.get_separate_statistics():
            cumul += stat
        return cumul
