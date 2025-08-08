import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from dataloader.Song import Song
from model.HarmonicNet import HarmonicNet, HarmonicNetOutput
from settings import Settings
from train.rnn_losses import np_midi_loss
from train.utils import midi_to_label_matrices


class HarmonicRNN(HarmonicNet):
    def __init__(self):
        """
        Creates a RNN to perform the task of audio to midi conversion
        """
        super().__init__()  # type: ignore

        assert Settings.hidden_size % 2 == 0

        self.__encoder = nn.GRU(
            input_size=Settings.sample_rate,
            hidden_size=Settings.hidden_size // 2,
            num_layers=Settings.encoder_num_layers,
            batch_first=True,
            bidirectional=True,
            device=Settings.device,
        )

        self.__decoder = nn.GRU(
            input_size=0,
            hidden_size=Settings.hidden_size,
            num_layers=Settings.decoder_num_layers,
            batch_first=True,
            device=Settings.device,
        )

        self.__linear_output = nn.Linear(
            in_features=Settings.hidden_size,
            out_features=6,
            device=Settings.device,
        )

        self.__num_msg_generator = nn.Linear(
            in_features=Settings.hidden_size,
            out_features=1,
            device=Settings.device,
        )

        self.__ticks_per_beat_generator = nn.Linear(
            in_features=Settings.hidden_size,
            out_features=1,
            device=Settings.device,
        )

    def forward(
        self, batched_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given an audio file, uses the RNN to create its corresponding midi

        ---------------------------------------------------------------------
        PARAMETERS
        ----------
        - batched_input: the input wav to be converted into midi
            It must be a torch tensor\\[_b_, _t_, _s_], where:
            - _b_ is the index of the audio within the batch
            - _t_ is the time in seconds of a portion of the song
            - _s_ is the index of the sample within the second
            For example, batched_input[1, 2, 3] is the 4th data sample of the
            3rd second of the 2nd audio of the batch, which corresponds to
            the (2 \\* Settings.sample_rate + 3) overall sample for the song.

            In other words, the shape of the array must be
            (Settings.batch_size*, Settings.seconds, Settings.sample_rate)\\
            *or less, in case of incomplete batch.

        ---------------------------------------------------------------------
        OUTPUT
        ------
        A tuple of two tensors, that contain respectively:
        - a tensor that contains the np representations of the midis
        - a tensor that indicates how many midi messages compose each
            generated song
        """
        batch_size = batched_input.shape[0]

        _, hidden_states = self.__encoder(batched_input)
        hidden_states = hidden_states.reshape((-1, batch_size, Settings.hidden_size))

        ticks_per_beat = self.__ticks_per_beat_generator(hidden_states).flatten()

        num_messages = self.__num_msg_generator(hidden_states).flatten()
        midi = torch.tensor(
            np.empty(shape=(batch_size, Settings.max_midi_messages, 6)),
            dtype=torch.float32,
        )

        out, _ = self.__decoder.forward(
            torch.tensor(
                np.zeros(
                    shape=(batch_size, Settings.max_midi_messages, 0), dtype=np.float32
                )
            ),
            hidden_states,
        )

        shape = out.shape
        out = out.reshape((-1, Settings.hidden_size))
        midi = self.__linear_output(out).reshape((*shape[:-1], 6))

        return midi, num_messages, ticks_per_beat

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
    ) -> None:
        (midis_np, _, ticks_per_beats, nums_messages), audios = batch

        self.__input_audios = audios.reshape(
            (
                audios.shape[0],  # leave batch items untouched
                -1,  # all the seconds
                Settings.sample_rate,  # samples per secon
            )
        )

        self.__ground_truth_midi = torch.Tensor(midis_np)
        self.__ground_truth_values = {
            "ticks per beat": torch.Tensor(ticks_per_beats),
            "num messages": torch.Tensor(nums_messages),
        }

    def exec_forward(self) -> None:
        pred_midi, pred_len, pred_tpb = self.forward(torch.Tensor(self.__input_audios))

        self.__pred_midi = pred_midi
        self.__pred_values = {"ticks per beat": pred_tpb, "num messages": pred_len}

    def get_loss(self) -> torch.Tensor:
        return np_midi_loss(
            self.__pred_midi,
            self.__pred_values["num messages"],
            self.__pred_values["ticks per beat"],
            self.__ground_truth_midi,
            self.__ground_truth_values["num messages"],
            self.__ground_truth_values["ticks per beat"],
        )

    def __get_midi_and_images(
        self, original_midis: torch.Tensor
    ) -> list[HarmonicNetOutput]:
        midis = [
            Song.from_np(
                original_midis[i].to(torch.uint16).numpy(),  # type: ignore
                None,
                int(self.__pred_values["ticks per beat"][i]),
                int(self.__pred_values["num messages"][i]),
            ).get_midi()
            for i in range(len(original_midis))
        ]

        yos: list[torch.Tensor] = []
        yps: list[torch.Tensor] = []
        yns: list[torch.Tensor] = []

        for midi in midis:
            yo, yn = midi_to_label_matrices(
                midi, Settings.sample_rate, Settings.hop_length
            )

            yos.append(torch.Tensor(yo))
            yps.append(torch.Tensor(yn))
            yns.append(torch.Tensor(yn))

        return [
            HarmonicNetOutput(midi, yo, yp, yn)
            for midi, yo, yp, yn in zip(midis, yos, yps, yns)
        ]

    def get_network_input_internal(self) -> list[HarmonicNetOutput]:
        return self.__get_midi_and_images(self.__ground_truth_midi)

    def get_network_output_internal(self) -> list[HarmonicNetOutput]:
        return self.__get_midi_and_images(self.__pred_midi)
