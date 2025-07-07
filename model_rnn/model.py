import numpy as np
import torch
import torch.nn as nn

from settings import Settings


class HarmonicRNN(nn.Module):
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

    def forward(self, batched_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

        return midi, num_messages
