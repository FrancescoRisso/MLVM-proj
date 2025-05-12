import numpy as np
import torch
import torch.nn as nn

from dataloader.Song import END_OF_TRACK
from settings import Settings


class HarmonicRNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.__encoder = nn.GRU(
            input_size=Settings.sample_rate,
            hidden_size=Settings.hidden_size,
            num_layers=Settings.encoder_num_layers,
            batch_first=True,
            bidirectional=True,
            device=Settings.device,
        )

        self.__decoder = nn.GRU(
            input_size=6,
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

    def forward(self, batched_input: np.ndarray):
        _, hidden_states = self.__encoder(torch.tensor(batched_input))

        batch_size = batched_input.shape[0]

        num_messages = torch.tensor(np.empty(shape=(batch_size,)), dtype=torch.uint32)
        midi = torch.tensor(
            np.empty(shape=(batch_size, Settings.max_midi_messages, 6)),
            dtype=torch.float32,
        )

        for batch_item in range(batch_size):
            _, hidden = self.__decoder.forward(
                torch.tensor(np.zeros(shape=(1, 6), dtype=np.float32)),
                hidden_states[batch_item],
            )

            midi[batch_item, 0] = self.__linear_output(hidden)
            last_midi_msg = midi[batch_item, 0]
            messages = 1

            while (
                round(float(last_midi_msg[0])) != END_OF_TRACK
                and messages < Settings.max_midi_messages - 1
            ):
                _, hidden = self.__decoder(last_midi_msg.reshape(1, -1), hidden)
                messages += 1
                midi[batch_item, messages] = self.__linear_output(hidden)
                last_midi_msg = midi[batch_item, messages]

            num_messages[batch_item] = messages + 1

        return midi, num_messages
