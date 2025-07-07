from typing import Callable

import torch

from dataloader.Song import VALID_FIELDS_PER_MSG_TYPE


def np_midi_loss(
    pred_midi: torch.Tensor,
    pred_len: torch.Tensor,
    target_midi: torch.Tensor,
    target_len: torch.Tensor,
) -> torch.Tensor:
    target_len = target_len.to(torch.int32)
    target_midi = target_midi.to(torch.int32)

    max_msg_per_input = torch.max(pred_len, target_len)
    min_msg_per_input = torch.min(pred_len, target_len)

    my_sigmoid: Callable[[torch.Tensor], torch.Tensor] = (
        lambda x: 2 / (1 + torch.exp(-0.5 * x)) - 1
    )

    # Start by penalizing for missing or extra midi messages
    # Each message added or missing is considered as an error of 600
    loss_wrong_num_msg = 600 * torch.sum(max_msg_per_input - min_msg_per_input)

    # Create a mask to identify the valid messages for each song
    mask = torch.zeros(size=pred_midi.shape[:-1], dtype=torch.bool)
    for input_num in range(target_len.shape[0]):
        valid_msgs = int(min_msg_per_input[input_num])
        mask[input_num, :valid_msgs] = True

    # All the following applies only to the cells of valid messages

    # Penalize putting a message at the wrong time
    tick_errors = torch.abs(target_midi[:, :, 0] - pred_midi[:, :, 0])
    loss_wrong_time = torch.sum(mask * tick_errors)

    # Penalize choosing the wrong message type
    msg_type_delta = torch.abs(target_midi[:, :, 1] - pred_midi[:, :, 1])
    loss_wrong_message = torch.sum(mask * my_sigmoid(msg_type_delta))

    # Penalize wrong message fields

    # For each message, which fields are used?
    fields_meaningful = torch.zeros(size=target_midi.shape, dtype=torch.bool)
    for msgtype in range(1, 8):
        fields_meaningful[torch.where(target_midi[:, :, 1] == msgtype)] = (
            VALID_FIELDS_PER_MSG_TYPE[msgtype]
        )
    fields_meaningful[:, :, :2] = False  # tick and message type already accounted for
    for i in range(6):
        # field is meaningful only if the whole message is meaningful
        fields_meaningful[:, :, i] *= mask

    field_deltas = torch.abs(target_midi - pred_midi)
    loss_field_values = torch.sum(field_deltas * fields_meaningful)

    return loss_wrong_num_msg + loss_wrong_time + loss_wrong_message + loss_field_values
