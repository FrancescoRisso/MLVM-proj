import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from dataloader.Song import Song
from model.HarmonicNet import HarmonicNet, HarmonicNetOutput
from model.postprocessing import posteriorgrams_to_midi
from model.preprocessing import preprocess
from settings import Settings as s
from train.losses import harmoniccnn_loss
from train.utils import midi_to_label_matrices, to_numpy, to_tensor


# ---------- CNN Blocks ----------
def get_conv_net(
    channels: list[int], ks: tuple[int, int], s: tuple[int, int], p: tuple[int, int]
):
    layers: list[nn.Module] = []
    for i in range(len(channels) - 1):
        layers.append(
            nn.Conv2d(channels[i], channels[i + 1], kernel_size=ks, stride=s, padding=p)
        )
        layers.append(nn.BatchNorm2d(channels[i + 1]))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


# ---------- Full Model ----------
class HarmonicCNN(HarmonicNet):
    def __init__(self):
        super().__init__()  # type: ignore

        # Yp branch
        self.block_a1 = get_conv_net([3, 16], ks=(5, 5), s=(1, 1), p=(2, 2))
        self.block_a2 = get_conv_net([16, 8], ks=(3, 39), s=(1, 1), p=(1, 19))
        self.conv_a3 = nn.Conv2d(8, 1, kernel_size=(5, 5), padding=(2, 2))

        # Yn branch (uses yp as input)
        self.conv_c1 = nn.Conv2d(
            1, 32, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3)
        )
        self.relu_c2 = nn.ReLU()
        self.conv_c3 = nn.Conv2d(32, 1, kernel_size=(7, 3), padding=(3, 1))

        # Yo branch (uses xb + yn as input)
        self.block_b1 = get_conv_net([3, 32], ks=(5, 5), s=(1, 1), p=(2, 2))
        self.conv_b2 = nn.Conv2d(33, 1, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x: torch.Tensor):
        x = preprocess(x)

        # --- Yp branch ---
        xa = self.block_a1(x)
        xa = self.block_a2(xa)
        yp_logits = self.conv_a3(xa)  # logits
        yp = torch.sigmoid(yp_logits)  # only for internal use

        # --- Yn and Yo branch ---
        xb = self.block_b1(x)

        if not s.remove_yn:
            # --- Yn branch ---
            yn_logits = self.conv_c3(self.relu_c2(self.conv_c1(yp)))  # logits
            yn = torch.sigmoid(yn_logits)  # solo per yo

            # --- Yo branch ---
            concat = torch.cat([xb, yn], dim=1)  # uses "activated" yn output
        else:
            # --- Yn branch ---
            yn_logits = None

            # --- Yo branch ---
            concat = torch.cat([xb, yp], dim=1)

        yo_logits = self.conv_b2(concat)  # logits

        return (
            yo_logits,
            yp_logits,  # return yp_logits instead of yn_logits
            yn_logits,
        )  # postprocessing(Y0, YN) TODO da usare sigmoide anche prima

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
        (midis_np, tempos, ticks_per_beats, nums_messages), audios = batch

        yo_true_batch: list[torch.Tensor] = []
        yn_true_batch: list[torch.Tensor] = []
        yp_true_batch: list[torch.Tensor] = []

        audio_input_batch: list[torch.Tensor] = []

        for i in range(midis_np.shape[0]):
            midi = Song.from_np(
                midis_np[i], tempos[i], ticks_per_beats[i], nums_messages[i]
            ).get_midi()
            yo, yp = midi_to_label_matrices(
                midi, s.sample_rate, s.hop_length, n_bins=88
            )
            yn = yp  # In this case, yp is the same as yn

            yo_true_batch.append(to_tensor(yo).to(device))
            yp_true_batch.append(to_tensor(yp).to(device))
            if not s.remove_yn:
                yn_true_batch.append(to_tensor(yn).to(device))

            audio_input_batch.append(audios[i].to(device))

        self.__audio_input = torch.stack(audio_input_batch)
        self.__ground_truth: dict[str, torch.Tensor | None] = {
            "yo": torch.stack(yo_true_batch),
            "yp": torch.stack(yp_true_batch),
            "yn": None if s.remove_yn else torch.stack(yn_true_batch),
        }

    def exec_forward(self) -> None:
        yo_pred, yp_pred, yn_pred = self.forward(self.__audio_input)

        assert isinstance(yo_pred, torch.Tensor)
        assert isinstance(yp_pred, torch.Tensor) or yp_pred is None
        assert isinstance(yn_pred, torch.Tensor)

        self.__output: dict[str, torch.Tensor] = {
            "yo": yo_pred,
            "yp": yp_pred if yp_pred is not None else yn_pred,
            "yn": yn_pred,
        }

    def get_loss(self) -> torch.Tensor:
        yp_pred, yo_pred, yn_pred = (
            self.__output["yp"],
            self.__output["yo"],
            self.__output["yn"],
        )

        assert yo_pred is not None and yp_pred is not None
        assert self.__ground_truth["yo"] is not None
        assert self.__ground_truth["yp"] is not None

        yo_pred = yo_pred.squeeze(1)
        yp_pred = yp_pred.squeeze(1)
        if not s.remove_yn:
            assert yn_pred is not None
            yn_pred = yn_pred.squeeze(1)

        loss = harmoniccnn_loss(
            yo_pred,  # yo_logits
            yp_pred,  # yp_logits
            self.__ground_truth["yo"],  # yo_true
            self.__ground_truth["yp"],  # yp_true
            yn_pred,  # yn_logits (opzionale)
            self.__ground_truth["yn"],  # yn_true (opzionale)
            label_smoothing=s.label_smoothing,
            weighted=s.weighted,
        )

        return sum(loss.values())  # type: ignore

    def __get_midi_and_images(
        self, images: dict[str, torch.Tensor]
    ) -> list[HarmonicNetOutput]:
        return [
            HarmonicNetOutput(
                posteriorgrams_to_midi(
                    to_numpy(yo[0]),  # type: ignore
                    to_numpy(yp[0]),  # type: ignore
                    to_numpy(yn[0] if yn is not None else yp[0]),  # type: ignore
                    velocity=100,  # type: ignore
                    frame_rate=s.sample_rate / s.hop_length,
                ),
                yo,
                yp,
                yn,
            )
            for yo, yp, yn in zip(images["yo"], images["yp"], images["yn"])
        ]

    def get_network_output_internal(self) -> list[HarmonicNetOutput]:
        return self.__get_midi_and_images(self.__output)

    def get_network_input_internal(self) -> list[HarmonicNetOutput]:
        self.__ground_truth["yn"] = (
            self.__ground_truth["yn"]
            if self.__ground_truth["yn"] is not None
            else self.__ground_truth["yp"]
        )

        return self.__get_midi_and_images({k: v.unsqueeze(0) for k, v in self.__ground_truth.items()})  # type: ignore
