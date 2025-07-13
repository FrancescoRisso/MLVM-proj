import numpy as np
import numpy.typing as npt
import pretty_midi  # type: ignore
import torch
from settings import Settings as s
from settings import Model
from model.model import HarmonicCNN
import torchaudio

"""
    Converts onset, pitch, and note posteriorgrams to a MIDI file.

    Args:
        Yo (np.ndarray): Onset posteriorgram (time x pitch).
        Yp (np.ndarray): Pitch posteriorgram (time x pitch).
        Yn (np.ndarray): Note posteriorgram (time x pitch).
        threshold (float): Threshold for activation.
        frame_rate (float): Frames per second.
        velocity (int): MIDI velocity for notes.
        return_path (bool): Whether to return the path instead of the MIDI object.
        output_path (str): File path to save the MIDI file.

    Returns:
        str or pretty_midi.PrettyMIDI: MIDI file path or object.
"""


def posteriorgrams_to_midi(
    Yo: npt.NDArray[np.float32],
    Yp: npt.NDArray[np.float32],
    Yn: npt.NDArray[np.float32],
    threshold: float = 0.5,
    frame_rate: int = 100,
    velocity: int = 100,
    return_path: bool | str = False,
    output_path: str = "trial_audio\output.mid",
):
    # If input is 3D (B, T, P), take first batch
    if Yo.ndim == 3:
        Yo = Yo[0]
    if Yp.ndim == 3:
        Yp = Yp[0]
    if Yn.ndim == 3:
        Yn = Yn[0]

    # Apply thresholding to the posteriorgrams
    onsets = Yo > threshold
    pitches = Yp > threshold
    notes = Yn > threshold

    time_per_frame = 1.0 / frame_rate
    note_events = []
    num_frames, num_pitches = Yo.shape
    # Iterate through the frames and pitches to find note events
    for t in range(num_frames):
        for pitch in range(num_pitches):
            if onsets[t, pitch] and pitches[t, pitch]:
                # Track how long the note is active
                duration = 0
                for dt in range(t, num_frames):
                    if notes[dt, pitch]:
                        duration += 1
                    else:
                        break
                if duration > 0:
                    start_time = t * time_per_frame
                    end_time = (t + duration) * time_per_frame
                    note_events.append(  # type: ignore
                        (pitch + 21, start_time, end_time)
                    )  # MIDI pitch offset (21 = A0)

    # Create MIDI
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
    for pitch, start, end in note_events:  # type: ignore
        midi_note = pretty_midi.Note(
            velocity=velocity, pitch=pitch, start=start, end=end  # type: ignore
        )
        instrument.notes.append(midi_note)  # type: ignore
    midi.instruments.append(instrument)  # type: ignore

    if return_path:
        midi.write(output_path)  # type: ignore
        return output_path
    else:
        return midi


def postprocess(yo: torch.Tensor, yp: torch.Tensor, yn: torch.Tensor):
    yo_np, yp_np, yn_np = [  # type: ignore
        x.squeeze().detach().cpu().numpy() for x in (yo, yp, yn)  # type: ignore
    ]  # Remove batch dimension
    
    midi = posteriorgrams_to_midi(yo_np, yp_np, yn_np, threshold=0.5, frame_rate=100, return_path=True)  # type: ignore

    return midi

def model_eval(
        model_path: str | None, audio_path: str
) -> pretty_midi.PrettyMIDI | str:
    
    device = s.device
    print(f"Evaluating on {device}")

    if s.model == Model.RNN:
        return
        
    if model_path is not None:
        model = HarmonicCNN()
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise ValueError("No model found insert a valid model")
    
    model.eval()

    audio = torchaudio.load(audio_path)[0]  # Load audio file

    with torch.no_grad():
        # Assume audio is already preprocessed and ready for model input
        yo_pred, yp_pred, yn_pred = model(audio)
        

        yo_pred = torch.sigmoid(yo_pred)
        yo_pred = yo_pred.squeeze(1)
        yp_pred = yp_pred.squeeze(1)
        yn_pred = yn_pred.squeeze(1) if s.remove_yn == False else yp_pred

        midi = postprocess(yo_pred, yp_pred, yn_pred)
        return midi
