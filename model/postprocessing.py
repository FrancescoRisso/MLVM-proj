import numpy as np
import numpy.typing as npt
import pretty_midi  # type: ignore
import torch

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
    output_path: str = "output.mid",
):
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
    
    midi = posteriorgrams_to_midi(yo_np, yp_np, yn_np, threshold=0.5, frame_rate=100)  # type: ignore

    return midi
