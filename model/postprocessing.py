import numpy as np
import numpy.typing as npt
import pretty_midi
import matplotlib.pyplot as plt
import torch
import torchaudio
from model.model import HarmonicCNN
from settings import Model
from settings import Settings as s


def posteriorgrams_to_midi(
    Yo: npt.NDArray[np.float32],
    Yp: npt.NDArray[np.float32],
    Yn: npt.NDArray[np.float32],
    threshold: float = 0.4,
    frame_rate: float | None = None,
    velocity: int = 100,
    return_path: bool | str = False,
    output_path: str = "trial_audio/output.mid",
    audio_duration: float | None = None,
    min_duration: float = 0.03,
    debug_visual: bool = True
):
    # Remove batch dimension if present
    if Yo.ndim == 3:
        Yo = Yo[0]
    if Yp.ndim == 3:
        Yp = Yp[0]
    if Yn.ndim == 3:
        Yn = Yn[0]

    # Restrict pitch range to first 88 bins (like piano)
    max_pitch_bins = min(Yo.shape[1], 88)
    Yo = Yo[:, :max_pitch_bins]
    Yp = Yp[:, :max_pitch_bins]
    Yn = Yn[:, :max_pitch_bins]

    onsets = Yo > threshold
    pitches = Yp > threshold
    notes = Yn > threshold

    print(f"Yo shape: {Yo.shape}, max: {Yo.max():.4f}")
    print(f"Yp shape: {Yp.shape}, max: {Yp.max():.4f}")
    print(f"Yn shape: {Yn.shape}, max: {Yn.max():.4f}")
    print("Onset activations:", np.sum(onsets))
    print("Pitch activations:", np.sum(pitches))
    print("Note activations:", np.sum(notes))

    if debug_visual:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
        axs[0].imshow(Yo.T, aspect='auto', origin='lower', cmap='hot')
        axs[0].set_title("Onset Posteriorgram")
        axs[1].imshow(Yp.T, aspect='auto', origin='lower', cmap='hot')
        axs[1].set_title("Pitch Posteriorgram")
        axs[2].imshow(Yn.T, aspect='auto', origin='lower', cmap='hot')
        axs[2].set_title("Note Posteriorgram")
        axs[2].set_xlabel("Frame Index")
        plt.tight_layout()
        plt.show()

    num_frames, num_pitches = Yo.shape

    if frame_rate is None:
        if audio_duration is None:
            raise ValueError("Provide either frame_rate or audio_duration.")
        frame_rate = num_frames / audio_duration

    time_per_frame = 1.0 / frame_rate
    note_events = []

    # Main note extraction loop
    for t in range(num_frames):
        for pitch in range(num_pitches):
            if onsets[t, pitch] and pitches[t, pitch]:
                start_time = t * time_per_frame
                end_time = start_time + time_per_frame
                for dt in range(t + 1, num_frames):
                    if not notes[dt, pitch]:
                        break
                    end_time = (dt + 1) * time_per_frame
                if end_time - start_time >= min_duration:
                    midi_pitch = pitch + 21
                    if 0 <= midi_pitch <= 127:
                        note_events.append((midi_pitch, start_time, end_time))

    print(f"Detected {len(note_events)} notes")

    # Create MIDI
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    for pitch, start, end in note_events:
        midi_note = pretty_midi.Note(
            velocity=min(int(velocity), 127),
            pitch=int(pitch),
            start=start,
            end=end
        )
        instrument.notes.append(midi_note)
    midi.instruments.append(instrument)

    if return_path:
        midi.write(output_path)
        return output_path
    else:
        return midi


def postprocess(yo, yp, yn, audio_length: int, sample_rate: int):
    yo_np, yp_np, yn_np = [x.squeeze(0).detach().cpu().numpy() for x in (yo, yp, yn)]
    duration_sec = audio_length / sample_rate
    midi = posteriorgrams_to_midi(
        yo_np, yp_np, yn_np,
        threshold=0.48,
        audio_duration=duration_sec,
        return_path=True,
        debug_visual=True
    )
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
    audio = audio.to(device)

    with torch.no_grad():
        # Assume audio is already preprocessed and ready for model input
        yo_pred, yp_pred, yn_pred = model(audio)
        
        yo_pred = torch.sigmoid(yo_pred)
        yp_pred = torch.sigmoid(yp_pred)
        yn_pred = torch.sigmoid(yn_pred) if s.remove_yn == False else yp_pred

        yo_pred = yo_pred.squeeze(1) 
        yp_pred = yp_pred.squeeze(1)
        yn_pred = yn_pred.squeeze(1) if s.remove_yn == False else yp_pred

        midi = postprocess(yo_pred, yp_pred, yn_pred, audio_length=audio.shape[-1], sample_rate=s.sample_rate)
        return midi
