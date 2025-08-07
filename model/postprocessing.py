import numpy as np
import numpy.typing as npt
import pretty_midi
from settings import Settings as s


def posteriorgrams_to_midi(
    Yo: npt.NDArray[np.float32],
    Yp: npt.NDArray[np.float32],
    Yn: npt.NDArray[np.float32],
    frame_rate: float | None = None,
    velocity: int = 100,
    audio_duration: float | None = None,
    debug: bool = False
):

    # Restrict pitch range to first 88 bins (like piano)
    max_pitch_bins = min(Yo.shape[0], 88)
    Yo = Yo[:max_pitch_bins,:]
    Yp = Yp[:max_pitch_bins,:]
    Yn = Yn[:max_pitch_bins,:]

    onsets = Yo > s.threshold
    pitches = Yp > s.threshold
    notes = Yn > s.threshold

    if debug:
        import matplotlib.pyplot as plt

        print(f"Yo shape: {Yo.shape}, max: {Yo.max():.4f}")
        print(f"Yp shape: {Yp.shape}, max: {Yp.max():.4f}")
        print(f"Yn shape: {Yn.shape}, max: {Yn.max():.4f}")
        print("Onset activations:", np.sum(onsets))
        print("Pitch activations:", np.sum(pitches))
        print("Note activations:", np.sum(notes))

        plt.imshow(Yp, aspect='auto', origin='lower', cmap='hot')
        plt.title("Pitch Posteriorgram (Yp)")
        plt.xlabel("Frame Index")
        plt.ylabel("Pitch Index")
        plt.colorbar()
        plt.show()

    num_pitches, num_frames = Yo.shape

    if frame_rate is None:
        if audio_duration is None:
            raise ValueError("Provide either frame_rate or audio_duration.")
        frame_rate = num_frames / audio_duration

    time_per_frame = 1.0 / frame_rate
    min_duration = time_per_frame * 0.99
    note_events = []

    # Main note extraction loop
    for t in range(num_frames):
        for pitch in range(num_pitches):
            if pitches[pitch, t] and (t == 0 or not pitches[pitch, t - 1]):
                start_time = t * time_per_frame
                end_time = start_time + time_per_frame
                for dt in range(t + 1, num_frames):
                    if not pitches[pitch, dt]:
                        break
                    end_time = (dt + 1) * time_per_frame
                if end_time - start_time >= min_duration:
                    midi_pitch = pitch + 21
                    if 0 <= midi_pitch <= 127:
                        note_events.append((midi_pitch, start_time, end_time))

    if debug:
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

    return midi

def postprocess(yo, yp, yn, audio_length: int, sample_rate: int):
    yo_np, yp_np, yn_np = [x.squeeze(0).detach().cpu().numpy() for x in (yo, yp, yn)]
    duration_sec = audio_length / sample_rate
    midi = posteriorgrams_to_midi(
        yo_np, yp_np, yn_np,
        audio_duration=duration_sec,
        debug=False
    )
    return midi
