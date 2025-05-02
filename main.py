import pretty_midi
import librosa
import numpy as np
import torch
import matplotlib.pyplot as plt


def midi_to_yo_yn(midi_path, sr=22050, hop_length=512, fmin=librosa.note_to_hz("C1"), n_bins_yn=88):
    midi = pretty_midi.PrettyMIDI(midi_path)
    duration = min(midi.get_end_time(), 60.0)
    n_frames = int(np.ceil(duration * sr / hop_length))

    freqs_yn = librosa.cqt_frequencies(n_bins=n_bins_yn, fmin=fmin)

    yo = np.zeros((n_bins_yn, n_frames), dtype=np.float32)
    yn = np.zeros((n_bins_yn, n_frames), dtype=np.float32)
    yp = np.zeros((n_bins_yn, n_frames), dtype=np.float32)

    for instrument in midi.instruments:
        for note in instrument.notes:
            if note.start > 60.0:
                continue
            start_frame = int(note.start * sr / hop_length)
            end_frame = int(min(note.end, 60.0) * sr / hop_length)
            pitch_hz = librosa.midi_to_hz(note.pitch)
            bin_index = np.argmin(np.abs(freqs_yn - pitch_hz))

            if start_frame < n_frames:
                yo[bin_index, start_frame] = 1.0
            yn[bin_index, start_frame:end_frame] = 1.0
            yp[bin_index, start_frame:end_frame] = 1.0

    return torch.tensor(yo), torch.tensor(yn), torch.tensor(yp)



def save_midi_from_labels(yo, yp, yn, sr, hop_length, file_name="output_with_velocity.mid"):
    midi = pretty_midi.PrettyMIDI()
    freqs_yp = librosa.cqt_frequencies(n_bins=yp.shape[0], fmin=librosa.note_to_hz("C1"))
    instrument = pretty_midi.Instrument(program=0)

    for bin_idx in range(yo.shape[0]):
        frame_idx = 0
        while frame_idx < yo.shape[1]:
            if yo[bin_idx, frame_idx] == 1.0:
                start_time = frame_idx * hop_length / sr
                end_frame = frame_idx
                while end_frame < yn.shape[1] and yn[bin_idx, end_frame] == 1.0:
                    end_frame += 1
                end_time = end_frame * hop_length / sr

                if end_time - start_time < 0.01:
                    end_time = start_time + 0.1

                pitch_hz = freqs_yp[bin_idx]
                pitch_midi = int(round(librosa.hz_to_midi(pitch_hz)))
                velocity_value = 100

                note = pretty_midi.Note(
                    velocity=velocity_value, pitch=pitch_midi, start=start_time, end=end_time
                )
                instrument.notes.append(note)

                frame_idx = end_frame
            else:
                frame_idx += 1

    midi.instruments.append(instrument)
    midi.write(file_name)


def plot_matrices(yo, yp, yn):
    plt.figure(figsize=(12, 10))
    plt.subplot(3, 1, 1)
    plt.imshow(yo.numpy(), aspect='auto', cmap='viridis')
    plt.title("Yo - Onsets")
    plt.colorbar()

    plt.subplot(3, 1, 2)
    plt.imshow(yp.numpy(), aspect='auto', cmap='magma')
    plt.title("Yp - Pitch Durations")
    plt.colorbar()

    plt.subplot(3, 1, 3)
    plt.imshow(yn.numpy(), aspect='auto', cmap='inferno')
    plt.title("Yn - Note Durations")
    plt.colorbar()

    plt.tight_layout()
    plt.show()


def main():
    midi_path = "test.midi"
    sr = 22050
    hop_length = 512

    yo, yn, yp = midi_to_yo_yn(midi_path, sr, hop_length)

    print(f"yo sum: {yo.sum()}, yp sum: {yp.sum()}, yn sum: {yn.sum()}")

    plot_matrices(yo, yp, yn)

    save_midi_from_labels(yo, yp, yn, sr, hop_length, file_name="output_with_velocity.mid")
    print("Saved as output_with_velocity.mid")


if __name__ == "__main__":
    main()
