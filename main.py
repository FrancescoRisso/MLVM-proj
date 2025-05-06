import mido
import librosa
import numpy as np
import torch
import pretty_midi
import matplotlib.pyplot as plt


from train.train import train


def midi_to_yo_yn(midi_path, sr=22050, hop_length=512, n_bins_yn=88):
    # Carica il file MIDI con mido
    mido_midi = mido.MidiFile(midi_path)
    ticks_per_beat = mido_midi.ticks_per_beat
    min_pitch = 21  # Pitch corrispondente a "A0"
    max_pitch = min_pitch + n_bins_yn

    current_tempo = 500000  # Tempo di default (120bpm)
    tick_acc = 0
    time_sec = 0.0

    active_notes = {}
    notes = []

    # Estrazione delle note dal file MIDI
    for msg in mido.merge_tracks(mido_midi.tracks):
        tick_acc += msg.time
        if msg.type == 'set_tempo':
            current_tempo = msg.tempo
        time_sec = mido.tick2second(tick_acc, ticks_per_beat, current_tempo)

        if msg.type == 'note_on' and msg.velocity > 0:
            active_notes[msg.note] = time_sec
        elif msg.type in ['note_off', 'note_on'] and msg.velocity == 0:
            start = active_notes.pop(msg.note, None)
            if start is not None and min_pitch <= msg.note < max_pitch:
                notes.append((msg.note, start, time_sec))

    # Calcola la durata massima
    max_time = max((end for _, _, end in notes), default=0.0)
    n_frames = int(np.ceil(max_time * sr / hop_length))

    # Matrici per le etichette
    yo = np.zeros((n_bins_yn, n_frames), dtype=np.float32)
    yn = np.zeros((n_bins_yn, n_frames), dtype=np.float32)

    # Compilazione delle matrici per onsets, durata delle note, e durata del pitch
    for pitch, start, end in notes:
        p = pitch - min_pitch
        start_frame = int(np.floor(start * sr / hop_length))
        end_frame = int(np.ceil(end * sr / hop_length))
        if start_frame < n_frames:
            yo[p, start_frame] = 1.0
        yn[p, start_frame:end_frame] = 1.0
        

    return torch.tensor(yo), torch.tensor(yn)

def save_midi_from_labels_using_yn(yn, sr, hop_length, file_name="reconstructed_from_yn.mid"):
    midi = pretty_midi.PrettyMIDI()
    # Frequenze che corrispondono a ciascun bin in yn
    freqs = librosa.cqt_frequencies(n_bins=yn.shape[0], fmin=librosa.note_to_hz("A0"))

    # Verifica delle frequenze
    print("Frequenze corrispondenti ai bin di yn:")
    print(freqs)

    # Strumento per creare il MIDI
    instrument = pretty_midi.Instrument(program=0)

    # Ricostruzione del MIDI usando solo la matrice yn
    for bin_idx in range(yn.shape[0]):
        active = False
        start_frame = 0
        for frame_idx in range(yn.shape[1]):
            if yn[bin_idx, frame_idx] == 1.0 and not active:
                active = True
                start_frame = frame_idx
            elif (yn[bin_idx, frame_idx] == 0.0 or frame_idx == yn.shape[1] - 1) and active:
                end_frame = frame_idx
                start_time = start_frame * hop_length / sr
                end_time = end_frame * hop_length / sr
                pitch_hz = freqs[bin_idx]
                pitch_midi = int(round(librosa.hz_to_midi(pitch_hz)))
                note = pretty_midi.Note(velocity=100, pitch=pitch_midi, start=start_time, end=end_time)
                instrument.notes.append(note)
                active = False

    midi.instruments.append(instrument)
    midi.write(file_name)

def plot_matrices(yo, yp, yn):
    # Funzione per visualizzare le matrici
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

def compare_midi_files(file1, file2, time_tolerance=1, pitch_tolerance=0):
    """
    Confronta due file MIDI per verificarne la similarità.
    
    Args:
    - file1 (str): Percorso del primo file MIDI.
    - file2 (str): Percorso del secondo file MIDI.
    - time_tolerance (float): Tolleranza per la differenza nei tempi di inizio e fine delle note (in secondi).
    - pitch_tolerance (float): Tolleranza per la differenza nel pitch delle note (in semitoni).
    
    Returns:
    - bool: True se i file sono simili entro le tolleranze specificate, False altrimenti.
    """
    # Carica i file MIDI
    midi1 = pretty_midi.PrettyMIDI(file1)
    midi2 = pretty_midi.PrettyMIDI(file2)
    
    # Estrai le note da entrambi i file
    notes1 = []
    for instrument in midi1.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                notes1.append((note.pitch, note.start, note.end))
    
    notes2 = []
    for instrument in midi2.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                notes2.append((note.pitch, note.start, note.end))
    
    # Ordina le note per pitch e start time
    notes1.sort(key=lambda x: (x[0], x[1]))
    notes2.sort(key=lambda x: (x[0], x[1]))
    
    # Confronta le note
    if len(notes1) != len(notes2):
        return False
    
    for note1, note2 in zip(notes1, notes2):
        pitch1, start1, end1 = note1
        pitch2, start2, end2 = note2
        
        # Confronta il pitch
        if abs(pitch1 - pitch2) > pitch_tolerance:
            return False
        
        # Confronta i tempi di inizio e fine
        if abs(start1 - start2) > time_tolerance or abs(end1 - end2) > time_tolerance:
            return False
    
    return True

def prova():
        
    # Esempio di utilizzo
    file1 = 'test.midi'
    file2 = 'reconstructed_from_yn.mid'


    if compare_midi_files(file1, file2):
        print("I file MIDI sono simili.")
    else:
        print("I file MIDI non sono simili.")

    midi_path = "test.midi"  # Percorso al file MIDI di input
    sr = 22050  # Frequenza di campionamento
    hop_length = 512  # Lunghezza della finestra per il calcolo

    # Conversione del MIDI in matrici
    yo, yn, yp = midi_to_yo_yn(midi_path, sr, hop_length)

    # Visualizzazione delle matrici
    print(f"yo sum: {yo.sum()}, yp sum: {yp.sum()}, yn sum: {yn.sum()}")
    plot_matrices(yo, yp, yn)

    # Salvataggio del MIDI ricostruito usando solo yn
    save_midi_from_labels_using_yn(yn, sr, hop_length, file_name="reconstructed_from_yn.mid")
    print("Saved as reconstructed_from_yn.mid")


def main():
    # Esegui il training del modello
    train()


if __name__ == "__main__":
    main()
