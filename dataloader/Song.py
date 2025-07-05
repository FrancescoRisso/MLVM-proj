from __future__ import annotations

import os
import random
from typing import Iterable

import librosa
import numpy as np
import torch
from mido import MidiFile, MidiTrack, second2tick
from mido.messages import BaseMessage, Message
from mido.midifiles.meta import MetaMessage

from dataloader.MidiToWav import midi_to_wav
from settings import Settings

SET_TEMPO = 1
END_OF_TRACK = 2
CONTROL_CHANGE = 3
NOTE_ON = 4
NOTE_OFF = 5
PROGRAM_CHANGE = 6
TIME_SIGNATURE = 7

VALID_FIELDS_PER_MSG_TYPE = np.empty(shape=(8, 6), dtype=np.bool)

# fmt: off
# ms_type 0 for messages after song end
VALID_FIELDS_PER_MSG_TYPE[0]              = (0, 0, 0, 0, 0, 0)
VALID_FIELDS_PER_MSG_TYPE[SET_TEMPO]      = (1, 1, 1, 0, 0, 0)
VALID_FIELDS_PER_MSG_TYPE[END_OF_TRACK]   = (1, 1, 0, 0, 0, 0)
VALID_FIELDS_PER_MSG_TYPE[CONTROL_CHANGE] = (1, 1, 1, 1, 0, 0)
VALID_FIELDS_PER_MSG_TYPE[NOTE_ON]        = (1, 1, 1, 1, 0, 0)
VALID_FIELDS_PER_MSG_TYPE[NOTE_OFF]       = (1, 1, 1, 0, 0, 0)
VALID_FIELDS_PER_MSG_TYPE[PROGRAM_CHANGE] = (1, 1, 1, 0, 0, 0)
VALID_FIELDS_PER_MSG_TYPE[TIME_SIGNATURE] = (1, 1, 1, 1, 1, 1)
# fmt: on

VALID_FIELDS_PER_MSG_TYPE = torch.tensor(VALID_FIELDS_PER_MSG_TYPE)


class Song:
    def __init__(self, midi: MidiFile, tempo=None, wav_path: None | str = None):
        """
        Creates a new song, starting from a midi object

        ---------------------------------------------------------------------
        PARAMETERS
        ----------
        - midi: the midi representation of the song
        - tempo: the tempo of the song, if present (otherwhise, it will be
            computed)
        - wav_path: the path where there is the pre-generated wav, if
                available (or None otherwhise)
        """
        self.__midi = midi
        self.__tempo = tempo
        self.__ticks_per_beat = midi.ticks_per_beat
        self.__wav_path = wav_path

        self.__update_song_tempo()

    @classmethod
    def from_path(cls, path: str, wav_path: None | str = None) -> Song:
        """
        Creates a new song, reading the midi from a file

        ---------------------------------------------------------------------
        PARAMETERS
        ----------
        - path: the path to the .mid (or .midi) file
        - wav_path: the path where there is the pre-generated wav, if
                available (or None otherwhise)

        ---------------------------------------------------------------------
        OUTPUT
        ------
        The song
        """
        return Song(MidiFile(path), wav_path=wav_path)

    @classmethod
    def from_tracks(
        cls,
        tracks: Iterable[MidiTrack],
        ticks_per_beat: int,
        tempo: int | None = None,
        wav_path: None | str = None,
    ) -> Song:
        """
        Creates a new song that contains some specific tracks

        ---------------------------------------------------------------------
        PARAMETERS
        ----------
        - tracks: the midi tracks to put in the song
        - ticks_per_beat: the ticks_per_beat setting in which the tracks are
            written
        - tempo: the tempo of the song, if present (otherwhise, it will be
            computed)
        - wav_path: the path where there is the pre-generated wav, if
                available (or None otherwhise)

        ---------------------------------------------------------------------
        OUTPUT
        ------
        The song
        """
        return Song(
            MidiFile(tracks=tracks, ticks_per_beat=ticks_per_beat),
            tempo,
            wav_path=wav_path,
        )

    @classmethod
    def from_np(
        cls,
        data: np.ndarray,
        tempo: int,
        ticks_per_beat: int,
        num_messages: int,
        wav_path: None | str = None,
    ) -> Song:
        """
        Creates a new song from our numpy representation

        ---------------------------------------------------------------------
        PARAMETERS
        ----------
        - data: the numpy representation of the midi
        - tempo: the tempo of the song, if present (otherwhise, it will be
            computed)
        - ticks_per_beat: the ticks_per_beat setting in which the tracks are
            written
        - num_messages: the number of meaningful messages in the midi
        - wav_path: the path where there is the pre-generated wav, if
                available (or None otherwhise)

        ---------------------------------------------------------------------
        OUTPUT
        ------
        The song
        """
        track = MidiTrack()

        for i in range(num_messages):
            midi_msg = np_to_msg(data[i])
            if midi_msg is not None:
                track.append(midi_msg)

        return Song.from_tracks(
            [track], tempo=tempo, ticks_per_beat=ticks_per_beat, wav_path=wav_path
        )

    def __update_song_tempo(self) -> None:
        """
        Finds the "set_tempo" meta message within the midi, and sets the
        tempo to that value.

        If the tempo is already set, it does nothing
        """
        if self.__tempo is not None:
            return

        for track in self.__midi.tracks:
            assert isinstance(track, MidiTrack)

            for msg in track:
                if isinstance(msg, MetaMessage) and msg.type == "set_tempo":
                    self.__tempo = msg.tempo
                    return

    def cut(self, start_second: float, end_second: float) -> Song:
        """
        Creates a new song as the portion of this song included between two
        timestamps.

        ---------------------------------------------------------------------
        PARAMETERS
        ----------
        - start_second: the timestamp, expressed in second, where the song
            slice should start
        - end_second: the timestamp, expressed in second, where the song
            slice should end. If it exceeds the song natural end, the slice
            will end at the natural song end

        ---------------------------------------------------------------------
        OUTPUT
        ------
        The song obtained by the slicing process
        """
        control_track, music_track = self.__midi.tracks[0:2]

        assert isinstance(music_track, MidiTrack)

        start_tick = second2tick(start_second, self.__ticks_per_beat, self.__tempo)
        end_tick = second2tick(end_second, self.__ticks_per_beat, self.__tempo)

        cut_track = MidiTrack()

        # Store the list of notes that are playing at the cur_tick.
        # This is useful for adding notes started before the cut, and
        # for stopping notes that continue after the cut.
        # Midi can handle 128 different notes (middle C = 60), therefore
        # I'm storing the velocity of each note in a separate array cell
        running_notes: list[int | None] = [None for _ in range(128)]

        # Similarly, store the last control channel update for each channel
        control_channels_status: list[int | None] = [None for _ in range(128)]

        # Meta messages are ignored

        cur_tick = 0
        for midi_msg in music_track:
            assert isinstance(midi_msg, BaseMessage)

            # If the part to be cut is finished, exit the loop
            if cur_tick + midi_msg.time > end_tick:
                break

            # Compute current tick (cumulative)
            cur_tick += midi_msg.time

            # If it's a note, update the running_notes variable
            if midi_msg.type in ["note_on", "note_off"]:
                midi_msg = self.__process_note(midi_msg, running_notes)

            if cur_tick < start_tick:
                # Store control messages to add them at the start of the track
                if midi_msg.type == "control_change":
                    control_channels_status[midi_msg.control] = midi_msg.value

            else:
                if len(cut_track) == 0:
                    # Add all the control messages and the running notes
                    self.__first_message_of_cut(
                        cut_track, running_notes, control_channels_status
                    )
                    midi_msg.time = cur_tick - start_tick

                cut_track.append(midi_msg)

        self.__turn_off_running_notes(cut_track, running_notes, end_tick - cur_tick)

        for m in control_track:
            if m.time == 0:
                cut_track.insert(0, m)
            else:
                break

        return Song.from_tracks(
            [cut_track], self.__ticks_per_beat, self.__tempo, wav_path=self.__wav_path
        )

    def __turn_off_running_notes(
        self, cut_track: MidiTrack, running_notes: list[int | None], time: int
    ) -> None:
        """
        Given a list of the notes that are still running at the end of the
        clip, adds a "note_off" event for each of them at the end of the
        clip.
        Finally, it adds a "end_of_track" message at the end of the track.

        ---------------------------------------------------------------------
        PARAMETERS
        ----------
        - cut_track: the track where to append the "note_off" messages
        - running_notes: a list where running_notes[i] is either _None_ to
            mean that the midi note _i_ is not being played, or a number if
            the note is being played
        """
        for note, vel in enumerate(running_notes):
            if vel != None:
                msg = Message("note_off", channel=0, note=note, velocity=0, time=time)
                cut_track.append(msg)
                time = 0

        cut_track.append(MetaMessage("end_of_track", time=time))

    def __first_message_of_cut(
        self,
        cut_track: MidiTrack,
        running_notes: list[int | None],
        control_channels_status: list[int | None],
    ) -> None:
        """
        Given a recap of the previous control messages and the notes that are
        currently being played, this function adds to the track the required
        message to start in the same state (controls and notes) as in the
        original song at that moment

        ---------------------------------------------------------------------
        PARAMETERS
        ----------
        - cut_track: the track where to add all the messages
        - running_notes: a list where running_notes[i] is either _None_ to
            mean that the midi note _i_ is not being played, or a number _N_
            to mean that the midi note _i_ is being played with _velocity=N_
        - control_channels_status: a list where control_channels_status[i]
            contains the last value set for the control channel _i_, or
            _None_ if the control channel _i_ was never set
        ---------------------------------------------------------------------
        OUTPUT
        ------

        """
        for ctrl, val in enumerate(control_channels_status):
            if val != None:
                msg = Message(
                    "control_change", channel=0, control=ctrl, value=val, time=0
                )
                cut_track.append(msg)

        for note, vel in enumerate(running_notes):
            if vel != None:
                msg = Message("note_on", channel=0, note=note, velocity=vel, time=0)
                cut_track.append(msg)

    def __process_note(
        self, midi_msg: Message, running_notes: list[int | None]
    ) -> Message:
        """
        Given a "note_on/off" message, updates the list of notes currently
        being played with the information contained in the message.

        It also converts ( "note_on", velocity=0 ) messages to "note_off"
        messages

        ---------------------------------------------------------------------
        PARAMETERS
        ----------
        - midi_msg: the message containing the "note_on/off" event
        - running_notes: a list where running_notes[i] is either _None_ to
            mean that the midi note _i_ is not being played, or a number _N_
            to mean that the midi note _i_ is being played with _velocity=N_

        ---------------------------------------------------------------------
        OUTPUT
        ------
        A message equivalent to the input one, that has always type
        "note_off" if the velocity is 0
        """
        if midi_msg.type == "note_on" and midi_msg.velocity != 0:
            running_notes[midi_msg.note] = midi_msg.velocity
            return midi_msg

        else:
            # note_on with velocity=0 is equivalent to note_off
            running_notes[midi_msg.note] = None
            return Message(
                "note_off",
                channel=midi_msg.channel,
                note=midi_msg.note,
                velocity=midi_msg.velocity,
                time=midi_msg.time,
            )

    def save(self, fname: str) -> None:
        """
        Stores a midi song to a file

        ---------------------------------------------------------------------
        PARAMETERS
        ----------
        - fname: the file where to store the midi
        """
        self.__midi.save(fname)

    def get_midi(self) -> MidiFile:
        """
        Returns the current song as plain midi object

        ---------------------------------------------------------------------
        OUTPUT
        ------
        The song as MidiFile
        """
        return self.__midi

    def choose_cut_boundary(
        self, duration: None | float | tuple[float, float]
    ) -> None | tuple[float, float]:
        """
        Given a cutting duration option, chooses the start and end second
        where to cut the song

        ---------------------------------------------------------------------
        PARAMETERS
        ----------
        - duration: how long the crop should be, in one of the following
            formats:
            - None: don't crop the song
            - num: create a random crop of exactly num seconds (equivalent to
                the previous one if num > song length)
            - (min, max): select a random duration between min and max, then
                select a random crop of that duration (max is cropped to the
                song length, and the option is equivalent to None if also min
                is greater than the song length)

        ---------------------------------------------------------------------
        OUTPUT
        ------
        - None if the full song should be used
        - a tuple in the form (start, end), where there are the timestamps
            where the song should be cropped
        """

        if duration is None:
            return None

        if isinstance(duration, tuple):
            min, max = duration

            if min > max or min > self.__midi.length:
                return None

            if max > self.__midi.length:
                max = self.__midi.length

            duration = random.uniform(min, max)

        else:
            if duration > self.__midi.length:
                return None

        start = random.uniform(0, self.__midi.length - duration)
        return start, start + duration

    def to_wav(self, verbose: bool = False) -> np.ndarray:
        """
        Create a wav representation of the song

        ---------------------------------------------------------------------
        PARAMETERS
        ----------
        - verbose: if the output of the synthesizer should be printed or not
            (default: False)

        ---------------------------------------------------------------------
        OUTPUT
        ------
        The wav data, as loaded by librosa

        If the synthesizer (fluidsynth) is not installed, the program crashes
        instructing the user to install it.
        """
        try:
            self.save(Settings.tmp_midi_file)

            midi_to_wav(
                Settings.tmp_midi_file,
                Settings.tmp_audio_file,
                Settings.audio_font_path,
                sample_rate=Settings.sample_rate,
            )

            y, _ = librosa.load(
                Settings.tmp_audio_file,
                sr=Settings.sample_rate,
                duration=Settings.seconds,
            )

            os.remove(Settings.tmp_midi_file)
            os.remove(Settings.tmp_audio_file)

            return y

        except FileNotFoundError:
            os.remove(Settings.tmp_midi_file)
            print("Error: fluidsynth may not be installed")
            print('Please install it with "sudo apt install fluidsynth"')
            exit(-1)

    def load_cut_wav(self, start: int, end: int) -> np.ndarray:
        """
        Loads a cut of the corresponding audio file

        ---------------------------------------------------------------------
        PARAMETERS
        ----------
        - start: the second in which the cut should start
        - end: the second in which the cut should end

        ---------------------------------------------------------------------
        OUTPUT
        ------
        The wav data, as loaded by librosa
        """

        assert self.__wav_path is not None

        song, _ = librosa.load(
            self.__wav_path,
            sr=Settings.sample_rate,
            duration=end,
        )

        samples_to_keep = Settings.seconds * Settings.sample_rate
        return song[-samples_to_keep:]

    def to_np(self) -> tuple[np.ndarray, int, int, int]:
        """
        Converts a song to a numpy array

        ---------------------------------------------------------------------
        OUTPUT
        ------
        A tuple containing:
        - the np array with the song
        - the tempo of the song
        - the ticks per beat of the song
        - the number of midi messages in the song
        """
        res = np.zeros(shape=(Settings.max_midi_messages, 6), dtype=np.uint16)

        idx = 0
        for msg in self.__midi.tracks[0]:
            if idx >= Settings.max_midi_messages:
                raise Exception(
                    f"A song with {len(self.__midi.tracks[0])} messages does not fit on the allocated {Settings.max_midi_messages}"
                )

            if self.__msg_to_np(msg, res[idx]):
                idx += 1
            else:
                print(f'WARN: To np not implemented for message "{msg}"')

        return res, self.__tempo, self.__ticks_per_beat, len(self.__midi.tracks[0])

    def __msg_to_np(self, msg: Message | MetaMessage, res: np.ndarray) -> bool:
        """
        Stores a midi message into a np array in a suitable format

        ---------------------------------------------------------------------
        PARAMETERS
        ----------
        - msg: the message to save
        - res: the array where to save the message

        ---------------------------------------------------------------------
        OUTPUT
        ----------
        if the message was recognised and saved
        """

        if isinstance(msg.time, float):
            msg.time = second2tick(msg.time, self.__ticks_per_beat, self.__tempo)

        if isinstance(msg, MetaMessage):
            if msg.type == "time_signature":
                res[:] = (
                    msg.time,
                    TIME_SIGNATURE,
                    msg.numerator,
                    msg.denominator,
                    msg.clocks_per_click,
                    msg.notated_32nd_notes_per_beat,
                )

            elif msg.type == "set_tempo":
                res[:3] = (msg.time, SET_TEMPO, msg.tempo // 1000)

            elif msg.type == "end_of_track":
                res[:2] = (msg.time, END_OF_TRACK)

            else:
                return False

        # not MetaMessage
        else:
            if msg.type == "control_change":
                res[:4] = (msg.time, CONTROL_CHANGE, msg.control, msg.value)

            elif msg.type == "note_off" or (
                msg.type == "note_on" and msg.velocity == 0
            ):
                res[:3] = (msg.time, NOTE_OFF, msg.note)

            elif msg.type == "note_on":
                res[:4] = (msg.time, NOTE_ON, msg.note, msg.velocity)

            elif msg.type == "program_change":
                res[:3] = (msg.time, PROGRAM_CHANGE, msg.program)

            else:
                return False

        return True


def np_to_msg(arr: np.ndarray) -> Message | MetaMessage | None:
    """
    Converts a midi message from np stored to real message

    ---------------------------------------------------------------------
    PARAMETERS
    ----------
    - arr: the midi message in the np format

    ---------------------------------------------------------------------
    OUTPUT
    ------
    The real (meta) message
    """

    if arr[1] == TIME_SIGNATURE:
        return MetaMessage(
            "time_signature",
            time=int(arr[0]),
            numerator=int(arr[2]),
            denominator=int(arr[3]),
            clocks_per_click=int(arr[4]),
            notated_32nd_notes_per_beat=int(arr[5]),
        )

    if arr[1] == SET_TEMPO:
        return MetaMessage("set_tempo", time=int(arr[0]), tempo=1000 * int(arr[2]))

    if arr[1] == CONTROL_CHANGE:
        return Message(
            "control_change",
            time=int(arr[0]),
            control=int(arr[2]),
            value=int(arr[3]),
        )

    if arr[1] == NOTE_OFF:
        return Message("note_off", time=int(arr[0]), note=int(arr[2]))

    if arr[1] == NOTE_ON:
        return Message(
            "note_on", time=int(arr[0]), note=int(arr[2]), velocity=int(arr[3])
        )

    if arr[1] == PROGRAM_CHANGE:
        return Message("program_change", time=int(arr[0]), program=int(arr[2]))

    if arr[1] == END_OF_TRACK:
        return MetaMessage("end_of_track", time=int(arr[0]))

    print(f'WARN: Unrecognized np message type "{arr[1]}"')
    return None
