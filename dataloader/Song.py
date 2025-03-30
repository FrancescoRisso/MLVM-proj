from __future__ import annotations

import random
from typing import Iterable

from mido import MidiFile, MidiTrack, second2tick  # type: ignore
from mido.messages import BaseMessage, Message  # type: ignore
from mido.midifiles.meta import MetaMessage  # type: ignore


class Song:
    def __init__(self, midi: MidiFile, tempo=None):
        """
        Creates a new song, starting from a midi object

        ---------------------------------------------------------------------
        PARAMETERS
        ----------
        - midi: the midi representation of the song
        - tempo: the tempo of the song, if present (otherwhise, it will be
            computed)
        """
        self.__midi = midi
        self.__tempo = tempo
        self.__ticks_per_beat = midi.ticks_per_beat

        self.__update_song_tempo()

    @classmethod
    def from_path(cls, path: str) -> Song:
        """
        Creates a new song, reading the midi from a file

        ---------------------------------------------------------------------
        PARAMETERS
        ----------
        - path: the path to the .mid (or .midi) file

        ---------------------------------------------------------------------
        OUTPUT
        ------
        The song
        """
        return Song(MidiFile(path))

    @classmethod
    def from_tracks(
        cls, tracks: Iterable[MidiTrack], ticks_per_beat: int, tempo: int | None = None
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

        ---------------------------------------------------------------------
        OUTPUT
        ------
        The song
        """
        return Song(MidiFile(tracks=tracks, ticks_per_beat=ticks_per_beat), tempo)

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

        return Song.from_tracks(
            [control_track, cut_track], self.__ticks_per_beat, self.__tempo
        )

    def __turn_off_running_notes(
        self, cut_track: MidiTrack, running_notes: list[int | None], time: int
    ) -> None:
        """
        Given a list of the notes that are still running at the end of the
        clip, adds a "note_off" event for each of them at the end of the clip

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
