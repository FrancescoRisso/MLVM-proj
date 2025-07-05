import subprocess


def midi_to_wav(
    midi_file: str,
    out_file: str,
    sound_font: str,
    verbose: bool = False,
    sample_rate: int = 44100,
    gain: float = 0.2,
) -> None:
    """
    Converts a midi file to an audio file

    ---------------------------------------------------------------------
    PARAMETERS
    ----------
    - midi_file: the path of the midi file to convert
    - out_file: the path where to store the result
    - sound_font: the path where the audio font is
    - verbose: if the output from fluidsynth should be printed or not
    - sample_rate: the sample rate for the output audio file
    - gain: the gain for fluidsynth
    """

    if verbose:
        stdout = None
    else:
        stdout = subprocess.DEVNULL

    params = ["fluidsynth", "-ni"]
    params = [*params, "-g", str(gain)]
    params = [*params, "-F", out_file]
    params = [*params, "-r", str(sample_rate)]
    params = [*params, sound_font, midi_file]

    subprocess.call(params, stdout=stdout)
