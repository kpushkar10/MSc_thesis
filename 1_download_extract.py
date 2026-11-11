"""Step 1: download match videos, extract their audio, delete the videos.

This file is built up chunk by chunk.
Chunk 2: turn ONE video file into ONE audio file.
Chunk 3: check whether an extracted audio file is usable.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000          # audio samples kept per second (16 kHz, also what VGGish/YAMNet need later)
MAX_DURATION_DIFF_S = 1.0    # audio and video length may differ by at most 1 second
SILENCE_RMS = 1e-4           # a second quieter than this (-80 dBFS) counts as digital silence
MAX_SILENT_FRACTION = 0.02   # more than 2% digital silence = broken source audio


def extract_audio(video, out):
    """Convert a video file into a mono, 16 kHz FLAC audio file."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)  # create the output folder if it is missing

    command = [
        "ffmpeg",
        "-nostdin",                  # never wait for keyboard input
        "-hide_banner",              # skip ffmpeg's long start-up text
        "-loglevel", "error",        # only print something if there is an error
        "-y",                        # overwrite the output file if it already exists
        "-i", str(video),            # the input video
        "-map", "0:a:0",             # use the first audio stream of the first input
        "-vn",                       # ignore the video picture
        "-af", "aresample=async=1:first_pts=0",  # keep audio timing faithful to the video
        "-ac", "1",                  # mix stereo down to 1 channel (mono)
        "-ar", str(SAMPLE_RATE),     # resample to 16000 samples per second
        "-c:a", "flac",              # store as FLAC (lossless, about half the size of WAV)
        str(out),                    # the output file
    ]
    subprocess.run(command, check=True)  # run it; raise an error if ffmpeg fails


def duration(path):
    """Length of a media file in seconds, read with ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def audio_problem(audio, video):
    """Return a short description of what is wrong with the audio, or None if it looks fine."""
    samples, sr = sf.read(audio, dtype="float32")   # the audio as one long array of numbers

    # 1. the audio should be as long as the video
    audio_seconds = len(samples) / sr
    video_seconds = duration(video)
    if abs(audio_seconds - video_seconds) > MAX_DURATION_DIFF_S:
        return f"audio is {audio_seconds:.1f} s long but the video is {video_seconds:.1f} s"

    # 2. the audio should not be mostly digital silence
    n_seconds = len(samples) // sr                            # number of whole seconds
    per_second = samples[: n_seconds * sr].reshape(n_seconds, sr)  # one row per second
    rms = np.sqrt((per_second ** 2).mean(axis=1))             # loudness of each second
    silent_fraction = (rms < SILENCE_RMS).mean()              # share of silent seconds
    if silent_fraction > MAX_SILENT_FRACTION:
        return f"{silent_fraction:.0%} of the audio is digital silence"

    return None   # no problem found


if __name__ == "__main__":
    # usage:
    #   python 1_download_extract.py extract <video file> <output .flac file>
    #   python 1_download_extract.py check   <audio .flac file> <video file>
    action = sys.argv[1]
    if action == "extract":
        extract_audio(sys.argv[2], sys.argv[3])
        print("saved", sys.argv[3])
    elif action == "check":
        problem = audio_problem(sys.argv[2], sys.argv[3])
        print("OK" if problem is None else "PROBLEM: " + problem)
