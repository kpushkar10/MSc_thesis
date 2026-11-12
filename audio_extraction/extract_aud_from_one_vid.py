import subprocess
import os
import sys
from pathlib import Path
import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000 #beacuse we convert aud tract to 16KHz

# to check if the audio is useable
MAX_DURATION_DIFF_S = 1.0
SILENCE_RMS = 1e-4
MAX_SILENT_FRACTION = 0.02


def extract_audio(video, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    command = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-i", str(video),
            "-map", "0:a:0",
            "-vn",
            "-af", "aresample=async=1:first_pts=0",
            "-ac", "1",
            "-ar", str(SAMPLE_RATE),
            "-c:a", "flac",
            str(out),
        ]

    subprocess.run(command, check='True')

def vid_duration(vid_path):
    # returns duration of video in seconds
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(vid_path)],
                            capture_output=True, text=True, check=True,)
    return float(result.stdout.strip())

def audio_problem(audio_path, video_path):
    samples, sr = sf.read(audio_path, dtype="float32")

    # checking if lenght if audio and video is same
    # 1 sec diff is accepted
    aud_seconds = len(samples)/sr
    vid_seconds = vid_duration(video_path)
    # print(aud_seconds, vid_seconds)

    if abs(aud_seconds - vid_seconds) > MAX_DURATION_DIFF_S:
        return f"audio is {aud_seconds:.1f} s long but the video is {vid_seconds:.1f}"

    n_seconds = len(samples) // sr
    per_second = samples[: n_seconds * sr].reshape(n_seconds, sr)
    rms = np.sqrt((per_second ** 2).mean(axis=1))
    silent_fraction = (rms < SILENCE_RMS).mean()
    if silent_fraction > MAX_SILENT_FRACTION:
        return f"{silent_fraction:.0%} of the audio is digital silence"

    return None



if __name__ == "__main__":
    video_path, audio_path = "data/SoccerNet/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/1_224p.mkv", "data/audio/trail/1.flac"
    if not os.path.exists(audio_path):
        extract_audio(video_path, audio_path)
        print("audio saved at:",audio_path)
    else:
        print("audio already present")


    audio_prob = audio_problem(audio_path, video_path)
    print("OK. No problems in the audio" if audio_prob is None else "PROBLEM: " + audio_prob)

# Video --> stream|codec_name=vorbis|sample_rate=48000|channels=2 | format|duration=2700.003000

# Audio --> stream|codec_name=flac|sample_rate=16000|channels=1 | format|duration=2700.017688