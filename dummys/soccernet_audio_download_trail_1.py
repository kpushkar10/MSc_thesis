"""

"""
import json
import os
import subprocess
import pathlib

from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.utils import getListGames

ROOT = "data/SoccerNet"
SPLIT = "train"
GAME = getListGames(SPLIT)[0]  # swap in any game path from getListGames(...)
HALF = "1"                     # "1" or "2"

dl = SoccerNetDownloader(LocalDirectory=ROOT)
dl.password = "s0cc3rn3t"

game_dir = os.path.join(ROOT, GAME)
video = os.path.join(game_dir, f"{HALF}_224p.mkv")
wav = os.path.join(game_dir, f"{HALF}_audio_16k_mono.wav")

# download video if does not exist
if not os.path.exists(video):
    dl.downloadGame(game=GAME, files=["Labels-v2.json", f"{HALF}_224p.mkv"], spl=SPLIT)
else:
    print(f"Skipping Download. Video already downloaded. path = {video}")


print("From Code ---> Match:", GAME)
print("From Code ---> Video size: %.0f MB" % (os.path.getsize(video) / 1e6))

# 1) Check that the file actually has an audio stream
probe = subprocess.run(
    ["ffprobe", "-v", "error", "-select_streams", "a",
     "-show_entries", "stream=codec_name,sample_rate,channels:format=duration",
     "-of", "json", video],
    capture_output=True, text=True, check=True)
info = json.loads(probe.stdout)
print("Audio streams:", info.get("streams"))
print("Duration: %.1f min" % (float(info["format"]["duration"]) / 60))
if not info.get("streams"):
    raise SystemExit("No audio stream in this file.")

# 2) Extract audio: mono, 16 kHz
if not os.path.exists(wav):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", video,
                "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav], check=True)
    print("Audio written: %s (%.0f MB)" % (wav, os.path.getsize(wav) / 1e6))
else:
    print("Audio already present. Audio written: %s (%.0f MB)" % (wav, os.path.getsize(wav) / 1e6))


# # 3) convert wav to FLAC
flac = os.path.join(game_dir, f"{HALF}_audio_16k_mono.flac")
if not os.path.exists(flac):
    for wav in pathlib.Path("data/SoccerNet").rglob("*.wav"):
        flac = wav.with_name(wav.name.split("_")[0] + ".flac")   # 1_audio_16k_mono.wav -> 1.flac
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(wav),
                        "-c:a", "flac", "-compression_level", "8", str(flac)], check=True)
        print(f"{wav.name}: {wav.stat().st_size/1e6:.0f} MB -> {flac.stat().st_size/1e6:.0f} MB")