import os
import subprocess

from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.utils import getListGames

games_list = getListGames('train')
print("Total number of games in train set",len(games_list))
print(games_list[0])

ROOT = "data/SoccerNet_720p"
current_game = games_list[0]
HALF = 1

downloader = SoccerNetDownloader(LocalDirectory=ROOT)
downloader.password = "s0cc3rn3t"

game_dir = os.path.join(ROOT, current_game)
video_name = os.path.join(game_dir, f"{HALF}_720p.mkv")
game_wav = os.path.join(game_dir, f"{HALF}_audio_16k_mono.wav")

# download match video if it doesnot Exist
if not os.path.exists(video_name):
    downloader.downloadGame(game=current_game, files=["Lables-v2.json", f"{HALF}_720p.mkv"], spl="train")
else:
    print(f"Skipping Download. Video already downloaded. path = {video_name}")

# extract audio from the video
if not os.path.exists(game_wav):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", video_name,
    "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", game_wav], check=True)
    print("Audio written: %s (%.0f MB)" % (game_wav, os.path.getsize(game_wav) / 1e6))
else:
    print("Audio already present. Audio written: %s (%.0f MB)" % (game_wav, os.path.getsize(game_wav) / 1e6))
