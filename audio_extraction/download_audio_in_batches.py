"""
download the videos -> extract the audio -> check it -> delete the videos.
A match with broken audio is replaced by a reserve game from the same league.
"""

import importlib
import shutil
import sys
from pathlib import Path

import pandas as pd
from SoccerNet.Downloader import SoccerNetDownloader

MATCHES_CSV = Path("matches_100.csv")
VIDEO_DIR = Path("data/videos")
AUDIO_DIR = Path("data/audio")

USED_SPLITS = ["train", "val", "test"]

HALVES = [1, 2]
BATCH_SIZE = 10

extract_aud_from_one_vid = importlib.import_module("extract_aud_from_one_vid")

def video_path(game, half):
    return VIDEO_DIR/game/f"{half}_224p.mkv"

def audio_path(game, half):
    return AUDIO_DIR / game / f"{half}.flac"

def half_done(game, half):
    return audio_path(game, half).exists() and not video_path(game, half).exists()

def match_done(game):
    return all(half_done(game, half) for half in HALVES)

def download(game):
    """Download the 224p video of every half of this game"""
    files = [video_path(game, half).name for half in HALVES if not half_done(game, half)]
    SoccerNetDownloader(LocalDirectory=str(VIDEO_DIR)).downloadGame(game=game, files=files)

def process(game):
    """Extract and check each unfinished half; delete its video only if the audio is fine.

    Returns None if the whole match is fine, otherwise a short description of the problem.
    """
    for half in HALVES:
        if half_done(game, half):
            continue                          # finished in an earlier run
        video, audio = video_path(game, half), audio_path(game, half)
        extract_aud_from_one_vid.extract_audio(video, audio)
        problem = extract_aud_from_one_vid.audio_problem(audio, video)
        if problem is not None:
            return f"half {half}: {problem}"  # stop here; the video is kept until replace() runs
        video.unlink()                        # the audio is fine, so the video is not needed any more
    return None

def replace(game, problem):
    """Mark a broken game as "bad" and give its split to the next reserve game of the same league."""
    matches = pd.read_csv(MATCHES_CSV)
    i = matches.index[matches["game"] == game][0]           # row number of the broken game
    league, split = matches.at[i, "league"], matches.at[i, "split"]
    matches.at[i, "split"] = "bad"
    matches.at[i, "problem"] = problem                      # keep the reason, for the thesis write-up

    spares = matches.index[(matches["split"] == "reserve") & (matches["league"] == league)]
    if len(spares) == 0:
        print("----> no reserve game left for", league)
    else:
        matches.at[spares[0], "split"] = split              # the first spare takes the broken game's place
        print("----> replaced by", matches.at[spares[0], "game"], f"({split})")
    matches.to_csv(MATCHES_CSV, index=False)

    shutil.rmtree(VIDEO_DIR / game, ignore_errors=True)     # throw away the broken game's video
    shutil.rmtree(AUDIO_DIR / game, ignore_errors=True) 

def todo_games():
    """The chosen games (train / val / test) that are not finished yet."""
    matches = pd.read_csv(MATCHES_CSV)
    chosen = matches[matches["split"].isin(USED_SPLITS)]["game"]
    return [game for game in chosen if not match_done(game)]

def run(limit=None):
    """Process all unfinished games in batches."""
    while True:
        todo = todo_games()[:limit] # "all of them"
        if not todo:
            break # everything is finished

        for start in range(0, len(todo), BATCH_SIZE):
            batch = todo[start:start + BATCH_SIZE]
            print(f"\n=== batch: games {start + 1}-{start + len(batch)} of {len(todo)} ===")
            for game in batch: # 1. download the whole batch
                print("download", game)
                download(game)

            for game in batch: # 2. extract, check, delete video
                problem = process(game)

                if problem is None:
                    print("OK  ", game)

                else:
                    print("BAD ", game, "-", problem)
                    replace(game, problem)

        if limit is not None:
            break # a test run does one pass only

    print("\nunfinished games left:", len(todo_games()))

if __name__ == "__main__":
    # limit = int(sys.argv[1]) if len(sys.argv) > 1 else None   # a number = test run on that many games
    
    run()