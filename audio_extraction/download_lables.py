import json
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download
from SoccerNet.utils import getListGames

LABELS_DIR = Path("data/labels")   # all the label will be saved here, every game will have it's own folder
GAMES_CSV = Path("games.csv") # to store info abt every game

def download_labels():
    """Download Labels-v2.json for every SoccerNet game """
    if Path.exists(LABELS_DIR):
        print("Lables already downloaded!!")
        return
    
    snapshot_download(
        repo_id="SoccerNet/SN-Labels",        # the Hugging Face dataset that holds all SoccerNet labels
        repo_type="dataset",                  # it is a dataset, not a model
        allow_patterns=["*/Labels-v2.json"],  # only fetch the Labels-v2.json files, nothing else
        local_dir=LABELS_DIR,                 # save them under data/labels/
    )

def game_row(game):
    labels = json.loads((LABELS_DIR / game / "Labels-v2.json").read_text())
    
    league, season, _ = game.split("/")
    home_goals, away_goals = labels["gameScore"].split(" - ")
    labelled_goals = sum(1 for a in labels["annotations"] if a["label"] == "Goal")

    return {
        "game": game, #id
        "league": league,
        "season": season,
        "home": labels["gameHomeTeam"],
        "away": labels["gameAwayTeam"],
        "score": labels["gameScore"],
        "score_goals": int(home_goals) + int(away_goals),   # goals according to the final score
        "labelled_goals": labelled_goals,                   # goals in the labels
    }


if __name__ == "__main__":
    download_labels()

    games = getListGames(["train", "valid", "test"])   # all 500 games labels
    table = pd.DataFrame([game_row(g) for g in games])

    is_goalless = table["score_goals"] == 0 # 0-0 games
    is_mismatch = table["labelled_goals"] != table["score_goals"] # labels and actual score disagree
    table["eligible"] = ~is_goalless & ~is_mismatch # keeping only games with goals

    table.to_csv(GAMES_CSV, index=False)

    print("games in total:      ", len(table))
    print("0-0 games:           ", is_goalless.sum())
    print("score mismatch games:", (is_mismatch & ~is_goalless).sum())
    print("eligible games:      ", table["eligible"].sum())
    print()
    print("eligible games per league:")
    print(table[table["eligible"]]["league"].value_counts().to_string())
    print()
    print("saved", GAMES_CSV)
