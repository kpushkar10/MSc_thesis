"""
Downloads lables
Selects 100 matches. Splits in to train, val, test (70/15/15)
"""
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download
from SoccerNet.utils import getListGames

from sklearn.model_selection import train_test_split

LABELS_DIR = Path("data/labels")   # all the label will be saved here, every game will have it's own folder
GAMES_CSV = Path("games.csv") # to store info abt every game
MATCHES_CSV = Path("matches_100.csv")

N_MATCHES = 100
N_TRAIN = 70 # how many go to training (the other 30 become 15 val + 15 test)
SEED = 42

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

def choose_and_split(eligible_matches):
    """Pick N_MATCHES games and split them 70/15/15. Every league keeps its share in each part."""
    # 1. 100 chosen games, 330 left over as reserves
    chosen_matches, reserve = train_test_split(
        eligible_matches,
        train_size = N_MATCHES,
        stratify = eligible_matches["league"],    # each league gets its fair share of the 100
        random_state = SEED,              # same seed -> same choice every time
    )

    # 2. the 100 chosen games: 70 train, 30 left over
    train, rest = train_test_split(
        chosen_matches, train_size=N_TRAIN, stratify=chosen_matches["league"], random_state=SEED,
    )

    # 3. the 30 left over: 15 validation, 15 test
    val, test = train_test_split(
        rest, test_size=0.5, stratify=rest["league"], random_state=SEED,
    )

    # put the four parts back into one table, with a column saying which part each game is in
    return pd.concat([
        train.assign(split="train"),
        val.assign(split="val"),
        test.assign(split="test"),
        reserve.assign(split="reserve"),   # spares, used only if a chosen game has broken audio
    ])

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

    # choose 100 matches and split them
    eligible_matches = table[table["eligible"]].drop(columns="eligible")   # the 430 usable games
    matches = choose_and_split(eligible_matches)
    matches.to_csv(MATCHES_CSV, index=False)

    print()
    print("games per split:")
    print(matches["split"].value_counts().to_string())
    print()
    print("chosen games per league and split:")
    chosen = matches[matches["split"] != "reserve"]
    print(pd.crosstab(chosen["league"], chosen["split"], margins=True).to_string())
    print()
    print("saved", MATCHES_CSV)
