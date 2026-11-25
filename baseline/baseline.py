import json
import sys
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.ndimage import uniform_filter1d

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from utils import utils  # the file utils/utils.py

FLOOR_DB = -80.0 # anything quieter than -80 dB counts as -80 dB (same limit as the silence check)
SMOOTH_WINDOWS = 5 # average over 5 windows = 3 s of audio, so one short noise does not make a peak
MERGE_S = 60.0     # events less than 60 s apart are one event (a goal = cheer + celebration + replay)
RESULTS_DIR = Path("results")

THRESHOLDS = np.round(np.arange(0.50, 1.00, 0.01), 2)   # the thresholds try
TUNE_SPLIT = "val"                                       # the threshold is chosen on these matches ONLY
THRESHOLD_JSON = RESULTS_DIR / "baseline_threshold.json" 
EVENTS_CSV = RESULTS_DIR / "baseline_events.csv"
USED_SPLITS = ["train", "val", "test"]                   # all 100 chosen matches

CURVE_COLOR = "#2a78d6"
GOAL_COLOR = "#eb6834"
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3df"
F1_COLOR = "#2a78d6"
PRECISION_COLOR = "#eb6834"
RECALL_COLOR = "#1baf7a"

def window_db(audio_file):
    """Loudness in dB of every window (1s long, every 0.5s) of one half."""
    samples, sr = sf.read(str(audio_file), dtype="float32")
    rms = librosa.feature.rms(
        y=samples,
        frame_length=utils.WINDOW,   # 16000 samples = 1 s
        hop_length=utils.HOP,        # 8000 samples = 0.5 s
        center=False,                # window 0 starts at 0 s, the same grid as utils.window_times()
    )[0]
    floor = 10 ** (FLOOR_DB / 20)                  # -80 dB as an RMS value = 0.0001
    return 20 * np.log10(np.maximum(rms, floor))

def loudness_curve(game):
    """The 0-1 loudness curve of one match: {1: array, 2: array}, one value per window."""
    db = {half: window_db(utils.audio_path(game, half)) for half in utils.HALVES}

    # smooth: each value becomes the average of itself and its 2 neighbours on each side
    smooth = {half: uniform_filter1d(db[half], size=SMOOTH_WINDOWS, mode="nearest") for half in utils.HALVES}

    # scale to 0-1 using the quietest and loudest moment of the WHOLE match (both halves together)
    low = min(smooth[half].min() for half in utils.HALVES)
    high = max(smooth[half].max() for half in utils.HALVES)
    return {half: (smooth[half] - low) / (high - low) for half in utils.HALVES}

def plot_curve(game, curve, out_file):
    """Save a picture of the loudness curve of both halves, with a dashed line at every goal."""
    goals = utils.load_goals(game)
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharey=True)   # 2 rows: half 1 top, half 2 below

    for ax, half in zip(axes, utils.HALVES):
        minutes = utils.window_times(len(curve[half])) / 60 # window centres, in minutes
        ax.plot(minutes, curve[half], color=CURVE_COLOR, linewidth=0.8)
        for goal in goals[half]:
            ax.axvline(goal / 60, color=GOAL_COLOR, linestyle="--", linewidth=1.5)
            ax.text(goal / 60, 1.02, " goal", color=TEXT_COLOR, fontsize=8, va="bottom")

        ax.set_xlim(0, minutes[-1])
        ax.set_ylim(0, 1)
        ax.set_title(f"half {half}", loc="left", fontsize=10, color=TEXT_COLOR)
        ax.set_ylabel("loudness (0-1)", color=TEXT_COLOR)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
        ax.tick_params(colors=TEXT_COLOR)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)                      # remove the box lines

    axes[-1].set_xlabel("minutes into the half", color=TEXT_COLOR)
    fig.suptitle(game, fontsize=10, color=TEXT_COLOR)
    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=120)
    plt.close(fig)


# ---------------- detecting events ----------------

def detect(curve, threshold):
    """Event times (seconds) in ONE half: the moments where the curve rises above the threshold."""
    times = utils.window_times(len(curve))

    # 1. find every stretch above the threshold; remember its start time and its highest value
    events = []                                   # each item: [start time, highest value]
    inside = False                                # currelty inside a stretch that is above the threshold?
    for t, value in zip(times, curve):
        if value > threshold and not inside:      # the curve just went above: a new stretch starts
            events.append([t, value])
            inside = True
        elif value > threshold:                   # still inside curve & above threshold: keep track of the highest value
            events[-1][1] = max(events[-1][1], value)
        else:                                     # below the threshold: the stretch has ended
            inside = False

    # 2. merge events less than 60s apart, keeping the loudest one
    merged = []
    for t, peak in events:
        if merged and t - merged[-1][0] < MERGE_S:   # too close to the previous event: same moment
            if peak > merged[-1][1]:
                merged[-1] = [t, peak]               # this one is louder, so it replaces the previous one
        else:
            merged.append([t, peak])                 # far enough away: a new event

    return [float(t) for t, peak in merged]          # just the times, as plain Python numbers


def detect_match(curve, threshold):
    """detect() for both halves: {1: [seconds, ...], 2: [seconds, ...]}, the same shape as the goals."""
    return {half: detect(curve[half], threshold) for half in utils.HALVES}


def check_detect():
    """A made-up curve with known bumps, where we know which events should come out."""
    curve = np.zeros(1000)          # 1000 windows = about 500 s of silence (value 0)
    curve[100:110] = 0.90           # bump A: starts at window 100 = 50.5 s
    curve[140:146] = 0.95           # bump B: starts at 70.5 s, only 20 s after A, and louder
    curve[400:406] = 0.80           # bump C: starts at 200.5 s
    curve[700:706] = 0.60           # bump D: starts at 350.5 s, but it is quiet

    cases = [
        # (threshold, expected event times,  why)
        (0.70, [70.5, 200.5],         "A and B merge (B is louder), C stays, D is too quiet"),
        (0.50, [70.5, 200.5, 350.5],  "a lower threshold also catches D"),
        (0.85, [70.5],                "only A and B are above 0.85, and they merge"),
        (0.99, [],                    "nothing is that loud"),
    ]
    for threshold, expected, why in cases:
        found = detect(curve, threshold)
        assert found == expected, f"threshold {threshold}: got {found}, expected {expected}"
        print(f"OK  threshold {threshold:.2f} -> {found}   ({why})")


def check_curve(number):
    """plot one TRAINING match and see whether goals line up with loud moments."""
    game = utils.games_in("train")[number]
    curve = loudness_curve(game)
    goals = utils.load_goals(game)

    print(game)
    for half in utils.HALVES:
        times = utils.window_times(len(curve[half]))
        print(f"\nhalf {half}: {len(curve[half])} windows, typical loudness {np.median(curve[half]):.2f}")
        for goal in goals[half]:
            near = np.abs(times - goal) <= 30                       # windows up to 30 s around the goal
            print(f"  goal at {utils.mmss(goal)}: loudest value within 30 s = {curve[half][near].max():.2f}")

    out_file = RESULTS_DIR / f"loudness_curve_train_{number}.png"
    plot_curve(game, curve, out_file)
    print("\nsaved", out_file)


def check_detect_on_match(number, threshold):
    """run detect() on one TRAINING match and score it."""
    game = utils.games_in("train")[number]
    curve = loudness_curve(game)
    goals = utils.load_goals(game)
    events = detect_match(curve, threshold)

    print(f"\n{game}   (threshold {threshold})")
    for half in utils.HALVES:
        print(f"  half {half}: goals  {[utils.mmss(t) for t in goals[half]]}")
        print(f"          events {[utils.mmss(t) for t in events[half]]}")
    score = utils.score_match(goals, events)
    print(f"  tp={score['tp']} fp={score['fp']} fn={score['fn']}  "
          f"precision={score['precision']:.2f} recall={score['recall']:.2f} F1={score['f1']:.2f}")


# ---------------- tuning the threshold (validation matches only) ----------------

def sweep(split):
    """Try every threshold on every match of one split. One row per (match, threshold)."""
    assert split != "test", "never tune on the test matches"
    rows = []
    for game in utils.games_in(split):
        curve = loudness_curve(game)                 # the slow part, so it is done once per match
        goals = utils.load_goals(game)
        for threshold in THRESHOLDS:
            events = detect_match(curve, threshold)
            score = utils.score_match(goals, events)
            rows.append({"game": game, "threshold": threshold, **score})   # ** = copy all of score's keys
        print("scored", game)
    return pd.DataFrame(rows)


def summarise(per_match):
    """Average over the matches: one row per threshold (each match counts the same)."""
    per_match = per_match.assign(events=per_match["tp"] + per_match["fp"])   # events found per match
    return per_match.groupby("threshold")[["precision", "recall", "f1", "events"]].mean().reset_index()


def choose_threshold(summary):
    """The threshold with the highest mean F1 (if several tie, the lowest of them)."""
    best = summary.loc[summary["f1"].idxmax()]
    return float(best["threshold"]), best


def plot_sweep(summary, chosen, out_file):
    """Mean F1, precision and recall against the threshold, with the chosen threshold marked."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    lines = [("F1", "f1", F1_COLOR), ("precision", "precision", PRECISION_COLOR), ("recall", "recall", RECALL_COLOR)]
    for name, column, color in lines:
        ax.plot(summary["threshold"], summary[column], color=color, linewidth=2, label=name)
        ax.text(summary["threshold"].iloc[-1] + 0.005, summary[column].iloc[-1], name,
                color=TEXT_COLOR, fontsize=9, va="center")             # name at the right end of the line

    ax.axvline(chosen, color=TEXT_COLOR, linestyle="--", linewidth=1)
    ax.text(chosen, 1.02, f" chosen {chosen:.2f}", color=TEXT_COLOR, fontsize=9, va="bottom")

    ax.set_xlim(THRESHOLDS[0], THRESHOLDS[-1] + 0.06)                   # room for the line names
    ax.set_ylim(0, 1)
    ax.set_xlabel("threshold", color=TEXT_COLOR)
    ax.set_ylabel(f"mean over {TUNE_SPLIT} matches", color=TEXT_COLOR)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.tick_params(colors=TEXT_COLOR)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", frameon=False, labelcolor=TEXT_COLOR)
    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=120)
    plt.close(fig)


def tune():
    """sweep the thresholds on the validation matches, pick the best, save everything."""
    per_match = sweep(TUNE_SPLIT)
    summary = summarise(per_match)
    threshold, best = choose_threshold(summary)

    RESULTS_DIR.mkdir(exist_ok=True)
    per_match.to_csv(RESULTS_DIR / "threshold_sweep_val_per_match.csv", index=False)
    summary.to_csv(RESULTS_DIR / "threshold_sweep_val.csv", index=False)
    plot_sweep(summary, threshold, RESULTS_DIR / "threshold_sweep_val.png")
    chosen = {
        "threshold": threshold,
        "tuned_on": TUNE_SPLIT,
        "n_matches": int(per_match["game"].nunique()),
        "mean_f1": round(float(best["f1"]), 4),
        "mean_precision": round(float(best["precision"]), 4),
        "mean_recall": round(float(best["recall"]), 4),
        "mean_events_per_match": round(float(best["events"]), 2),
    }
    (RESULTS_DIR / "baseline_threshold.json").write_text(json.dumps(chosen, indent=2))

    print("\nevery 5th threshold:")
    print(summary.iloc[::5].round(3).to_string(index=False))
    print(f"\nchosen threshold: {threshold:.2f}  (mean F1 {best['f1']:.3f}, "
          f"precision {best['precision']:.3f}, recall {best['recall']:.3f}, {best['events']:.1f} events per match)")
    if threshold in (THRESHOLDS[0], THRESHOLDS[-1]):
        print("WARNING: the best threshold is at the edge of the range; try a wider range")
    else:
        print("OK  the best threshold is inside the range, not at an edge")
    print("saved results/threshold_sweep_val.csv, threshold_sweep_val_per_match.csv, "
          "threshold_sweep_val.png, baseline_threshold.json")


# ---------------- applying the frozen threshold to all matches ----------------

def load_threshold():
    """It is read from the file, never typed in again."""
    return json.loads(THRESHOLD_JSON.read_text())["threshold"]


def apply_all():
    """detect events in all 100 matches with the frozen threshold, and save them.

    Train and val matches will be scored. Test matches will NOT scored here:
    they are scored once, at the end, together with the other models (O6)
    """
    threshold = load_threshold()
    rows = []      # one row per detected event
    scores = []    # one row per scored match (train and val only)

    for split in USED_SPLITS:
        for game in utils.games_in(split):
            events = detect_match(loudness_curve(game), threshold)
            for half in utils.HALVES:
                for t in events[half]:
                    rows.append({"game": game, "split": split, "half": half,
                                 "time_s": t, "threshold": threshold})
            if split != "test":
                score = utils.score_match(utils.load_goals(game), events)
                scores.append({"game": game, "split": split, **score})
            n_events = len(events[1]) + len(events[2])
            print(f"{split:5s} {n_events:3d} events  {game}")

    table = pd.DataFrame(rows)
    table.to_csv(EVENTS_CSV, index=False)
    scores = pd.DataFrame(scores)

    # ---- checks ----
    print(f"\nthreshold used: {threshold}  (from {THRESHOLD_JSON})")
    print("\nevents per split:")
    print(table.groupby("split").agg(matches=("game", "nunique"), events=("time_s", "size")).to_string())
    all_games = [g for split in USED_SPLITS for g in utils.games_in(split)]
    silent = [g for g in all_games if g not in set(table["game"])]
    print(f"\nmatches with no event at all: {len(silent)}")
    for game in silent:
        print("  ", game)

    print("\nmean scores (train and val only):")
    print(scores.groupby("split")[["precision", "recall", "f1"]].mean().round(3).to_string())
    val_f1 = scores[scores["split"] == "val"]["f1"].mean()
    tuned_f1 = json.loads(THRESHOLD_JSON.read_text())["mean_f1"]
    assert abs(val_f1 - tuned_f1) < 0.001, "val F1 differs from chunk 11: something changed"
    print(f"OK  val F1 {val_f1:.3f} is the same as on validation matches")
    assert "test" not in set(scores["split"]), "test matches must not be scored yet"
    print("OK  no test match was scored")
    print("\nsaved", EVENTS_CSV)


if __name__ == "__main__":
    # usage:
    #   python baseline/baseline.py curve 0         plot training match 0
    #   python baseline/baseline.py detect 0 0.9    test detect(), then try it on training match 0
    #   python baseline/baseline.py tune            choose the threshold on the validation matches
    #   python baseline/baseline.py apply           detect events in all 100 matches, save them
    action = sys.argv[1]
    number = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    if action == "curve":
        check_curve(number)
    elif action == "detect":
        threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.9
        check_detect()
        check_detect_on_match(number, threshold)
    elif action == "tune":
        tune()
    elif action == "apply":
        apply_all()