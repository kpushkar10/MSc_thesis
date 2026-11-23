"""
Utility Module
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

MATCHES_CSV = Path("matches_100.csv")
LABELS_DIR = Path("data/labels")
AUDIO_DIR = Path("data/audio")

SAMPLE_RATE = 16000

WINDOW_S = 1.0 # 1 s   = 16000 samples
HOP_S = 0.5  # 0.5 s =  8000 samples

WINDOW = int(WINDOW_S * SAMPLE_RATE)
HOP = int(HOP_S * SAMPLE_RATE)

HALVES = [1, 2]
TOLERANCE_S = 5.0 # a detection counts as a hit if it is at most 5 s from a goal

def games_in(split):
    """reads CSV and returns the names of all games in one split"""
    matches = pd.read_csv(MATCHES_CSV)
    return matches[matches["split"] == split]["game"].tolist()

def audio_path(game, half):
    return AUDIO_DIR / game / f"{half}.flac"

def load_goals(game):
    """Goal times of one game as {1: [seconds, ...], 2: [seconds, ...]}.
    The seconds are counted from the start of that half, like the audio file of that half.
    """
    labels = json.loads((LABELS_DIR / game / "Labels-v2.json").read_text())
    goals = {1: [], 2: []}
    for a in labels["annotations"]:
        if a["label"] == "Goal":
            half = int(a["gameTime"].split(" - ")[0])          # "2 - 13:10" -> 2
            goals[half].append(int(a["position"]) / 1000)       # milliseconds -> seconds
    return {half: sorted(times) for half, times in goals.items()}

def n_windows(n_samples):
    """How many whole windows fit into n_samples of audio."""
    return 1 + (n_samples - WINDOW) // HOP

def window_times(n):
    """The centre time in seconds of each of n windows: 0.5, 1.0, 1.5, 2.0, ..."""
    return np.arange(n) * HOP_S + WINDOW_S / 2

def mmss(seconds):
    """137.4 -> "02:17" (easier to compare with the match clock)."""
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

def count_hits(goal_times, detection_times, tolerance=TOLERANCE_S):
    """How many goals of ONE half were found. Each detection can be used for one goal only."""
    unused = list(detection_times)                 # detections not yet matched to a goal
    hits = 0
    for goal in sorted(goal_times):
        close = [d for d in unused if abs(d - goal) <= tolerance]   # detections near this goal
        if close:
            nearest = min(close, key=lambda d: abs(d - goal))       # the closest one
            unused.remove(nearest)                                  # it cannot be used again
            hits += 1
    return hits

def score_match(goals, detections, tolerance=TOLERANCE_S):
    """Compare detections with goals for one match.

    goals and detections both look like {1: [seconds, ...], 2: [seconds, ...]}.
    Returns the counts (tp, fp, fn) and precision, recall and F1.
    """
    tp = sum(count_hits(goals.get(h, []), detections.get(h, []), tolerance) for h in HALVES)
    n_goals = sum(len(goals.get(h, [])) for h in HALVES)
    n_detections = sum(len(detections.get(h, [])) for h in HALVES)

    fp = n_detections - tp      # detections that are not near any goal (false alarms)
    fn = n_goals - tp           # goals that no detection found (misses)

    precision = tp / n_detections if n_detections > 0 else 0.0   # of what we flagged, how much was a goal
    recall = tp / n_goals if n_goals > 0 else 0.0                # of the goals, how many we found
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}

def check_scoring():
    """Small hand-made cases where the right answer is already known"""
    goals = {1: [100.0], 2: []} # one goal, 100 s into the first half
    cases = [
        # (description,                     detections,                expected tp, fp, fn)
        ("exact hit",                       {1: [100.0]},              1, 0, 0),
        ("hit exactly 5 s away",            {1: [105.0]},              1, 0, 0),
        ("5.5 s away = too far",            {1: [105.5]},              0, 1, 1),
        ("two detections for one goal",     {1: [99.0, 102.0]},        1, 1, 0),
        ("right time, wrong half",          {2: [100.0]},              0, 1, 1),
        ("no detections at all",            {},                        0, 0, 1),
    ]
    for name, detections, tp, fp, fn in cases:
        s = score_match(goals, detections)
        assert (s["tp"], s["fp"], s["fn"]) == (tp, fp, fn), name   # stop with an error if it is wrong
        print(f"OK  {name:32s} tp={s['tp']} fp={s['fp']} fn={s['fn']}  "
              f"P={s['precision']:.2f} R={s['recall']:.2f} F1={s['f1']:.2f}")

    # one detection between two goals that are 8 s apart: it may count for only one of them
    s = score_match({1: [100.0, 108.0]}, {1: [104.0]})
    assert (s["tp"], s["fp"], s["fn"]) == (1, 0, 1)
    print(f"OK  {'one detection, two close goals':32s} tp={s['tp']} fp={s['fp']} fn={s['fn']}  "
          f"P={s['precision']:.2f} R={s['recall']:.2f} F1={s['f1']:.2f}")


if __name__ == "__main__":
    # check: goal times and number of windows for one training match that has its audio
    game = next(g for g in games_in("train") if audio_path(g, 1).exists() and audio_path(g, 2).exists())
    goals = load_goals(game)
    print(game)
    print("---> Goals",goals)
    for half in [1, 2]:
        n_samples = sf.info(str(audio_path(game, half))).frames   # length in samples, without loading the audio
        n = n_windows(n_samples)
        times = window_times(n)
        print(f"\nhalf {half}: {n_samples / SAMPLE_RATE:.1f} s of audio -> {n} windows")
        print(f"  first window centres: {times[:3]} ... last: {times[-1]:.1f} s")
        print(f"  goals: {[mmss(t) for t in goals[half]]}  ({goals[half]} s)")