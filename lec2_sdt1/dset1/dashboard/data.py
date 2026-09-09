from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "osfstorage" / "Data"
SOURCE_DIR = ROOT / "osfstorage" / "Code" / "Source Data"

RAW_COLUMNS = [
    "block",
    "trial_in_block",
    "design_code",
    "analysis_code",
    "cue_location_code",
    "left_orientation_code",
    "right_orientation_code",
    "test_location_code",
    "stimulated_hemifield_code",
    "response_code",
    "correct",
    "saccade_latency_ms",
    "landing_position_deg",
]

EXPERIMENT_LABELS = {
    "Exp1": "Experiment 1 · V1/V2 · contrast response",
    "Exp2a": "Experiment 2a · V1/V2 · time course",
    "Exp2b": "Experiment 2b · rFEF+ · time course",
}
CONTRAST_LEVELS = {1: 2, 2: 7, 3: 13, 4: 24, 5: 46, 6: 85}
TIME_BIN_LABELS = {
    1: "200–150",
    2: "150–100",
    3: "100–50",
    4: "50–0",
}
TIME_BIN_MIDPOINTS = {1: 175, 2: 125, 3: 75, 4: 25}
SIDE_LABELS = {0: "Neutral", 1: "Left", 2: "Right"}

# The processed Figure 2 source arrays are best reproduced when Experiment 1
# saccade trials at 350 ms or later are omitted. This is an empirical bridge
# between the deposited CSVs and .mat files, not a stated paper cutoff. See
# RECONCILIATION.md for the comparison and limitations.
EXP1_FIGURE_ALIGNED_LATENCY_MS = 350
ANALYSIS_PROFILE_LABELS = {
    "figure_aligned": "Figure-aligned approximation",
    "all_rows": "All deposited rows",
}


@lru_cache(maxsize=1)
def load_trials() -> pd.DataFrame:
    """Load all subject CSVs and add cautiously decoded analysis columns."""
    frames: list[pd.DataFrame] = []
    for path in sorted(DATA_DIR.glob("*/*_resMat.csv")):
        frame = pd.read_csv(path, header=None, names=RAW_COLUMNS)
        frame["experiment"] = path.parent.name
        frame["participant"] = path.name.split("_")[0]
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No trial files were found below {DATA_DIR}")

    data = pd.concat(frames, ignore_index=True)
    data["participant_id"] = data["experiment"] + " / " + data["participant"]
    data["site"] = np.where(data["experiment"].eq("Exp2b"), "rFEF+", "V1/V2")
    data["cue_location"] = data["cue_location_code"].map(SIDE_LABELS)
    data["test_location"] = data["test_location_code"].map(SIDE_LABELS)
    data["stimulated_hemifield"] = data["stimulated_hemifield_code"].map(SIDE_LABELS)
    data["test_stimulation"] = np.where(
        data["test_location_code"].eq(data["stimulated_hemifield_code"]),
        "Stimulated",
        "Not stimulated",
    )
    data["validity"] = np.select(
        [
            data["cue_location_code"].eq(0),
            data["cue_location_code"].eq(data["test_location_code"]),
        ],
        ["Neutral", "Valid"],
        default="Invalid",
    )
    data["stimulus_code"] = np.where(
        data["test_location_code"].eq(1),
        data["left_orientation_code"],
        data["right_orientation_code"],
    ).astype(int)
    data["correct"] = data["correct"].astype(bool)

    is_exp1 = data["experiment"].eq("Exp1")
    data["contrast_pct"] = np.where(
        is_exp1, data["analysis_code"].map(CONTRAST_LEVELS), np.nan
    )
    data["planned_tms_ms"] = np.where(
        ~is_exp1, (data["design_code"] - 1) * 50, np.nan
    )
    data["time_bin"] = np.where(
        ~is_exp1, data["analysis_code"].map(TIME_BIN_LABELS), None
    )
    data["time_before_saccade_ms"] = np.where(
        ~is_exp1, data["analysis_code"].map(TIME_BIN_MIDPOINTS), np.nan
    )
    return data


def apply_analysis_profile(
    data: pd.DataFrame, profile: str = "figure_aligned"
) -> pd.DataFrame:
    """Apply one of the viewer's transparent trial-inclusion profiles.

    Experiment 2 source d-primes are reproduced from every deposited row, so
    both profiles leave those experiments untouched. For Experiment 1, the
    figure-aligned profile retains all neutral/fixation trials and saccade
    trials with the deposited latency below 350 ms. The alternative exposes
    every row exactly as deposited.
    """
    if profile not in ANALYSIS_PROFILE_LABELS:
        raise ValueError(
            f"Unknown analysis profile {profile!r}; expected one of "
            f"{sorted(ANALYSIS_PROFILE_LABELS)}"
        )
    if profile == "all_rows":
        return data.copy()

    exp1_saccade = data["experiment"].eq("Exp1") & data["validity"].ne("Neutral")
    included = ~exp1_saccade | data["saccade_latency_ms"].lt(
        EXP1_FIGURE_ALIGNED_LATENCY_MS
    )
    return data.loc[included].copy()


def _correct_rate(successes: int, total: int, method: str) -> float:
    if total == 0:
        return np.nan
    if method == "loglinear":
        return (successes + 0.5) / (total + 1)
    rate = successes / total
    if method == "paper":
        if rate == 0:
            return 0.01
        if rate == 1:
            return 0.99
    return rate


def compute_sdt(
    data: pd.DataFrame,
    dimensions: list[str],
    correction: str = "paper",
    signal_code: int = -1,
    min_class_trials: int = 1,
) -> pd.DataFrame:
    """Compute subject-first SDT metrics for each requested experimental cell.

    The selected signal code is treated as the "yes" class. The paper defines
    counter-clockwise as signal but the deposited CSVs have no codebook, so the
    dashboard exposes the sign convention explicitly.
    """
    group_columns = ["experiment", "participant"]
    group_columns.extend(column for column in dimensions if column not in group_columns)
    rows: list[dict[str, object]] = []

    for keys, group in data.groupby(group_columns, observed=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        signal = group[group["stimulus_code"].eq(signal_code)]
        noise = group[group["stimulus_code"].eq(-signal_code)]
        n_signal, n_noise = len(signal), len(noise)
        if n_signal < min_class_trials or n_noise < min_class_trials:
            continue

        hits = int(signal["response_code"].eq(signal_code).sum())
        false_alarms = int(noise["response_code"].eq(signal_code).sum())
        hit_rate_raw = hits / n_signal
        fa_rate_raw = false_alarms / n_noise
        hit_rate = _correct_rate(hits, n_signal, correction)
        fa_rate = _correct_rate(false_alarms, n_noise, correction)
        z_hit, z_fa = norm.ppf(hit_rate), norm.ppf(fa_rate)

        row.update(
            {
                "n_trials": len(group),
                "n_signal": n_signal,
                "n_noise": n_noise,
                "hits": hits,
                "false_alarms": false_alarms,
                "hit_rate": hit_rate_raw,
                "false_alarm_rate": fa_rate_raw,
                "dprime": z_hit - z_fa,
                "criterion": -0.5 * (z_hit + z_fa),
                "accuracy": group["correct"].mean(),
                "mean_saccade_latency_ms": group["saccade_latency_ms"].mean(),
            }
        )
        rows.append(row)

    result = pd.DataFrame(rows)
    if not result.empty:
        result.replace([np.inf, -np.inf], np.nan, inplace=True)
    return result


@lru_cache(maxsize=None)
def load_mat_struct(stem: str) -> dict[str, np.ndarray]:
    path = SOURCE_DIR / f"{stem}.mat"
    raw = loadmat(path)[stem][0, 0]
    return {name: np.asarray(raw[name], dtype=float) for name in raw.dtype.names}


def dataset_summary() -> dict[str, int]:
    data = load_trials()
    return {
        "trials": len(data),
        "participants": data["participant_id"].nunique(),
        "experiments": data["experiment"].nunique(),
        "figures": len(list((ROOT / "results").glob("*.png"))),
    }
