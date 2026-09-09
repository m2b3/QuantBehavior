"""Reproduce the Experiment 1 CSV-to-MAT reconciliation audit."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm

from dashboard.data import load_mat_struct, load_trials


CELL_COLUMNS = ["validity", "test_stimulation", "contrast_pct"]
CONTRASTS = [2, 7, 13, 24, 46, 85]


def dprime_cells(data: pd.DataFrame) -> pd.DataFrame:
    """Vectorized paper-method d-prime for each participant and Figure 2 cell."""
    trials = data.copy()
    trials["yes"] = trials["response_code"].eq(-1).astype(int)
    grouped = (
        trials.groupby(
            ["participant", *CELL_COLUMNS, "stimulus_code"], observed=True
        )["yes"]
        .agg(["size", "sum"])
        .unstack("stimulus_code")
    )
    hit_rate = grouped[("sum", -1)] / grouped[("size", -1)]
    false_alarm_rate = grouped[("sum", 1)] / grouped[("size", 1)]
    hit_rate = hit_rate.mask(hit_rate.eq(0), 0.01).mask(hit_rate.eq(1), 0.99)
    false_alarm_rate = false_alarm_rate.mask(false_alarm_rate.eq(0), 0.01).mask(
        false_alarm_rate.eq(1), 0.99
    )
    values = pd.Series(
        norm.ppf(hit_rate) - norm.ppf(false_alarm_rate), index=hit_rate.index
    )
    return values.rename("dprime").reset_index()


def source_cells() -> pd.DataFrame:
    source = load_mat_struct("fig2_psa")
    rows = []
    for source_row in range(10):
        for validity in ("Valid", "Neutral", "Invalid"):
            for test_stimulation, suffix in (
                ("Stimulated", "T"),
                ("Not stimulated", "n"),
            ):
                for contrast_index, contrast in enumerate(CONTRASTS):
                    rows.append(
                        {
                            "source_row": source_row + 1,
                            "validity": validity,
                            "test_stimulation": test_stimulation,
                            "contrast_pct": contrast,
                            "dprime": source[f"{validity.lower()}_{suffix}"][
                                source_row, contrast_index
                            ],
                        }
                    )
    return pd.DataFrame(rows)


def score_candidate(
    data: pd.DataFrame,
) -> tuple[float, float, float, list[tuple[str, int]]]:
    observed = dprime_cells(data)
    source = source_cells()

    observed_group = observed.groupby(CELL_COLUMNS, observed=True)["dprime"].mean()
    source_group = source.groupby(CELL_COLUMNS, observed=True)["dprime"].mean()
    group_error = observed_group.subtract(source_group).abs()

    source["participant"] = source["source_row"].map(lambda row: f"S{row:02d}")
    same_id = observed.merge(
        source,
        on=["participant", *CELL_COLUMNS],
        suffixes=("_csv", "_source"),
    )
    same_id_mae = float(
        same_id["dprime_csv"].subtract(same_id["dprime_source"]).abs().mean()
    )

    participants = sorted(observed["participant"].unique())
    source_rows = sorted(source["source_row"].unique())
    costs = np.empty((len(participants), len(source_rows)))
    for participant_index, participant in enumerate(participants):
        left = observed.loc[observed["participant"].eq(participant)].set_index(
            CELL_COLUMNS
        )["dprime"]
        for row_index, source_row in enumerate(source_rows):
            right = source.loc[source["source_row"].eq(source_row)].set_index(
                CELL_COLUMNS
            )["dprime"]
            costs[participant_index, row_index] = left.subtract(right).abs().mean()

    raw_indices, source_indices = linear_sum_assignment(costs)
    assignment = [
        (participants[raw_index], source_rows[source_index])
        for raw_index, source_index in zip(raw_indices, source_indices, strict=True)
    ]
    individual_mae = float(costs[raw_indices, source_indices].mean())
    return float(group_error.mean()), same_id_mae, individual_mae, assignment


def main() -> None:
    trials = load_trials()
    exp1 = trials.loc[trials["experiment"].eq("Exp1")].copy()
    neutral = exp1["validity"].eq("Neutral")
    participant_median = exp1.groupby("participant", observed=True)[
        "landing_position_deg"
    ].transform("median")
    centered_landing = exp1["landing_position_deg"].sub(participant_median).abs()

    candidates: list[tuple[str, Callable[[pd.DataFrame], pd.Series]]] = [
        ("all rows", lambda frame: pd.Series(True, index=frame.index)),
        ("latency <= 300", lambda frame: neutral | frame["saccade_latency_ms"].le(300)),
        ("latency <= 350", lambda frame: neutral | frame["saccade_latency_ms"].le(350)),
        ("latency < 350", lambda frame: neutral | frame["saccade_latency_ms"].lt(350)),
        (
            "latency <= 350 and median landing radius <= 2.5",
            lambda frame: neutral
            | (frame["saccade_latency_ms"].le(350) & centered_landing.le(2.5)),
        ),
    ]

    print("Experiment 1 candidate rules")
    print(
        "rule                                             rows  group_MAE  "
        "same_ID_MAE  assigned_MAE"
    )
    best_assignment: list[tuple[str, int]] = []
    for name, inclusion in candidates:
        selected = exp1.loc[inclusion(exp1)]
        group_mae, same_id_mae, individual_mae, assignment = score_candidate(selected)
        print(
            f"{name:48} {len(selected):5d}  {group_mae:9.4f}  "
            f"{same_id_mae:11.4f}  {individual_mae:12.4f}"
        )
        if name == "latency < 350":
            best_assignment = assignment

    print("\nDiagnostic source-row assignment under latency < 350")
    print(", ".join(f"{participant} -> row {row}" for participant, row in best_assignment))
    print("Excel labels rows 1-10 as S01-S10; this diagnostic is not a relabeling.")
    print("See RECONCILIATION.md before interpreting identity or landing position.")


if __name__ == "__main__":
    main()
