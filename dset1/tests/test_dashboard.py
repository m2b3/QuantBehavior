import numpy as np
import pandas as pd

from dashboard.data import apply_analysis_profile, compute_sdt, load_trials


def test_decoded_response_matches_correctness():
    trials = load_trials()
    predicted = trials["stimulus_code"].eq(trials["response_code"])
    assert predicted.equals(trials["correct"])


def test_known_sdt_values_without_correction():
    # H=.8 and FA=.2 yield d'=1.6832 and an unbiased criterion.
    data = pd.DataFrame(
        {
            "experiment": ["X"] * 20,
            "participant": ["S01"] * 20,
            "stimulus_code": [-1] * 10 + [1] * 10,
            "response_code": [-1] * 8 + [1] * 2 + [-1] * 2 + [1] * 8,
            "correct": [True] * 8 + [False] * 4 + [True] * 8,
            "saccade_latency_ms": [250] * 20,
            "condition": ["A"] * 20,
        }
    )
    result = compute_sdt(data, ["condition"], correction="none")
    assert np.isclose(result.loc[0, "dprime"], 1.683242467, atol=1e-8)
    assert np.isclose(result.loc[0, "criterion"], 0.0, atol=1e-12)


def test_paper_extreme_rate_replacement():
    data = pd.DataFrame(
        {
            "experiment": ["X"] * 4,
            "participant": ["S01"] * 4,
            "stimulus_code": [-1, -1, 1, 1],
            "response_code": [-1, -1, 1, 1],
            "correct": [True] * 4,
            "saccade_latency_ms": [250] * 4,
        }
    )
    result = compute_sdt(data, [], correction="paper")
    assert np.isclose(result.loc[0, "dprime"], 4.652695749, atol=1e-8)
    assert np.isclose(result.loc[0, "criterion"], 0.0, atol=1e-12)


def test_reversing_signal_preserves_dprime_and_flips_criterion():
    trials = load_trials()
    sample = trials[
        trials["experiment"].eq("Exp1")
        & trials["participant"].eq("S01")
        & trials["validity"].eq("Valid")
    ]
    negative = compute_sdt(sample, ["contrast_pct"], signal_code=-1)
    positive = compute_sdt(sample, ["contrast_pct"], signal_code=1)
    merged = negative.merge(positive, on="contrast_pct", suffixes=("_neg", "_pos"))
    assert np.allclose(merged["dprime_neg"], merged["dprime_pos"])
    assert np.allclose(merged["criterion_neg"], -merged["criterion_pos"])


def test_figure_aligned_profile_only_filters_late_exp1_saccades():
    trials = load_trials()
    aligned = apply_analysis_profile(trials, "figure_aligned")

    exp1 = aligned[aligned["experiment"].eq("Exp1")]
    assert len(exp1) == 22_930
    assert exp1.loc[
        exp1["validity"].ne("Neutral"), "saccade_latency_ms"
    ].lt(350).all()
    assert len(aligned[aligned["experiment"].ne("Exp1")]) == len(
        trials[trials["experiment"].ne("Exp1")]
    )


def test_all_rows_profile_is_non_filtering():
    trials = load_trials()
    assert len(apply_analysis_profile(trials, "all_rows")) == len(trials)
