import numpy as np

from dashboard.data import apply_analysis_profile, load_trials
from reconcile_exp1 import score_candidate


def test_figure_aligned_profile_improves_exp1_source_match():
    trials = load_trials()
    exp1 = trials.loc[trials["experiment"].eq("Exp1")]

    all_rows_mae, _, _, _ = score_candidate(exp1)
    aligned_mae, _, _, _ = score_candidate(
        apply_analysis_profile(exp1, "figure_aligned")
    )

    assert np.isclose(all_rows_mae, 0.06044, atol=0.00001)
    assert np.isclose(aligned_mae, 0.05331, atol=0.00001)
    assert aligned_mae < all_rows_mae
