# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.24.0",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "scipy",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Reconstructing $d'$ and criterion from open trial data

    This notebook starts with the deposited CSV files from Hanning, Fernández & Carrasco (2023). It asks two questions:

    1. Can we reconstruct the published sensitivity measure, $d'$?
    2. What does the same response table show about criterion, $c$?

    The paper plotted $d'$. It did **not** report criterion. The right-hand column below is therefore a new descriptive analysis, not a reproduction of a paper figure.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.stats import norm

    return Path, norm, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Read and decode the CSV files

    The repository contains 26 participant files: 10 from Experiment 1, 9 from Experiment 2a, and 7 from Experiment 2b. Each file has 13 numeric columns and no header or deposited codebook.

    The stimulus and response columns can nevertheless be identified: selecting the orientation at the tested location and comparing it with the response reproduces the recorded correct/incorrect value on every trial. Cue validity and whether the tested location was stimulated follow from the same internal relationships.
    """)
    return


@app.cell
def _(Path, np, pd):
    DATA_DIRECTORY = Path(__file__).parent / "osfstorage" / "Data"

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

    _frames = []
    for _path in sorted(DATA_DIRECTORY.glob("*/*_resMat.csv")):
        _frame = pd.read_csv(_path, header=None, names=RAW_COLUMNS)
        _frame["experiment"] = _path.parent.name
        _frame["participant"] = _path.name.split("_")[0]
        _frames.append(_frame)

    trials = pd.concat(_frames, ignore_index=True)
    trials["test_stimulation"] = np.where(
        trials["test_location_code"].eq(trials["stimulated_hemifield_code"]),
        "Stimulated",
        "Not stimulated",
    )
    trials["validity"] = np.select(
        [
            trials["cue_location_code"].eq(0),
            trials["cue_location_code"].eq(trials["test_location_code"]),
        ],
        ["Neutral", "Valid"],
        default="Invalid",
    )
    trials["stimulus_code"] = np.where(
        trials["test_location_code"].eq(1),
        trials["left_orientation_code"],
        trials["right_orientation_code"],
    ).astype(int)
    trials["correct"] = trials["correct"].astype(bool)
    trials["contrast_pct"] = np.where(
        trials["experiment"].eq("Exp1"),
        trials["analysis_code"].map({1: 2, 2: 7, 3: 13, 4: 24, 5: 46, 6: 85}),
        np.nan,
    )
    trials["time_before_saccade_ms"] = np.where(
        trials["experiment"].ne("Exp1"),
        trials["analysis_code"].map({1: 175, 2: 125, 3: 75, 4: 25}),
        np.nan,
    )

    assert len(_frames) == 26
    assert len(trials) == 32_525
    assert trials["stimulus_code"].eq(trials["response_code"]).equals(
        trials["correct"]
    )
    return DATA_DIRECTORY, RAW_COLUMNS, trials


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Reconstruct the analysis cells

    The paper used

    \[
    d'=\Phi^{-1}(H)-\Phi^{-1}(F).
    \]

    The same hit rate, $H$, and false-alarm rate, $F$, also give

    \[
    c=-\frac{1}{2}\left[\Phi^{-1}(H)+\Phi^{-1}(F)\right].
    \]

    Rates equal to 0 or 1 are replaced by .01 or .99, following the paper. Metrics are calculated within each participant and experimental cell before taking group means.

    The deposited CSVs do not say whether `-1` or `+1` means counter-clockwise. Here `-1` is called the signal. Reversing that choice leaves $d'$ unchanged and reverses the sign of $c$.
    """)
    return


@app.cell
def _(norm, np, pd, trials):
    # The final Experiment 1 exclusions cannot be recovered from the CSVs.
    # Keeping neutral trials and saccade trials below 350 ms gives the closest
    # group-level match to the deposited Figure 2 source values.
    _exp1_saccade = trials["experiment"].eq("Exp1") & trials["validity"].ne(
        "Neutral"
    )
    analysis_trials = trials.loc[
        ~_exp1_saccade | trials["saccade_latency_ms"].lt(350)
    ].copy()

    _experiment_axes = {
        "Exp1": "contrast_pct",
        "Exp2a": "time_before_saccade_ms",
        "Exp2b": "time_before_saccade_ms",
    }
    _rows = []
    for _experiment, _x_column in _experiment_axes.items():
        _experiment_trials = analysis_trials.loc[
            analysis_trials["experiment"].eq(_experiment)
        ]
        _group_columns = [
            "participant",
            _x_column,
            "validity",
            "test_stimulation",
        ]
        for _keys, _cell in _experiment_trials.groupby(
            _group_columns, observed=True, dropna=False
        ):
            _participant, _x_value, _validity, _stimulation = _keys
            _signal = _cell.loc[_cell["stimulus_code"].eq(-1)]
            _noise = _cell.loc[_cell["stimulus_code"].eq(1)]
            _hit_rate = _signal["response_code"].eq(-1).mean()
            _false_alarm_rate = _noise["response_code"].eq(-1).mean()
            _hit_rate = 0.01 if _hit_rate == 0 else 0.99 if _hit_rate == 1 else _hit_rate
            _false_alarm_rate = (
                0.01
                if _false_alarm_rate == 0
                else 0.99
                if _false_alarm_rate == 1
                else _false_alarm_rate
            )
            _z_hit = norm.ppf(_hit_rate)
            _z_false_alarm = norm.ppf(_false_alarm_rate)
            _rows.append(
                {
                    "experiment": _experiment,
                    "participant": _participant,
                    "x_value": _x_value,
                    "validity": _validity,
                    "test_stimulation": _stimulation,
                    "dprime": _z_hit - _z_false_alarm,
                    "criterion": -0.5 * (_z_hit + _z_false_alarm),
                    "n_trials": len(_cell),
                }
            )

    cell_metrics = pd.DataFrame(_rows)
    assert len(analysis_trials) == 31_420
    assert len(cell_metrics) == 616
    assert np.isfinite(cell_metrics[["dprime", "criterion"]]).all().all()
    return analysis_trials, cell_metrics


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Sensitivity and criterion
    """)
    return


@app.cell
def _(cell_metrics, np, plt):
    _colors = {"Valid": "#655ba4", "Neutral": "#777a83", "Invalid": "#3c9878"}
    _experiments = [
        ("Exp1", "Experiment 1 · V1/V2", "Contrast (%)"),
        ("Exp2a", "Experiment 2a · V1/V2", "Time before saccade onset (ms)"),
        ("Exp2b", "Experiment 2b · rFEF+", "Time before saccade onset (ms)"),
    ]
    _conditions = [
        ("Valid", "Stimulated"),
        ("Valid", "Not stimulated"),
        ("Neutral", "Stimulated"),
        ("Neutral", "Not stimulated"),
        ("Invalid", "Stimulated"),
        ("Invalid", "Not stimulated"),
    ]

    reconstruction_figure, _axes = plt.subplots(3, 2, figsize=(13, 12))
    for _row_index, (_experiment, _title, _x_label) in enumerate(_experiments):
        _experiment_metrics = cell_metrics.loc[
            cell_metrics["experiment"].eq(_experiment)
        ]
        for _column_index, (_metric, _metric_label) in enumerate(
            [("dprime", "$d'$"), ("criterion", "Criterion $c$")]
        ):
            _axis = _axes[_row_index, _column_index]
            for _validity, _stimulation in _conditions:
                _values = _experiment_metrics.loc[
                    _experiment_metrics["validity"].eq(_validity)
                    & _experiment_metrics["test_stimulation"].eq(_stimulation)
                ]
                if _values.empty:
                    continue
                _summary = (
                    _values.groupby("x_value", observed=True)[_metric]
                    .agg(["mean", "std", "count"])
                    .reset_index()
                    .sort_values("x_value")
                )
                _summary["sem"] = _summary["std"] / np.sqrt(_summary["count"])
                _axis.errorbar(
                    _summary["x_value"],
                    _summary["mean"],
                    yerr=_summary["sem"],
                    color=_colors[_validity],
                    linestyle="-" if _stimulation == "Stimulated" else "--",
                    marker="o",
                    markerfacecolor=(
                        _colors[_validity]
                        if _stimulation == "Stimulated"
                        else "white"
                    ),
                    capsize=3,
                    label=f"{_validity} · {_stimulation}",
                )
            if _metric == "criterion":
                _axis.axhline(0, color="black", linewidth=1, linestyle=":")
            if _experiment == "Exp1":
                _axis.set_xscale("log")
                _axis.set_xticks([2, 7, 13, 24, 46, 85])
                _axis.set_xticklabels([2, 7, 13, 24, 46, 85])
            else:
                _axis.set_xticks([175, 125, 75, 25])
                _axis.set_xlim(190, 10)
            _axis.set_title(f"{_title} · {_metric_label}")
            _axis.set_xlabel(_x_label)
            _axis.set_ylabel(_metric_label)
            _axis.grid(alpha=0.25)

    _handles, _labels = _axes[0, 0].get_legend_handles_labels()
    reconstruction_figure.legend(
        _handles,
        _labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=3,
        frameon=False,
    )
    reconstruction_figure.suptitle(
        "Figure 1. Participant-first reconstruction from deposited CSV trials",
        y=0.995,
    )
    reconstruction_figure.tight_layout(rect=[0, 0, 1, 0.89])
    plt.show()
    return (reconstruction_figure,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **How to read Figure 1.** The left column shows sensory separation. Higher $d'$ means that the two orientations produced more distinguishable response distributions. The right column shows response preference. The dotted line is $c=0$. Under the chosen `-1 = signal` convention, positive $c$ means fewer `-1` responses and negative $c$ means more. A nonzero criterion is response bias, not poorer vision.

    Filled circles are trials at the stimulated test location. Open circles are trials at the non-stimulated location. Error bars are standard errors across participant-level values. The criterion patterns are descriptive: they were not reported or tested in the paper.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Why reconstruction was necessary

    The deposit contains trial CSVs, processed Excel/MATLAB figure values, and plotting code. It does not contain the code that transformed the trial files into the processed figure values, and it does not contain a CSV data dictionary.

    Experiment 1 also lacks the blink flags, fixation traces, target coordinates, and target-relative landing errors needed to repeat every exclusion in the Methods. Its CSVs contain 24,035 rows; the paper reports 22,679 analyzed trials. The transparent rule used above—retain neutral trials and saccade trials below 350 ms—leaves 22,930 rows and gives the closest group-level $d'$ match among the defensible rules we could test.

    This is a common open-data situation. Data may be available while the codebook, intermediate table, or exact preprocessing chain is not. “Open” makes checking possible; it does not guarantee that every transformation is recoverable.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. One participant does not align perfectly

    Nine of the ten Experiment 1 CSV participants closely match a processed source row after allowing for an apparent row reordering. `S07` remains a poor source-row match. This does **not** make `S07` a behavioral outlier: its mean cell-level $d'=1.334$ and $c=0.035$ are both within the ordinary range of the other participants.

    The mismatch could reflect missing offline exclusions, changed participant ordering, or different raw and processed export versions. The available files cannot decide among these explanations. `S07` is therefore retained rather than silently excluded.

    This too is common in open-data analysis: a mismatch is evidence to document, not permission to remove a participant until the cause is known.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. What reproduced—and what is new

    | Analysis path | Result |
    |---|---|
    | Deposited processed source data → Figures 2, 3, S1, and S2 | All supplied plotted numerical results reproduced. The supplied Figure 3 MATLAB script mislabels the time direction; the paper's axis is correct. |
    | Experiment 2a and 2b CSV trials → published $d'$ cells | Reproduced exactly. |
    | Experiment 1 CSV trials → Figure 2 $d'$ cells | Close at group level; nine participants closely reconcile and one does not. |
    | CSV trials → criterion $c$ | New analysis. Criterion was not shown in the paper and is absent from its figure-source matrices. |

    Reproducing a figure from processed source values and reconstructing it from trials are different achievements. The former checks the published plot. The latter checks the missing path from observations to analysis-ready values.

    ---

    © 2026 Suresh Krishna. This educational notebook is licensed under
    [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
    [Source and attribution details](https://github.com/m2b3/QuantBehavior/blob/main/LICENSE-CONTENT.md).
    """)
    return


if __name__ == "__main__":
    app.run()
