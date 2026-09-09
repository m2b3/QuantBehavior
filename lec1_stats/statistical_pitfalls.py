# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.23.9",
#     "matplotlib",
#     "numpy",
#     "scipy",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Statistical evidence: how convincing patterns can mislead

    This notebook uses small simulations and natural-number examples to develop six connected ideas:

    1. [Spurious correlation and time as a confounder](#1-spurious-correlation-and-time-as-a-confounder)
    2. [P-values, confidence intervals, likelihood, and Bayes](#2-p-values-confidence-intervals-likelihood-and-bayes)
    3. [P-value distributions, multiplicity, and publication bias](#3-p-value-distributions-multiplicity-and-publication-bias)
    4. [Double-dipping and circular analysis](#4-double-dipping-and-circular-analysis)
    5. [Hierarchical data and invalid pooling](#5-hierarchical-data-and-invalid-pooling)
    6. [Confounding and Simpson's paradox](#6-confounding-and-simpsons-paradox)
    7. [Diagnostic tests, base rates, and predictive value](#7-diagnostic-tests-base-rates-and-predictive-value)
    8. [Selection and Berkson's fallacy](#8-selection-and-berksons-fallacy)

    The recurring question is not merely “is there a pattern?” It is: **what process could have produced the pattern, what comparison answers the question, and what information was selected before we saw it?**

    ---

    © 2026 Suresh Krishna. Except for separately attributed third-party material, this educational notebook is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats

    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return np, plt, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Spurious correlation and time as a confounder

    A correlation says that two measured quantities vary together. By itself, it does not say that changing one quantity would change the other.

    Two common ways to obtain a misleading correlation are:

    - **Chance:** inspect enough unrelated pairs and an impressive-looking one will eventually appear.
    - **A shared cause:** both variables change with something else. Time is especially dangerous because population, technology, prices, measurement practices, and many biological variables can all drift together.

    In the next demonstration, time affects both measurements. There is no direct arrow from measurement X to measurement Y.
    """)
    return


@app.cell
def _(mo):
    time_strength = mo.ui.slider(
        start=0.0,
        stop=2.0,
        step=0.1,
        value=1.0,
        label="Strength of the shared time trend",
        show_value=True,
    )
    searched_pairs = mo.ui.slider(
        start=1,
        stop=100,
        step=1,
        value=20,
        label="Unrelated pairs inspected",
        show_value=True,
    )
    mo.hstack([time_strength, searched_pairs], widths="equal", gap=2)
    return searched_pairs, time_strength


@app.cell
def _(mo, np, plt, stats, time_strength):
    _rng = np.random.default_rng(601)
    _n = 80
    _time = np.linspace(0, 10, _n)
    _strength = time_strength.value
    _x = 2.0 + 0.9 * _strength * _time + _rng.normal(0, 2.0, _n)
    _y = 8.0 + 1.1 * _strength * _time + _rng.normal(0, 2.0, _n)

    _raw_r, _raw_p = stats.pearsonr(_x, _y)
    _x_residual = _x - np.polyval(np.polyfit(_time, _x, 1), _time)
    _y_residual = _y - np.polyval(np.polyfit(_time, _y, 1), _time)
    _residual_r, _residual_p = stats.pearsonr(_x_residual, _y_residual)

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4))
    _points = _axes[0].scatter(
        _x,
        _y,
        c=_time,
        cmap="viridis",
        edgecolor="white",
        linewidth=0.5,
    )
    _fit = np.polyfit(_x, _y, 1)
    _line_x = np.linspace(_x.min(), _x.max(), 100)
    _axes[0].plot(_line_x, np.polyval(_fit, _line_x), color="black")
    _axes[0].set(
        xlabel="Measurement X",
        ylabel="Measurement Y",
        title=f"Before accounting for time: r = {_raw_r:.2f}",
    )
    _fig.colorbar(_points, ax=_axes[0], label="Time")

    _axes[1].scatter(
        _x_residual,
        _y_residual,
        color="#7b3294",
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
    )
    _residual_fit = np.polyfit(_x_residual, _y_residual, 1)
    _residual_line_x = np.linspace(_x_residual.min(), _x_residual.max(), 100)
    _axes[1].plot(
        _residual_line_x,
        np.polyval(_residual_fit, _residual_line_x),
        color="black",
    )
    _axes[1].axhline(0, color="0.75", linewidth=1)
    _axes[1].axvline(0, color="0.75", linewidth=1)
    _axes[1].set(
        xlabel="X after removing its time trend",
        ylabel="Y after removing its time trend",
        title=f"After accounting for time: r = {_residual_r:.2f}",
    )
    _fig.suptitle("Figure 1. A shared time trend can manufacture an association")
    _fig.tight_layout()

    mo.vstack(
        [
            _fig,
            mo.callout(
                mo.md(
                    f"""
                    Before accounting for time, X and Y have correlation **{_raw_r:.2f}**. After subtracting each variable's linear time trend, their remaining correlation is **{_residual_r:.2f}**. The raw association was mostly information about time, not evidence that X changes Y.
                    """
                ),
                kind="info",
            ),
        ]
    )
    return


@app.cell
def _(mo, np, plt, searched_pairs, stats):
    _rng = np.random.default_rng(882)
    _observations = 30
    _number_searched = searched_pairs.value
    # Generate the full set once, then reveal a longer prefix as the slider moves.
    # The search is therefore nested: adding candidates cannot make the winner weaker.
    _xs = _rng.normal(size=(100, _observations))[:_number_searched]
    _ys = _rng.normal(size=(100, _observations))[:_number_searched]
    _correlations = np.array(
        [stats.pearsonr(_xs[_i], _ys[_i]).statistic for _i in range(_number_searched)]
    )
    _winner = int(np.argmax(np.abs(_correlations)))
    _winner_test = stats.pearsonr(_xs[_winner], _ys[_winner])

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4))
    _axes[0].hist(
        _correlations,
        bins=np.linspace(-1, 1, 21),
        color="#80cdc1",
        edgecolor="white",
    )
    _axes[0].axvline(_correlations[_winner], color="#c51b7d", linewidth=3)
    _axes[0].set(
        xlabel="Observed correlation",
        ylabel="Number of pairs",
        title=f"All {_number_searched} unrelated pairs",
    )

    _axes[1].scatter(
        _xs[_winner],
        _ys[_winner],
        color="#c51b7d",
        edgecolor="white",
        linewidth=0.5,
    )
    _winner_fit = np.polyfit(_xs[_winner], _ys[_winner], 1)
    _winner_line = np.linspace(_xs[_winner].min(), _xs[_winner].max(), 100)
    _axes[1].plot(
        _winner_line,
        np.polyval(_winner_fit, _winner_line),
        color="black",
    )
    _axes[1].set(
        xlabel="Unrelated X",
        ylabel="Unrelated Y",
        title=f"The most impressive pair: r = {_winner_test.statistic:.2f}",
    )
    _fig.suptitle("Figure 2. Looking first and reporting the winner")
    _fig.tight_layout()

    mo.vstack(
        [
            _fig,
            mo.callout(
                mo.md(
                    f"The selected pair has an ordinary, unadjusted p-value of **{_winner_test.pvalue:.3f}**. That number describes a test chosen in advance; it does not account for the fact that we inspected **{_number_searched}** pairs and deliberately kept the strongest one. Move the slider and watch the winner become more extreme."
                ),
                kind="warn",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The jelly-bean lesson

    [xkcd 882, “Significant”](https://xkcd.com/882/) is a compact version of the same problem: test many jelly-bean colours, announce only the colour whose result crosses the usual cutoff, and make one chance result look like a planned discovery.

    <p style="text-align:center">
      <a href="https://xkcd.com/882/" target="_blank" rel="noopener noreferrer">
        <img src="https://imgs.xkcd.com/comics/significant.png" alt="xkcd 882, Significant: repeated tests of jelly-bean colours produce one nominally significant association" style="max-width:900px;width:100%" />
      </a>
    </p>

    Comic by Randall Munroe, [xkcd 882](https://xkcd.com/882/), licensed [CC BY-NC 2.5](https://creativecommons.org/licenses/by-nc/2.5/). The notebook simulations are original and do not assume that jelly beans cause acne.

    The important distinction is between **a question specified before seeing the data** and **a winner selected after many questions were tried**. We return to the numerical consequences in Section 3.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. P-values, confidence intervals, likelihood, and Bayes
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Five related ideas answer different questions

    | Quantity | The question it answers |
    |---|---|
    | **P-value** | If a specified null model generated repeated data, how unusual is a result at least this extreme? |
    | **Confidence interval** | Which parameter values would survive the corresponding repeated-sampling tests? |
    | **Likelihood** | With the observed data held fixed, which parameter values predicted these data better? |
    | **Likelihood ratio** | How much better did one specified model predict the observed data than another? |
    | **Bayesian posterior** | After combining an explicit prior distribution with the likelihood, what uncertainty remains about the parameter or model? |

    These are not interchangeable translations. A likelihood is a function of a parameter for fixed data; it does not become a probability distribution over that parameter until a prior is supplied and the result is normalized. For two simple hypotheses, posterior odds equal prior odds multiplied by their likelihood ratio.

    The next demonstration observes heads and tails from one coin. The null model says the chance of heads is one half; a comparison model says seven tenths. The Bayesian panel instead allows every possible bias and starts with a symmetric prior centered on one half.
    """)
    return


@app.cell
def _(mo):
    coin_heads = mo.ui.slider(
        0,
        20,
        step=1,
        value=14,
        label="Heads observed in 20 flips",
        show_value=True,
    )
    coin_prior_strength = mo.ui.slider(
        2,
        40,
        step=2,
        value=4,
        label="Prior strength around a fair coin",
        show_value=True,
    )
    mo.hstack([coin_heads, coin_prior_strength], widths="equal", gap=2)
    return coin_heads, coin_prior_strength


@app.cell
def _(coin_heads, coin_prior_strength, mo, np, plt, stats):
    _trials = 20
    _heads = coin_heads.value
    _possible_heads = np.arange(_trials + 1)
    _null_probabilities = stats.binom.pmf(_possible_heads, _trials, 0.5)
    _at_least_as_extreme = (
        np.abs(_possible_heads - _trials / 2)
        >= abs(_heads - _trials / 2)
    )
    _p_value = stats.binomtest(_heads, _trials, p=0.5).pvalue

    _theta = np.linspace(0.001, 0.999, 800)
    _likelihood = stats.binom.pmf(_heads, _trials, _theta)
    _relative_likelihood = _likelihood / np.max(_likelihood)
    _likelihood_fair = stats.binom.pmf(_heads, _trials, 0.5)
    _likelihood_seventy = stats.binom.pmf(_heads, _trials, 0.7)
    _likelihood_ratio = _likelihood_seventy / _likelihood_fair
    _point_model_posterior = _likelihood_ratio / (1 + _likelihood_ratio)

    _prior_alpha = coin_prior_strength.value / 2
    _prior_beta = coin_prior_strength.value / 2
    _posterior_alpha = _prior_alpha + _heads
    _posterior_beta = _prior_beta + _trials - _heads
    _prior_density = stats.beta.pdf(_theta, _prior_alpha, _prior_beta)
    _posterior_density = stats.beta.pdf(
        _theta,
        _posterior_alpha,
        _posterior_beta,
    )
    _credible_low, _credible_high = stats.beta.ppf(
        [0.025, 0.975],
        _posterior_alpha,
        _posterior_beta,
    )
    _confidence_interval = stats.binomtest(_heads, _trials).proportion_ci(
        confidence_level=0.95,
        method="exact",
    )

    _fig, _axes = plt.subplots(1, 3, figsize=(13, 3.8))
    _bar_colors = np.where(_at_least_as_extreme, "#d73027", "#bdbdbd")
    _axes[0].bar(
        _possible_heads,
        _null_probabilities,
        color=_bar_colors,
        edgecolor="white",
    )
    _axes[0].axvline(_heads, color="black", linestyle="--")
    _axes[0].set(
        xlabel="Heads in 20 flips",
        ylabel="Probability if the coin is fair",
        title=f"P-value: tail area = {_p_value:.3f}",
    )

    _axes[1].plot(_theta, _relative_likelihood, color="#2166ac", linewidth=2)
    _axes[1].scatter(
        [0.5, 0.7],
        [
            _likelihood_fair / np.max(_likelihood),
            _likelihood_seventy / np.max(_likelihood),
        ],
        color=["black", "#d73027"],
        zorder=3,
    )
    _axes[1].set(
        xlabel="Possible chance of heads",
        ylabel="Likelihood relative to its maximum",
        title=f"Likelihood ratio: L(.70)/L(.50) = {_likelihood_ratio:.2f}",
    )

    _axes[2].plot(_theta, _prior_density, color="0.55", label="Prior")
    _axes[2].plot(_theta, _posterior_density, color="#1b9e77", linewidth=2, label="Posterior")
    _axes[2].axvspan(
        _credible_low,
        _credible_high,
        color="#1b9e77",
        alpha=0.18,
        label="95% credible interval",
    )
    _axes[2].set(
        xlabel="Possible chance of heads",
        ylabel="Density",
        title="Bayes: prior updated by the likelihood",
    )
    _axes[2].legend(fontsize=8)
    _fig.suptitle("Figure 3A. The same data viewed through different inferential questions")
    _fig.tight_layout()

    mo.vstack(
        [
            _fig,
            mo.callout(
                mo.md(
                    f"""
                    - The null-model p-value is **{_p_value:.3f}**.
                    - The 95% confidence interval runs from **{_confidence_interval.low:.2f}** to **{_confidence_interval.high:.2f}**.
                    - The observed sequence is **{_likelihood_ratio:.2f} times** as likely under the seven-tenths model as under the fair-coin model.
                    - If those two point models had equal prior odds, the data would update the probability assigned to the seven-tenths model to **{_point_model_posterior:.2f}**.
                    - Under the displayed continuous prior, the 95% Bayesian credible interval runs from **{_credible_low:.2f}** to **{_credible_high:.2f}**.

                    Similar-looking numbers can have different definitions because they condition on different things.
                    """
                ),
                kind="info",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What did a small p-value establish?

    Suppose a drug study tests “no effect,” obtains a p-value below 0.05, and favors “some effect.” Choose the statement that logically follows from that result alone.
    """)
    return


@app.cell
def _(mo):
    p_quiz = mo.ui.radio(
        options={
            "1. The no-effect hypothesis is false.": "1",
            "2. The some-effect hypothesis is true.": "2",
            "3. The no-effect hypothesis is probably false.": "3",
            "4. The some-effect hypothesis is probably true.": "4",
            "5. Both 1 and 2.": "5",
            "6. Both 3 and 4.": "6",
            "7. None of the above.": "7",
        },
        value=None,
        label="What has been shown?",
    )
    p_quiz
    return (p_quiz,)


@app.cell
def _(mo, p_quiz):
    if p_quiz.value is None:
        _quiz_result = mo.callout(
            mo.md("Choose an answer, then interpret the p-value in the direction in which it was calculated."),
            kind="neutral",
        )
    elif p_quiz.value == "7":
        _quiz_result = mo.callout(
            mo.md(
                "**Correct: none of the above.** The p-value asks how surprising these data would be if the no-effect model were true. It is not the chance that the hypothesis is true or false, and it does not logically prove either hypothesis."
            ),
            kind="success",
        )
    else:
        _quiz_result = mo.callout(
            mo.md(
                "That conclusion is stronger than the p-value permits. A p-value conditions on the no-effect model; it does not assign a probability to that model or its alternative."
            ),
            kind="warn",
        )
    _quiz_result
    return


@app.cell
def _(mo):
    observed_t = mo.ui.slider(
        start=0.0,
        stop=4.0,
        step=0.1,
        value=2.0,
        label="Absolute observed t statistic",
        show_value=True,
    )
    null_df = mo.ui.slider(
        start=5,
        stop=100,
        step=1,
        value=20,
        label="Degrees of freedom",
        show_value=True,
    )
    mo.hstack([observed_t, null_df], widths="equal", gap=2)
    return null_df, observed_t


@app.cell
def _(mo, np, null_df, observed_t, plt, stats):
    _df = null_df.value
    _t_observed = observed_t.value
    _x = np.linspace(-5, 5, 1200)
    _density = stats.t.pdf(_x, df=_df)
    _p_value = 2 * stats.t.sf(_t_observed, df=_df)
    _left_tail = _x <= -_t_observed
    _right_tail = _x >= _t_observed

    _fig, _ax = plt.subplots(figsize=(10, 3.8))
    _ax.plot(_x, _density, color="black", label="What repeated studies produce if the null model is true")
    _ax.fill_between(
        _x[_left_tail],
        0,
        _density[_left_tail],
        color="#d73027",
        alpha=0.45,
        label="At least this far from zero",
    )
    _ax.fill_between(
        _x[_right_tail],
        0,
        _density[_right_tail],
        color="#d73027",
        alpha=0.45,
    )
    _ax.axvline(_t_observed, color="#d73027", linestyle="--")
    _ax.axvline(-_t_observed, color="#d73027", linestyle="--")
    _ax.set(
        xlabel="t statistic under the no-effect model",
        ylabel="Density",
        title=f"Figure 3B. Two-sided p-value = {_p_value:.3f}",
    )
    _ax.legend(fontsize=8)
    _fig.tight_layout()

    mo.vstack(
        [
            _fig,
            mo.md(
                f"**Plain language:** if the no-effect model and its assumptions generated repeated studies, a result at least this far from zero would occur with chance **{_p_value:.3f}**. This calculation does not say there is a {_p_value:.3f} chance of no effect."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Confidence intervals, error bars, and paired observations

    A confidence interval shows the range of effect sizes that remain reasonably compatible with the data and model. It makes uncertainty visible in the units of the effect. Always say whether an error bar is a standard deviation, standard error, or confidence interval; they answer different questions.

    Pairing matters when the same individuals are measured twice. A paired test asks whether the **within-person changes** are consistently above or below zero. Two separate groups of error bars throw away that pairing.
    """)
    return


@app.cell
def _(mo):
    paired_n = mo.ui.slider(
        start=6,
        stop=60,
        step=1,
        value=20,
        label="Number of paired participants",
        show_value=True,
    )
    paired_effect = mo.ui.slider(
        start=0.0,
        stop=8.0,
        step=0.5,
        value=3.0,
        label="Average within-person change",
        show_value=True,
    )
    mo.hstack([paired_n, paired_effect], widths="equal", gap=2)
    return paired_effect, paired_n


@app.cell
def _(mo, np, paired_effect, paired_n, plt, stats):
    _rng = np.random.default_rng(91)
    _n = paired_n.value
    _before = _rng.normal(50, 10, _n)
    _change = paired_effect.value + _rng.normal(0, 4, _n)
    _after = _before + _change
    _paired_test = stats.ttest_rel(_after, _before)
    _mean_change = float(np.mean(_change))
    _change_se = stats.sem(_change)
    _margin = stats.t.ppf(0.975, df=_n - 1) * _change_se
    _ci_low, _ci_high = _mean_change - _margin, _mean_change + _margin

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4))
    for _person in range(_n):
        _axes[0].plot([0, 1], [_before[_person], _after[_person]], color="0.75", linewidth=1)
    _axes[0].scatter(np.zeros(_n), _before, color="#4575b4", label="Before", zorder=3)
    _axes[0].scatter(np.ones(_n), _after, color="#d73027", label="After", zorder=3)
    _axes[0].set_xticks([0, 1], ["Before", "After"])
    _axes[0].set(ylabel="Measurement", title="Keep each person's observations connected")
    _axes[0].legend()

    _axes[1].axhline(0, color="black", linestyle="--")
    _axes[1].scatter(np.ones(_n), _change, color="#7b3294", alpha=0.7)
    _axes[1].errorbar(
        1.35,
        _mean_change,
        yerr=_margin,
        fmt="o",
        color="black",
        capsize=7,
        label="Mean change and 95% CI",
    )
    _axes[1].set_xlim(0.75, 1.6)
    _axes[1].set_xticks([1, 1.35], ["Individuals", "Mean"])
    _axes[1].set(ylabel="After minus before", title="The paired analysis uses the changes")
    _axes[1].legend(fontsize=8)
    _fig.suptitle("Figure 4. Pairing can remove irrelevant person-to-person variation")
    _fig.tight_layout()

    mo.vstack(
        [
            _fig,
            mo.callout(
                mo.md(
                    f"The estimated mean change is **{_mean_change:.2f}**, with a 95% confidence interval from **{_ci_low:.2f}** to **{_ci_high:.2f}**. The paired-test p-value is **{_paired_test.pvalue:.3g}**."
                ),
                kind="info",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Absence of evidence is not evidence of absence

    A large p-value can arise because the effect is small, because the study is noisy, or because the sample is too small to distinguish those possibilities. “We did not detect an effect” is therefore not the same statement as “we showed that no meaningful effect exists.”

    To gather evidence of practical absence, define the smallest effect that would matter and estimate the effect precisely enough to rule out effects at least that large. Equivalence tests are one formal way to do this.
    """)
    return


@app.cell
def _(mo):
    absence_n = mo.ui.slider(
        start=8,
        stop=400,
        step=8,
        value=24,
        label="Sample size",
        show_value=True,
    )
    meaningful_effect = mo.ui.slider(
        start=0.1,
        stop=0.8,
        step=0.05,
        value=0.3,
        label="Smallest standardized effect that matters",
        show_value=True,
    )
    mo.hstack([absence_n, meaningful_effect], widths="equal", gap=2)
    return absence_n, meaningful_effect


@app.cell
def _(absence_n, meaningful_effect, mo, np, plt, stats):
    _estimate = 0.05
    _se = 1 / np.sqrt(absence_n.value)
    _margin = stats.norm.ppf(0.975) * _se
    _ci_low, _ci_high = _estimate - _margin, _estimate + _margin
    _delta = meaningful_effect.value
    _p_value = 2 * stats.norm.sf(abs(_estimate / _se))
    _equivalent = _ci_low > -_delta and _ci_high < _delta
    _status = (
        "The interval is narrow enough to support practical absence."
        if _equivalent
        else "The result is inconclusive: meaningful effects have not been ruled out."
    )

    _limit = max(0.9, abs(_ci_low) + 0.1, abs(_ci_high) + 0.1)
    _fig, _ax = plt.subplots(figsize=(10, 2.8))
    _ax.axvspan(-_delta, _delta, color="#a6dba0", alpha=0.45, label="Effects too small to matter")
    _ax.axvline(0, color="black", linestyle="--")
    _ax.errorbar(
        _estimate,
        0,
        xerr=_margin,
        fmt="o",
        color="#5e3c99",
        capsize=8,
        linewidth=2,
        label="Estimate and 95% CI",
    )
    _ax.set_xlim(-_limit, _limit)
    _ax.set_ylim(-0.7, 0.7)
    _ax.set_yticks([])
    _ax.set(xlabel="Standardized effect", title="Figure 5. A non-significant result may be precise or uninformative")
    _ax.legend(loc="upper right", fontsize=8)
    _fig.tight_layout()

    mo.vstack(
        [
            _fig,
            mo.callout(
                mo.md(
                    f"The ordinary no-effect test has p = **{_p_value:.3f}**, but that is not the key question here. The 95% interval runs from **{_ci_low:.2f}** to **{_ci_high:.2f}**. **{_status}**"
                ),
                kind="success" if _equivalent else "warn",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Difference of significance is not significance of the difference

    Suppose treatment A is significantly different from zero and treatment B is not. That does **not** establish that A and B differ. The scientific question requires a direct test of the contrast between A and B—often an interaction.

    The example below assumes two independent effect estimates. One interval narrowly excludes zero; the other narrowly includes zero. Their wide overlap should already suggest that their difference is uncertain.
    """)
    return


@app.cell
def _(mo, np, plt, stats):
    _effect_a, _se_a = 0.55, 0.25
    _effect_b, _se_b = 0.40, 0.22
    _difference = _effect_a - _effect_b
    _difference_se = np.sqrt(_se_a**2 + _se_b**2)
    _estimates = np.array([_effect_a, _effect_b, _difference])
    _ses = np.array([_se_a, _se_b, _difference_se])
    _p_values = 2 * stats.norm.sf(np.abs(_estimates / _ses))
    _margins = stats.norm.ppf(0.975) * _ses

    _fig, _ax = plt.subplots(figsize=(10, 3.5))
    _positions = np.arange(3)
    _colors = ["#2166ac", "#67a9cf", "#762a83"]
    for _position, _estimate_value, _margin_value, _color in zip(
        _positions, _estimates, _margins, _colors
    ):
        _ax.errorbar(
            _position,
            _estimate_value,
            yerr=_margin_value,
            fmt="o",
            color=_color,
            capsize=7,
            markersize=8,
        )
    _ax.axhline(0, color="black", linestyle="--")
    _ax.set_xticks(_positions, ["Effect A", "Effect B", "A minus B"])
    _ax.set(ylabel="Estimate and 95% CI", title="Figure 6. Test the comparison you want to claim")
    for _position, _p in zip(_positions, _p_values):
        _ax.text(_position, _estimates[_position] + _margins[_position] + 0.07, f"p = {_p:.3f}", ha="center")
    _ax.set_ylim(-0.65, 1.2)
    _fig.tight_layout()

    mo.vstack(
        [
            _fig,
            mo.callout(
                mo.md(
                    f"A has p = **{_p_values[0]:.3f}** and B has p = **{_p_values[1]:.3f}**, but their direct difference has p = **{_p_values[2]:.3f}**. The first two tests compare each effect with zero; the third asks whether the effects differ from each other."
                ),
                kind="warn",
            ),
            mo.md(
                "See Nieuwenhuis, Forstmann, and Wagenmakers (2011), [*Erroneous analyses of interactions in neuroscience*](https://doi.org/10.1038/nn.2886)."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. P-value distributions, multiplicity, and publication bias

    Several related processes can make a scientific record look stronger than the underlying evidence:

    - **Garden of forking paths:** reasonable-looking choices depend on features noticed in the data, silently creating many possible analyses.
    - **Fishing or p-hacking:** analyses are tried until a favorable result appears, after which only the successful route is reported.
    - **HARKing:** a result observed in the data is rewritten as if it had been predicted in advance.
    - **File-drawer effect:** studies with disappointing results remain unseen, so the published literature is a selected sample.

    These labels describe different behaviors and mechanisms, but they share a consequence: the reported p-value no longer represents the full process that selected the result.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What a distribution of p-values can reveal

    Across many valid continuous tests whose null hypotheses are all true, p-values are uniformly spread between zero and one: about the same number fall in every equal-width bin. Real effects add extra small p-values, with the shape depending on effect size and statistical power. Selection and analytical flexibility can reshape the distribution again.

    Therefore, a single p-value and a collection of p-values answer different questions. A histogram can diagnose broad patterns, but it cannot by itself tell whether small values arose from real effects, selective reporting, model failure, or misconduct.
    """)
    return


@app.cell
def _(mo):
    pdist_sample_size = mo.ui.slider(
        5,
        100,
        step=5,
        value=25,
        label="Sample size per simulated test",
        show_value=True,
    )
    pdist_effect = mo.ui.slider(
        0.0,
        0.8,
        step=0.05,
        value=0.3,
        label="Standardized effect when a signal exists",
        show_value=True,
    )
    mo.hstack([pdist_sample_size, pdist_effect], widths="equal", gap=2)
    return pdist_effect, pdist_sample_size


@app.cell
def _(mo, np, pdist_effect, pdist_sample_size, plt, stats):
    _rng = np.random.default_rng(307)
    _experiments = 5000
    _null_z = _rng.normal(0, 1, _experiments)
    _signal_z = _rng.normal(
        pdist_effect.value * np.sqrt(pdist_sample_size.value),
        1,
        _experiments,
    )
    _null_p = 2 * stats.norm.sf(np.abs(_null_z))
    _signal_p = 2 * stats.norm.sf(np.abs(_signal_z))
    _bins = np.linspace(0, 1, 21)

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    _axes[0].hist(_null_p, bins=_bins, color="0.6", edgecolor="white")
    _axes[0].axhline(_experiments / 20, color="black", linestyle="--", label="Uniform expectation")
    _axes[0].set(
        xlabel="p-value",
        ylabel="Simulated tests",
        title="Every null hypothesis is true",
    )
    _axes[0].legend(fontsize=8)
    _axes[1].hist(_signal_p, bins=_bins, color="#1b9e77", edgecolor="white")
    _axes[1].set(
        xlabel="p-value",
        title="Every test contains the displayed effect",
    )
    _fig.suptitle("Figure 7A. P-value distributions depend on the data-generating process")
    _fig.tight_layout()

    _null_below = 100 * np.mean(_null_p < 0.05)
    _signal_below = 100 * np.mean(_signal_p < 0.05)
    mo.vstack(
        [
            _fig,
            mo.callout(
                mo.md(
                    f"Under the null, **{_null_below:.1f}%** of simulated p-values fall below 0.05. With the displayed effect and sample size, **{_signal_below:.1f}%** do. The remaining large p-values do not prove that those simulated effects were absent; they are missed effects."
                ),
                kind="info",
            ),
        ]
    )
    return


@app.cell
def _(mo):
    family_size = mo.ui.slider(
        start=1,
        stop=100,
        step=1,
        value=20,
        label="Independent null hypotheses tested",
        show_value=True,
    )
    family_size
    return (family_size,)


@app.cell
def _(family_size, mo, np, plt):
    _tests = family_size.value
    _alpha = 0.05
    _rng = np.random.default_rng(311)
    _repetitions = 10000
    _false_positive_counts = np.sum(
        _rng.random((_repetitions, _tests)) < _alpha,
        axis=1,
    )
    _chance_any = 1 - (1 - _alpha) ** _tests
    _empirical_chance = np.mean(_false_positive_counts > 0)
    _maximum_shown = min(int(_false_positive_counts.max()), 10)
    _bins = np.arange(-0.5, _maximum_shown + 1.5, 1)

    _fig, _ax = plt.subplots(figsize=(9, 3.5))
    _ax.hist(
        np.minimum(_false_positive_counts, _maximum_shown),
        bins=_bins,
        color="#fdae61",
        edgecolor="white",
    )
    _ax.set_xticks(range(_maximum_shown + 1))
    _ax.set(
        xlabel="False-positive results in one family of tests",
        ylabel="Simulated repetitions",
        title="Figure 7B. Multiplicity turns a rare event into a common event",
    )
    _fig.tight_layout()

    mo.vstack(
        [
            _fig,
            mo.callout(
                mo.md(
                    f"With **{_tests}** independent tests and no real effects, the chance of at least one p-value below 0.05 is about **{100 * _chance_any:.1f}%**. The simulation gives **{100 * _empirical_chance:.1f}%**. This is the arithmetic behind the jelly-bean comic."
                ),
                kind="info",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The file drawer changes what the literature looks like

    Under a true no-effect model, estimates scatter symmetrically around zero. Small studies scatter more widely than large studies. If positive or “significant” studies are easier to publish, the observed literature is no longer a symmetric sample of all completed work.

    A funnel plot can reveal some forms of asymmetry, but it is a diagnostic rather than proof of a particular cause. Heterogeneity, study quality, and selective reporting can produce similar-looking patterns.
    """)
    return


@app.cell
def _(mo):
    file_drawer_keep = mo.ui.slider(
        start=0,
        stop=100,
        step=5,
        value=10,
        label="Percent of non-significant studies retained",
        show_value=True,
    )
    file_drawer_keep
    return (file_drawer_keep,)


@app.cell
def _(file_drawer_keep, mo, np, plt, stats):
    _rng = np.random.default_rng(1201)
    _studies = 500
    _standard_errors = _rng.uniform(0.06, 0.45, _studies)
    _estimates = _rng.normal(0, _standard_errors)
    _p_values = 2 * stats.norm.sf(np.abs(_estimates / _standard_errors))
    _significant_positive = (_p_values < 0.05) & (_estimates > 0)
    _retain_other = _rng.random(_studies) < file_drawer_keep.value / 100
    _published = _significant_positive | _retain_other

    _fig, _axes = plt.subplots(1, 3, figsize=(12, 3.8))
    _axes[0].scatter(_estimates, _standard_errors, color="0.65", alpha=0.55, s=18)
    _axes[0].axvline(0, color="black", linestyle="--")
    _axes[0].invert_yaxis()
    _axes[0].set(xlabel="Estimated effect", ylabel="Standard error", title="All completed studies")

    _axes[1].scatter(
        _estimates[_published],
        _standard_errors[_published],
        color="#d73027",
        alpha=0.65,
        s=20,
    )
    _axes[1].axvline(0, color="black", linestyle="--")
    _axes[1].invert_yaxis()
    _axes[1].set(xlabel="Estimated effect", ylabel="Standard error", title="The visible literature")

    _axes[2].hist(
        _p_values[_published],
        bins=np.linspace(0, 1, 21),
        color="#d73027",
        edgecolor="white",
    )
    _axes[2].axvline(0.05, color="black", linestyle="--")
    _axes[2].set(xlabel="Reported p-value", ylabel="Published studies", title="Selected p-values")
    _fig.suptitle("Figure 8. Selective visibility distorts effects and p-values")
    _fig.tight_layout()

    _all_mean = float(np.mean(_estimates))
    _published_mean = float(np.mean(_estimates[_published]))
    mo.vstack(
        [
            _fig,
            mo.callout(
                mo.md(
                    f"The average effect across all simulated studies is **{_all_mean:.3f}**. Among the **{int(np.sum(_published))}** visible studies it is **{_published_mean:.3f}**, even though the true effect is zero. A pile-up near a threshold can be a warning sign, but a histogram alone cannot identify intent or prove misconduct."
                ),
                kind="warn",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### False discovery rate and the Benjamini–Hochberg procedure

    Correcting for multiple tests is not one universal operation. A family-wise method asks to avoid even one false positive in the family. False-discovery-rate control asks a different question: among the results called discoveries, how large a false fraction are we willing to tolerate on average under repeated use?

    The Benjamini–Hochberg procedure sorts the p-values, compares them with a rising boundary, and keeps the initial set that passes. The simulation knows which hypotheses really contain a signal, so it can reveal the false discoveries that real data do not label for us.

    A **q-value** is the smallest false-discovery-rate level at which a particular result would enter the discovery set. It is not the probability that this individual result is false.
    """)
    return


@app.cell
def _(mo):
    fdr_tests = mo.ui.slider(20, 300, step=10, value=100, label="Number of tests", show_value=True)
    fdr_signal_share = mo.ui.slider(0, 50, step=5, value=20, label="Percent with a real signal", show_value=True)
    fdr_effect = mo.ui.slider(0.0, 4.0, step=0.25, value=2.5, label="Signal strength", show_value=True)
    fdr_level = mo.ui.slider(0.01, 0.25, step=0.01, value=0.10, label="Target FDR", show_value=True)
    mo.vstack(
        [
            mo.hstack([fdr_tests, fdr_signal_share], widths="equal", gap=2),
            mo.hstack([fdr_effect, fdr_level], widths="equal", gap=2),
        ]
    )
    return fdr_effect, fdr_level, fdr_signal_share, fdr_tests


@app.cell
def _(fdr_effect, fdr_level, fdr_signal_share, fdr_tests, mo, np, plt, stats):
    _rng = np.random.default_rng(1301)
    _m = fdr_tests.value
    _is_signal = _rng.random(_m) < fdr_signal_share.value / 100
    _z = _rng.normal(fdr_effect.value * _is_signal, 1, _m)
    _p = 2 * stats.norm.sf(np.abs(_z))
    _order = np.argsort(_p)
    _sorted_p = _p[_order]
    _ranks = np.arange(1, _m + 1)
    _bh_boundary = fdr_level.value * _ranks / _m
    _passing = np.flatnonzero(_sorted_p <= _bh_boundary)
    if _passing.size:
        _cutoff = _sorted_p[_passing[-1]]
        _bh_discovery = _p <= _cutoff
    else:
        _cutoff = 0.0
        _bh_discovery = np.zeros(_m, dtype=bool)
    _raw_discovery = _p < 0.05

    _raw_total = int(np.sum(_raw_discovery))
    _raw_false = int(np.sum(_raw_discovery & ~_is_signal))
    _bh_total = int(np.sum(_bh_discovery))
    _bh_false = int(np.sum(_bh_discovery & ~_is_signal))

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4))
    _axes[0].hist(
        _p[~_is_signal],
        bins=np.linspace(0, 1, 21),
        alpha=0.75,
        color="0.65",
        label="No real signal",
    )
    _axes[0].hist(
        _p[_is_signal],
        bins=np.linspace(0, 1, 21),
        alpha=0.75,
        color="#1b9e77",
        label="Real signal",
    )
    _axes[0].set(xlabel="p-value", ylabel="Tests", title="P-values from a mixture")
    _axes[0].legend(fontsize=8)

    _axes[1].scatter(_ranks, _sorted_p, s=18, color="#7570b3", label="Ordered p-values")
    _axes[1].plot(_ranks, _bh_boundary, color="#d95f02", label="Benjamini–Hochberg boundary")
    if _bh_total:
        _axes[1].scatter(
            _ranks[:_bh_total],
            _sorted_p[:_bh_total],
            s=26,
            color="#1b9e77",
            label="Called discoveries",
        )
    _axes[1].set_yscale("log")
    _axes[1].set(xlabel="Rank", ylabel="Ordered p-value (log scale)", title="Keep the initial set below the boundary")
    _axes[1].legend(fontsize=8)
    _fig.suptitle("Figure 9. False-discovery-rate control across many tests")
    _fig.tight_layout()

    mo.vstack(
        [
            _fig,
            mo.callout(
                mo.md(
                    f"Using p < 0.05 separately calls **{_raw_total}** results discoveries, including **{_raw_false}** false discoveries in this run. Benjamini–Hochberg calls **{_bh_total}**, including **{_bh_false}** false discoveries. FDR control is a repeated-procedure guarantee, not a promise about the exact fraction in every single experiment."
                ),
                kind="info",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Double-dipping and circular analysis

    Neuroscience often measures many voxels, neurons, time points, or features. Selecting the feature that looks most related to behavior and then reporting that relationship from the **same data** uses the noise twice: once to select and again to estimate or test. This is **double-dipping**, or circular analysis.

    Selection itself is necessary and is not automatically circular. The problem occurs when the selection statistic and the reported statistic are not independent under the null model. Common safeguards include an independent localizer, a genuinely orthogonal contrast, held-out data, cross-validation, or a single model that includes selection uncertainty.

    The simulation below contains no brain–behavior relationship in any feature. It chooses the strongest apparent relationship in a training sample and then carries that same feature into independent test data.
    """)
    return


@app.cell
def _(mo):
    double_dip_features = mo.ui.slider(
        10,
        1000,
        step=10,
        value=200,
        label="Null neural features searched",
        show_value=True,
    )
    double_dip_features
    return (double_dip_features,)


@app.cell
def _(double_dip_features, mo, np, plt, stats):
    _rng = np.random.default_rng(409)
    _participants = 40
    _maximum_features = 1000
    _train_behavior = _rng.normal(size=_participants)
    _test_behavior = _rng.normal(size=_participants)
    _train_features = _rng.normal(size=(_maximum_features, _participants))
    _test_features = _rng.normal(size=(_maximum_features, _participants))

    _train_centered = _train_features - _train_features.mean(axis=1, keepdims=True)
    _behavior_centered = _train_behavior - _train_behavior.mean()
    _train_correlations = (_train_centered @ _behavior_centered) / np.sqrt(
        np.sum(_train_centered**2, axis=1) * np.sum(_behavior_centered**2)
    )
    _searched_correlations = _train_correlations[: double_dip_features.value]
    _winner = int(np.argmax(np.abs(_searched_correlations)))
    _training_result = stats.pearsonr(
        _train_features[_winner],
        _train_behavior,
    )
    _test_result = stats.pearsonr(
        _test_features[_winner],
        _test_behavior,
    )

    _fig, _axes = plt.subplots(1, 3, figsize=(13, 3.8))
    _axes[0].hist(
        _searched_correlations,
        bins=np.linspace(-0.7, 0.7, 29),
        color="#92c5de",
        edgecolor="white",
    )
    _axes[0].axvline(_training_result.statistic, color="#d73027", linewidth=3)
    _axes[0].set(
        xlabel="Training correlation",
        ylabel="Neural features",
        title="Select the most extreme feature",
    )

    for _ax, _feature, _behavior, _result, _title, _color in [
        (
            _axes[1],
            _train_features[_winner],
            _train_behavior,
            _training_result,
            "Same training data",
            "#d73027",
        ),
        (
            _axes[2],
            _test_features[_winner],
            _test_behavior,
            _test_result,
            "Independent test data",
            "#1b9e77",
        ),
    ]:
        _ax.scatter(_feature, _behavior, color=_color, alpha=0.75)
        _fit = np.polyfit(_feature, _behavior, 1)
        _line = np.linspace(_feature.min(), _feature.max(), 100)
        _ax.plot(_line, np.polyval(_fit, _line), color="black")
        _ax.set(
            xlabel="Selected neural feature",
            ylabel="Behavior",
            title=f"{_title}: r = {_result.statistic:.2f}",
        )
    _fig.suptitle("Figure 10. Selecting and testing on the same noise exaggerates evidence")
    _fig.tight_layout()

    mo.vstack(
        [
            _fig,
            mo.callout(
                mo.md(
                    f"After searching **{double_dip_features.value}** null features, the selected training correlation is **{_training_result.statistic:.2f}** with an invalid post-selection p-value of **{_training_result.pvalue:.3g}**. For the same selected feature in independent data, r = **{_test_result.statistic:.2f}** and p = **{_test_result.pvalue:.3g}**. A validation result can differ by chance, but unlike the training result its test was not used to choose the feature."
                ),
                kind="warn",
            ),
            mo.md(
                "Reference: Kriegeskorte, Simmons, Bellgowan, and Baker (2009), [*Circular analysis in systems neuroscience: the dangers of double dipping*](https://doi.org/10.1038/nn.2303)."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Hierarchical data and invalid pooling

    Trials are nested within neurons, neurons within animals, repeated sessions within people, and people within experimental groups. Measurements in the same upper-level unit share history and conditions, so they are not independent replicas of the experiment.

    Three strategies answer different questions:

    - **Complete pooling** erases subject identity. Treating every trial or neuron as an independent subject is pseudoreplication.
    - **No pooling** analyzes each subject separately but does not by itself provide a population-level conclusion.
    - **Hierarchical or partial-pooling methods** represent both within-subject and between-subject variation. Mixed-effects models, sufficient-summary-statistic approaches, and hierarchical bootstraps are examples.

    The correct experimental unit is determined by the claim. If the claim generalizes to animals, adding neurons from the same few animals does not create new independent animals.
    """)
    return


@app.cell
def _(mo):
    subject_separation = mo.ui.slider(
        0.0,
        6.0,
        step=0.25,
        value=4.0,
        label="Shared separation between subjects",
        show_value=True,
    )
    subject_separation
    return (subject_separation,)


@app.cell
def _(mo, np, plt, stats, subject_separation):
    _rng = np.random.default_rng(511)
    _subjects = 4
    _samples = 20
    _base_offsets = np.linspace(-1.5, 1.5, _subjects)
    _offsets = subject_separation.value * _base_offsets
    _colors = ["#2166ac", "#67a9cf", "#ef8a62", "#b2182b"]
    _x_parts = []
    _y_parts = []
    _subject_labels = []
    _within_correlations = []

    for _subject, _offset in enumerate(_offsets):
        _x_noise = _rng.normal(0, 0.8, _samples)
        _x_noise = _x_noise - _x_noise.mean()
        _y_noise = _rng.normal(0, 0.8, _samples)
        _y_noise = _y_noise - _y_noise.mean()
        # Remove the sample projection so the intended within-subject relation is zero.
        _y_noise = _y_noise - _x_noise * (
            np.dot(_x_noise, _y_noise) / np.dot(_x_noise, _x_noise)
        )
        _x_subject = _offset + _x_noise
        _y_subject = _offset + _y_noise
        _x_parts.append(_x_subject)
        _y_parts.append(_y_subject)
        _subject_labels.extend([_subject] * _samples)
        _within_correlations.append(
            stats.pearsonr(_x_subject, _y_subject).statistic
        )

    _x_pooled = np.concatenate(_x_parts)
    _y_pooled = np.concatenate(_y_parts)
    _subject_labels = np.array(_subject_labels)
    _pooled_result = stats.pearsonr(_x_pooled, _y_pooled)

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True)
    for _subject, _color in enumerate(_colors):
        _mask = _subject_labels == _subject
        _axes[0].scatter(
            _x_pooled[_mask],
            _y_pooled[_mask],
            color=_color,
            alpha=0.8,
            label=f"Subject {_subject + 1}",
        )
        _within_fit = np.polyfit(_x_pooled[_mask], _y_pooled[_mask], 1)
        _within_line = np.linspace(
            _x_pooled[_mask].min(),
            _x_pooled[_mask].max(),
            30,
        )
        _axes[0].plot(
            _within_line,
            np.polyval(_within_fit, _within_line),
            color=_color,
            linestyle="--",
        )
        _axes[1].scatter(
            _x_pooled[_mask],
            _y_pooled[_mask],
            color=_color,
            alpha=0.8,
        )

    _pooled_fit = np.polyfit(_x_pooled, _y_pooled, 1)
    _pooled_line = np.linspace(_x_pooled.min(), _x_pooled.max(), 100)
    _axes[1].plot(
        _pooled_line,
        np.polyval(_pooled_fit, _pooled_line),
        color="black",
        linewidth=2.5,
    )
    _axes[0].set(
        xlabel="Neural measurement X",
        ylabel="Behavioral measurement Y",
        title="Within each subject: no association",
    )
    _axes[0].legend(fontsize=8)
    _axes[1].set(
        xlabel="Neural measurement X",
        title=f"Pooled observations: r = {_pooled_result.statistic:.2f}",
    )
    _fig.suptitle("Figure 11. Subject offsets can create a spurious pooled correlation")
    _fig.tight_layout()

    _within_display = ", ".join(f"{_r:.2f}" for _r in _within_correlations)
    mo.vstack(
        [
            _fig,
            mo.callout(
                mo.md(
                    f"The four within-subject correlations are **{_within_display}**. Pooling all repeated observations produces r = **{_pooled_result.statistic:.2f}** and p = **{_pooled_result.pvalue:.3g}** because the subject-specific baselines line up. The small p-value answers a pooled-observation question, not the within-subject brain–behavior question."
                ),
                kind="warn",
            ),
            mo.md(
                "This reproduces the principle in Dowding and Haufe (2018), [*Powerful statistical inference for nested data using sufficient summary statistics*](https://doi.org/10.3389/fnhum.2018.00103). Their illustrative pooled example obtains r = 0.98 with an extremely small p-value despite no detectable relationship within any subject."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### More neurons are not more animals

    In the next null simulation, each group contains eight animals. Neurons from the same animal share an animal-specific offset. A chance difference between these particular sets of animals is not a population treatment effect.

    The naive analysis counts every neuron as an independent replicate. As more neurons are measured, its standard error collapses around the chance difference between these animals. The subject-level analysis recognizes that the independent units for an animal-level claim are the animals.
    """)
    return


@app.cell
def _(mo):
    units_per_subject = mo.ui.slider(
        1,
        200,
        step=1,
        value=50,
        label="Neurons measured per animal",
        show_value=True,
    )
    units_per_subject
    return (units_per_subject,)


@app.cell
def _(mo, np, plt, stats, units_per_subject):
    _rng = np.random.default_rng(14)
    _animals_per_group = 8
    _maximum_units = 200
    _animal_offsets = _rng.normal(0, 1, 2 * _animals_per_group)
    _unit_noise = _rng.normal(
        0,
        1,
        (2 * _animals_per_group, _maximum_units),
    )
    _observations = (
        _animal_offsets[:, None]
        + _unit_noise[:, : units_per_subject.value]
    )
    _group_a = _observations[:_animals_per_group]
    _group_b = _observations[_animals_per_group:]
    _flat_a = _group_a.ravel()
    _flat_b = _group_b.ravel()
    _animal_means_a = _group_a.mean(axis=1)
    _animal_means_b = _group_b.mean(axis=1)
    _naive_test = stats.ttest_ind(_flat_b, _flat_a, equal_var=False)
    _animal_test = stats.ttest_ind(
        _animal_means_b,
        _animal_means_a,
        equal_var=False,
    )

    def _welch_interval(_first, _second):
        _difference_value = float(np.mean(_first) - np.mean(_second))
        _first_term = np.var(_first, ddof=1) / len(_first)
        _second_term = np.var(_second, ddof=1) / len(_second)
        _se_value = np.sqrt(_first_term + _second_term)
        _df_value = (_first_term + _second_term) ** 2 / (
            _first_term**2 / (len(_first) - 1)
            + _second_term**2 / (len(_second) - 1)
        )
        _margin_value = stats.t.ppf(0.975, _df_value) * _se_value
        return _difference_value, _margin_value

    _naive_difference, _naive_margin = _welch_interval(_flat_b, _flat_a)
    _animal_difference, _animal_margin = _welch_interval(
        _animal_means_b,
        _animal_means_a,
    )

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4.3))
    _jitter_a = _rng.normal(-0.08, 0.025, len(_flat_a))
    _jitter_b = _rng.normal(1.08, 0.025, len(_flat_b))
    _axes[0].scatter(_jitter_a, _flat_a, color="#92c5de", alpha=0.12, s=12)
    _axes[0].scatter(_jitter_b, _flat_b, color="#f4a582", alpha=0.12, s=12)
    _axes[0].scatter(
        np.zeros(_animals_per_group),
        _animal_means_a,
        color="#2166ac",
        edgecolor="white",
        s=70,
        label="Animal means",
        zorder=3,
    )
    _axes[0].scatter(
        np.ones(_animals_per_group),
        _animal_means_b,
        color="#b2182b",
        edgecolor="white",
        s=70,
        zorder=3,
    )
    _axes[0].set_xticks([0, 1], ["Group A", "Group B"])
    _axes[0].set(
        ylabel="Neural measurement",
        title="Pale points are neurons; dark points are animals",
    )
    _axes[0].legend(fontsize=8)

    _axes[1].axvline(0, color="black", linestyle="--")
    _axes[1].errorbar(
        [_naive_difference, _animal_difference],
        [1, 0],
        xerr=[_naive_margin, _animal_margin],
        fmt="o",
        color="#762a83",
        capsize=7,
    )
    _axes[1].set_yticks(
        [1, 0],
        [
            f"Naive: {2 * _animals_per_group * units_per_subject.value} neurons",
            f"Group level: {2 * _animals_per_group} animals",
        ],
    )
    _axes[1].set(
        xlabel="Group B minus group A, with 95% CI",
        title="The unit counted as independent changes uncertainty",
    )
    _fig.suptitle("Figure 12. Pseudoreplication makes a chance group difference look precise")
    _fig.tight_layout()

    mo.vstack(
        [
            _fig,
            mo.callout(
                mo.md(
                    f"Treating all **{2 * _animals_per_group * units_per_subject.value} neurons** as independent gives p = **{_naive_test.pvalue:.3g}**. Treating the **{2 * _animals_per_group} animals** as the independent units gives p = **{_animal_test.pvalue:.3g}**. Adding neurons improves each animal's measurement, but it does not add animals to the population comparison."
                ),
                kind="warn",
            ),
            mo.md(
                "Saravanan, Berman, and Sober (2020), [*Application of the hierarchical bootstrap to multi-level data in neuroscience*](https://nbdt.scholasticahq.com/article/13927-application-of-the-hierarchical-bootstrap-to-multi-level-data-in-neuroscience), demonstrate this issue for neurons nested in animals and trials, and show how resampling each hierarchical level can control false positives while retaining information."
            ),
            mo.md(
                "A separate group-comparison error is shown in [Difference of significance is not significance of the difference](#difference-of-significance-is-not-significance-of-the-difference): compare effects with a direct interaction or contrast, as emphasized by Nieuwenhuis, Forstmann, and Wagenmakers (2011)."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Confounding and Simpson's paradox

    ### Which surgeon is better?

    A surgeon with a 99-in-100 success record may treat routine cases, while a surgeon with a 60-in-100 record may receive patients who are unlikely to survive without a difficult operation. The rates are precise if the samples are large, but precision does not repair an unfair comparison. We need comparable patients, risk adjustment, stratification, or—when ethical and feasible—randomization.

    **Simpson's paradox** is a visible form of the problem: an association within every subgroup can reverse when the groups are pooled because the groups occur in very different proportions.

    The admissions example uses the exact counts from the slides:

    | Stream | Men admitted | Women admitted |
    |---|---:|---:|
    | Information technology | 300 of 800 | 100 of 100 |
    | Humanities | 10 of 200 | 100 of 900 |
    | Overall | 310 of 1000 | 200 of 1000 |

    Women have the higher admission rate within both streams, yet men have the higher rate after the streams are pooled. Most women applied to the more selective stream; most men applied to the less selective one.
    """)
    return


@app.cell
def _(mo, np, plt):
    _labels = ["Information\ntechnology", "Humanities", "Overall"]
    _men_rates = np.array([300 / 800, 10 / 200, 310 / 1000])
    _women_rates = np.array([100 / 100, 100 / 900, 200 / 1000])
    assert np.all(_women_rates[:2] > _men_rates[:2])
    assert _men_rates[2] > _women_rates[2]
    _x = np.arange(len(_labels))
    _width = 0.36

    _fig, _ax = plt.subplots(figsize=(10, 4.2))
    _men_bars = _ax.bar(_x - _width / 2, _men_rates, _width, color="#4575b4", label="Men")
    _women_bars = _ax.bar(_x + _width / 2, _women_rates, _width, color="#d73027", label="Women")
    _ax.set_xticks(_x, _labels)
    _ax.set_ylim(0, 1.12)
    _ax.set(ylabel="Admission rate", title="Figure 13. Pooling reverses the comparison")
    _ax.legend()
    for _bars in (_men_bars, _women_bars):
        for _bar in _bars:
            _ax.text(
                _bar.get_x() + _bar.get_width() / 2,
                _bar.get_height() + 0.025,
                f"{100 * _bar.get_height():.1f}%",
                ha="center",
                fontsize=9,
            )
    _fig.tight_layout()

    mo.vstack(
        [
            _fig,
            mo.callout(
                mo.md("There is no arithmetic contradiction. The pooled comparison changes the weighting of the two streams. Whether to pool is a scientific and causal question, not merely a statistical preference."),
                kind="info",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A continuous version: BMI, blood pressure, and age

    Confounding does not require a table. In the simulated data below, blood pressure rises with BMI within every age group. Older groups also tend to have higher blood pressure and lower BMI. Ignoring age therefore produces an overall trend in the opposite direction.
    """)
    return


@app.cell
def _(mo, np, plt, stats):
    _rng = np.random.default_rng(418)
    _group_names = ["20s", "40s", "60s"]
    _bmi_centers = [31.0, 25.5, 21.0]
    _bp_centers = [108.0, 127.0, 146.0]
    _colors = ["#4575b4", "#1a9850", "#f46d43"]
    _all_bmi = []
    _all_bp = []

    _fig, _ax = plt.subplots(figsize=(10, 4.5))
    _within_text = []
    for _name, _bmi_center, _bp_center, _color in zip(
        _group_names, _bmi_centers, _bp_centers, _colors
    ):
        _bmi = _rng.normal(_bmi_center, 1.3, 18)
        _bp = _bp_center + 1.5 * (_bmi - _bmi_center) + _rng.normal(0, 2.2, 18)
        _within_r = stats.pearsonr(_bmi, _bp).statistic
        _within_text.append(f"{_name}: {_within_r:.2f}")
        _all_bmi.append(_bmi)
        _all_bp.append(_bp)
        _ax.scatter(_bmi, _bp, color=_color, alpha=0.8, label=f"Age {_name}")
        _fit = np.polyfit(_bmi, _bp, 1)
        _line = np.linspace(_bmi.min(), _bmi.max(), 50)
        _ax.plot(_line, np.polyval(_fit, _line), color=_color, linestyle="--")

    _all_bmi = np.concatenate(_all_bmi)
    _all_bp = np.concatenate(_all_bp)
    _overall_r = stats.pearsonr(_all_bmi, _all_bp).statistic
    _overall_fit = np.polyfit(_all_bmi, _all_bp, 1)
    _overall_line = np.linspace(_all_bmi.min(), _all_bmi.max(), 100)
    _ax.plot(
        _overall_line,
        np.polyval(_overall_fit, _overall_line),
        color="black",
        linewidth=2.5,
        label="Overall trend",
    )
    _ax.set(
        xlabel="BMI",
        ylabel="Systolic blood pressure",
        title="Figure 14. Within-group and pooled trends can point in opposite directions",
    )
    _ax.legend(ncol=2, fontsize=8)
    _fig.tight_layout()

    mo.vstack(
        [
            _fig,
            mo.callout(
                mo.md(
                    f"The correlations within age groups are **{', '.join(_within_text)}**. The pooled correlation is **{_overall_r:.2f}**. Age changes the typical location of both variables, so the black line answers a different—and misleading—question."
                ),
                kind="warn",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Diagnostic tests, base rates, and predictive value

    These terms condition on different groups:

    - **Sensitivity**, also called **recall**, asks: among people who have the disease, how many test positive?
    - **Specificity** asks: among people who do not have the disease, how many test negative?
    - **Precision**, also called **positive predictive value (PPV)**, reverses the question: among people whose test is positive, how many actually have the disease?

    A test can have excellent sensitivity and specificity while a positive result is still uncertain when the disease is rare. Work with people rather than abstract percentages: start with 100 people, divide them by disease status, and then apply the test separately to each group.

    **Correction to the slides:** slide 19 says specificity is 99%, while slide 20's “1 in 6” answer uses 95%. The default below reproduces slide 20 consistently. Move specificity to 99% to see the other scenario.
    """)
    return


@app.cell
def _(mo):
    prevalence_count = mo.ui.slider(
        1,
        30,
        step=1,
        value=1,
        label="People with disease out of 100",
        show_value=True,
    )
    sensitivity_setting = mo.ui.slider(
        50,
        100,
        step=1,
        value=100,
        label="Sensitivity (%)",
        show_value=True,
    )
    specificity_setting = mo.ui.slider(
        80,
        100,
        step=1,
        value=95,
        label="Specificity (%)",
        show_value=True,
    )
    mo.vstack(
        [
            prevalence_count,
            mo.hstack([sensitivity_setting, specificity_setting], widths="equal", gap=2),
        ]
    )
    return prevalence_count, sensitivity_setting, specificity_setting


@app.cell
def _(mo, np, plt, prevalence_count, sensitivity_setting, specificity_setting):
    _disease = prevalence_count.value
    _healthy = 100 - _disease
    _true_positive = int(np.floor(_disease * sensitivity_setting.value / 100 + 0.5))
    _false_negative = _disease - _true_positive
    _true_negative = int(np.floor(_healthy * specificity_setting.value / 100 + 0.5))
    _false_positive = _healthy - _true_negative
    _positive_tests = _true_positive + _false_positive

    _categories = (
        ["True positive"] * _true_positive
        + ["False negative"] * _false_negative
        + ["False positive"] * _false_positive
        + ["True negative"] * _true_negative
    )
    assert len(_categories) == 100
    _palette = {
        "True positive": "#2166ac",
        "False negative": "#762a83",
        "False positive": "#d73027",
        "True negative": "#d9d9d9",
    }
    _positions = np.arange(100)
    _x = _positions % 10
    _y = 9 - _positions // 10

    _fig, _ax = plt.subplots(figsize=(9, 6))
    for _category in _palette:
        _mask = np.array([_item == _category for _item in _categories])
        _ax.scatter(
            _x[_mask],
            _y[_mask],
            s=210,
            color=_palette[_category],
            edgecolor="#225ea8" if _category == "True negative" else "white",
            linewidth=0.8,
            label=f"{_category}: {int(np.sum(_mask))}",
        )
    _ax.set_aspect("equal")
    _ax.set_xlim(-0.8, 9.8)
    _ax.set_ylim(-0.8, 9.8)
    _ax.axis("off")
    _ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    _ax.set_title("Figure 15. Test outcomes among 100 people")
    _fig.tight_layout()

    if _positive_tests:
        _ppv_words = f"{_true_positive} of the {_positive_tests} positive tests come from people with disease"
    else:
        _ppv_words = "there are no positive tests in this rounded group"

    mo.vstack(
        [
            _fig,
            mo.callout(
                mo.md(
                    f"""
                    - **Sensitivity / recall:** among the **{_disease}** people with disease, **{_true_positive}** test positive and **{_false_negative}** test negative.
                    - **Specificity:** among the **{_healthy}** people without disease, **{_true_negative}** test negative and **{_false_positive}** test positive.
                    - **Precision / PPV:** {_ppv_words}.

                    At the slide-20 defaults, this is **1 true positive and 5 false positives: only 1 of 6 positive tests indicates disease**. At 99% specificity it becomes approximately **1 true positive and 1 false positive: about 1 of 2**.
                    """
                ),
                kind="info",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Selection and Berkson's fallacy

    Confounding asks whether a shared cause helps produce both measured variables. **Collider bias** is different: it appears when we restrict attention to a group selected using both variables.

    Consider physical attractiveness and acting ability. Suppose they are unrelated in the full population, but either quality can help an actor become highly successful. Among only successful actors, a person with less of one quality generally needed more of the other to pass the selection barrier. Conditioning on success can therefore create a negative association that was absent in the population.

    The causal shape is:

    **Attractiveness → selected as successful ← acting ability**

    “Selected as successful” is called a collider because the two arrows collide there.
    """)
    return


@app.cell
def _(mo):
    selection_rate = mo.ui.slider(
        5,
        80,
        step=5,
        value=20,
        label="Percent selected as successful",
        show_value=True,
    )
    selection_rate
    return (selection_rate,)


@app.cell
def _(mo, np, plt, selection_rate, stats):
    _rng = np.random.default_rng(622)
    _population = 2500
    _attractiveness = _rng.normal(size=_population)
    _acting = _rng.normal(size=_population)
    _selection_score = _attractiveness + _acting + _rng.normal(0, 0.35, _population)
    _threshold = np.quantile(_selection_score, 1 - selection_rate.value / 100)
    _selected = _selection_score >= _threshold
    _all_r = stats.pearsonr(_attractiveness, _acting).statistic
    _selected_r = stats.pearsonr(_attractiveness[_selected], _acting[_selected]).statistic

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    _axes[0].scatter(_attractiveness, _acting, color="0.65", alpha=0.2, s=12)
    _axes[0].set(
        xlabel="Physical attractiveness",
        ylabel="Acting ability",
        title=f"Full population: r = {_all_r:.2f}",
    )
    _axes[1].scatter(
        _attractiveness[_selected],
        _acting[_selected],
        color="#c51b7d",
        alpha=0.55,
        s=16,
    )
    _fit = np.polyfit(_attractiveness[_selected], _acting[_selected], 1)
    _line = np.linspace(_attractiveness[_selected].min(), _attractiveness[_selected].max(), 100)
    _axes[1].plot(_line, np.polyval(_fit, _line), color="black")
    _axes[1].set(
        xlabel="Physical attractiveness",
        title=f"Only successful actors: r = {_selected_r:.2f}",
    )
    _fig.suptitle("Figure 16. Conditioning on a common consequence creates an association")
    _fig.tight_layout()

    mo.vstack(
        [
            _fig,
            mo.callout(
                mo.md(
                    f"The traits have correlation **{_all_r:.2f}** in the simulated population but **{_selected_r:.2f}** after selecting the top **{selection_rate.value}%** on their combined contribution to success. The selection rule—not a trade-off in the population—creates the negative correlation."
                ),
                kind="warn",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The same structure appears when we analyze only admitted students, hospitalized patients, survey respondents, published papers, or any other group whose inclusion depends on two causes under study. “Control for everything” is not safe advice: controlling for a collider can introduce bias.

    ## A compact checklist

    Before interpreting a statistical pattern, ask:

    1. Was this question chosen before or after looking at the data?
    2. How many outcomes, subgroups, transformations, or models were inspected?
    3. Were feature selection and effect estimation performed with independent information?
    4. What is the independent experimental unit: trials, neurons, animals, or people?
    5. What does the uncertainty interval still permit?
    6. Was the claimed comparison tested directly?
    7. Could time or another shared cause explain both variables?
    8. Does pooling hide important subgroups or erase a hierarchy?
    9. What base rate applies to the population being tested?
    10. Was the analyzed sample selected by a consequence of the variables?

    ## References and further reading

    - Randall Munroe, [xkcd 882: “Significant”](https://xkcd.com/882/), CC BY-NC 2.5.
    - Altman & Bland (1995), [*Absence of evidence is not evidence of absence*](https://doi.org/10.1136/bmj.311.7003.485).
    - Nieuwenhuis, Forstmann, & Wagenmakers (2011), [*Erroneous analyses of interactions in neuroscience*](https://doi.org/10.1038/nn.2886).
    - Kriegeskorte, Simmons, Bellgowan, & Baker (2009), [*Circular analysis in systems neuroscience: the dangers of double dipping*](https://doi.org/10.1038/nn.2303).
    - Benjamini & Hochberg (1995), [*Controlling the false discovery rate: a practical and powerful approach to multiple testing*](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x).
    - Masicampo & Lalande (2012), [*A peculiar prevalence of p values just below .05*](https://doi.org/10.1080/17470218.2012.711335).
    - Dowding & Haufe (2018), [*Powerful statistical inference for nested data using sufficient summary statistics*](https://doi.org/10.3389/fnhum.2018.00103).
    - Saravanan, Berman, & Sober (2020), [*Application of the hierarchical bootstrap to multi-level data in neuroscience*](https://nbdt.scholasticahq.com/article/13927-application-of-the-hierarchical-bootstrap-to-multi-level-data-in-neuroscience).
    - Hernán, Hernández-Díaz, & Robins (2004), [*A structural approach to selection bias*](https://doi.org/10.1097/01.ede.0000135174.63482.43).

    These demonstrations deliberately use simple generating processes. Real analyses require design knowledge, justified models, uncertainty checks, and careful definition of the population and estimand.
    """)
    return


if __name__ == "__main__":
    app.run()
