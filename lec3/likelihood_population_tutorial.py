# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.23.9",
#     "matplotlib>=3.9",
#     "numpy>=2.1",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import math
    from statistics import NormalDist

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle

    normal = NormalDist()

    def circular_delta(a, b):
        """Signed a-b distance in degrees, wrapped to [-180, 180)."""
        return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0

    def direction_tuning(stimuli, preferences, kappa=2.2):
        """Unit-height von Mises tuning, with broadcasting over stimuli."""
        stimuli = np.asarray(stimuli, dtype=float)
        preferences = np.asarray(preferences, dtype=float)
        delta = np.deg2rad(
            circular_delta(stimuli[..., np.newaxis], preferences)
        )
        return np.exp(kappa * (np.cos(delta) - 1.0))

    def relative_likelihood(log_values):
        shifted = np.asarray(log_values) - np.max(log_values)
        return np.exp(shifted)

    def clean_axes(axes, grid=True):
        for axis in np.atleast_1d(axes).flat:
            axis.spines[["top", "right"]].set_visible(False)
            if grid:
                axis.grid(alpha=0.13)

    return (
        NormalDist,
        Rectangle,
        circular_delta,
        clean_axes,
        direction_tuning,
        math,
        mo,
        normal,
        np,
        plt,
        relative_likelihood,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.Html(
        r"""
        <style>
          :root {
            --ink: #182338;
            --muted: #5f6b7c;
            --paper: #f6f3ec;
            --card: #ffffff;
            --line: #dce3ec;
            --spike: #d97706;
            --likelihood: #7c3aed;
            --tuning: #2563a8;
            --teal: #16846b;
            --coral: #cf5c5c;
          }
          body { background: var(--paper); color: var(--ink); }
          .likelihood-hero {
            margin: 0 0 1.25rem;
            padding: 2.15rem 2.35rem;
            border-radius: 18px;
            color: white;
            background:
              radial-gradient(circle at 86% 18%, rgba(255,255,255,.12), transparent 23%),
              linear-gradient(125deg, #17233a, #30466d);
            box-shadow: 0 12px 34px rgba(24, 35, 56, .13);
          }
          .likelihood-hero h1 {
            margin: 0 0 .55rem;
            color: white !important;
            font-family: Georgia, "Times New Roman", serif;
            font-size: clamp(2rem, 4vw, 3.25rem);
            line-height: 1.06;
          }
          .likelihood-hero p {
            max-width: 900px;
            margin: 0;
            color: #dbe4f2;
            line-height: 1.58;
          }
          .lesson-card {
            padding: 1.05rem 1.3rem;
            margin: .7rem 0 1.15rem;
            border: 1px solid var(--line);
            border-radius: 14px;
            background: var(--card);
          }
          .formula-card {
            padding: .9rem 1.15rem;
            border-left: 4px solid var(--likelihood);
            border-radius: 5px 12px 12px 5px;
            background: #f4f0ff;
          }
          .concept-chain {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: .65rem;
            margin: .8rem 0 1.1rem;
          }
          .concept-step {
            position: relative;
            min-height: 92px;
            padding: .8rem .9rem;
            border: 1px solid var(--line);
            border-top: 4px solid var(--tuning);
            border-radius: 11px;
            background: white;
          }
          .concept-step:nth-child(2) { border-top-color: var(--spike); }
          .concept-step:nth-child(3) { border-top-color: var(--teal); }
          .concept-step:nth-child(4) { border-top-color: var(--likelihood); }
          .concept-step strong { display: block; margin-bottom: .25rem; }
          .concept-step span { color: var(--muted); font-size: .86rem; line-height: 1.42; }
          .metric-row {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: .65rem;
            margin: .75rem 0;
          }
          .metric {
            padding: .72rem .82rem;
            border: 1px solid var(--line);
            border-radius: 10px;
            background: white;
          }
          .metric strong { display: block; font-size: 1.18rem; }
          .metric span { color: var(--muted); font-size: .78rem; }
          .small-note { color: var(--muted); font-size: .88rem; line-height: 1.55; }
          table { width: 100%; }
          th, td { padding: .45rem .6rem !important; }
          .marimo-cell.interactive .console-output-area,
          .marimo-cell.interactive .output-area {
            max-height: none !important;
            overflow: visible !important;
          }
          @media (max-width: 820px) {
            .concept-chain, .metric-row { grid-template-columns: 1fr 1fr; }
            .likelihood-hero { padding: 1.5rem; }
          }
          @media (max-width: 540px) {
            .concept-chain, .metric-row { grid-template-columns: 1fr; }
          }
        </style>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div class="likelihood-hero">
      <h1>Likelihood and population decoding</h1>
      <p>How a model of neural responses defines a likelihood function, how population responses can represent that function, and which assumptions justify a weighted readout. Interactive figures distinguish response probability, relative likelihood, posterior probability, and decision variables.</p>
    </div>

    <div class="concept-chain">
      <div class="concept-step"><strong>1 · Response model</strong><span>For each fixed stimulus, specify a distribution over possible responses.</span></div>
      <div class="concept-step"><strong>2 · Observed data</strong><span>Fix the response vector measured on one trial.</span></div>
      <div class="concept-step"><strong>3 · Likelihood</strong><span>Evaluate the probability of that same response under each candidate stimulus.</span></div>
      <div class="concept-step"><strong>4 · Population readout</strong><span>Determine when these evaluations reduce to weighted sums of spike counts.</span></div>
    </div>

    This tutorial accompanies Jazayeri & Movshon (2006). All numerical examples
    below are illustrative models; they do not reproduce recorded neural data.

    ## 1. Response probability, likelihood, and posterior probability

    Let $r$ denote a neural response and $\theta$ a stimulus. A response model
    specifies $P(r\mid\theta)$: for a **fixed stimulus**, it assigns probabilities
    to the possible responses. Once a response $r_{\mathrm{obs}}$ has been
    observed, the likelihood is $L(\theta)=P(r_{\mathrm{obs}}\mid\theta)$.
    The same model is evaluated with the **response fixed and stimulus varied**.
    The conditioning has not been reversed: likelihood is not
    $P(\theta\mid r_{\mathrm{obs}})$.

    In the left panel, a neuron's response is grouped into low, medium, or high
    spike-count categories within a fixed observation window. Each row is a
    probability distribution over these mutually exclusive, exhaustive
    categories, so it sums to one. Selecting an observed category highlights
    one column. The middle panel plots its entries against candidate direction:
    this is the likelihood function. The direction with the largest entry is
    the maximum-likelihood estimate among these three candidates.

    The right panel adds a prior over directions. Each likelihood value is
    multiplied by its prior probability, then the three products are divided by
    their sum. The result is a posterior distribution over directions. Change
    the prior while keeping the response fixed: the likelihood stays unchanged,
    but the posterior can change substantially.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    observed_response = mo.ui.radio(
        options=["low count", "medium count", "high count"],
        value="high count",
        inline=True,
        label="Observed response category",
    )
    direction_prior = mo.ui.radio(
        options=["equal probabilities", "leftward more frequent"],
        value="equal probabilities",
        inline=True,
        label="Prior over directions",
    )
    return direction_prior, observed_response


@app.cell(hide_code=True)
def _(Rectangle, clean_axes, direction_prior, mo, np, observed_response, plt):
    _responses = ["low count", "medium count", "high count"]
    _worlds = ["leftward", "upward", "rightward"]
    _forward = np.array(
        [
            [0.70, 0.25, 0.05],
            [0.25, 0.55, 0.20],
            [0.05, 0.25, 0.70],
        ]
    )
    _chosen = _responses.index(observed_response.value)
    _likelihood = _forward[:, _chosen]
    _best = int(np.argmax(_likelihood))
    _prior = (
        np.ones(3) / 3
        if direction_prior.value == "equal probabilities"
        else np.array([0.85, 0.10, 0.05])
    )
    _posterior = _likelihood * _prior
    _posterior /= _posterior.sum()

    _fig, (_ax_table, _ax_like, _ax_post) = plt.subplots(
        1, 3, figsize=(13.4, 4.2), gridspec_kw={"width_ratios": [1.4, 1, 1]}
    )
    _fig.patch.set_facecolor("white")

    _image = _ax_table.imshow(
        _forward, cmap="Blues", vmin=0, vmax=0.75, aspect="auto"
    )
    for _row in range(3):
        for _column in range(3):
            _ax_table.text(
                _column,
                _row,
                f"{_forward[_row, _column]:.2f}",
                ha="center",
                va="center",
                color="white" if _forward[_row, _column] > 0.48 else "#182338",
                fontsize=12,
                fontweight="bold" if _column == _chosen else "normal",
            )
    _ax_table.add_patch(
        Rectangle(
            (_chosen - 0.48, -0.48),
            0.96,
            2.96,
            fill=False,
            edgecolor="#d97706",
            linewidth=3.0,
        )
    )
    _ax_table.set_xticks(range(3), _responses)
    _ax_table.set_yticks(range(3), _worlds)
    _ax_table.set(
        title="Response probabilities P(r | direction)",
        xlabel="possible neural response",
        ylabel="candidate direction",
    )
    _ax_table.tick_params(length=0)
    for _spine in _ax_table.spines.values():
        _spine.set_visible(False)
    _colorbar = _fig.colorbar(_image, ax=_ax_table, fraction=0.045, pad=0.04)
    _colorbar.set_label("probability")

    _bar_colors = ["#7c3aed" if i == _best else "#b8a4ea" for i in range(3)]
    _ax_like.barh(_worlds, _likelihood, color=_bar_colors, height=0.58)
    for _row, _value in enumerate(_likelihood):
        _ax_like.text(
            _value + 0.018,
            _row,
            f"{_value:.2f}",
            va="center",
            color="#182338",
            fontweight="bold" if _row == _best else "normal",
        )
    _ax_like.set(
        title="Likelihood for the observed category",
        xlabel="P(observed category | direction)",
        xlim=(0, 0.8),
    )
    _ax_like.invert_yaxis()
    _positions = np.arange(3)
    _ax_post.barh(_positions - 0.17, _prior, height=0.30, color="#bac4d2", label="prior")
    _ax_post.barh(_positions + 0.17, _posterior, height=0.30, color="#16846b", label="posterior")
    _ax_post.set_yticks(_positions, _worlds)
    _ax_post.invert_yaxis()
    _ax_post.set(title="Probabilities over directions", xlabel="probability", xlim=(0, 1))
    _ax_post.legend(frameon=False, fontsize=8)
    clean_axes([_ax_like, _ax_post])
    _fig.tight_layout(w_pad=2.4)

    mo.vstack(
        [
            observed_response,
            direction_prior,
            _fig,
            mo.callout(
                mo.md(
                    f"""
                    The observed response was **{observed_response.value}**.
                    Its probabilities under leftward, upward, and rightward
                    motion are **{_likelihood[0]:.2f}**, **{_likelihood[1]:.2f}**,
                    and **{_likelihood[2]:.2f}**, respectively. The
                    maximum-likelihood direction is **{_worlds[_best]}**.
                    The likelihood values sum to **{_likelihood.sum():.2f}**;
                    no requirement makes this sum equal to one. In contrast,
                    each set of prior or posterior bars sums to one.

                    A likelihood ratio compares two entries in the middle
                    panel. A ratio of 14 means that this response is 14 times
                    as probable under one candidate as under the other. It
                    does not mean the candidate itself is 14 times as probable
                    unless the prior odds are one. Posterior odds combine
                    the prior odds with this likelihood ratio.
                    """
                ),
                kind="info",
            ),
        ],
        gap=0.7,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Continuous responses and plotting scales.** For a continuous measurement,
    the response model supplies a probability *density*, rather than a
    probability at a single exact value. Probabilities are areas over response
    intervals. A density can exceed one and has units reciprocal to those of
    the response. Its likelihood is still obtained by holding the observed
    response fixed and varying the candidate stimulus.

    Multiplying every likelihood value by the same positive constant leaves
    all likelihood ratios unchanged. In a **relative likelihood** plot we divide
    by the maximum, giving a peak of one. This is a plotting convention, not
    normalization into a probability distribution. The width shows how quickly
    relative support falls away from the best-fitting stimulus; it is not by
    itself a posterior standard deviation or a confidence interval.

    ## 2. Combining conditionally independent measurements

    Consider two measurements of the same scalar stimulus. Each has Gaussian
    error centered on the true stimulus, with a known standard deviation. The
    sliders specify the two observed measurements and their noise standard
    deviations. For each candidate stimulus, the model evaluates the density
    of each observed measurement. Their product is the joint likelihood if
    the errors are **independent conditional on that stimulus**.

    In the upper panel, both likelihoods are divided by their individual
    maxima. Multiplying these relative curves gives the correct shape of the
    joint likelihood. The lower panel divides that product by its own maximum
    to make its width easy to compare. All these rescalings are constant across
    candidate stimuli for the selected data and noise settings.

    With equal noise, the joint maximum lies halfway between the measurements.
    With unequal noise, it lies closer to the measurement with the smaller
    standard deviation. A narrower input curve falls faster as a candidate
    moves away from its measurement, so it contributes more to locating the
    joint maximum. Both measurements refer to one common stimulus; combining
    measurements of different stimuli would require a different model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    clue_one_center = mo.ui.slider(
        start=-80,
        stop=80,
        step=5,
        value=-25,
        debounce=True,
        show_value=True,
        full_width=True,
        label="Observed measurement 1",
    )
    clue_two_center = mo.ui.slider(
        start=-80,
        stop=80,
        step=5,
        value=30,
        debounce=True,
        show_value=True,
        full_width=True,
        label="Observed measurement 2",
    )
    clue_width = mo.ui.slider(
        start=12,
        stop=60,
        step=4,
        value=32,
        debounce=True,
        show_value=True,
        full_width=True,
        label="Noise standard deviation: measurement 1",
    )
    clue_two_width = mo.ui.slider(
        start=12, stop=60, step=4, value=32, debounce=True,
        show_value=True, full_width=True,
        label="Noise standard deviation: measurement 2",
    )
    clue_controls = mo.vstack(
        [clue_one_center, clue_two_center, clue_width, clue_two_width], gap=0.5
    )
    return clue_controls, clue_one_center, clue_two_center, clue_two_width, clue_width


@app.cell(hide_code=True)
def _(
    clean_axes,
    clue_controls,
    clue_one_center,
    clue_two_center,
    clue_two_width,
    clue_width,
    mo,
    np,
    plt,
):
    _x = np.linspace(-110, 110, 881)
    _width = clue_width.value
    _width_two = clue_two_width.value
    _l1 = np.exp(-0.5 * ((_x - clue_one_center.value) / _width) ** 2)
    _l2 = np.exp(-0.5 * ((_x - clue_two_center.value) / _width_two) ** 2)
    _joint = _l1 * _l2
    _joint /= _joint.max()
    _joint_sd = 1.0 / np.sqrt(1.0 / _width**2 + 1.0 / _width_two**2)
    _best_x = _joint_sd**2 * (
        clue_one_center.value / _width**2 + clue_two_center.value / _width_two**2
    )

    _fig, (_ax_parts, _ax_joint) = plt.subplots(
        2, 1, figsize=(10.8, 6.7), sharex=True
    )
    _fig.patch.set_facecolor("white")
    _ax_parts.plot(_x, _l1, color="#2563a8", linewidth=2.5, label="measurement 1")
    _ax_parts.fill_between(_x, 0, _l1, color="#2563a8", alpha=0.12)
    _ax_parts.plot(_x, _l2, color="#d97706", linewidth=2.5, label="measurement 2")
    _ax_parts.fill_between(_x, 0, _l2, color="#d97706", alpha=0.12)
    _ax_parts.set(
        title="Individual likelihoods, each scaled to peak at 1",
        ylabel="relative likelihood",
        ylim=(0, 1.05),
    )
    _ax_parts.legend(frameon=False, ncols=2)

    _ax_joint.plot(_x, _joint, color="#7c3aed", linewidth=3.0)
    _ax_joint.fill_between(_x, 0, _joint, color="#7c3aed", alpha=0.20)
    _ax_joint.axvline(_best_x, color="#182338", linestyle="--", linewidth=1.4)
    _ax_joint.scatter([_best_x], [1], color="#182338", s=55, zorder=4)
    _ax_joint.text(
        _best_x,
        1.08,
        f"maximum-likelihood estimate = {_best_x:.1f}",
        ha="center",
        fontsize=9.5,
    )
    _ax_joint.set(
        title="Joint likelihood: product rescaled to peak at 1",
        xlabel="candidate stimulus",
        ylabel="relative likelihood",
        xlim=(-110, 110),
        ylim=(0, 1.22),
    )
    clean_axes([_ax_parts, _ax_joint])
    _fig.tight_layout(h_pad=2.0)

    mo.vstack(
        [
            clue_controls,
            _fig,
            mo.callout(
                mo.md(
                    f"""
                    The joint maximum is **{_best_x:.1f}**. Its Gaussian width
                    parameter is **{_joint_sd:.1f}**, compared with **{_width}**
                    and **{_width_two}** for the individual measurements.
                    With equal noise, combining two independent measurements
                    reduces this width by a factor of approximately 1.41.
                    Under a flat prior over the real line, these particular
                    Gaussian likelihoods also give Gaussian posteriors with
                    those standard deviations.

                    Move the measurements farther apart without changing the
                    noise. The joint width stays the same: in this fixed-noise,
                    common-stimulus model, disagreement changes the location
                    and unscaled height of the product, not its curvature.
                    A narrow relative likelihood therefore does not establish
                    that the model fits the observations well. Assessing
                    disagreement requires checking the measurement difference
                    against its predicted noise, or comparing an explicit
                    alternative model. The rescaled peak height cannot do this.
                    """
                ),
                kind="info",
            ),
        ],
        gap=0.7,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Log likelihood and repeated observations

    Taking logarithms turns a product of likelihood factors into a sum. Because
    the logarithm is strictly increasing, the maximum stays at the same
    stimulus. A difference between two log-likelihood values is the logarithm
    of their likelihood ratio. This preserves relative evidence while avoiding
    numerical underflow from multiplying many small numbers. It does not make
    dependent observations independent.

    Here the response on each trial is binary. The left curve specifies the
    probability of response 1 at each stimulus; the probability of response 0
    is its complement. Suppose the stimulus is fixed across trials and every
    observed response is 1. For independent trials, the likelihood is the
    product of one copy of the left curve per observed trial. This is a model
    for actual repeated observations, not permission to count one observation
    more than once.

    Increase the number of trials. The middle panel shows the probability of
    the entire response sequence becoming smaller, even at its maximum.
    Simultaneously, the differences between candidates in the right panel grow:
    the data discriminate more strongly between stimuli. The absolute
    probability of a longer data sequence and the precision of stimulus
    estimation are different quantities.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    repeated_clues = mo.ui.slider(
        start=1,
        stop=30,
        step=1,
        value=8,
        debounce=True,
        show_value=True,
        full_width=True,
        label="Number of independent trials, all with observed response 1",
    )
    return (repeated_clues,)


@app.cell(hide_code=True)
def _(clean_axes, mo, np, plt, repeated_clues):
    _x = np.linspace(-100, 100, 700)
    _single = 0.82 * np.exp(-0.5 * ((_x - 18.0) / 31.0) ** 2) + 0.015
    _number = repeated_clues.value
    _product = _single**_number
    _single_log = np.log(_single)
    _log_sum = _number * _single_log
    _winner = float(_x[np.argmax(_single)])
    _product_peak = float(np.max(_product))
    _log_peak = float(np.max(_log_sum))

    _fig, _axes = plt.subplots(1, 3, figsize=(12.2, 3.8))
    _fig.patch.set_facecolor("white")
    _axes[0].plot(_x, _single, color="#2563a8", linewidth=2.6)
    _axes[0].fill_between(_x, 0, _single, color="#2563a8", alpha=0.15)
    _axes[0].set(
        title="One observed response: r = 1",
        xlabel="candidate stimulus",
        ylabel="P(r = 1 | stimulus)",
        ylim=(0, 1.02),
    )

    _axes[1].plot(_x, _product, color="#d97706", linewidth=2.6)
    _axes[1].fill_between(_x, 0, _product, color="#d97706", alpha=0.18)
    _axes[1].set(
        title=f"Joint likelihood of {_number} responses",
        xlabel="candidate stimulus",
        ylabel="product",
        ylim=(0, 1.02),
    )

    _axes[2].plot(
        _x, _single_log, color="#9d89d8", linewidth=1.8, linestyle="--",
        label="one log contribution",
    )
    _axes[2].plot(
        _x, _log_sum, color="#7c3aed", linewidth=2.8,
        label=f"sum of {_number}",
    )
    _axes[2].fill_between(_x, _log_sum, np.min(_log_sum), color="#7c3aed", alpha=0.10)
    _axes[2].set(
        title="Sum of individual log likelihoods",
        xlabel="candidate stimulus",
        ylabel="log likelihood",
    )
    _axes[2].legend(frameon=False, fontsize=8)

    for _axis in _axes:
        _axis.axvline(_winner, color="#182338", linestyle=":", linewidth=1.25)
    clean_axes(_axes)
    _fig.tight_layout(w_pad=1.8)

    mo.vstack(
        [
            repeated_clues,
            _fig,
            mo.Html(
                f"""
                <div class="metric-row">
                  <div class="metric"><strong>{_number}</strong><span>likelihood factors</span></div>
                  <div class="metric"><strong>{_product_peak:.3g}</strong><span>maximum joint likelihood</span></div>
                  <div class="metric"><strong>{_log_peak:.2f}</strong><span>maximum log likelihood</span></div>
                  <div class="metric"><strong>{_winner:.0f}</strong><span>maximum-likelihood estimate</span></div>
                </div>
                """
            ),
            mo.callout(
                mo.md(r"""
                The middle and right panels represent the same joint
                likelihood on different vertical scales. A drop of 2 log units
                from the maximum corresponds to a likelihood ratio of about
                0.14 relative to the maximum; a drop of 5 corresponds to about
                0.007. Larger vertical differences imply stronger relative
                discrimination between those candidates.

                An additive constant independent of stimulus can be removed
                from a log likelihood without changing these differences.
                Multiplying a log likelihood by a constant is different: it
                changes likelihood ratios and the inferred precision, even
                though it preserves the maximum when the constant is positive.
                """),
                kind="info",
            ),
            mo.accordion(
                {
                    "Log-likelihood identity": mo.md(
                        r"""If conditionally independent observations supply factors
                        $L_1,L_2,\ldots$, then
                        $\log(L_1L_2\cdots)=\log L_1+\log L_2+\cdots$.
                        Multiplication becomes addition; the location of the
                        maximum is unchanged."""
                    )
                }
            ),
        ],
        gap=0.7,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. A weighted population readout

    Twelve neurons have preferred directions spaced evenly around a circle.
    Each has the same von Mises tuning shape: a smooth, periodic mean-response
    curve centered on its preference. Counts are measured over a fixed time
    window. For this example, the decoder assumes independent Poisson counts
    with these means. The next two sections explain that assumption and the
    resulting calculation in detail.

    The upper-left panel contains a constructed count pattern obtained by
    rounding a tuning profile. It is a possible observation under the model,
    not a random trial or a measured dataset. Changing its center or strength
    constructs a different observation. Changing only the candidate direction
    leaves those counts unchanged and selects a different output unit.

    Read the top row from left to right: observed counts, cosine connection
    weights for one candidate, and the products of count and weight. Their sum
    is the candidate's weighted score. The weight scale is the tuning
    concentration (1.8 here), as required by this response model; an arbitrary
    scale would preserve the peak but give incorrect likelihood ratios.

    Repeating the calculation for all candidate directions produces the lower
    curve. The Poisson total-expected-count term is included when computing
    this curve; it is nearly constant for this evenly spaced population.
    Exponentiating the log likelihood after subtracting its maximum gives
    the plotted relative likelihood. The input axis indexes **neurons by
    preference**; the output axis indexes **candidate stimuli**. Similar-looking
    axes do not make the count profile itself a likelihood function.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    population_center = mo.ui.slider(
        start=-150,
        stop=150,
        step=15,
        value=45,
        debounce=True,
        show_value=True,
        full_width=True,
        label="Center of the constructed count profile (degrees)",
    )
    population_strength = mo.ui.slider(
        start=2,
        stop=12,
        step=1,
        value=8,
        debounce=True,
        show_value=True,
        full_width=True,
        label="Peak expected count used to construct the response",
    )
    candidate_direction = mo.ui.slider(
        start=-180,
        stop=180,
        step=15,
        value=0,
        debounce=True,
        show_value=True,
        full_width=True,
        label="Candidate direction tested by the highlighted output unit",
    )
    population_controls = mo.vstack(
        [population_center, population_strength, candidate_direction], gap=0.5
    )
    return (
        candidate_direction,
        population_center,
        population_controls,
        population_strength,
    )


@app.cell(hide_code=True)
def _(
    candidate_direction,
    circular_delta,
    clean_axes,
    direction_tuning,
    mo,
    np,
    plt,
    population_center,
    population_controls,
    population_strength,
    relative_likelihood,
):
    _preferences = np.arange(-180, 180, 30, dtype=float)
    _profile = direction_tuning(
        np.array([population_center.value]), _preferences, kappa=1.8
    )[0]
    _counts = np.rint(population_strength.value * _profile).astype(int)
    _candidate = candidate_direction.value
    _weights = 1.8 * np.cos(np.deg2rad(circular_delta(_candidate, _preferences)))
    _contributions = _counts * _weights
    _candidate_score = float(np.sum(_contributions))

    _candidate_grid = np.linspace(-180, 180, 721)
    _weight_bank = np.cos(
        np.deg2rad(
            circular_delta(
                _candidate_grid[:, np.newaxis], _preferences[np.newaxis, :]
            )
        )
    )
    _expected_counts = population_strength.value * direction_tuning(
        _candidate_grid, _preferences, kappa=1.8
    )
    _scores = 1.8 * (_weight_bank @ _counts) - _expected_counts.sum(axis=1)
    _relative = relative_likelihood(_scores)
    _peak_direction = float(_candidate_grid[np.argmax(_scores)])
    _selected_relative = float(
        np.interp(_candidate, _candidate_grid, _relative)
    )
    _resultant_x = float(np.sum(_counts * np.cos(np.deg2rad(_preferences))))
    _resultant_y = float(np.sum(_counts * np.sin(np.deg2rad(_preferences))))
    _vector_mean = float(np.rad2deg(np.arctan2(_resultant_y, _resultant_x)))

    _fig = plt.figure(figsize=(13.0, 8.1))
    _fig.patch.set_facecolor("white")
    _grid_spec = _fig.add_gridspec(
        2, 3, height_ratios=[1.0, 1.05], width_ratios=[1, 1, 1]
    )
    _ax_spikes = _fig.add_subplot(_grid_spec[0, 0])
    _ax_weights = _fig.add_subplot(_grid_spec[0, 1])
    _ax_contributions = _fig.add_subplot(_grid_spec[0, 2])
    _ax_likelihood = _fig.add_subplot(_grid_spec[1, :2])
    _ax_compass = _fig.add_subplot(_grid_spec[1, 2], projection="polar")
    _bar_width = 23
    _ax_spikes.bar(
        _preferences, _counts, width=_bar_width, color="#d97706", alpha=0.86
    )
    _ax_spikes.set(
        title="1 · The observed population response",
        xlabel="input neuron's preferred direction",
        ylabel="observed spike count",
        xlim=(-195, 195),
    )

    _weight_colors = np.where(_weights >= 0, "#16846b", "#cf5c5c")
    _ax_weights.bar(
        _preferences, _weights, width=_bar_width,
        color=_weight_colors, alpha=0.82
    )
    _ax_weights.axhline(0, color="#6b7280", linewidth=1.0)
    _ax_weights.set(
        title=f"2 · Fixed weights for candidate {_candidate:+.0f}°",
        xlabel="input neuron's preferred direction",
        ylabel="connection weight",
        xlim=(-195, 195),
        ylim=(-2.0, 2.0),
    )

    _contribution_colors = np.where(
        _contributions >= 0, "#16846b", "#cf5c5c"
    )
    _ax_contributions.bar(
        _preferences, _contributions, width=_bar_width,
        color=_contribution_colors, alpha=0.84
    )
    _ax_contributions.axhline(0, color="#6b7280", linewidth=1.0)
    _ax_contributions.set(
        title=f"3 · Spike × weight; add them = {_candidate_score:.1f}",
        xlabel="input neuron's preferred direction",
        ylabel="contribution to score",
        xlim=(-195, 195),
    )

    _ax_likelihood.plot(
        _candidate_grid, _relative, color="#7c3aed", linewidth=2.9
    )
    _ax_likelihood.fill_between(
        _candidate_grid, 0, _relative, color="#7c3aed", alpha=0.18
    )
    _ax_likelihood.axvline(
        _peak_direction, color="#182338", linestyle="--", linewidth=1.4
    )
    _ax_likelihood.scatter(
        [_candidate], [_selected_relative], color="#d97706", s=70,
        edgecolor="white", linewidth=1.0, zorder=4,
        label=f"selected output: {_candidate:+.0f}°",
    )
    _ax_likelihood.set(
        title="4 · Population relative likelihood under the Poisson model",
        xlabel="candidate direction",
        ylabel="relative likelihood",
        xlim=(-180, 180),
        ylim=(0, 1.08),
    )
    _ax_likelihood.legend(frameon=False, fontsize=8.5)

    _angles = np.deg2rad(_preferences)
    _ax_compass.bar(
        _angles,
        _counts,
        width=np.deg2rad(22),
        color="#d97706",
        alpha=0.34,
        edgecolor="#d97706",
        linewidth=0.8,
    )
    _mean_angle = np.deg2rad(_vector_mean)
    _mean_radius = max(1.0, float(np.max(_counts)) * 1.15)
    _ax_compass.plot(
        [_mean_angle, _mean_angle], [0, _mean_radius],
        color="#7c3aed", linewidth=4.0, solid_capstyle="round",
    )
    _ax_compass.scatter(
        [_mean_angle], [_mean_radius], color="#7c3aed", s=70,
        edgecolor="white", linewidth=0.9, zorder=5,
    )
    _ax_compass.set_theta_zero_location("E")
    _ax_compass.set_thetagrids(
        [0, 90, 180, 270], labels=["0°", "+90°", "±180°", "−90°"]
    )
    _ax_compass.set_yticklabels([])
    _ax_compass.set_ylim(0, _mean_radius * 1.08)
    _ax_compass.set_title(
        f"5 · Population vector\ndirection = {_vector_mean:+.1f}°",
        va="bottom",
        pad=18,
    )
    _ax_compass.grid(alpha=0.18)

    _cartesian_axes = [
        _ax_spikes, _ax_weights, _ax_contributions, _ax_likelihood
    ]
    for _axis in _cartesian_axes:
        _axis.set_xticks(np.arange(-180, 181, 60))
    clean_axes(_cartesian_axes)
    _fig.tight_layout(h_pad=2.5, w_pad=1.5)
    mo.vstack(
        [
            population_controls,
            _fig,
            mo.callout(
                mo.md(
                    f"""
                    The selected candidate, **{_candidate:+.0f}°**, has
                    likelihood **{_selected_relative:.3g}** times the maximum.
                    The maximum-likelihood direction is **{_peak_direction:+.1f}°**.
                    Neither number is the posterior probability of an exact
                    direction. The peak is fixed at one by the plotting scale;
                    its height contains no information about response strength.

                    The circular panel assigns each spike a unit vector in its
                    neuron's preferred direction and sums the vectors. The
                    resulting direction is **{_vector_mean:+.1f}°**. For cosine
                    weights and a constant expected total, this direction is
                    also the likelihood maximum. The vector's length sets the
                    amplitude of the cosine log-likelihood profile: a longer
                    resultant produces a narrower relative likelihood. Keeping
                    only its direction loses that concentration information;
                    keeping both vector components retains it in this model.

                    This equivalence depends on the tuning and noise model.
                    Heterogeneous tuning or a varying expected total need not
                    give a cosine log likelihood, so a population-vector angle
                    need not be the maximum-likelihood estimate. Negative
                    cosine weights describe signed contributions to the score,
                    not negative probabilities or negative output firing rates.
                    """
                ),
                kind="info",
            ),
            mo.Html(
                r"""
                <div class="formula-card"><strong>Readout operation:</strong>
                for each candidate, add <em>spike count × that candidate's
                connection weight</em> across the input population. A bank of
                candidates performs the same operation in parallel.</div>
                """
            ),
        ],
        gap=0.7,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. From a tuning curve to a single-neuron likelihood

    A tuning curve specifies the neuron's **expected spike count** at each
    direction. If tuning is expressed as a firing rate, multiply it by the
    observation duration to obtain this expected count. The mean alone does
    not specify response probabilities: two count distributions with the same
    mean can assign different probabilities to an observed count.

    Here we assume a Poisson count distribution. Its variance equals its mean,
    so higher expected counts also have larger absolute standard deviations.
    Once the mean is specified, all count probabilities are determined. This
    example includes a small additive baseline in its mean tuning curve and
    evaluates the full Poisson likelihood, without assuming cosine weights.

    The left panel varies direction and displays the predicted mean. Three
    colored points select means for the count distributions in the middle
    panel. Each middle curve is a probability mass function over nonnegative
    integers; only counts up to 20 are shown. The vertical line fixes the
    observed count. Its intersections with the three curves give that count's
    probability under the three selected directions. Evaluating this probability
    at every direction and dividing by its maximum gives the right panel.

    A useful visual comparison is between the observed-count horizontal line
    in the first panel and the peaks in the last panel. For a positive count
    within the tuning curve's mean range, the likelihood is largest where the
    predicted mean equals the observed count. There are usually two such
    directions, one on each flank. A single neuron's preferred direction is
    therefore not generally the maximum-likelihood explanation of its response.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    poisson_observed = mo.ui.slider(
        start=0,
        stop=18,
        step=1,
        value=8,
        debounce=True,
        show_value=True,
        full_width=True,
        label="Observed spike count n",
    )
    poisson_width = mo.ui.slider(
        start=0.8,
        stop=4.0,
        step=0.2,
        value=2.2,
        debounce=True,
        show_value=True,
        full_width=True,
        label="Tuning concentration (larger = narrower)",
    )
    poisson_controls = mo.vstack([poisson_observed, poisson_width], gap=0.5)
    return poisson_controls, poisson_observed, poisson_width


@app.cell(hide_code=True)
def _(
    clean_axes,
    circular_delta,
    direction_tuning,
    math,
    mo,
    np,
    plt,
    poisson_controls,
    poisson_observed,
    poisson_width,
):
    _directions = np.linspace(-180, 180, 721)
    _preference = 35.0
    _means = 0.35 + 11.0 * direction_tuning(
        _directions, np.array([_preference]), kappa=poisson_width.value
    )[:, 0]
    _n_observed = poisson_observed.value

    def _poisson_probability(count, rate):
        count = np.asarray(count, dtype=float)
        rate = np.asarray(rate, dtype=float)
        _log_factorial = np.array(
            [math.lgamma(float(value) + 1.0) for value in count.flat]
        ).reshape(count.shape)
        return np.exp(count * np.log(rate) - rate - _log_factorial)

    _likelihood = _poisson_probability(_n_observed, _means)
    _relative = _likelihood / np.max(_likelihood)
    _min_mean = 0.35 + 11.0 * np.exp(-2.0 * poisson_width.value)
    _max_mean = 11.35
    if _n_observed <= _min_mean:
        _best_directions = np.array([float(circular_delta(_preference + 180, 0))])
        _count_note = (
            "The observed count is below the smallest predicted mean. "
            "The maximum occurs opposite the preferred direction, where the mean is smallest. "
            "For zero spikes, every candidate still has likelihood exp(−mean); silence is informative."
        )
    elif _n_observed >= _max_mean:
        _best_directions = np.array([_preference])
        _count_note = (
            "The observed count exceeds the largest predicted mean. "
            "The maximum occurs at the preferred direction, which predicts the largest count. "
            "That candidate is only the best among the specified candidates; "
            "a relative peak of one does not say the observed count was probable."
        )
    else:
        _offset = np.rad2deg(np.arccos(
            1.0 + np.log((_n_observed - 0.35) / 11.0) / poisson_width.value
        ))
        _best_directions = circular_delta(
            np.array([_preference - _offset, _preference + _offset]), 0
        )
        _count_note = (
            "Two directions predict a mean equal to the observed count and have equal maximum "
            "likelihood. The observation does not distinguish these two flanks of the tuning curve."
        )
    _best_text = " and ".join(f"{angle:+.1f}°" for angle in _best_directions)
    _example_directions = np.array([-75.0, 35.0, 120.0])
    _example_rates = 0.35 + 11.0 * direction_tuning(
        _example_directions, np.array([_preference]), kappa=poisson_width.value
    )[:, 0]
    _possible_counts = np.arange(0, 21)
    _example_colors = ["#2563a8", "#d97706", "#16846b"]

    _fig, _axes = plt.subplots(1, 3, figsize=(12.7, 4.2))
    _fig.patch.set_facecolor("white")
    _axes[0].plot(_directions, _means, color="#2563a8", linewidth=2.6)
    for _direction, _rate, _color in zip(
        _example_directions, _example_rates, _example_colors
    ):
        _axes[0].scatter([_direction], [_rate], color=_color, s=55, zorder=4)
        _axes[0].plot(
            [_direction, _direction], [0, _rate], color=_color,
            linestyle=":", linewidth=1.1,
        )
    _axes[0].set(
        title="1 · Tuning predicts a mean",
        xlabel="candidate direction",
        ylabel="predicted mean spikes",
        xlim=(-180, 180),
        ylim=(0, max(12.5, _n_observed + 1.5)),
    )
    _axes[0].axhline(
        _n_observed, color="#7c3aed", linestyle="--", linewidth=1.2,
        label=f"observed count = {_n_observed}",
    )
    _axes[0].legend(frameon=False, fontsize=8)

    for _direction, _rate, _color in zip(
        _example_directions, _example_rates, _example_colors
    ):
        _probabilities = _poisson_probability(_possible_counts, _rate)
        _axes[1].plot(
            _possible_counts, _probabilities, marker="o", markersize=3.5,
            color=_color, linewidth=1.7,
            label=f"{_direction:+.0f}° → mean {_rate:.1f}",
        )
        if _n_observed <= 20:
            _axes[1].scatter(
                [_n_observed], [_poisson_probability(_n_observed, _rate)],
                color=_color, s=70, edgecolor="white", linewidth=0.8, zorder=5,
            )
    _axes[1].axvline(
        _n_observed, color="#7c3aed", linestyle="--", linewidth=1.5
    )
    _axes[1].set(
        title=f"2 · How probable is n = {_n_observed}?",
        xlabel="possible observed count",
        ylabel="probability",
        xlim=(-0.5, 20.5),
        ylim=(0, 0.72),
    )
    _axes[1].legend(frameon=False, fontsize=7.7)

    _axes[2].plot(_directions, _relative, color="#7c3aed", linewidth=2.8)
    _axes[2].fill_between(
        _directions, 0, _relative, color="#7c3aed", alpha=0.18
    )
    for _best in _best_directions:
        _axes[2].axvline(_best, color="#182338", linestyle="--", linewidth=1.3)
    _axes[2].set(
        title=f"3 · Likelihood with n = {_n_observed} fixed",
        xlabel="candidate direction",
        ylabel="relative likelihood",
        xlim=(-180, 180),
        ylim=(0, 1.08),
    )
    for _axis in (_axes[0], _axes[2]):
        _axis.set_xticks(np.arange(-180, 181, 90))
    clean_axes(_axes)
    _fig.tight_layout(w_pad=1.5)

    mo.vstack(
        [
            poisson_controls,
            _fig,
            mo.callout(
                mo.md(
                    f"""
                    {_count_note} The maximizing direction(s) are
                    **{_best_text}**. Other neurons with different preferences
                    can distinguish these candidates if their conditional
                    response distributions differ at those directions.

                    Keep the count fixed and change tuning concentration. This
                    changes the model's predicted means and therefore where
                    each candidate falls in the count distribution. It does
                    not change the observed data. The likelihood depends on
                    both the data and this assumed response model.
                    """
                ),
                kind="info",
            ),
            mo.accordion(
                {
                    "Poisson probability and log-likelihood terms": mo.md(
                        r"""For mean count $f(\theta)$ and observed count $n$,
                        the Poisson probability is
                        $P(n\mid\theta)=f(\theta)^n e^{-f(\theta)}/n!$.
                        Its log contains a spike-dependent term
                        $n\log f(\theta)$, the expected-count term
                        $-f(\theta)$, and $-\log(n!)$. The last term is constant
                        across directions for a fixed observation. The
                        expected-count term generally is not."""
                    )
                }
            ),
        ],
        gap=0.7,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. When a weighted sum represents the log likelihood

    For independent Poisson neurons, each candidate's population log likelihood
    contains three terms: observed counts weighted by log expected counts,
    minus the total expected count, minus a term involving the observed-count
    factorials. The factorial term is constant across candidates on a fixed
    trial. The total expected count can vary across candidates even though the
    observed count vector is fixed.

    The expected-count term is part of the Poisson probability model, not a
    separate correction applied only when too few spikes were observed. For
    example, a silent neuron contributes minus its expected count to the log
    likelihood. Directions predicting a large response from that silent neuron
    are less likely than directions predicting a small response. Keeping only
    observed-count contributions would miss this information.

    In an evenly sampled population of shifted copies of a smooth tuning curve,
    the **total expected count is approximately constant across direction**.
    Subtracting it shifts the entire log-likelihood curve vertically without
    changing its shape. The weighted sum then represents the log likelihood
    up to a candidate-independent offset. Exact constancy is the mathematical
    condition; dense uniform coverage is one way to approximate it. Identical
    tuning curves are not necessary if heterogeneous curves also sum to a
    constant. A finite set of evenly spaced neurons need not give exact constancy.

    The first panel plots each neuron's mean tuning curve. The middle panel
    sums those curves vertically at each direction and expresses the sum as a
    percentage deviation from its mean. The final panel compares the full
    Poisson likelihood with the exponentiated weighted score that omits the
    expected-count term; each curve is scaled separately to peak at one.
    Here the observations are randomly sampled Poisson counts. Changing the
    trial number draws a new count vector at the selected generating direction.

    Select reduced coverage near 90° or unequal gains. The expected total now
    depends on direction. The omitted term can alter both the maximum and the
    profile around it. The orange line identifies the generating direction,
    which need not equal either estimate on a noisy trial.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    coverage_mode = mo.ui.radio(
        options=["uniform coverage", "reduced coverage near 90°", "unequal gains"],
        value="uniform coverage",
        inline=True,
        label="Population architecture",
    )
    assumption_direction = mo.ui.slider(
        start=-150,
        stop=150,
        step=15,
        value=30,
        debounce=True,
        show_value=True,
        full_width=True,
        label="Direction that generated this trial",
    )
    assumption_seed = mo.ui.slider(
        start=1,
        stop=30,
        step=1,
        value=7,
        debounce=True,
        show_value=True,
        full_width=True,
        label="Trial number (changes the random spike pattern)",
    )
    assumption_controls = mo.vstack(
        [coverage_mode, assumption_direction, assumption_seed], gap=0.5
    )
    return (
        assumption_controls,
        assumption_direction,
        assumption_seed,
        coverage_mode,
    )


@app.cell(hide_code=True)
def _(
    assumption_controls,
    assumption_direction,
    assumption_seed,
    circular_delta,
    clean_axes,
    coverage_mode,
    direction_tuning,
    mo,
    np,
    plt,
    relative_likelihood,
):
    _preferences = np.arange(-170, 171, 20, dtype=float)
    _candidates = np.linspace(-180, 180, 721)
    _gains = np.ones_like(_preferences)
    if coverage_mode.value == "reduced coverage near 90°":
        _gains[
            np.abs(circular_delta(_preferences, 90.0)) <= 45.0
        ] = 0.08
    elif coverage_mode.value == "unequal gains":
        _gains = 1.0 + 0.62 * np.cos(
            np.deg2rad(circular_delta(_preferences, -65.0))
        )

    _base_tuning = direction_tuning(
        _candidates, _preferences, kappa=2.15
    )
    _rate_matrix = 8.5 * _base_tuning * _gains[np.newaxis, :]
    _true_rates = 8.5 * direction_tuning(
        np.array([assumption_direction.value]), _preferences, kappa=2.15
    )[0] * _gains
    _rng = np.random.default_rng(assumption_seed.value)
    _observed = _rng.poisson(_true_rates)

    _log_rates = np.log(np.maximum(_rate_matrix, 1e-12))
    _weighted_spikes = _log_rates @ _observed
    _population_budget = np.sum(_rate_matrix, axis=1)
    _budget_deviation = 100.0 * (
        _population_budget / np.mean(_population_budget) - 1.0
    )
    if np.max(np.abs(_budget_deviation)) < 1e-8:
        _budget_deviation = np.zeros_like(_budget_deviation)
    _full_log = _weighted_spikes - _population_budget
    _full_relative = relative_likelihood(_full_log)
    _shortcut_relative = relative_likelihood(_weighted_spikes)
    _full_peak = float(_candidates[np.argmax(_full_log)])
    _shortcut_peak = float(_candidates[np.argmax(_weighted_spikes)])
    _budget_range = float(np.ptp(_budget_deviation))

    _fig, _axes = plt.subplots(1, 3, figsize=(13.0, 4.45))
    _fig.patch.set_facecolor("white")
    _tuning_colors = plt.get_cmap("twilight")(
        np.linspace(0.04, 0.96, len(_preferences))
    )
    for _index, (_preference, _color) in enumerate(
        zip(_preferences, _tuning_colors)
    ):
        _axes[0].plot(
            _candidates,
            _rate_matrix[:, _index],
            color=_color,
            linewidth=1.0,
            alpha=0.78,
        )
    _axes[0].set(
        title="Individual tuning curves",
        xlabel="candidate direction",
        ylabel="expected spikes per neuron",
        xlim=(-180, 180),
    )

    _axes[1].plot(
        _candidates, _budget_deviation, color="#d97706", linewidth=2.8
    )
    _axes[1].fill_between(
        _candidates,
        0.0,
        _budget_deviation,
        color="#d97706",
        alpha=0.15,
    )
    _axes[1].axhline(0.0, color="#6b7280", linewidth=1.0)
    _axes[1].set(
        title=f"Expected total; range = {_budget_range:.1f}%",
        xlabel="candidate direction",
        ylabel="deviation from mean (%)",
        xlim=(-180, 180),
    )

    _axes[2].plot(
        _candidates,
        _full_relative,
        color="#182338",
        linewidth=2.8,
        label=f"full Poisson: {_full_peak:+.0f}°",
    )
    _axes[2].plot(
        _candidates,
        _shortcut_relative,
        color="#7c3aed",
        linewidth=2.2,
        linestyle="--",
        label=f"expected-count term omitted: {_shortcut_peak:+.0f}°",
    )
    _axes[2].axvline(
        assumption_direction.value,
        color="#d97706",
        linewidth=1.4,
        linestyle=":",
        label=f"generating direction: {assumption_direction.value:+.0f}°",
    )
    _axes[2].set(
        title="Effect of the expected-count term",
        xlabel="candidate direction",
        ylabel="exponentiated score / its maximum",
        xlim=(-180, 180),
        ylim=(0, 1.08),
    )
    _axes[2].legend(
        frameon=False, fontsize=7.6, loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
    )
    for _axis in _axes:
        _axis.set_xticks(np.arange(-180, 181, 90))
    clean_axes(_axes)
    _fig.tight_layout(w_pad=1.45)

    _peak_gap = abs(float(circular_delta(_shortcut_peak, _full_peak)))
    _conclusion = (
        "The expected total is effectively constant, so the curves coincide to plotting precision."
        if coverage_mode.value == "uniform coverage"
        else f"The expected total varies across directions. On this trial the two maxima differ by {_peak_gap:.1f}°."
    )

    mo.vstack(
        [
            assumption_controls,
            _fig,
            mo.callout(
                mo.md(
                    f"""
                    **{coverage_mode.value.capitalize()}:** {_conclusion}

                    The solid curve is a relative likelihood under the stated
                    model. When the expected total varies, the dashed curve is
                    an approximation and cannot be interpreted as that model's
                    relative likelihood. Unequal coverage does not prevent
                    likelihood decoding: a candidate-specific offset equal to
                    minus the expected total restores the full calculation.
                    It does prevent the uncorrected weighted sum from being
                    sufficient for this calculation.

                    If stimulus strength multiplies all mean tuning curves by
                    a common positive gain without changing their shape, its
                    contribution to the count-weighted log term is independent
                    of direction. With a constant expected total across
                    directions, the same direction weights can then be used
                    across gains. More observed spikes can still sharpen the
                    direction likelihood. These cancellations concern direction
                    at a given strength; they do not generally justify dropping
                    the same terms when comparing strengths or signal presence.
                    """
                ),
                kind="warn" if coverage_mode.value != "uniform coverage" else "success",
            ),
            mo.accordion(
                {
                    "Population log-likelihood terms": mo.md(
                        r"""For independent Poisson neurons, the direction-
                        dependent part of the population log likelihood is

                        $$\sum_i n_i\log f_i(\theta)-\sum_i f_i(\theta).$$

                        The first sum is observed spikes times fixed weights.
                        The second is the total expected population activity.
                        If that second sum is flat across $\theta$, removing it
                        changes neither the curve's shape nor its maximum. For
                        pure von Mises mean tuning, $\log f_i(\theta)$ reduces,
                        up to direction-independent terms, to a cosine weight
                        scaled by the tuning concentration. An additive firing
                        baseline generally prevents that exact reduction."""
                    )
                }
            ),
        ],
        gap=0.7,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Correlation and the variability of a decision variable

    The product rule above requires conditional independence. Correlated
    responses generally require a joint response model: multiplying the
    individual-neuron probabilities does not recover their joint probability.
    Even zero pairwise correlation is insufficient to establish independence
    in general. In the jointly Gaussian example here, zero correlation does
    imply independence.

    This figure uses two continuous Gaussian responses, not integer Poisson
    spike counts. For each stimulus class, the mean of each response and its
    variance (4 response units squared) stay fixed. The correlation slider
    changes only the covariance within each class. Thus it controls **noise
    correlation conditional on stimulus**, not correlation induced by pooling
    trials with different stimuli. Both classes have the same covariance.

    Each point in the left panel is one pair of responses on a simulated trial.
    Positive correlation elongates each cloud along the positive diagonal;
    negative correlation elongates it along the opposite diagonal. The middle
    panel projects the same trials onto a difference or sum of responses. The
    overlap after projection determines error probability at the dashed
    threshold. The final panel gives the analytically predicted accuracy,
    assuming equal class probabilities and equal costs for the two errors.

    In the opponent example, the class means differ along the difference
    direction. Positive common fluctuations mostly run parallel to the decision
    boundary and partly cancel in the subtraction. In the sum example, the
    means differ along the positive diagonal, so those same fluctuations
    increase variability along the readout. Correlation's effect depends on
    its orientation relative to the stimulus-dependent mean difference.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    correlation_code = mo.ui.radio(
        options=["opponent: neuron 1 − neuron 2", "shared: neuron 1 + neuron 2"],
        value="opponent: neuron 1 − neuron 2",
        inline=True,
        label="Readout geometry",
    )
    correlation_strength = mo.ui.slider(
        start=-0.85,
        stop=0.85,
        step=0.05,
        value=0.45,
        debounce=True,
        show_value=True,
        full_width=True,
        label="Within-class response correlation",
    )
    correlation_seed = mo.ui.slider(
        start=1,
        stop=20,
        step=1,
        value=4,
        debounce=True,
        show_value=True,
        full_width=True,
        label="Example trial cloud",
    )
    correlation_controls = mo.vstack(
        [correlation_code, correlation_strength, correlation_seed], gap=0.5
    )
    return (
        correlation_code,
        correlation_controls,
        correlation_seed,
        correlation_strength,
    )


@app.cell(hide_code=True)
def _(
    clean_axes,
    correlation_code,
    correlation_controls,
    correlation_seed,
    correlation_strength,
    mo,
    normal,
    np,
    plt,
):
    _opponent = correlation_code.value.startswith("opponent")
    if _opponent:
        _mean_a = np.array([10.0, 6.0])
        _mean_b = np.array([6.0, 10.0])
        _weights = np.array([1.0, -1.0])
        _readout_name = "difference r₁ − r₂"
    else:
        _mean_a = np.array([10.0, 10.0])
        _mean_b = np.array([6.0, 6.0])
        _weights = np.array([1.0, 1.0])
        _readout_name = "sum r₁ + r₂"

    _rho = correlation_strength.value
    _single_variance = 4.0
    _covariance = _single_variance * np.array([[1.0, _rho], [_rho, 1.0]])
    _rng = np.random.default_rng(correlation_seed.value)
    _trials_a = _rng.multivariate_normal(_mean_a, _covariance, size=900)
    _trials_b = _rng.multivariate_normal(_mean_b, _covariance, size=900)
    _scores_a = _trials_a @ _weights
    _scores_b = _trials_b @ _weights
    _mean_score_a = float(_mean_a @ _weights)
    _mean_score_b = float(_mean_b @ _weights)
    _threshold = 0.5 * (_mean_score_a + _mean_score_b)
    _score_variance = float(_weights @ _covariance @ _weights)
    _score_sd = np.sqrt(_score_variance)
    _separation = abs(_mean_score_a - _mean_score_b)
    _accuracy = normal.cdf(_separation / (2.0 * _score_sd))

    _rho_grid = np.linspace(-0.85, 0.85, 241)
    _variance_curve = np.array(
        [
            float(
                _weights
                @ (_single_variance * np.array([[1.0, r], [r, 1.0]]))
                @ _weights
            )
            for r in _rho_grid
        ]
    )
    _accuracy_curve = np.array(
        [normal.cdf(_separation / (2.0 * np.sqrt(v))) for v in _variance_curve]
    )

    _fig, _axes = plt.subplots(1, 3, figsize=(13.0, 4.2))
    _fig.patch.set_facecolor("white")
    _axes[0].scatter(
        _trials_a[:450, 0], _trials_a[:450, 1], s=12,
        color="#2563a8", alpha=0.28, label="stimulus A",
    )
    _axes[0].scatter(
        _trials_b[:450, 0], _trials_b[:450, 1], s=12,
        color="#d97706", alpha=0.28, label="stimulus B",
    )
    _axes[0].scatter(
        [_mean_a[0], _mean_b[0]], [_mean_a[1], _mean_b[1]],
        s=85, color=["#2563a8", "#d97706"], edgecolor="white", zorder=5,
    )
    _line = np.linspace(0, 16, 100)
    if _opponent:
        _axes[0].plot(
            _line, _line - _threshold, color="#7c3aed",
            linestyle="--", linewidth=1.5, label="decision boundary",
        )
    else:
        _axes[0].plot(
            _line, _threshold - _line, color="#7c3aed",
            linestyle="--", linewidth=1.5, label="decision boundary",
        )
    _axes[0].set(
        title=f"Joint responses at correlation {_rho:+.2f}",
        xlabel="neuron 1 response",
        ylabel="neuron 2 response",
        xlim=(0, 16),
        ylim=(0, 16),
    )
    _axes[0].set_aspect("equal", adjustable="box")
    _axes[0].legend(frameon=False, fontsize=7.8, loc="upper left")

    _score_min = min(np.min(_scores_a), np.min(_scores_b))
    _score_max = max(np.max(_scores_a), np.max(_scores_b))
    _bins = np.linspace(_score_min, _score_max, 38)
    _axes[1].hist(
        _scores_a, bins=_bins, density=True, color="#2563a8",
        alpha=0.50, label="stimulus A",
    )
    _axes[1].hist(
        _scores_b, bins=_bins, density=True, color="#d97706",
        alpha=0.50, label="stimulus B",
    )
    _axes[1].axvline(
        _threshold, color="#7c3aed", linestyle="--", linewidth=1.8
    )
    _axes[1].set(
        title=f"Project onto the {_readout_name}",
        xlabel="pooled decision variable",
        ylabel="density",
    )
    _axes[1].legend(frameon=False, fontsize=8)

    _axes[2].plot(
        _rho_grid, _accuracy_curve, color="#182338", linewidth=2.7
    )
    _axes[2].scatter(
        [_rho], [_accuracy], color="#7c3aed", s=75,
        edgecolor="white", linewidth=0.8, zorder=5,
    )
    _axes[2].axhline(0.5, color="#9ca3af", linestyle=":", linewidth=1.1)
    _axes[2].set(
        title=f"Predicted classification accuracy: {_accuracy:.1%}",
        xlabel="pairwise correlation",
        ylabel="accuracy with the specified readout",
        xlim=(-0.85, 0.85),
        ylim=(0.48, 1.01),
    )
    clean_axes(_axes)
    _fig.tight_layout(w_pad=1.5)

    if abs(_rho) < 1e-10:
        _effect = "At zero correlation, the variance of either sum or difference is 8."
    elif _rho > 0:
        _effect = (
            "Positive correlated fluctuations partly cancel in the subtraction, narrowing the decision variable."
            if _opponent
            else "Positive correlated fluctuations add together, widening the decision variable."
        )
    else:
        _effect = (
            "Negative correlation widens a difference readout."
            if _opponent else "Negative correlation partly cancels in a sum readout."
        )
    mo.vstack(
        [
            correlation_controls,
            _fig,
            mo.callout(
                mo.md(
                    f"""
                    The two individual neurons have unchanged means and
                    variances; only their trial-to-trial relationship changed.
                    The pooled variance is now **{_score_variance:.2f}** and
                    predicted accuracy is **{_accuracy:.1%}**. {_effect}

                    For these symmetric Gaussian examples, the selected sum
                    or difference remains the optimal linear readout direction
                    as correlation changes. Only its scale in the exact log
                    likelihood ratio changes. In a general population,
                    covariance can also change the optimal relative weights.
                    The result here therefore illustrates correlation geometry,
                    not a general rule that fixed weights are optimal.

                    In the [paper's supplement](https://www.cns.nyu.edu/~tony/Publications/jazayeri-movshon-2006-supp.pdf),
                    decoder weights are derived without accounting for
                    correlations, while predicted behavioral variability
                    includes a specified correlation structure. Computing the
                    distribution of a decoder's output and constructing a
                    correlation-aware decoder are separate operations.
                    """
                ),
                kind="info",
            ),
        ],
        gap=0.7,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Detection, estimation, and discrimination

    A direction likelihood compares candidate directions within a specified
    signal model. **Detection also requires a signal-absent model.** The
    statistically relevant quantity for two specified hypotheses is the
    likelihood ratio: the probability of the observed response under one
    hypothesis divided by its probability under the other. The height of a
    likelihood curve that was separately rescaled on each trial is insufficient
    for this comparison.

    Here the signal model has 18 independent Poisson neurons with von Mises
    mean tuning (peak expected count 8, concentration 1.9). The noise model
    gives every neuron the same expected count, chosen to match the population's
    average expected total under the signal model. This makes response pattern,
    rather than simply total count, informative about signal presence. Every
    point on the purple curve is the log likelihood ratio for a signal at that
    direction versus this *same* noise model. Expected-count terms are retained.

    The trial controls generate a single count vector. At signal fraction zero,
    it is sampled from the noise model; at one, from the signal model at the
    selected direction. Intermediate fractions interpolate the generating
    means, while the decoder continues to compare the two fixed endpoint
    models. The same observed vector is used in all three panels.

    Detection tests the specified direction 0° against noise. Estimation
    selects the maximum over all signal directions. Discrimination compares
    only the two specified directions. Subtracting their plotted values
    cancels the common noise reference and gives the log likelihood ratio
    between the two signal alternatives. The reference therefore supports
    detection while preserving all within-signal direction comparisons.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    task_direction = mo.ui.slider(
        start=-120,
        stop=120,
        step=10,
        value=30,
        debounce=True,
        show_value=True,
        full_width=True,
        label="Generating direction of the signal component (degrees)",
    )
    task_separation = mo.ui.slider(
        start=20,
        stop=180,
        step=10,
        value=100,
        debounce=True,
        show_value=True,
        full_width=True,
        label="Separation of the two known alternatives",
    )
    detection_criterion = mo.ui.slider(
        start=-20.0,
        stop=60.0,
        step=5.0,
        value=0.0,
        debounce=True,
        show_value=True,
        full_width=True,
        label="Detection criterion: log likelihood ratio",
    )
    task_signal_fraction = mo.ui.slider(
        start=0.0, stop=1.0, step=0.1, value=1.0, debounce=True,
        show_value=True, full_width=True,
        label="Generating signal fraction (0 = noise model, 1 = signal model)",
    )
    task_seed = mo.ui.slider(
        start=1, stop=30, step=1, value=4, debounce=True,
        show_value=True, full_width=True, label="Trial number",
    )
    task_controls = mo.vstack(
        [task_direction, task_signal_fraction, task_seed, task_separation, detection_criterion], gap=0.5
    )
    return detection_criterion, task_controls, task_direction, task_seed, task_separation, task_signal_fraction


@app.cell(hide_code=True)
def _(
    clean_axes,
    detection_criterion,
    direction_tuning,
    mo,
    np,
    plt,
    task_controls,
    task_direction,
    task_seed,
    task_separation,
    task_signal_fraction,
):
    _preferences = np.arange(-180, 180, 20, dtype=float)
    _grid = np.linspace(-180, 180, 721)
    _signal_means = 8.0 * direction_tuning(_grid, _preferences, kappa=1.9)
    # Use one full period without its duplicated endpoint for the mean.
    _noise_means = np.full(len(_preferences), _signal_means[:-1].mean())
    _generating_signal_means = 8.0 * direction_tuning(
        np.array([task_direction.value]), _preferences, kappa=1.9
    )[0]
    _trial_means = (
        task_signal_fraction.value * _generating_signal_means
        + (1.0 - task_signal_fraction.value) * _noise_means
    )
    _counts = np.random.default_rng(task_seed.value).poisson(_trial_means)
    # Count-factorial terms cancel between these two models for the same data.
    _log_likelihood = (
        np.log(_signal_means / _noise_means) @ _counts
        - _signal_means.sum(axis=1) + _noise_means.sum()
    )
    _peak = float(_grid[np.argmax(_log_likelihood)])
    _known_direction = 0.0
    _known_value = float(np.interp(_known_direction, _grid, _log_likelihood))
    _detect = _known_value >= detection_criterion.value
    _left_alt = -task_separation.value / 2.0
    _right_alt = task_separation.value / 2.0
    _left_value = float(np.interp(_left_alt, _grid, _log_likelihood))
    _right_value = float(np.interp(_right_alt, _grid, _log_likelihood))
    _difference = _right_value - _left_value
    _choice = (
        "equal likelihoods" if np.isclose(_difference, 0, atol=1e-10)
        else f"choose {(_right_alt if _difference > 0 else _left_alt):+.0f}°"
    )
    _lower_limit = min(-5.0, float(np.min(_log_likelihood)) - 5.0, detection_criterion.value - 5.0)
    _upper_limit = max(5.0, float(np.max(_log_likelihood)) + 5.0, detection_criterion.value + 5.0)

    _fig, _axes = plt.subplots(1, 3, figsize=(13.0, 4.0), sharey=True)
    _fig.patch.set_facecolor("white")
    for _axis in _axes:
        _axis.plot(_grid, _log_likelihood, color="#7c3aed", linewidth=2.6)
        _axis.fill_between(
            _grid, _lower_limit, _log_likelihood,
            color="#7c3aed", alpha=0.12,
        )
        _axis.set(
            xlabel="candidate direction",
            xlim=(-180, 180),
            ylim=(_lower_limit, _upper_limit),
        )
        _axis.set_xticks(np.arange(-180, 181, 90))
        _axis.axhline(0, color="#9ca3af", linewidth=0.9, linestyle=":")

    _axes[0].axhline(
        detection_criterion.value, color="#d97706", linestyle="--",
        linewidth=1.5, label="criterion",
    )
    _axes[0].scatter([0], [_known_value], color="#182338", s=65, zorder=4)
    _axes[0].set(
        title=f"0° detection: {'signal' if _detect else 'noise'} response",
        ylabel="log likelihood ratio: signal / noise",
    )
    _axes[0].legend(frameon=False, fontsize=8)

    _axes[1].axvline(_peak, color="#182338", linestyle="--", linewidth=1.4)
    _axes[1].scatter(
        [_peak], [np.max(_log_likelihood)], color="#182338", s=65, zorder=4
    )
    _axes[1].set(title=f"Direction estimate: {_peak:+.0f}°")

    _axes[2].scatter(
        [_left_alt, _right_alt], [_left_value, _right_value],
        color=["#2563a8", "#d97706"], s=[65, 65], zorder=4,
    )
    _axes[2].vlines(
        [_left_alt, _right_alt], 0, [_left_value, _right_value],
        colors=["#2563a8", "#d97706"], linestyles=":", linewidth=1.4,
    )
    _axes[2].set(title=f"Discrimination: {_choice}")
    clean_axes(_axes)
    _fig.tight_layout(w_pad=1.5)

    mo.vstack(
        [
            task_controls,
            _fig,
            mo.callout(
                mo.md(
                    f"""
                    **Detection:** at 0°, the log likelihood ratio is
                    **{_known_value:.2f}**, compared with criterion
                    **{detection_criterion.value:.2f}**. Zero means equal
                    likelihood under the specified signal and noise hypotheses.
                    A zero criterion minimizes classification error for equal
                    prior probabilities and equal error costs. Other priors or
                    costs generally require a different criterion. Detecting
                    a signal of *unknown* direction is a different composite
                    hypothesis problem; it requires specifying how directions
                    are combined, rather than simply reusing the 0° test.

                    **Estimation:** the maximum-likelihood direction is
                    **{_peak:+.0f}°**. This rule always returns a direction,
                    including on noise-only trials, because it searches only
                    signal directions. It is not evidence by itself that a
                    signal was present. A Bayesian estimate additionally
                    depends on the prior and the loss assigned to estimation
                    errors; for example, squared error favors a posterior mean
                    in a noncircular scalar problem.

                    **Discrimination:** the alternatives are
                    **{_left_alt:+.0f}°** and **{_right_alt:+.0f}°**. The right
                    minus left log likelihood is **{_difference:.2f}**. Its sign
                    selects the larger likelihood under equal priors and equal
                    error costs. Two points on a sharply peaked curve can both
                    have low values if neither specified alternative is near
                    the estimate. Forced choice still compares those two points.
                    """
                ),
                kind="info",
            ),
        ],
        gap=0.7,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(r"""
            ## 9. What the representation contains

            The [paper](https://www.cns.nyu.edu/~tony/Publications/jazayeri-movshon-2006.pdf)
            proposes a feedforward population transformation whose output
            represents a log-likelihood function under specified encoding
            assumptions. An output unit corresponds to a candidate stimulus;
            its weights depend on the input neurons' tuning and response
            statistics. This supplies a shared representation for several
            perceptual tasks. It is a model of a possible computation, rather
            than direct evidence that recorded neurons explicitly implement
            each step shown here.

            The examples distinguish several operations that can preserve the
            same maximum while retaining different statistical information:

            | Operation | What is preserved, and what changes? |
            |---|---|
            | Multiply a likelihood by a positive constant independent of stimulus | All likelihood ratios and the maximum are preserved. Absolute scale changes. |
            | Add a constant independent of stimulus to log likelihood | All log-likelihood differences and the maximum are preserved. |
            | Multiply log likelihood by a positive constant | The maximum is preserved, but likelihood ratios and concentration change. |
            | Keep only the maximizing stimulus | The estimate is retained, but the rest of the profile is discarded. |
            | Multiply by a prior and normalize | This constructs a posterior; relative probabilities and the maximum can change. |

            A candidate-independent factor may depend on the observed response.
            It can be removed when comparing directions within that trial, but
            its removal does not automatically permit comparing heights across
            trials or between signal and noise models. Which terms may be
            removed depends on the hypotheses being compared.

            The population examples use exact Poisson probabilities, not just
            a qualitative assumption of variable firing. Other response models
            can also admit linear log-likelihood readouts, but this requires
            checking the form of their stimulus-dependent terms. The
            [supplement](https://www.cns.nyu.edu/~tony/Publications/jazayeri-movshon-2006-supp.pdf)
            discusses extensions beyond Poisson statistics. Approximate
            mean–variance proportionality alone is insufficient to establish
            that log tuning curves give the appropriate weights.

            When examining a new decoder, first identify the response model,
            the observation held fixed, and the candidate stimuli on the axis.
            Then determine whether its output is a log likelihood, an
            approximation to one, or a decision score. Finally specify the
            prior and decision objective before calling the downstream
            behavior optimal. These distinctions make the visual profile
            quantitatively interpretable.
            """),
            mo.callout(
                mo.md(r"""
                **Reference.** M. Jazayeri and
                J. A. Movshon, “Optimal representation of sensory information
                by neural populations,” *Nature Neuroscience* 9, 690–696
                (2006), [paper](https://www.cns.nyu.edu/~tony/Publications/jazayeri-movshon-2006.pdf)
                and [supplement](https://www.cns.nyu.edu/~tony/Publications/jazayeri-movshon-2006-supp.pdf).
                The categorical, Gaussian, and Poisson examples here are
                separate teaching models. Their parameters are chosen for
                illustration, not estimated from the paper's experiments.
                """),
                kind="info",
            ),
        ],
        gap=0.8,
    )
    return


if __name__ == "__main__":
    app.run()
