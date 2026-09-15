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
        return np.exp(np.clip(shifted, -40.0, 0.0))

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
      <h1>From spikes to likelihood</h1>
      <p>A visual tutorial about a beautifully simple possibility: a sensory population can describe not just one guessed stimulus, but how well every possible stimulus explains the spikes that actually occurred. We build the idea without a spike-count distribution, introduce Poisson noise only when it becomes useful, and finish with the assumptions and correlations that determine when the shortcut works.</p>
    </div>

    <div class="concept-chain">
      <div class="concept-step"><strong>1 · Predict</strong><span>A tuning curve says what a neuron tends to do for each possible stimulus.</span></div>
      <div class="concept-step"><strong>2 · Observe</strong><span>On one trial, the population emits a particular pattern of spikes.</span></div>
      <div class="concept-step"><strong>3 · Ask backward</strong><span>Which possible stimuli make that observed pattern unsurprising?</span></div>
      <div class="concept-step"><strong>4 · Pool</strong><span>Weighted sums of spikes can answer that question for every stimulus.</span></div>
    </div>

    This is a conceptual companion to Jazayeri & Movshon (2006), not a
    line-by-line reconstruction. Keep one distinction in view throughout:

    > **Probability predicts data from a proposed world. Likelihood compares
    > proposed worlds using data we have already observed.**

    ## 1. Turn the question around

    Begin with one neuron and no distributional machinery. Suppose its trial
    response has only three useful labels: **quiet**, **medium**, or **busy**.
    The left panel is a forward model: each row asks what the neuron might do
    if that row's direction were really present. Select the response that was
    actually observed. The highlighted column, read vertically, becomes a
    likelihood over the three candidate causes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    observed_response = mo.ui.radio(
        options=["quiet", "medium", "busy"],
        value="busy",
        inline=True,
        label="The neuron was observed to be…",
    )
    return (observed_response,)


@app.cell(hide_code=True)
def _(Rectangle, clean_axes, mo, np, observed_response, plt):
    _responses = ["quiet", "medium", "busy"]
    _worlds = ["leftward", "straight", "rightward"]
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

    _fig, (_ax_table, _ax_like) = plt.subplots(
        1, 2, figsize=(11.2, 4.1), gridspec_kw={"width_ratios": [1.35, 1]}
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
        title="Forward probabilities: what response would each world produce?",
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
        title=f"Read the '{observed_response.value}' column as likelihood",
        xlabel="support for each candidate world",
        xlim=(0, 0.8),
    )
    _ax_like.invert_yaxis()
    clean_axes([_ax_like])
    _fig.tight_layout(w_pad=2.4)

    mo.vstack(
        [
            observed_response,
            _fig,
            mo.callout(
                mo.md(
                    f"""
                    The observed response was **{observed_response.value}**.
                    That event had probability **{_likelihood[0]:.2f}** in a
                    leftward world, **{_likelihood[1]:.2f}** in a straight
                    world, and **{_likelihood[2]:.2f}** in a rightward world.
                    So **{_worlds[_best]}** is best supported by this neuron.

                    Rows are probability distributions and sum to one. The
                    highlighted column is a likelihood function: it compares
                    candidate causes and need not sum to one. Turning it into
                    probabilities *of the causes* would additionally require
                    prior probabilities and normalization.
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
    ## 2. Independent clues multiply

    Now imagine two independent sources of sensory evidence. Each produces a
    broad likelihood curve rather than a single answer. Move where each clue
    points and how precise both clues are. The lower panel keeps only places
    that both clues support.

    Independence here is conditional: **once a candidate stimulus is fixed,
    knowing clue 1 does not help predict clue 2**. That is the condition that
    lets their support multiply.
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
        label="Clue 1 points toward",
    )
    clue_two_center = mo.ui.slider(
        start=-80,
        stop=80,
        step=5,
        value=30,
        debounce=True,
        show_value=True,
        full_width=True,
        label="Clue 2 points toward",
    )
    clue_width = mo.ui.slider(
        start=12,
        stop=60,
        step=4,
        value=32,
        debounce=True,
        show_value=True,
        full_width=True,
        label="Uncertainty of each clue (width)",
    )
    clue_controls = mo.vstack(
        [clue_one_center, clue_two_center, clue_width], gap=0.5
    )
    return clue_controls, clue_one_center, clue_two_center, clue_width


@app.cell(hide_code=True)
def _(
    clean_axes,
    clue_controls,
    clue_one_center,
    clue_two_center,
    clue_width,
    mo,
    np,
    plt,
):
    _x = np.linspace(-110, 110, 700)
    _width = clue_width.value
    _l1 = 0.96 * np.exp(-0.5 * ((_x - clue_one_center.value) / _width) ** 2)
    _l2 = 0.96 * np.exp(-0.5 * ((_x - clue_two_center.value) / _width) ** 2)
    _joint = _l1 * _l2
    _best_x = float(_x[np.argmax(_joint)])
    _peak_support = float(np.max(_joint))

    _fig, (_ax_parts, _ax_joint) = plt.subplots(
        2, 1, figsize=(10.8, 6.7), sharex=True
    )
    _fig.patch.set_facecolor("white")
    _ax_parts.plot(_x, _l1, color="#2563a8", linewidth=2.5, label="clue 1")
    _ax_parts.fill_between(_x, 0, _l1, color="#2563a8", alpha=0.12)
    _ax_parts.plot(_x, _l2, color="#d97706", linewidth=2.5, label="clue 2")
    _ax_parts.fill_between(_x, 0, _l2, color="#d97706", alpha=0.12)
    _ax_parts.set(
        title="Each clue supports a range of possible stimuli",
        ylabel="likelihood",
        ylim=(0, 1.05),
    )
    _ax_parts.legend(frameon=False, ncols=2)

    _ax_joint.plot(_x, _joint, color="#7c3aed", linewidth=3.0)
    _ax_joint.fill_between(_x, 0, _joint, color="#7c3aed", alpha=0.20)
    _ax_joint.axvline(_best_x, color="#182338", linestyle="--", linewidth=1.4)
    _ax_joint.scatter([_best_x], [_peak_support], color="#182338", s=55, zorder=4)
    _ax_joint.text(
        _best_x,
        min(0.98, _peak_support + 0.09),
        f"best joint explanation ≈ {_best_x:.0f}",
        ha="center",
        fontsize=9.5,
    )
    _ax_joint.set(
        title="Joint support: multiply at every candidate stimulus",
        xlabel="candidate stimulus",
        ylabel="likelihood product",
        xlim=(-110, 110),
        ylim=(0, 1.05),
    )
    clean_axes([_ax_parts, _ax_joint])
    _fig.tight_layout(h_pad=2.0)

    _agreement = abs(clue_one_center.value - clue_two_center.value)
    _reading = (
        "The clues overlap strongly, so their joint peak remains high."
        if _agreement <= _width
        else "The clues disagree relative to their uncertainty, so even their best compromise has weak absolute support."
    )
    mo.vstack(
        [
            clue_controls,
            _fig,
            mo.callout(
                mo.md(
                    f"""
                    At each horizontal location, the blue height is multiplied
                    by the orange height. The winner is near **{_best_x:.0f}**,
                    but the height of the winning product is only
                    **{_peak_support:.3f}**. {_reading}

                    This distinction matters: the *location* of the peak says
                    which explanation is best; its *shape and scale* describe
                    uncertainty and compatibility. A point estimate throws
                    most of that information away.
                    """
                ),
                kind="success",
            ),
        ],
        gap=0.7,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Logs turn repeated multiplication into accumulation

    Probabilities below one shrink rapidly when multiplied. That is awkward
    numerically and awkward for a neural circuit. Logs change the bookkeeping:
    every new clue contributes an amount that can simply be added. The best
    explanation does not move, because the logarithm preserves ordering.

    Move the number of agreeing observations. The middle panel rapidly becomes
    visually tiny; the right panel remains a clean accumulated evidence
    landscape.
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
        label="Number of independent, similarly shaped clues",
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
        title="One clue",
        xlabel="candidate stimulus",
        ylabel="likelihood",
        ylim=(0, 1.02),
    )

    _axes[1].plot(_x, _product, color="#d97706", linewidth=2.6)
    _axes[1].fill_between(_x, 0, _product, color="#d97706", alpha=0.18)
    _axes[1].set(
        title=f"Multiply {_number} copies",
        xlabel="candidate stimulus",
        ylabel="product",
        ylim=(0, max(0.06, _product_peak * 1.15)),
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
        title="Add in log space",
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
                  <div class="metric"><strong>{_product_peak:.3g}</strong><span>product at the winner</span></div>
                  <div class="metric"><strong>{_log_peak:.2f}</strong><span>sum of logs at the winner</span></div>
                  <div class="metric"><strong>{_winner:.0f}</strong><span>same winner in every panel</span></div>
                </div>
                """
            ),
            mo.callout(
                mo.md(r"""
                **Nothing inferential changed. Only the coordinate system did.**
                A product of likelihood factors and a sum of their logarithms
                rank every candidate in exactly the same order. This is the
                bridge to accumulation: evidence from cells, cues, or moments
                can arrive separately and be added to one running score.
                """),
                kind="info",
            ),
            mo.accordion(
                {
                    "The entire log rule in one line": mo.md(
                        r"""If independent evidence supplies factors
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
    ## 4. A population readout is multiplication followed by addition

    We can now replace abstract clues with neurons. Twelve input neurons prefer
    twelve different motion directions. The first panel shows the spike pattern
    observed on one trial.

    An output unit asks one concrete question: *how well does this pattern
    support my candidate direction?* Its fixed connection weights are largest
    for input neurons whose preferences agree with that candidate, near zero
    for orthogonal inputs, and negative for opposite inputs. Multiply every
    spike count by its connection weight, then add the contributions.

    Move the **candidate direction** slowly. You are rotating one output unit's
    weight profile across the unchanged input pattern. The final panel shows
    what happens when a whole bank of output units asks all candidate questions
    at once.
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
        label="Center of the observed population response",
    )
    population_strength = mo.ui.slider(
        start=2,
        stop=12,
        step=1,
        value=8,
        debounce=True,
        show_value=True,
        full_width=True,
        label="Response strength (roughly, spikes at the peak)",
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
    _counts = np.rint(0.5 + population_strength.value * _profile).astype(int)
    _candidate = candidate_direction.value
    _weights = np.cos(np.deg2rad(circular_delta(_candidate, _preferences)))
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
    _scores = 0.42 * (_weight_bank @ _counts)
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
        ylabel="spikes",
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
        ylim=(-1.12, 1.12),
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
        title=f"4 · All rotated readouts form a likelihood profile",
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
        f"5 · Spike-weighted compass\nmean direction = {_vector_mean:+.1f}°",
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
    _candidate_relation = (
        "aligned with" if abs(circular_delta(_candidate, _peak_direction)) <= 15
        else "away from"
    )

    mo.vstack(
        [
            population_controls,
            _fig,
            mo.callout(
                mo.md(
                    f"""
                    The selected output is **{_candidate_relation}** the
                    population and therefore has relative likelihood
                    **{_selected_relative:.3f}**. The best-supported direction
                    is **{_peak_direction:+.1f}°**.

                    In this circular, symmetric example, the same peak can be
                    found as a very simple **spike-weighted circular average**:
                    give each spike a unit arrow pointing in its neuron's
                    preferred direction, add the arrows, and read the direction
                    of the result. That gives **{_vector_mean:+.1f}°**. The full
                    output curve retains more than that average: its width and
                    height retain information about uncertainty and evidence
                    strength.
                    """
                ),
                kind="success",
            ),
            mo.Html(
                r"""
                <div class="formula-card"><strong>The neural operation:</strong>
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
    ## 5. Only now do we need Poisson noise

    A tuning curve is a forward prediction of a neuron's **mean** spike count.
    A mean alone is not yet a likelihood. We also need a rule saying how
    variable single trials are around that mean.

    Jazayeri and Movshon use Poisson-like spike counts. For a Poisson neuron,
    specifying the tuning-curve mean specifies the entire distribution of
    possible counts. The figure follows one neuron through the conversion:

    1. candidate direction → predicted mean count;
    2. predicted mean → distribution of possible observed counts;
    3. hold the observed count fixed and read its probability backward over
       candidate directions.

    Change the spike count that was actually observed. A quiet response can be
    evidence *against* directions that predict a vigorous response.
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
    _best = float(_directions[np.argmax(_likelihood)])
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
        ylim=(0, 12.5),
    )

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
    _axes[2].axvline(_best, color="#182338", linestyle="--", linewidth=1.3)
    _axes[2].set(
        title=f"3 · Read P(n={_n_observed} | direction) backward",
        xlabel="candidate direction",
        ylabel="relative likelihood",
        xlim=(-180, 180),
        ylim=(0, 1.08),
    )
    for _axis in (_axes[0], _axes[2]):
        _axis.set_xticks(np.arange(-180, 181, 90))
    clean_axes(_axes)
    _fig.tight_layout(w_pad=1.5)

    _quiet_note = (
        "Because the neuron was quiet, directions far from its preference can be more plausible than its preferred direction."
        if _n_observed <= 2
        else "Because the neuron fired vigorously, directions near its preference receive the strongest support."
    )
    mo.vstack(
        [
            poisson_controls,
            _fig,
            mo.callout(
                mo.md(
                    f"""
                    {_quiet_note} The best-supported direction for this one
                    neuron is currently near **{_best:+.1f}°**. Notice that a
                    single neuron's likelihood may be broad or even have two
                    equally good directions; a population resolves such
                    ambiguities by multiplying evidence across neurons.

                    The important modeling link is now explicit: **the tuning
                    curve supplies the expected count, and the noise law turns
                    that expectation into the probability of the count that
                    actually occurred.**
                    """
                ),
                kind="info",
            ),
            mo.accordion(
                {
                    "Optional: the one Poisson formula": mo.md(
                        r"""For mean count $f(\theta)$ and observed count $n$,
                        the Poisson probability is
                        $P(n\mid\theta)=f(\theta)^n e^{-f(\theta)}/n!$.
                        Its log contains a spike-dependent term
                        $n\log f(\theta)$ and an expected-rate penalty
                        $-f(\theta)$. The next section shows why the paper can
                        often ignore that second term at the population level."""
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
    ## 6. The key shortcut: every direction spends the same population budget

    Here is the most important assumption behind the paper's simple weighted
    readout. With neurons evenly covering direction and having shifted copies
    of the same tuning shape, every direction produces roughly the same **total
    expected population activity**. Individual cells change, but the total
    budget stays flat.

    Why does that matter? A Poisson model rewards a candidate when observed
    spikes land in neurons it predicts, but it also penalizes a candidate for
    all the spikes it predicted and did not receive. If every candidate predicts
    the same total number of spikes, that penalty is the same horizontal offset
    everywhere. It can be dropped without changing the likelihood's shape or
    winner. What remains is just a weighted sum of observed spikes.

    Switch from **uniform coverage** to a missing wedge or lopsided gain. The
    population budget is no longer flat, and the shortcut can diverge from the
    full Poisson likelihood.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    coverage_mode = mo.ui.radio(
        options=["uniform coverage", "missing wedge", "lopsided gain"],
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
    if coverage_mode.value == "missing wedge":
        _gains[
            np.abs(circular_delta(_preferences, 90.0)) <= 45.0
        ] = 0.08
    elif coverage_mode.value == "lopsided gain":
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
        title=f"Population budget; range = {_budget_range:.1f}%",
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
        label=f"weighted-sum shortcut: {_shortcut_peak:+.0f}°",
    )
    _axes[2].axvline(
        assumption_direction.value,
        color="#d97706",
        linewidth=1.4,
        linestyle=":",
        label=f"generating direction: {assumption_direction.value:+.0f}°",
    )
    _axes[2].set(
        title="Full likelihood versus the shortcut",
        xlabel="candidate direction",
        ylabel="relative likelihood",
        xlim=(-180, 180),
        ylim=(0, 1.08),
    )
    _axes[2].legend(frameon=False, fontsize=7.6, loc="upper left")
    for _axis in _axes:
        _axis.set_xticks(np.arange(-180, 181, 90))
    clean_axes(_axes)
    _fig.tight_layout(w_pad=1.45)

    _peak_gap = abs(float(circular_delta(_shortcut_peak, _full_peak)))
    _conclusion = (
        "The population budget is effectively flat, so the omitted penalty is constant and the curves coincide."
        if coverage_mode.value == "uniform coverage"
        else f"The budget varies across directions. On this trial the shortcut and full model peaks differ by {_peak_gap:.1f}°."
    )

    mo.vstack(
        [
            assumption_controls,
            _fig,
            mo.callout(
                mo.md(
                    f"""
                    **{coverage_mode.value.capitalize()}:** {_conclusion}

                    This exposes two distinct links that are easy to blur:

                    - **Tuning + a noise law** says how probable each neuron's
                      observed count is under each candidate stimulus.
                    - **Homogeneous population coverage** makes the total-rate
                      penalty independent of direction, leaving a linear
                      weighted sum of counts.

                    The paper also assumes that stimulus strength scales tuning
                    curves without changing their shape. That lets the same
                    feedforward weights continue to represent direction when
                    motion coherence changes.
                    """
                ),
                kind="warn" if coverage_mode.value != "uniform coverage" else "success",
            ),
            mo.accordion(
                {
                    "Optional: reveal the bookkeeping": mo.md(
                        r"""For independent Poisson neurons, the direction-
                        dependent part of the population log likelihood is

                        $$\sum_i n_i\log f_i(\theta)-\sum_i f_i(\theta).$$

                        The first sum is observed spikes times fixed weights.
                        The second is the total expected population activity.
                        If that second sum is flat across $\theta$, removing it
                        changes neither the curve's shape nor its maximum. For
                        von Mises MT tuning, $\log f_i(\theta)$ reduces—up to
                        irrelevant constants—to a cosine weight."""
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
    ## 7. Correlation changes the reliability of a pool

    Independence justified multiplication. Real cortical neurons are
    correlated, so the joint response cloud is tilted rather than circular.
    What matters is not whether correlation exists in the abstract, but whether
    shared fluctuations point **along** or **across** the readout direction.

    Use the two toy codes below. In the opponent code the decision subtracts
    the neurons; shared fluctuations cancel. In the shared-strength code the
    decision adds them; shared fluctuations accumulate. The weights stay fixed
    while correlation changes the spread of the pooled decision variable.
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
        label="Pairwise response correlation",
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
        _readout_name = "difference n₁ − n₂"
    else:
        _mean_a = np.array([10.0, 10.0])
        _mean_b = np.array([6.0, 6.0])
        _weights = np.array([1.0, 1.0])
        _readout_name = "sum n₁ + n₂"

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
        color="#2563a8", alpha=0.28, label="world A",
    )
    _axes[0].scatter(
        _trials_b[:450, 0], _trials_b[:450, 1], s=12,
        color="#d97706", alpha=0.28, label="world B",
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
        alpha=0.50, label="world A",
    )
    _axes[1].hist(
        _scores_b, bins=_bins, density=True, color="#d97706",
        alpha=0.50, label="world B",
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
        title=f"Same weights, changing reliability: {_accuracy:.1%}",
        xlabel="pairwise correlation",
        ylabel="ideal accuracy for this fixed pool",
        xlim=(-0.85, 0.85),
        ylim=(0.48, 1.01),
    )
    clean_axes(_axes)
    _fig.tight_layout(w_pad=1.5)

    _effect = (
        "Positive shared fluctuations cancel in the subtraction, narrowing the decision variable."
        if _opponent and _rho > 0
        else (
            "Positive shared fluctuations add together, widening the decision variable."
            if (not _opponent and _rho > 0)
            else (
                "Negative correlation widens a difference readout."
                if _opponent else "Negative correlation partly cancels in a sum readout."
            )
        )
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

                    **What the paper does:** its feedforward decoder does not
                    adjust weights to exploit the correlation matrix. The
                    authors argue that stimulus-dependent correlations are not
                    straightforwardly available to a fixed biological decoder.
                    But they do include an empirically motivated correlation
                    structure when computing the variance—and therefore the
                    behavioral performance—of their model. Ignoring correlation
                    in the readout is not the same as pretending correlation
                    has no effect.
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
    ## 8. One likelihood landscape, several behaviors

    The payoff of representing the entire likelihood is reuse. The same sensory
    population need not be decoded by a new bespoke rule for every task. Move
    the generating direction and the two discrimination alternatives. Each
    panel receives the same log-likelihood curve and asks a different question.
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
        label="Center of the observed sensory response",
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
        start=-8.0,
        stop=10.0,
        step=1.0,
        value=5.0,
        debounce=True,
        show_value=True,
        full_width=True,
        label="Detection criterion on the fixed log-likelihood scale",
    )
    task_controls = mo.vstack(
        [task_direction, task_separation, detection_criterion], gap=0.5
    )
    return detection_criterion, task_controls, task_direction, task_separation


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
    task_separation,
):
    _preferences = np.arange(-180, 180, 20, dtype=float)
    _counts = np.rint(
        0.4
        + 8.0
        * direction_tuning(
            np.array([task_direction.value]), _preferences, kappa=1.9
        )[0]
    ).astype(int)
    _grid = np.linspace(-180, 180, 721)
    _bank = np.cos(
        np.deg2rad(
            (_grid[:, np.newaxis] - _preferences[np.newaxis, :] + 180)
            % 360
            - 180
        )
    )
    _log_likelihood = 0.36 * (_bank @ _counts)
    _peak = float(_grid[np.argmax(_log_likelihood)])
    _known_direction = 0.0
    _known_value = float(np.interp(_known_direction, _grid, _log_likelihood))
    _detect = _known_value >= detection_criterion.value
    _left_alt = -task_separation.value / 2.0
    _right_alt = task_separation.value / 2.0
    _left_value = float(np.interp(_left_alt, _grid, _log_likelihood))
    _right_value = float(np.interp(_right_alt, _grid, _log_likelihood))
    _chosen_alt = _left_alt if _left_value > _right_value else _right_alt
    _lower_limit = min(-11.0, float(np.min(_log_likelihood)) - 1.0)
    _upper_limit = max(11.0, float(np.max(_log_likelihood)) + 1.0)

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

    _axes[0].axhline(
        detection_criterion.value, color="#d97706", linestyle="--",
        linewidth=1.5, label="criterion",
    )
    _axes[0].scatter([0], [_known_value], color="#182338", s=65, zorder=4)
    _axes[0].set(
        title=f"Detection: {'present' if _detect else 'absent'}",
        ylabel="log-likelihood score (fixed reference)",
    )
    _axes[0].legend(frameon=False, fontsize=8)

    _axes[1].axvline(_peak, color="#182338", linestyle="--", linewidth=1.4)
    _axes[1].scatter(
        [_peak], [np.max(_log_likelihood)], color="#182338", s=65, zorder=4
    )
    _axes[1].set(title=f"Identification: choose the peak {_peak:+.0f}°")

    _axes[2].scatter(
        [_left_alt, _right_alt], [_left_value, _right_value],
        color=["#2563a8", "#d97706"], s=[65, 65], zorder=4,
    )
    _axes[2].vlines(
        [_left_alt, _right_alt], 0, [_left_value, _right_value],
        colors=["#2563a8", "#d97706"], linestyles=":", linewidth=1.4,
    )
    _axes[2].set(title=f"Discrimination: choose {_chosen_alt:+.0f}°")
    clean_axes(_axes)
    _fig.tight_layout(w_pad=1.5)

    mo.vstack(
        [
            task_controls,
            _fig,
            mo.callout(
                mo.md(
                    f"""
                    - **Detection** reads the likelihood at the expected
                      direction (here 0°) on a fixed log-likelihood scale and
                      compares **{_known_value:.2f}** with a criterion.
                    - **Identification** finds the largest value over all
                      candidates, here **{_peak:+.0f}°**.
                    - **Two-choice discrimination** reads only the two known
                      alternatives, **{_left_alt:+.0f}°** and
                      **{_right_alt:+.0f}°**, and chooses the larger value.

                    Different decision rules sit downstream of one reusable
                    sensory representation. With multiple cues, independent
                    log-likelihood landscapes can also be added before any of
                    these decisions is made.
                    """
                ),
                kind="success",
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
            ## The whole story, without the derivation

            1. A tuning curve predicts a neuron's mean response under each
               possible stimulus.
            2. A noise model turns each predicted mean into a probability for
               the response that actually occurred. Read across candidate
               stimuli, those probabilities form that neuron's likelihood.
            3. Independent neurons multiply their likelihoods. In log space,
               their contributions add.
            4. With Poisson-like variability, each observed spike contributes
               a fixed copy of the log tuning curve—a feedforward connection
               weight.
            5. With homogeneous coverage, total expected population activity
               is constant across stimuli. The remaining stimulus-dependent
               calculation is a weighted sum of spike counts.
            6. Correlations change the variability of that pooled sum. The
               paper ignores them when choosing decoder weights but includes
               them when predicting performance.

            The deep idea is not “the brain computes one best direction.” It is
            that a simple population transformation can preserve an entire
            landscape of sensory support, after which detection, identification,
            discrimination, cue combination, priors, and temporal accumulation
            become downstream operations on a common currency.
            """),
            mo.callout(
                mo.md(r"""
                **Source and scope.** This tutorial is based on M. Jazayeri and
                J. A. Movshon, “Optimal representation of sensory information
                by neural populations,” *Nature Neuroscience* 9, 690–696
                (2006), [paper](https://www.cns.nyu.edu/~tony/Publications/jazayeri-movshon-2006.pdf)
                and [supplement](https://www.cns.nyu.edu/~tony/Publications/jazayeri-movshon-2006-supp.pdf).
                Several displays deliberately use simplified or continuous
                toy responses to isolate concepts. They should not be read as
                new empirical fits or exact reproductions of the paper's
                simulations.
                """),
                kind="warn",
            ),
        ],
        gap=0.8,
    )
    return


if __name__ == "__main__":
    app.run()
