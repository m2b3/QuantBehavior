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

    normal = NormalDist()

    def normal_pdf(x, mean=0.0, sd=1.0):
        return np.exp(-0.5 * ((x - mean) / sd) ** 2) / (
            sd * np.sqrt(2.0 * np.pi)
        )

    return mo, normal, normal_pdf, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.Html(
        r"""
        <style>
          :root {
            --ink: #182338;
            --muted: #5f6b7c;
            --paper: #f7f4ed;
            --card: #ffffff;
            --noise: #2b6cb0;
            --signal: #d97706;
            --criterion: #7c3aed;
            --line: #dde3ea;
          }
          body { background: var(--paper); color: var(--ink); }
          .tutorial-hero {
            margin: 0 0 1.2rem;
            padding: 2.1rem 2.3rem;
            border-radius: 18px;
            color: white;
            background: linear-gradient(125deg, #17233a, #293d61);
            box-shadow: 0 12px 34px rgba(24, 35, 56, .12);
          }
          .tutorial-hero h1 {
            margin: 0 0 .55rem;
            color: #ffffff !important;
            font-size: 2.25rem;
          }
          .tutorial-hero p { margin: 0; color: #dbe4f2; line-height: 1.55; }
          .lesson-card {
            padding: 1.15rem 1.35rem;
            margin: .7rem 0 1.15rem;
            border: 1px solid var(--line);
            border-radius: 14px;
            background: var(--card);
          }
          .lesson-card h3 { margin-top: .1rem; }
          .formula-card {
            padding: 1rem 1.25rem;
            border-left: 4px solid var(--criterion);
            border-radius: 5px 12px 12px 5px;
            background: #f4f0ff;
          }
          .answer-card {
            padding: 1rem 1.25rem;
            border-left: 4px solid #16846b;
            border-radius: 5px 12px 12px 5px;
            background: #edf9f5;
          }
          .warning-card {
            padding: .85rem 1.1rem;
            border-left: 4px solid #d97706;
            border-radius: 5px 12px 12px 5px;
            background: #fff7e8;
          }
          .controls-row {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: .8rem;
            align-items: end;
          }
          .metric-row {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: .65rem;
            margin: .8rem 0;
          }
          .metric {
            padding: .72rem .8rem;
            border: 1px solid var(--line);
            border-radius: 10px;
            background: white;
          }
          .metric strong { display: block; font-size: 1.22rem; }
          .metric span { color: var(--muted); font-size: .78rem; }
          .small-note { color: var(--muted); font-size: .88rem; line-height: 1.55; }
          table { width: 100%; }
          th, td { padding: .45rem .6rem !important; }
          /* Editable WASM normally caps interactive cell outputs at 610px,
             which creates nested scrollbars. Let the document itself scroll. */
          .marimo-cell.interactive .console-output-area,
          .marimo-cell.interactive .output-area {
            max-height: none !important;
            overflow: visible !important;
          }
          @media (max-width: 700px) {
            .controls-row, .metric-row { grid-template-columns: 1fr; }
            .tutorial-hero { padding: 1.5rem; }
          }
        </style>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div class="tutorial-hero">
      <h1>Why is z(H) − z(FA) equal to d′?</h1>
      <p>An interactive, equal-variance Gaussian signal-detection tutorial. We begin with right-tail hit and false-alarm rates, watch the inverse-normal transform remove the criterion, and finish with criterion, reversed evidence, and finite corrections for rates of 0 or 1.</p>
    </div>

    ## 1. What do $\Phi$ and $z$ do?

    The standard normal **density** is the bell curve. Its **CDF**,
    $\Phi(q)=P(Z\le q)$, always counts area in the usual direction: from
    $-\infty$ up to $q$. The inverse CDF, $z(p)=\Phi^{-1}(p)$, reverses that
    lookup: give it a proportion $p$, and it returns the location whose area
    to the **left** is $p$.

    Hit and false-alarm rates are different: they are areas counted from a
    criterion out to $+\infty$. The first key trick is to reflect that
    right-tail area across zero. An area $R$ to the right of $+a$ is the same
    size as the area $R$ to the left of $-a$. Therefore the ordinary inverse
    CDF assigns the number $R$ the location $z(R)=-a$.

    Move the right-tail proportion below. The two equal shaded tails and the
    reflected locations $+a$ and $-a$ are marked in every view.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    cdf_probability = mo.ui.slider(
        start=0.01,
        stop=0.99,
        step=0.01,
        value=0.25,
        show_value=True,
        label="Right-tail proportion R",
    )
    cdf_probability
    return (cdf_probability,)


@app.cell(hide_code=True)
def _(cdf_probability, mo, normal, normal_pdf, np, plt):
    _r = cdf_probability.value
    _zr = normal.inv_cdf(_r)
    _a = -_zr
    _x = np.linspace(-3.5, 3.5, 700)
    _pdf = normal_pdf(_x)
    _cdf = np.array([normal.cdf(float(_value)) for _value in _x])
    _probs = np.linspace(0.001, 0.999, 700)
    _probits = np.array([normal.inv_cdf(float(_value)) for _value in _probs])

    _fig, _axes = plt.subplots(1, 3, figsize=(12, 3.25))
    _fig.patch.set_facecolor("#ffffff")

    _axes[0].plot(_x, _pdf, color="#243b64", linewidth=2.2)
    _axes[0].fill_between(
        _x, 0, _pdf, where=_x <= _zr, color="#7c3aed", alpha=0.28,
        label=rf"usual left-tail area $R={_r:.2f}$",
    )
    _axes[0].fill_between(
        _x, 0, _pdf, where=_x >= _a, color="#d97706", alpha=0.28,
        label=rf"right-tail rate $R={_r:.2f}$",
    )
    _axes[0].axvline(_zr, color="#7c3aed", linestyle="--", linewidth=1.7)
    _axes[0].axvline(_a, color="#d97706", linestyle="--", linewidth=1.7)
    _axes[0].text(
        _zr - 0.06,
        0.34,
        f"z(R) = {_zr:.2f}",
        ha="right",
        va="bottom",
        color="#5b21b6",
        fontsize=8.6,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76},
    )
    _axes[0].text(
        _a + 0.06,
        0.34,
        f"cutoff a = {_a:.2f}",
        ha="left",
        va="bottom",
        color="#b45309",
        fontsize=8.6,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76},
    )
    _axes[0].set(
        title="Reflect a right tail into a left tail",
        xlabel="standard score q",
    )
    _axes[0].set_yticks([])
    _axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.20),
                    frameon=False, fontsize=7.6)

    _axes[1].plot(_x, _cdf, color="#243b64", linewidth=2.2)
    _axes[1].plot([_x[0], _zr], [_r, _r], color="#7c3aed", linestyle="--")
    _axes[1].plot([_zr, _zr], [0, _r], color="#7c3aed", linestyle="--")
    _axes[1].scatter([_zr], [_r], color="#7c3aed", zorder=3)
    _axes[1].plot([_x[0], _a], [1 - _r, 1 - _r],
                  color="#d97706", linestyle=":")
    _axes[1].scatter([_a], [1 - _r], color="#d97706", zorder=3)
    _axes[1].set(
        title=rf"CDF: $\Phi({_zr:.2f})={_r:.2f}$",
        xlabel="standard score q",
        ylabel="left-tail probability p",
        ylim=(-0.03, 1.03),
    )

    _axes[2].plot(_probs, _probits, color="#243b64", linewidth=2.2)
    _axes[2].plot([0, _r], [_zr, _zr], color="#7c3aed", linestyle="--")
    _axes[2].plot([_r, _r], [-3.5, _zr], color="#7c3aed", linestyle="--")
    _axes[2].scatter([_r], [_zr], color="#7c3aed", zorder=3)
    _axes[2].set(
        title=r"Inverse CDF: $z(R)=\Phi^{-1}(R)$",
        xlabel="probability p",
        ylabel="standard score q",
        xlim=(-0.03, 1.03),
        ylim=(-3.5, 3.5),
    )

    for _axis in _axes:
        _axis.spines[["top", "right"]].set_visible(False)
        _axis.grid(alpha=0.13)
    _fig.tight_layout(rect=(0, 0.08, 1, 1))

    mo.vstack(
        [
            _fig,
            mo.callout(
                mo.md(
                    rf"""
                Start at the orange cutoff $a={_a:.3f}$ and count **backward
                from $+\infty$**: the orange right tail has proportion
                $R={_r:.2f}$. The $z$ function does not directly count from
                the right, so reflect the picture. The same proportion lies
                between $-\infty$ and the symmetric purple point
                $-a={_zr:.3f}$. Thus

                $$R=1-\Phi(a)=\Phi(-a) \quad\Longrightarrow\quad z(R)=-a.$$

                This does **not** change the probability. It only expresses a
                right-tail proportion in the left-counting coordinate system
                used by the ordinary CDF and inverse CDF. That reflection is
                the whole reason a right-tail rate produces the negative of
                its standardized cutoff. It works on **either side of the
                mean**, not only in the picture currently shown:

                - If $a>0$, the criterion is right of the mean. The right tail
                  is below $0.5$, and its reflected z-value $-a$ is negative.
                - If $a=0$, the criterion is at the mean. The right tail is
                  $0.5$, and $z(0.5)=0$.
                - If $a<0$, the criterion is left of the mean. The right tail
                  is above $0.5$, and its reflected z-value $-a$ is positive.

                Drag $R$ through $0.5$ to watch the orange cutoff and purple
                z-location swap sides continuously.
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
    ## 2. Begin at $d'=0$: hit rate and false-alarm rate

    Call the blue distribution **A (noise only)** and the orange distribution
    **B (signal)**. On every trial, the observer says **“yes, signal” when the
    evidence $X$ is greater than the criterion $\lambda$**.

    - The **hit rate** $H_R$ is the proportion of B/signal trials that land to
      the right of $\lambda$. In simple words: *when the signal was present,
      how often did the observer say yes?*
    - The **false-alarm rate** $FA_R$ is the proportion of A/noise trials that
      land to the right of $\lambda$. In simple words: *when there was no
      signal, how often did the observer nevertheless say yes?*

    Both are therefore right-tail areas:

    $$
    H_R=P(X_B>\lambda), \qquad FA_R=P(X_A>\lambda).
    $$

    Press **Set both shifts to 0** to begin. A and B then lie exactly on top of
    one another, so their shaded tails—and therefore $H_R$ and $FA_R$—have the
    same value. The exact current values are printed immediately below the
    controls and inside the figure.

    The equal variances are essential: they put both standardized cutoffs on
    the same scale. We use $\sigma=1$ here, so a one-unit horizontal movement
    is a one-standard-deviation movement.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    reset_dprime = mo.ui.button(
        value=0,
        on_click=lambda count: count + 1,
        label="Set both shifts to 0",
        tooltip="Put A and B back on top of one another, making d′ zero",
        full_width=True,
    )
    response_cutoff = mo.ui.slider(
        start=-2.5,
        stop=2.5,
        step=0.25,
        value=0.5,
        debounce=True,
        show_value=True,
        full_width=True,
        label=(
            r"Evidence-axis criterion $\lambda$ "
            r"(the right-tail yes/no cutoff)"
        ),
    )
    return reset_dprime, response_cutoff


@app.cell(hide_code=True)
def _(mo, reset_dprime):
    # Referencing the button makes this cell rerun after a click, creating two
    # new sliders whose frontend and Python values are both exactly zero.
    _reset_count = reset_dprime.value
    signal_shift = mo.ui.slider(
        start=-2.5,
        stop=2.5,
        step=0.25,
        value=0.0,
        debounce=True,
        show_value=True,
        full_width=True,
        label=r"Shift B (signal): $\mu_B/\sigma$ — try moving right to $+1$",
    )
    noise_shift = mo.ui.slider(
        start=-2.5,
        stop=2.5,
        step=0.25,
        value=0.0,
        debounce=True,
        show_value=True,
        full_width=True,
        label=r"Shift A (noise): $\mu_A/\sigma$ — try moving left to $-1$",
    )
    return noise_shift, signal_shift


@app.cell(hide_code=True)
def _(mo, noise_shift, reset_dprime, response_cutoff, signal_shift):
    main_controls = mo.vstack(
        [
            mo.hstack(
                [
                    mo.md(
                        "**Build $d'$ from two visible movements, one curve at a time.**"
                    ),
                    reset_dprime,
                ],
                widths=[3, 1],
                align="end",
                gap=1.0,
                wrap=True,
            ),
            mo.md(
                r"""
                First leave both shifts at zero. Then try **B = +1** while A
                stays at zero. Reset, and try **A = -1** while B stays at zero.
                Finally move both. Each slider advances in quarter-standard-
                deviation steps, so the suggested values are easy to select.
                The criterion is independent of those movements. Plots update
                when you release a handle, avoiding continuous WASM redraws
                while dragging.
                """
            ),
            signal_shift,
            noise_shift,
            response_cutoff,
        ],
        gap=0.65,
    )
    return (main_controls,)


@app.cell(hide_code=True)
def _(noise_shift, signal_shift):
    mu_signal = signal_shift.value
    mu_noise = noise_shift.value
    polarity = 1 if mu_signal >= mu_noise else -1
    return mu_noise, mu_signal, polarity


@app.cell(hide_code=True)
def _(mu_noise, mu_signal, normal, polarity, response_cutoff):
    cutoff = response_cutoff.value

    # Because sigma=1 and the response is X > cutoff:
    hit_right = normal.cdf(mu_signal - cutoff)
    false_alarm_right = normal.cdf(mu_noise - cutoff)
    z_hit_right = normal.inv_cdf(hit_right)
    z_false_alarm_right = normal.inv_cdf(false_alarm_right)
    dprime_signed = z_hit_right - z_false_alarm_right
    criterion_right = -0.5 * (z_hit_right + z_false_alarm_right)

    # Reorient evidence so that larger values always favor signal.
    hit_oriented = hit_right if polarity == 1 else 1.0 - hit_right
    false_alarm_oriented = (
        false_alarm_right if polarity == 1 else 1.0 - false_alarm_right
    )
    z_hit_oriented = normal.inv_cdf(hit_oriented)
    z_false_alarm_oriented = normal.inv_cdf(false_alarm_oriented)
    dprime_oriented = z_hit_oriented - z_false_alarm_oriented
    criterion_oriented = -0.5 * (
        z_hit_oriented + z_false_alarm_oriented
    )
    return (
        criterion_oriented,
        criterion_right,
        cutoff,
        dprime_oriented,
        dprime_signed,
        false_alarm_oriented,
        false_alarm_right,
        hit_oriented,
        hit_right,
        mu_noise,
        mu_signal,
        polarity,
        z_false_alarm_right,
        z_hit_right,
    )


@app.cell(hide_code=True)
def _(
    criterion_right,
    cutoff,
    dprime_signed,
    false_alarm_right,
    hit_right,
    main_controls,
    mo,
    mu_noise,
    mu_signal,
    normal,
    normal_pdf,
    np,
    plt,
    polarity,
    z_false_alarm_right,
    z_hit_right,
):
    _x = np.linspace(-4.5, 4.5, 900)
    _noise_pdf = normal_pdf(_x, mu_noise)
    _signal_pdf = normal_pdf(_x, mu_signal)
    _right = _x >= cutoff
    _evidence_midpoint = 0.5 * (mu_signal + mu_noise)
    _reference_z = -cutoff
    _reference_rate = normal.cdf(-cutoff)
    _accuracy = 0.5 * (hit_right + 1.0 - false_alarm_right)
    _zero_separation = abs(dprime_signed) < 1e-12

    _fig, (_ax, _mean_ax, _rate_ax) = plt.subplots(
        3,
        1,
        figsize=(10.5, 10.4),
        gridspec_kw={"height_ratios": [4.1, 1.45, 1.6]},
    )
    _fig.patch.set_facecolor("#ffffff")

    _ax.plot(
        _x, _noise_pdf, color="#2b6cb0", linewidth=2.4,
        label="A: noise only",
    )
    _ax.plot(
        _x,
        _signal_pdf,
        color="#d97706",
        linewidth=2.4,
        linestyle="--" if _zero_separation else "-",
        label="B: signal",
    )
    _ax.fill_between(
        _x,
        0,
        _noise_pdf,
        where=_right,
        color="#2b6cb0",
        alpha=0.26,
        label=rf"$FA_R={false_alarm_right:.3f}$",
    )
    _ax.fill_between(
        _x,
        0,
        _signal_pdf,
        where=_right,
        color="#d97706",
        alpha=0.26,
        label=rf"$H_R={hit_right:.3f}$",
    )
    _ax.axvline(cutoff, color="#7c3aed", linewidth=2, linestyle="--")
    _ax.text(
        cutoff,
        0.43,
        rf"criterion $\lambda={cutoff:.2f}$",
        ha="center",
        va="bottom",
        color="#5b21b6",
        fontsize=10,
    )
    _ax.axvline(mu_noise, color="#2b6cb0", linewidth=1, alpha=0.5)
    _ax.axvline(mu_signal, color="#d97706", linewidth=1, alpha=0.5)
    _ax.axvline(
        _evidence_midpoint,
        color="#7b8493",
        linewidth=1,
        linestyle=":",
        alpha=0.8,
    )
    _ax.annotate(
        "",
        xy=(mu_signal, 0.065),
        xytext=(mu_noise, 0.065),
        arrowprops={"arrowstyle": "<->", "color": "#182338", "lw": 1.4},
    )
    _ax.text(
        _evidence_midpoint,
        0.077,
        rf"mean gap $=(\mu_B-\mu_A)/\sigma=d'_R={dprime_signed:+.2f}$",
        ha="center",
        fontsize=10,
    )
    _ax.annotate(
        "",
        xy=(cutoff, 0.132),
        xytext=(_evidence_midpoint, 0.132),
        arrowprops={"arrowstyle": "<->", "color": "#7c3aed", "lw": 1.5},
    )
    _ax.text(
        0.5 * (cutoff + _evidence_midpoint),
        0.145,
        rf"criterion $c_R={criterion_right:+.2f}$",
        ha="center",
        color="#5b21b6",
        fontsize=9.5,
    )
    _ax.text(
        0.985,
        0.96,
        (
            f"Hit rate = {hit_right:.3f}\n"
            f"False-alarm rate = {false_alarm_right:.3f}\n"
            f"Accuracy = {_accuracy:.3f}\n"
            f"Criterion cR = {criterion_right:+.3f}"
        ),
        transform=_ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        linespacing=1.45,
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "white",
            "edgecolor": "#cbd3df",
            "alpha": 0.94,
        },
    )
    _ax.set(
        title="Evidence distributions and the two right-tail rates",
        xlabel="",
        ylabel="density",
        xlim=(-4.5, 4.5),
        ylim=(0, 0.47),
    )
    _ax.set_yticks([])
    _ax.legend(loc="upper left", frameon=False, ncols=2, fontsize=8.5)
    _ax.spines[["top", "right", "left"]].set_visible(False)

    # Upper pair: absolute evidence coordinates. The criterion really moves
    # here. Each colored arrow is the signed mean-minus-criterion distance.
    _b_mean_y, _a_mean_y = 0.66, 0.20
    _axis_limit = 5.25
    for _row_y in (_b_mean_y, _a_mean_y):
        _mean_ax.axhline(_row_y, color="#d6dbe3", linewidth=1.0, zorder=0)
    _mean_ax.axvline(
        cutoff, color="#7c3aed", linewidth=2.0, linestyle="--", zorder=1
    )
    _mean_ax.text(
        cutoff, 0.88, rf"$\lambda={cutoff:+.2f}$",
        ha="center", va="bottom", color="#5b21b6", fontsize=9.3,
    )
    _absolute_rows = [
        (_b_mean_y, mu_signal, "#d97706", rf"$\mu_B-\lambda={z_hit_right:+.2f}$"),
        (_a_mean_y, mu_noise, "#2b6cb0", rf"$\mu_A-\lambda={z_false_alarm_right:+.2f}$"),
    ]
    for _row_y, _mean, _color, _label in _absolute_rows:
        _mean_ax.scatter([cutoff], [_row_y], marker="|", s=150,
                         color="#7c3aed", zorder=3)
        if abs(_mean - cutoff) > 1e-12:
            _mean_ax.annotate(
                "", xy=(_mean, _row_y), xytext=(cutoff, _row_y),
                arrowprops={"arrowstyle": "-|>", "color": _color, "lw": 2.3},
            )
        _mean_ax.scatter([_mean], [_row_y], s=75, color=_color, zorder=4)
        _mean_ax.text(
            0.985, _row_y, _label,
            transform=_mean_ax.get_yaxis_transform(),
            ha="right", va="center", color=_color, fontsize=9.2,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88},
        )
    _mean_ax.set(
        title="Upper pair — absolute evidence axis: the purple criterion moves",
        xlabel="absolute evidence x",
        xlim=(-_axis_limit, _axis_limit),
        ylim=(-0.02, 1.02),
    )
    _mean_ax.set_yticks(
        [_b_mean_y, _a_mean_y],
        labels=[r"B mean $\mu_B$", r"A mean $\mu_A$"],
    )
    _mean_ax.tick_params(axis="y", length=0, labelsize=9)
    _mean_ax.spines[["top", "right", "left"]].set_visible(False)

    # Lower pair: exactly the same signed distances, but the coordinate axis
    # is re-centered so that the criterion is always zero. This is the
    # coordinate returned by z(right-tail rate) when sigma=1.
    _hit_y, _fa_y = 0.66, 0.20
    for _row_y in (_hit_y, _fa_y):
        _rate_ax.axhline(_row_y, color="#d6dbe3", linewidth=1.0, zorder=0)
    _rate_ax.axvline(
        0.0, color="#7c3aed", linewidth=2.0, linestyle="--", zorder=1
    )
    _rate_ax.text(
        0.0, 0.88, r"criterion-centered zero",
        ha="center", va="bottom", color="#5b21b6", fontsize=9.3,
    )
    _relative_rows = [
        (_hit_y, z_hit_right, "#d97706", rf"$z(H_R)={z_hit_right:+.2f}$"),
        (_fa_y, z_false_alarm_right, "#2b6cb0", rf"$z(FA_R)={z_false_alarm_right:+.2f}$"),
    ]
    for _row_y, _z_value, _color, _label in _relative_rows:
        _rate_ax.scatter([0.0], [_row_y], marker="|", s=150,
                         color="#7c3aed", zorder=3)
        if abs(_z_value) > 1e-12:
            _rate_ax.annotate(
                "", xy=(_z_value, _row_y), xytext=(0.0, _row_y),
                arrowprops={"arrowstyle": "-|>", "color": _color, "lw": 2.3},
            )
        _rate_ax.scatter([_z_value], [_row_y], s=75, color=_color, zorder=4)
        _rate_ax.text(
            0.985, _row_y, _label,
            transform=_rate_ax.get_yaxis_transform(),
            ha="right", va="center", color=_color, fontsize=9.2,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88},
        )
    _rate_ax.set(
        title="Lower pair — re-center at the criterion: the purple line is now zero",
        xlabel=r"signed distance from criterion $=(\mu-\lambda)/\sigma=z(\mathrm{rate})$",
        xlim=(-_axis_limit, _axis_limit),
        ylim=(-0.02, 1.02),
    )
    _rate_ax.set_yticks(
        [_hit_y, _fa_y],
        labels=[r"hit-rate $z(H_R)$", r"false-alarm $z(FA_R)$"],
    )
    _rate_ax.tick_params(axis="y", length=0, labelsize=9)
    _rate_ax.spines[["top", "right", "left"]].set_visible(False)

    _fig.tight_layout(h_pad=2.7)

    _ordering_note = (
        "The two means currently coincide, so the signed sensitivity is zero."
        if _zero_separation
        else (
            "B/signal is right of A/noise, so the signed value is positive."
            if polarity == 1
            else (
            "Signal is left of noise, but the response is still the right tail, "
            "so the signed value is negative."
            )
        )
    )
    mo.vstack(
        [
            main_controls,
            _fig,
            mo.callout(
                mo.md(
                    rf"""
                **Start by pressing “Set both shifts to 0.”** At that reference
                point, both curves have mean 0. With the current criterion
                $\lambda={cutoff:+.2f}$, their shared right-tail proportion is
                $H_R=FA_R={_reference_rate:.3f}$. In words, the observer says
                “yes” on {_reference_rate:.1%} of signal trials **and** on
                {_reference_rate:.1%} of noise-only trials, because the two
                evidence distributions are identical. The reflected CDF trick
                from section 1 gives both rates the same value
                $z(H_R)=z(FA_R)=-\lambda={_reference_z:+.3f}$, so their
                difference is zero.

                **Now try these two one-step experiments:**

                1. Move only **B/signal to $+1$**. More B trials cross the
                   fixed criterion, so the hit rate rises. Its z-value rises
                   by exactly the same $+1$; watch the orange arrow below
                   lengthen by the same amount as the orange arrow above.
                2. Reset, then move only **A/noise to $-1$**. Fewer A trials
                   cross the criterion, so the false-alarm rate falls. Its
                   z-value moves left by exactly $-1$; the two blue arrows
                   again have the same length and direction.
                3. To test the apparently awkward case, set **B = $+1$**,
                   **A = $0$**, and **$\lambda=+1.5$**. The criterion is now
                   to the right of even the signal mean. The hit rate is only
                   $0.309<0.5$, so $z(H_R)=-0.5$. The false-alarm rate is
                   $0.067$, so $z(FA_R)=-1.5$. Both z-values are negative,
                   but their difference is still
                   $-0.5-(-1.5)=1=d'$.

                In that third case, the standardized signal cutoff is
                $a_B=(\lambda-\mu_B)/\sigma=+0.5$. The hit rate is the area
                counted rightward from $+0.5$; reflecting the same area to the
                other side gives $z(H_R)=-0.5$. Thus the symmetric flip works
                separately around **each distribution's own mean**, even when
                the criterion lies to the right of that mean and the observed
                rate is below $0.5$.

                The current arrows and measured rates are:

                | curve | mean on upper axis | criterion | signed arrow from criterion | measured rate | matching endpoint below |
                |:--|--:|--:|--:|--:|--:|
                | B / signal | ${mu_signal:+.2f}$ | ${cutoff:+.2f}$ | ${mu_signal - cutoff:+.2f}$ | $H_R={hit_right:.3f}$ | $z(H_R)={z_hit_right:+.3f}$ |
                | A / noise | ${mu_noise:+.2f}$ | ${cutoff:+.2f}$ | ${mu_noise - cutoff:+.2f}$ | $FA_R={false_alarm_right:.3f}$ | $z(FA_R)={z_false_alarm_right:+.3f}$ |

                **Current horizontal gap:** upper pair
                $\mu_B-\mu_A={dprime_signed:+.3f}$; lower pair
                $z(H_R)-z(FA_R)={dprime_signed:+.3f}$.
                """
                ),
                kind="info",
            ),
            mo.callout(
                mo.md(
                    rf"""
                **Read the upper and lower pairs as the same two arrows in two
                coordinate systems.** In the upper pair, the scale is the
                absolute evidence axis. The purple line is the actual
                criterion and moves when you move $\lambda$. Each colored
                arrow runs from that criterion to its distribution mean.

                In the lower pair, we keep those arrow lengths and directions
                but **re-center the ruler at the criterion**. The purple line
                is therefore fixed at zero. The right-tail reflection from
                section 1 turns the orange signed distance into $z(H_R)$ and
                the blue signed distance into $z(FA_R)$. No diagonal mapping
                is needed: visually, it is the same arrow placed on a ruler
                whose zero is now the criterion.

                Try the sliders one at a time:

                - Move only B: only the orange mean and orange z endpoint move.
                - Move only A: only the blue mean and blue z endpoint move.
                - Move only $\lambda$: both means remain fixed above while the
                  purple criterion moves. Below, the criterion stays centered
                  at zero and **both** colored z endpoints move together by the
                  same amount.

                That last movement changes both rates, but it does not change
                the horizontal orange-to-blue gap. The upper gap and lower gap
                remain the same: $d'_R={dprime_signed:+.2f}$. Thus the picture
                preserves the role of $\lambda$ while showing why criterion
                placement disappears from the difference of the two z-values.

                $\lambda$ is the **decision criterion (cutoff) on the evidence
                axis**. Its standardized location relative to the midpoint is
                $c_R=[\lambda-(\mu_B+\mu_A)/2]/\sigma$. For the shift argument,
                the important fact is that the same $-\lambda$ offset appears
                in both z-values and cancels. {_ordering_note}

                The displayed accuracy assumes equally frequent signal and
                noise trials:
                $\mathrm{{accuracy}}=\tfrac12[H_R+(1-FA_R)]={_accuracy:.3f}$.
                """
                ),
                kind="success",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    roc_separation = mo.ui.slider(
        start=0.0,
        stop=3.5,
        step=0.1,
        value=1.6,
        debounce=True,
        show_value=True,
        full_width=True,
        label=r"ROC-row separation $|d'|$",
    )
    return (roc_separation,)


@app.cell(hide_code=True)
def _(mo, normal, normal_pdf, np, plt, roc_separation):
    _row_signed_dprime = roc_separation.value
    _row_mu_signal = _row_signed_dprime / 2.0
    _row_mu_noise = -_row_signed_dprime / 2.0
    _x = np.linspace(-4.5, 4.5, 700)
    _noise_pdf = normal_pdf(_x, _row_mu_noise)
    _signal_pdf = normal_pdf(_x, _row_mu_signal)
    _fixed_cutoffs = np.linspace(-2.4, 2.4, 9)
    _threshold_colors = plt.get_cmap("viridis")(
        np.linspace(0.08, 0.92, len(_fixed_cutoffs))
    )

    _fixed_false_alarms = np.array(
        [
            normal.cdf(_row_mu_noise - float(_value))
            for _value in _fixed_cutoffs
        ]
    )
    _fixed_hits = np.array(
        [
            normal.cdf(_row_mu_signal - float(_value))
            for _value in _fixed_cutoffs
        ]
    )
    _dense_cutoffs = np.linspace(-5.5, 5.5, 600)
    _roc_false_alarms = np.array(
        [
            normal.cdf(_row_mu_noise - float(_value))
            for _value in _dense_cutoffs
        ]
    )
    _roc_hits = np.array(
        [
            normal.cdf(_row_mu_signal - float(_value))
            for _value in _dense_cutoffs
        ]
    )

    # This SDT view deliberately has fixed criteria. It changes with d', but
    # it does not depend on the interactive criterion slider above.
    _fixed_fig, _fixed_ax = plt.subplots(figsize=(6.2, 4.8))
    _fixed_fig.patch.set_facecolor("#ffffff")
    _fixed_ax.plot(
        _x, _noise_pdf, color="#2b6cb0", linewidth=2.4, label="A: noise only"
    )
    _fixed_ax.plot(
        _x, _signal_pdf, color="#d97706", linewidth=2.4, label="B: signal"
    )
    for _fixed_cutoff, _threshold_color in zip(
        _fixed_cutoffs, _threshold_colors
    ):
        _fixed_ax.axvline(
            _fixed_cutoff,
            color=_threshold_color,
            linewidth=1.45,
            linestyle=":",
            alpha=0.9,
        )
        _fixed_ax.text(
            _fixed_cutoff,
            0.445,
            rf"${_fixed_cutoff:+.1f}$",
            ha="center",
            va="top",
            rotation=90,
            color=_threshold_color,
            fontsize=7.5,
        )
    _fixed_ax.set(
        title="Fixed criteria on the evidence distributions",
        xlabel=r"evidence $x$",
        ylabel="density",
        xlim=(-4.5, 4.5),
        ylim=(0, 0.47),
    )
    _fixed_ax.set_yticks([])
    _fixed_ax.legend(loc="upper left", frameon=False, ncols=2, fontsize=8.5)
    _fixed_ax.spines[["top", "right", "left"]].set_visible(False)
    _fixed_fig.tight_layout()

    _roc_fig, _roc_ax = plt.subplots(figsize=(6.2, 4.8))
    _roc_fig.patch.set_facecolor("#ffffff")
    _roc_ax.plot(
        [0, 1],
        [0, 1],
        color="#aab3c2",
        linewidth=1.2,
        linestyle="--",
        label="chance",
    )
    _roc_ax.plot(
        _roc_false_alarms,
        _roc_hits,
        color="#182338",
        linewidth=2.5,
        label=rf"ROC ($d'_R={_row_signed_dprime:+.1f}$)",
    )
    for _index, (
        _fixed_cutoff,
        _fixed_false_alarm,
        _fixed_hit,
        _threshold_color,
    ) in enumerate(
        zip(
            _fixed_cutoffs,
            _fixed_false_alarms,
            _fixed_hits,
            _threshold_colors,
        )
    ):
        _roc_ax.scatter(
            [_fixed_false_alarm],
            [_fixed_hit],
            marker="x",
            s=70,
            linewidths=2.0,
            color=_threshold_color,
            zorder=4,
        )
        _roc_ax.annotate(
            rf"$\lambda={_fixed_cutoff:+.1f}$",
            xy=(_fixed_false_alarm, _fixed_hit),
            xytext=(5, 7 if _index % 2 == 0 else -13),
            textcoords="offset points",
            fontsize=7.3,
            color=_threshold_color,
        )
    _roc_ax.set(
        title="The same criteria traced on the ROC",
        xlabel=r"false-alarm rate $FA_R$",
        ylabel=r"hit rate $H_R$",
        xlim=(-0.035, 1.035),
        ylim=(-0.035, 1.035),
    )
    _roc_ax.set_aspect("equal", adjustable="box")
    _roc_ax.grid(alpha=0.18)
    _roc_ax.spines[["top", "right"]].set_visible(False)
    _roc_ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    _roc_fig.tight_layout()

    _ordering_note = (
        "This independent ROC example keeps B/signal to the right of A/noise, "
        "so its ROC is above the diagonal whenever d' is positive."
    )
    mo.vstack(
        [
            mo.md(
                r"""
                ### One separation, many criteria

                The lower pair is independent of the purple criterion bar.
                Use its own $|d'|$ control below. Each colored vertical
                criterion on the SDT plot is the same-colored $\times$ on the
                right-tail ROC. This row uses the conventional orientation in
                which larger evidence favors B/signal.
                """
            ),
            roc_separation,
            mo.hstack(
                [_fixed_fig, _roc_fig],
                widths="equal",
                gap=1.0,
                wrap=True,
            ),
            mo.callout(
                mo.md(
                    rf"""
                    Moving a criterion travels along one ROC without changing
                    $d'$. Changing $|d'|$ changes the separation of the
                    distributions and therefore bends the entire ROC.
                    {_ordering_note}
                    """
                ),
                kind="info",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    criterion_oriented,
    criterion_right,
    dprime_oriented,
    dprime_signed,
    false_alarm_oriented,
    false_alarm_right,
    hit_oriented,
    hit_right,
    mo,
    mu_noise,
    mu_signal,
    polarity,
    z_false_alarm_right,
    z_hit_right,
):
    _orientation_explanation = (
        r"At zero separation there is no signal-favoring direction to orient."
        if abs(dprime_signed) < 1e-12
        else (
            r"No recoding is needed because larger evidence already favors signal."
            if polarity == 1
            else (
            r"To report a positive sensitivity, define oriented evidence "
            r"$X^*=-X$. The signal response is then the original **left** "
            r"tail. Equivalently, $H^*=1-H_R$ and $FA^*=1-FA_R$, so "
            r"$z(H^*)-z(FA^*)=z(FA_R)-z(H_R)$."
            )
        )
    )
    mo.vstack(
        [
            mo.md(
                rf"""
        ### The right-tail derivation

        Apply the complement identity from section 1 to each shaded area:

        $$
        \begin{{aligned}}
        H_R
          &=1-\Phi\!\left(\frac{{\lambda-\mu_B}}{{\sigma}}\right)
           =\Phi\!\left(\frac{{\mu_B-\lambda}}{{\sigma}}\right),\\
        FA_R
          &=1-\Phi\!\left(\frac{{\lambda-\mu_A}}{{\sigma}}\right)
           =\Phi\!\left(\frac{{\mu_A-\lambda}}{{\sigma}}\right).
        \end{{aligned}}
        $$

        Now $z=\Phi^{{-1}}$ cancels $\Phi$:

        $$
        \boxed{{d'_R=z(H_R)-z(FA_R)
        =\frac{{\mu_B-\lambda}}{{\sigma}}
        -\frac{{\mu_A-\lambda}}{{\sigma}}
        =\frac{{\mu_B-\mu_A}}{{\sigma}}}}
        $$

        In words, $d'$ is the **signal mean minus the noise mean, divided by
        their common standard deviation**. The $\lambda$ terms cancel. For the
        current settings, $(\mu_B-\mu_A)/\sigma={mu_signal - mu_noise:+.2f}$
        and the rate formula gives $d'_R={dprime_signed:+.2f}$.

        With unequal standard deviations, the difference instead contains
        $\lambda(1/\sigma_A-1/\sigma_B)$, so it changes with the criterion. That
        is precisely where the simple equal-variance identity stops applying.

        | $H_R$ | $FA_R$ | $z(H_R)$ | $z(FA_R)$ |
        |--:|--:|--:|--:|
        | {hit_right:.3f} | {false_alarm_right:.3f} | {z_hit_right:+.3f} | {z_false_alarm_right:+.3f} |

        ### What if signal is lower than noise?

        Keeping the right-tail rule gives a useful **signed** sensitivity:
        $d'_R={dprime_signed:+.2f}$. {_orientation_explanation}

        After orienting the evidence toward signal, the current rates are
        $H^*={hit_oriented:.3f}$ and $FA^*={false_alarm_oriented:.3f}$, giving
        $d'={dprime_oriented:.2f}=|d'_R|$.
        """
            ),
            mo.callout(
                mo.md(
                    r"""
        **So should it be $z(FA)-z(H)$?** Only when signal lies to the left and
        you intentionally reverse the evidence/response orientation. If both
        rates are fixed right tails, $z(H_R)-z(FA_R)$ is always the correct
        *signed* formula. State the convention rather than silently changing
        the subtraction.
        """
                ),
                kind="warn",
            ),
            mo.md(
                rf"""
        ## 3. Criterion comes from the midpoint of the two z values

        Their difference measures separation; their sum locates the cutoff.
        Starting from the same two transformed right-tail rates:

        $$
        \begin{{aligned}}
        z(H_R)+z(FA_R)
          &=\frac{{\mu_B+\mu_A-2\lambda}}{{\sigma}},\\
        \boxed{{c_R=-\tfrac12[z(H_R)+z(FA_R)]}}
          &=\boxed{{\frac{{\lambda-(\mu_B+\mu_A)/2}}{{\sigma}}}}.
        \end{{aligned}}
        $$

        Thus criterion is the cutoff minus the signal/noise mean midpoint,
        divided by the common standard deviation. It is also the **negative
        midpoint** of $z(H_R)$ and $z(FA_R)$. Therefore $c_R=0$ when the cutoff
        is at the midpoint of the evidence distributions. Positive values put
        the right-tail cutoff to its right; negative values put it to its left.

        If you orient evidence so larger values always favor signal, use
        $c= s\,c_R$, where $s=\operatorname{{sign}}(\mu_B-\mu_A)$. The current
        oriented value is $c={criterion_oriented:+.2f}$. In that convention,
        $c>0$ always means a conservative signal criterion.
        """
            ),
            mo.callout(
                mo.md(
                    rf"""
        For the current slider settings:

        $$
        c_R=-\frac12\left[({z_hit_right:+.3f})
        +({z_false_alarm_right:+.3f})\right]
        ={criterion_right:+.3f}.
        $$

        The evidence-axis calculation gives the same result:
        $[\lambda-(\mu_B+\mu_A)/2]/\sigma={criterion_right:+.3f}$.
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
    ## 4. Edge rates: why 0 and 1 become infinite

    The inverse CDF has $z(0)=-\infty$ and $z(1)=+\infty$. An observed rate
    of 0 or 1 usually means a finite sample contained no errors—not that the
    underlying probability is literally 0 or 1.

    Change the number of trials to see why a correction should know the
    denominator.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    edge_trials = mo.ui.slider(
        start=5,
        stop=200,
        step=5,
        value=20,
        show_value=True,
        label="Trials contributing to the rate (n)",
    )
    edge_case = mo.ui.radio(
        options=["0 successes", "n successes"],
        value="0 successes",
        inline=True,
        label="Observed edge",
    )
    mo.hstack([edge_trials, edge_case], widths="equal")
    return edge_case, edge_trials


@app.cell(hide_code=True)
def _(edge_case, edge_trials, mo, normal):
    _n = edge_trials.value
    _count = 0 if edge_case.value == "0 successes" else _n
    _raw_rate = _count / _n
    _raw_z = r"$-\infty$" if _count == 0 else r"$+\infty$"

    # Loglinear (half-count) correction: add 0.5 to both cells of the
    # binomial table. The total therefore increases by 1.
    _half_rate = (_count + 0.5) / (_n + 1.0)
    _half_z = normal.inv_cdf(_half_rate)

    # Boundary-only correction, often called the 1/(2N) adjustment.
    _boundary_rate = 1.0 / (2.0 * _n) if _count == 0 else 1.0 - 1.0 / (2.0 * _n)
    _boundary_z = normal.inv_cdf(_boundary_rate)

    _fixed_rate = 0.01 if _count == 0 else 0.99
    _fixed_z = normal.inv_cdf(_fixed_rate)

    mo.vstack(
        [
            mo.md(
                rf"""
        | Method | Rate used | z value | Main feature |
        |:--|--:|--:|:--|
        | Raw {_count}/{_n} | {_raw_rate:.4f} | {_raw_z} | Infinite |
        | Fixed .01/.99 clipping | {_fixed_rate:.4f} | {_fixed_z:+.3f} | Ignores $n$ |
        | Boundary $1/(2n)$ | {_boundary_rate:.4f} | {_boundary_z:+.3f} | Corrects only edges |
        | **Half-count (loglinear)** | **{_half_rate:.4f}** | **{_half_z:+.3f}** | Uses $n$; apply to every cell |
        """
            ),
            mo.callout(
                mo.md(
                    r"""
        A practical default is the **half-count/loglinear correction**. For
        hits $h$, misses $m$, false alarms $f$, and correct rejections $r$:

        $$
        \widetilde H=\frac{h+0.5}{h+m+1},\qquad
        \widetilde{FA}=\frac{f+0.5}{f+r+1}.
        $$

        Then compute
        $d'=z(\widetilde H)-z(\widetilde{FA})$ and
        $c=-\tfrac12[z(\widetilde H)+z(\widetilde{FA})]$.
        Apply the correction within each participant/condition **before** taking
        group averages, and report which correction you used.
        """
                ),
                kind="success",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Count-based implementation

    ```python
    from statistics import NormalDist

    def sdt_from_counts(hits, misses, false_alarms, correct_rejections):
        # Equal-variance SDT with a half-count correction.
        hit_rate = (hits + 0.5) / (hits + misses + 1)
        fa_rate = (false_alarms + 0.5) / (
            false_alarms + correct_rejections + 1
        )
        z = NormalDist().inv_cdf
        z_hit, z_fa = z(hit_rate), z(fa_rate)
        dprime = z_hit - z_fa
        criterion = -0.5 * (z_hit + z_fa)
        return dprime, criterion
    ```

    **Take-home:** with right tails, $z(H_R)-z(FA_R)$ equals the signed
    standardized mean difference. Reverse the evidence/tails explicitly if
    you want sensitivity oriented toward a leftward signal, use the matching
    sign for criterion, and correct finite-sample edge rates before applying
    $z$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(
                r"""
                ## 5. Why two intervals improve forced-choice performance

                Compare a single-observation classification task
                (**yes/no**, or 1I–2AFC in the convention used here) with a
                **2I–2AFC** task. Let

                $$
                N\sim\mathcal N(0,\sigma^2),\qquad
                S\sim\mathcal N(\Delta,\sigma^2),\qquad
                d'=\frac{\Delta}{\sigma},
                $$

                where larger evidence favors the signal.

                ### One interval: compare one observation with a criterion

                With equal priors, the optimal criterion is
                $\lambda=\Delta/2$. The two kinds of correct response have
                the same probability, so

                $$
                \begin{aligned}
                P_C^{\mathrm{1I}}
                  &=\tfrac12\left[P(S>\Delta/2)+P(N<\Delta/2)\right]\\
                  &=\Phi\!\left(\frac{\Delta}{2\sigma}\right)
                   =\boxed{\Phi\!\left(\frac{d'}{2}\right)}.
                \end{aligned}
                $$

                ### Two intervals: compare two independent observations

                One interval contains $S$ and the other contains $N$. Choose
                the interval with the larger observation. Orient the
                difference as target minus nontarget:

                $$
                D=S-N.
                $$

                Independence makes the variances **add**:

                $$
                E[D]=\Delta,\qquad
                \operatorname{Var}(D)=\sigma^2+\sigma^2=2\sigma^2,
                \qquad SD(D)=\sqrt{2}\,\sigma.
                $$

                A response is correct when $D>0$, hence

                $$
                P_C^{\mathrm{2I}}
                  =P(D>0)
                  =\Phi\!\left(\frac{\Delta}{\sqrt{2}\,\sigma}\right)
                  =\boxed{\Phi\!\left(\frac{d'}{\sqrt{2}}\right)}.
                $$

                Because $1/\sqrt{2}>1/2$, 2I–2AFC has higher expected accuracy
                than the equal-prior one-interval task whenever $d'>0$.
                Equivalently, if $D=X_2-X_1$, its conditional means are
                $-\Delta$ and $+\Delta$. Their standardized separation is

                $$
                d'_D=\frac{(+\Delta)-(-\Delta)}{\sqrt{2}\,\sigma}
                    =\boxed{\sqrt{2}\,d'}.
                $$
                """
            ),
            mo.callout(
                mo.md(
                    r"""
                    **Important correction:** the raw difference variable is
                    not narrower. Its standard deviation is
                    $\sqrt{2}\,\sigma$, not $\sigma/\sqrt{2}$. The advantage comes
                    from using two observations: the two target-location
                    distributions have twice the mean separation, producing a
                    net $\sqrt{2}$ improvement in standardized separation.
                    """
                ),
                kind="warn",
            ),
            mo.md(
                r"""
                The corresponding accuracy-to-sensitivity conversions are

                $$
                d'=2z(P_C^{\mathrm{1I}}),\qquad
                d'=\sqrt{2}\,z(P_C^{\mathrm{2I}}).
                $$

                Move the sensory $d'$ below. At zero, both tasks are at chance.
                As the distributions separate, watch the 2I margin move farther
                from the decision boundary and its accuracy rise faster.
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    forced_choice_dprime = mo.ui.slider(
        start=0.0,
        stop=3.5,
        step=0.1,
        value=1.6,
        debounce=True,
        show_value=True,
        full_width=True,
        label=r"Underlying sensory separation $d'=\Delta/\sigma$",
    )
    forced_choice_dprime
    return (forced_choice_dprime,)


@app.cell(hide_code=True)
def _(forced_choice_dprime, mo, normal, normal_pdf, np, plt):
    _dprime = forced_choice_dprime.value
    _delta = _dprime  # Set sigma=1 without loss of generality.
    _sigma = 1.0
    _difference_sd = np.sqrt(2.0) * _sigma
    _criterion_1i = _delta / 2.0
    _n_trials = 60_000

    _theory_1i = normal.cdf(_dprime / 2.0)
    _theory_2i = normal.cdf(_dprime / np.sqrt(2.0))

    # A reproducible Monte Carlo demonstration. In 1I, classify a randomly
    # selected noise or signal observation. In 2I, compare an independent
    # signal observation directly with an independent noise observation.
    _rng = np.random.default_rng(20260914)
    _is_signal = _rng.integers(0, 2, size=_n_trials).astype(bool)
    _one_observation = _rng.normal(0.0, _sigma, size=_n_trials)
    _one_observation = _one_observation + _is_signal * _delta
    _one_says_signal = _one_observation > _criterion_1i
    _simulated_1i = np.mean(_one_says_signal == _is_signal)

    _signal_observation = _rng.normal(_delta, _sigma, size=_n_trials)
    _noise_observation = _rng.normal(0.0, _sigma, size=_n_trials)
    _raw_difference = _signal_observation - _noise_observation
    _simulated_2i = np.mean(_raw_difference > 0.0)

    # Put both tasks on unit-SD, correctly-oriented decision margins. Correct
    # responses are to the right of zero in either distribution.
    _margin_1i = np.where(
        _is_signal,
        _one_observation - _criterion_1i,
        _criterion_1i - _one_observation,
    )
    _margin_2i = _raw_difference / _difference_sd

    _fig, _axes = plt.subplots(2, 2, figsize=(12.2, 8.0))
    _fig.patch.set_facecolor("#ffffff")

    # One-interval geometry.
    _x_1i = np.linspace(-4.2, _delta + 4.2, 850)
    _noise_density = normal_pdf(_x_1i, 0.0, _sigma)
    _signal_density = normal_pdf(_x_1i, _delta, _sigma)
    _axes[0, 0].plot(
        _x_1i, _noise_density, color="#2b6cb0", linewidth=2.2, label="Noise"
    )
    _axes[0, 0].plot(
        _x_1i,
        _signal_density,
        color="#d97706",
        linewidth=2.2,
        linestyle="--" if _dprime == 0 else "-",
        label="Signal",
    )
    _axes[0, 0].fill_between(
        _x_1i,
        0,
        _noise_density,
        where=_x_1i < _criterion_1i,
        color="#2b6cb0",
        alpha=0.18,
    )
    _axes[0, 0].fill_between(
        _x_1i,
        0,
        _signal_density,
        where=_x_1i > _criterion_1i,
        color="#d97706",
        alpha=0.18,
    )
    _axes[0, 0].axvline(
        _criterion_1i, color="#7c3aed", linestyle="--", linewidth=1.8
    )
    _axes[0, 0].text(
        _criterion_1i,
        0.415,
        rf"criterion $\Delta/2={_criterion_1i:.2f}$",
        ha="center",
        color="#5b21b6",
        fontsize=9,
    )
    _axes[0, 0].set(
        title=rf"1I / yes-no: $P_C=\Phi(d'/2)={_theory_1i:.3f}$",
        xlabel="one evidence observation",
        ylabel="density",
        ylim=(0, 0.45),
    )
    _axes[0, 0].legend(frameon=False, fontsize=8.5, ncols=2)

    # In 2I, D=X2-X1 has means +/-Delta depending on target location.
    _difference_limit = _delta + 4.8 * _difference_sd
    _x_difference = np.linspace(
        -_difference_limit, _difference_limit, 950
    )
    _target_first_density = normal_pdf(
        _x_difference, -_delta, _difference_sd
    )
    _target_second_density = normal_pdf(
        _x_difference, _delta, _difference_sd
    )
    _axes[0, 1].plot(
        _x_difference,
        _target_first_density,
        color="#2b6cb0",
        linewidth=2.2,
        label=r"target in I1: $E[D]=-\Delta$",
    )
    _axes[0, 1].plot(
        _x_difference,
        _target_second_density,
        color="#d97706",
        linewidth=2.2,
        linestyle="--" if _dprime == 0 else "-",
        label=r"target in I2: $E[D]=+\Delta$",
    )
    _axes[0, 1].fill_between(
        _x_difference,
        0,
        _target_first_density,
        where=_x_difference < 0,
        color="#2b6cb0",
        alpha=0.18,
    )
    _axes[0, 1].fill_between(
        _x_difference,
        0,
        _target_second_density,
        where=_x_difference > 0,
        color="#d97706",
        alpha=0.18,
    )
    _axes[0, 1].axvline(0, color="#7c3aed", linestyle="--", linewidth=1.8)
    _axes[0, 1].text(
        0,
        0.295,
        "choose I2 if D > 0",
        ha="center",
        color="#5b21b6",
        fontsize=9,
    )
    _axes[0, 1].set(
        title=rf"2I difference: $d'_D=\sqrt{{2}}d'={np.sqrt(2)*_dprime:.2f}$",
        xlabel=r"difference $D=X_2-X_1$",
        ylabel="density",
        ylim=(0, 0.32),
    )
    _axes[0, 1].legend(frameon=False, fontsize=8.2)

    # Simulation: orient both variables so correct is right of zero, then
    # standardize each by its own SD for an apples-to-apples comparison.
    _margin_limit = max(4.2, _dprime / np.sqrt(2.0) + 4.0)
    _margin_bins = np.linspace(-4.2, _margin_limit, 75)
    _margin_x = np.linspace(-4.2, _margin_limit, 700)
    _axes[1, 0].hist(
        _margin_1i,
        bins=_margin_bins,
        density=True,
        histtype="step",
        linewidth=1.2,
        color="#2b6cb0",
        alpha=0.65,
    )
    _axes[1, 0].hist(
        _margin_2i,
        bins=_margin_bins,
        density=True,
        histtype="step",
        linewidth=1.2,
        color="#16846b",
        alpha=0.65,
    )
    _axes[1, 0].plot(
        _margin_x,
        normal_pdf(_margin_x, _dprime / 2.0, 1.0),
        color="#2b6cb0",
        linewidth=2.3,
        label=rf"1I margin: mean $d'/2={_dprime/2:.2f}$",
    )
    _axes[1, 0].plot(
        _margin_x,
        normal_pdf(_margin_x, _dprime / np.sqrt(2.0), 1.0),
        color="#16846b",
        linewidth=2.3,
        label=(
            rf"2I margin: mean $d'/\sqrt{{2}}="
            rf"{_dprime/np.sqrt(2):.2f}$"
        ),
    )
    _axes[1, 0].axvline(0, color="#7c3aed", linestyle="--", linewidth=1.7)
    _axes[1, 0].text(
        0,
        0.415,
        "correct →",
        ha="left",
        color="#5b21b6",
        fontsize=9,
    )
    _axes[1, 0].set(
        title="Simulation: standardized correct-decision margins",
        xlabel="margin in its own SD units",
        ylabel="density",
        ylim=(0, 0.45),
    )
    _axes[1, 0].legend(frameon=False, fontsize=8.3)

    # The theoretical accuracy functions plus the current Monte Carlo result.
    _dprime_grid = np.linspace(0, 3.5, 250)
    _accuracy_1i_grid = np.array(
        [normal.cdf(float(_value) / 2.0) for _value in _dprime_grid]
    )
    _accuracy_2i_grid = np.array(
        [normal.cdf(float(_value) / np.sqrt(2.0)) for _value in _dprime_grid]
    )
    _axes[1, 1].plot(
        _dprime_grid,
        _accuracy_1i_grid,
        color="#2b6cb0",
        linewidth=2.4,
        label=r"1I theory: $\Phi(d'/2)$",
    )
    _axes[1, 1].plot(
        _dprime_grid,
        _accuracy_2i_grid,
        color="#16846b",
        linewidth=2.4,
        label=r"2I theory: $\Phi(d'/\sqrt{2})$",
    )
    _axes[1, 1].scatter(
        [_dprime],
        [_simulated_1i],
        marker="x",
        s=75,
        linewidths=2.0,
        color="#2b6cb0",
        zorder=4,
        label="1I simulation",
    )
    _axes[1, 1].scatter(
        [_dprime],
        [_simulated_2i],
        marker="x",
        s=75,
        linewidths=2.0,
        color="#16846b",
        zorder=4,
        label="2I simulation",
    )
    _axes[1, 1].axhline(
        0.5, color="#aab3c2", linestyle="--", linewidth=1.1
    )
    _axes[1, 1].axvline(
        _dprime, color="#7c3aed", linestyle=":", linewidth=1.2
    )
    _axes[1, 1].set(
        title="Two intervals gain accuracy",
        xlabel=r"underlying sensory $d'$",
        ylabel="proportion correct",
        xlim=(0, 3.5),
        ylim=(0.48, 1.005),
    )
    _axes[1, 1].legend(frameon=False, fontsize=8.0, ncols=2)

    for _axis in _axes.flat:
        _axis.spines[["top", "right"]].set_visible(False)
        _axis.grid(alpha=0.12)
    _axes[0, 0].set_yticks([])
    _axes[0, 1].set_yticks([])
    _fig.tight_layout(h_pad=2.2, w_pad=1.7)

    _gain_theory = _theory_2i - _theory_1i
    _gain_simulated = _simulated_2i - _simulated_1i
    mo.vstack(
        [
            _fig,
            mo.md(
                rf"""
                | Task | Analytical accuracy | Simulated accuracy |
                |:--|--:|--:|
                | 1I / yes-no | $\Phi(d'/2)$ = {_theory_1i:.4f} | {_simulated_1i:.4f} |
                | 2I–2AFC | $\Phi(d'/\sqrt{{2}})$ = {_theory_2i:.4f} | {_simulated_2i:.4f} |
                | **2I improvement** | **{_gain_theory:+.4f}** | **{_gain_simulated:+.4f}** |

                The simulation uses {_n_trials:,} trials per task and a fixed
                random seed. The upper-right raw difference distributions are
                **wider** because $SD(D)=\sqrt{{2}}\sigma$. The lower-left panel
                divides each task's decision margin by its own SD: both curves
                then have unit width, and the 2I curve sits farther to the
                right. Its larger area beyond zero is the accuracy advantage.
                """
            ),
            mo.accordion(
                {
                    "Show the minimal reproducible simulation": mo.md(
                        r"""
                This is the numerical core used above. Change `dprime` and run
                it again; the plotting code only visualizes these same decision
                variables and probabilities.

                ```python
                from statistics import NormalDist
                import numpy as np

                rng = np.random.default_rng(20260914)
                n_trials = 60_000
                sigma = 1.0
                dprime = 1.6
                delta = dprime * sigma

                # 1I / yes-no: classify one observation relative to Δ/2.
                is_signal = rng.integers(0, 2, size=n_trials).astype(bool)
                x = rng.normal(0.0, sigma, size=n_trials)
                x = x + is_signal * delta
                says_signal = x > delta / 2
                pc_1i_sim = np.mean(says_signal == is_signal)

                # 2I–2AFC: compare an independent S observation with N.
                signal = rng.normal(delta, sigma, size=n_trials)
                noise = rng.normal(0.0, sigma, size=n_trials)
                difference = signal - noise
                pc_2i_sim = np.mean(difference > 0)

                # Analytical predictions.
                phi = NormalDist().cdf
                pc_1i_theory = phi(dprime / 2)
                pc_2i_theory = phi(dprime / np.sqrt(2))

                print(f"1I: simulation={pc_1i_sim:.4f}, theory={pc_1i_theory:.4f}")
                print(f"2I: simulation={pc_2i_sim:.4f}, theory={pc_2i_theory:.4f}")
                ```
                """
                    )
                }
            ),
            mo.callout(
                mo.md(
                    r"""
                    This $\sqrt{2}$ relationship assumes independent interval
                    observations, equal Gaussian variances, an optimal rule,
                    and no interval bias. Lapses, unequal variance, correlated
                    interval noise, or order effects require a richer model.
                    """
                ),
                kind="info",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(r"""
        ### Endnote for real analyses

        The equations above are for understanding the geometry, **not a
        recommended production analysis pipeline**. For actual data, use a
        tested toolbox such as [Palamedes](https://www.palamedestoolbox.org/)
        (MATLAB/Python; psychometric and signal-detection routines) or
        [sensR](https://CRAN.R-project.org/package=sensR) (R; SDT estimates,
        uncertainty, and discrimination-test models).

        A toolbox will make conventions, finite-sample corrections, uncertainty,
        and model assumptions more explicit. For sparse, repeated-measures, or
        hierarchical data, prefer a likelihood-based SDT model with appropriate
        diagnostics over applying these plug-in formulae row by row.
        """),
        kind="warn",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    naka_n = mo.ui.slider(
        start=0.5,
        stop=6.0,
        step=0.25,
        value=2.0,
        debounce=True,
        show_value=True,
        full_width=True,
        label=r"Exponent $n$ (steepness)",
    )
    naka_c50 = mo.ui.slider(
        start=0.05,
        stop=0.95,
        step=0.05,
        value=0.30,
        debounce=True,
        show_value=True,
        full_width=True,
        label=r"Semisaturation $C_{50}$ (half-maximum input)",
    )
    naka_dmax = mo.ui.slider(
        start=0.5,
        stop=6.0,
        step=0.25,
        value=3.0,
        debounce=True,
        show_value=True,
        full_width=True,
        label=r"Maximum sensitivity $d'_{\max}$",
    )
    naka_controls = mo.vstack(
        [naka_n, naka_c50, naka_dmax],
        gap=0.55,
    )
    return naka_c50, naka_controls, naka_dmax, naka_n


@app.cell(hide_code=True)
def _(naka_c50, naka_controls, naka_dmax, naka_n, mo, np, plt):
    _n = naka_n.value
    _c50 = naka_c50.value
    _dmax = naka_dmax.value
    _contrast = np.linspace(0.0, 1.0, 700)

    def _naka_rushton(c, n, c50, dmax):
        return dmax * (c**n) / (c**n + c50**n)

    _current = _naka_rushton(_contrast, _n, _c50, _dmax)
    _half_max = _dmax / 2.0

    _n_values = [max(0.5, _n - 0.75), _n, min(6.0, _n + 0.75)]
    _c50_values = [max(0.05, _c50 - 0.15), _c50, min(0.95, _c50 + 0.15)]
    _dmax_values = [max(0.5, _dmax - 1.0), _dmax, min(6.0, _dmax + 1.0)]
    _comparison_colors = ["#2b6cb0", "#7c3aed", "#d97706"]
    _comparison_widths = [1.7, 2.8, 1.7]

    _fig, _axes = plt.subplots(2, 2, figsize=(11, 8.0))
    _fig.patch.set_facecolor("#ffffff")

    _axes[0, 0].plot(
        _contrast,
        _current,
        color="#7c3aed",
        linewidth=3.0,
        label="current curve",
    )
    _axes[0, 0].axvline(
        _c50, color="#d97706", linestyle="--", linewidth=1.7
    )
    _axes[0, 0].axhline(
        _half_max, color="#d97706", linestyle="--", linewidth=1.2
    )
    _axes[0, 0].axhline(
        _dmax, color="#5f6b7c", linestyle=":", linewidth=1.4
    )
    _axes[0, 0].scatter(
        [_c50], [_half_max], s=70, color="#d97706", zorder=4
    )
    _axes[0, 0].text(
        _c50,
        _half_max,
        rf"  $C_{{50}}={_c50:.2f},\ d'=d'_{{\max}}/2={_half_max:.2f}$",
        ha="left",
        va="bottom",
        color="#a34f05",
        fontsize=8.7,
    )
    _axes[0, 0].text(
        0.985,
        _dmax,
        rf"$d'_{{\max}}={_dmax:.2f}$  ",
        ha="right",
        va="bottom",
        color="#4b5563",
        fontsize=8.7,
    )
    _axes[0, 0].set(title="Current Naka–Rushton curve")

    for _value, _color, _width in zip(
        _n_values, _comparison_colors, _comparison_widths
    ):
        _axes[0, 1].plot(
            _contrast,
            _naka_rushton(_contrast, _value, _c50, _dmax),
            color=_color,
            linewidth=_width,
            label=rf"$n={_value:.2f}$",
        )
    _axes[0, 1].scatter(
        [_c50], [_half_max], color="#182338", s=32, zorder=4
    )
    _axes[0, 1].set(
        title=rf"Vary $n$; all curves cross at $C_{{50}}={_c50:.2f}$"
    )

    for _value, _color, _width in zip(
        _c50_values, _comparison_colors, _comparison_widths
    ):
        _axes[1, 0].plot(
            _contrast,
            _naka_rushton(_contrast, _n, _value, _dmax),
            color=_color,
            linewidth=_width,
            label=rf"$C_{{50}}={_value:.2f}$",
        )
        _axes[1, 0].scatter(
            [_value], [_half_max], color=_color, s=26, zorder=4
        )
    _axes[1, 0].set(title=r"Vary $C_{50}$: move the curve left or right")

    for _value, _color, _width in zip(
        _dmax_values, _comparison_colors, _comparison_widths
    ):
        _axes[1, 1].plot(
            _contrast,
            _naka_rushton(_contrast, _n, _c50, _value),
            color=_color,
            linewidth=_width,
            label=rf"$d'_{{\max}}={_value:.2f}$",
        )
        _axes[1, 1].axhline(
            _value, color=_color, linestyle=":", linewidth=0.8, alpha=0.65
        )
    _axes[1, 1].set(title=r"Vary $d'_{\max}$: raise or lower the ceiling")

    _upper_y = max(1.0, max(_dmax_values) * 1.13)
    for _axis in _axes.flat:
        _axis.set(
            xlabel="normalized contrast or intensity c",
            ylabel=r"sensitivity $d'(c)$",
            xlim=(0, 1.0),
            ylim=(0, _upper_y),
        )
        _axis.spines[["top", "right"]].set_visible(False)
        _axis.grid(alpha=0.14)
        _axis.legend(frameon=False, fontsize=8.0)
    _fig.tight_layout(h_pad=2.1, w_pad=1.6)

    mo.vstack(
        [
            mo.md(r"""
            ## Final endnote: the Naka–Rushton equation

            A common way to describe how sensitivity grows and then saturates
            with stimulus contrast or intensity is

            $$
            d'(c)=d'_{\max}\frac{c^n}{c^n+C_{50}^n}.
            $$

            Here $c$ is normalized stimulus contrast or intensity. Adjust the
            three parameters directly above the graph.
            """),
            naka_controls,
            _fig,
            mo.callout(
                mo.md(
                    rf"""
                    **Read the parameters visually:**

                    - $d'_{{\max}}={_dmax:.2f}$ is the upper asymptote: it
                      scales the curve vertically.
                    - $C_{{50}}={_c50:.2f}$ is the input at exactly half that
                      maximum. Consequently
                      $d'(C_{{50}})=d'_{{\max}}/2={_half_max:.2f}$ regardless
                      of $n$.
                    - $n={_n:.2f}$ controls steepness. Larger $n$ makes the
                      transition around $C_{{50}}$ sharper; smaller $n$ makes
                      it more gradual. It does not change the half-maximum
                      crossing or the asymptotic ceiling.

                    The three comparison panels vary one parameter around its
                    current slider value while holding the other two fixed.
                    In real data analysis, these parameters should be fitted
                    jointly with uncertainty rather than selected by eye.
                    """
                ),
                kind="info",
            ),
        ],
        gap=0.7,
    )
    return


if __name__ == "__main__":
    app.run()
