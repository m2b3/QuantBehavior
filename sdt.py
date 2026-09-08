# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.23.9",
#     "matplotlib",
#     "numpy",
#     "psignifit",
#     "scikit-learn",
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
    # Signal detection theory: from an ideal observer to psychometric functions

    We begin with the simplest equal-variance Gaussian model, then separate its core ideas from useful extensions and practical fitting.

    1. [The ideal equal-variance model](#1-the-ideal-equal-variance-model)
    2. [What changes in real observers?](#2-what-changes-in-real-observers)
    3. [Simulated yes/no data](#3-simulated-yesno-data)
    4. [Fit a psychometric function](#4-fit-a-psychometric-function)
    5. [From SDT to stimulus-dependent psychometric functions](#5-from-sdt-to-stimulus-dependent-psychometric-functions)
    6. [Appendix: selected extensions](#appendix-selected-extensions)

    ---

    © 2026 Suresh Krishna. This educational notebook is licensed under
    [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
    [Source and attribution details](https://github.com/m2b3/QuantBehavior/blob/main/LICENSE-CONTENT.md).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The ideal equal-variance model
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    from scipy.stats import skewnorm
    from sklearn.metrics import auc
    import psignifit as ps
    import psignifit.psigniplot as psp
    import math as math

    return auc, math, norm, np, plt, ps, psp, skewnorm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now let us consider an experiment where one half of trials contains the signal and the other half contains the noise, and what we get is a noisy measurement from the trial, that has some information about whether it was a signal trial or a noise trial. This situation is very general. For example (without paying attention to the relative proportions of signal and noise in these examples):

    1.   An airport security operator sees a bag under the X-ray scanner and tries to determine if it contains a forbidden item or not.
    2.   A radiologist sees an X-ray and tries to determine if there is a lesion in it or not.
    3.   You hear a sound and try to determine if your name was called or not.
    4.   You run a new test on a blood sample and try to determine if a certain condition is present or not.
    5.   Come up with a few more.

    We start with an intentionally idealized model:

    - signal and noise each produce a one-dimensional internal response;
    - both response distributions are Gaussian with the same variance;
    - trials are independent and the distributions are stationary;
    - the observer applies one fixed criterion;
    - there are no stimulus-independent guesses or lapses.

    When the signal is present the response tends to be larger, but the two distributions overlap.

    To think about: What does it mean to say that the response variable contains information about the presence of the signal ?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Three tasks that are easy to confuse

    | Task | What happens on one trial? | What must the observer report? | Most direct summary |
    |---|---|---|---|
    | **Yes/no detection** | One observation is shown. The signal may be present or absent. | “Signal present” or “signal absent.” | Hits and false alarms; sensitivity and response criterion. |
    | **Single-interval 2AFC identification** | One observation is shown. It is either category A or category B; one of them is always present. | “A” or “B”—no “unsure” response. | The same two-by-two outcome table after one category is designated “positive.” A decision criterion can still favor one response. |
    | **2I-2AFC detection** | Two intervals or locations are shown. One contains the signal and the other contains noise. | Which interval or location contained the signal? | Proportion correct and $d'$; the observer compares two observations. |

    Terminology is not perfectly consistent: **2AFC** is often used as shorthand for the two-interval task. Saying **1I-2AFC** or **2I-2AFC** removes the ambiguity.

    Sensitivity, specificity, recall, precision, and positive predictive value are most natural for yes/no detection or single-interval classification, where “positive” and “negative” outcomes exist. In ordinary 2I-2AFC detection the signal is present on every trial, so the task does not directly answer questions such as “given a positive test, what is the chance that disease is present?” It answers “which of these two observations contained the signal?” Catch trials or a separate yes/no task are needed to estimate detection specificity and predictive value.
    """)
    return


@app.cell
def _(norm, np, plt):
    # Generate x values - the response variable takes values from 0 to 100 (arbitrary)
    x = np.linspace(0, 100, 1000)
    diff_x = np.linspace(-100, 100, 1000)
    # Noise and signal distributions, respectively
    loc1 = 40  # mean
    scale1 = 10  # standard deviation
    loc2 = 60  # mean
    scale2 = 10  # standard deviation
    pdf1 = norm.pdf(x, loc1, scale1)
    pdf2 = norm.pdf(x, loc2, scale2)
    _criterion = (loc1 + loc2) / 2
    # Compute the PDF values
    plt.figure(figsize=(11, 4))
    plt.plot(x, pdf1, label=f'Noise trials: mean={loc1}, SD={scale1}', color='red')
    plt.plot(x, pdf2, label=f'Signal trials: mean={loc2}, SD={scale2}', color='blue')
    _right_of_criterion = x >= _criterion
    _left_of_criterion = x < _criterion
    plt.fill_between(
        x[_right_of_criterion],
        0,
        pdf2[_right_of_criterion],
        color='blue',
        alpha=0.20,
        label='H: hit area',
    )
    plt.fill_between(
        x[_right_of_criterion],
        0,
        pdf1[_right_of_criterion],
        color='red',
        alpha=0.30,
        hatch='//',
        label='F: false-alarm area',
    )
    plt.fill_between(
        x[_left_of_criterion],
        0,
        pdf1[_left_of_criterion],
        color='red',
        alpha=0.07,
    )
    # Plot the distributions
    _idx = np.argwhere(np.diff(np.sign(pdf2 - pdf1))).flatten()
    plt.plot(x[_idx], pdf2[_idx], 'ko')
    intersect_point = x[_idx]
    plt.axvline(
        _criterion,
        color='black',
        linestyle='--',
        label=f'Decision criterion k={_criterion:g} (c=0)',
    )
    _arrow_height = 1.08 * max(np.max(pdf1), np.max(pdf2))
    plt.annotate(
        '',
        xy=(loc2, _arrow_height),
        xytext=(loc1, _arrow_height),
        arrowprops={'arrowstyle': '<->', 'color': 'purple', 'linewidth': 2},
    )
    plt.text(
        loc1 + 0.25 * (loc2 - loc1),
        1.02 * _arrow_height,
        r"$d'$",
        color='purple',
        ha='center',
        va='bottom',
        fontsize=13,
        fontweight='bold',
        bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.85, 'pad': 1},
    )
    plt.title('Figure 1. Equal-variance Gaussian evidence in a yes/no task')
    plt.xlabel('Internal evidence')
    plt.ylabel('Density')
    plt.ylim(0, 1.28 * _arrow_height)
    plt.legend(fontsize=8, ncol=2, loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return diff_x, intersect_point, loc1, loc2, pdf1, pdf2, scale1, scale2, x


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **How to read Figure 1.** The red curve shows evidence on noise-only trials and the blue curve shows evidence on signal trials. The black dashed line is the observer's decision threshold: evidence to its right produces a “signal” response. The purple arrow marks $d'$, the distance between the two peaks measured in units of their common standard deviation. For the particular curves drawn here, that distance happens to be two standard deviations; this is an illustrative choice, not a fixed value of $d'$. The standardized criterion $c$ says where the black line lies relative to the midpoint between the peaks. Both measures are explained just below Figure 2.

    The area under the **blue** curve to the right of the black line is the hit rate. The area under the **red** curve to the right is the false-alarm rate. The red area to the left is specificity. These are areas under curves because they represent the chances of observations landing on each side of the decision line.

    More generally, response variables can be multidimensional (in which case the acceptance region may also be multidimensional). The acceptance region can also be much more complex, even in 1 dimension, if the underlying response variable distributions are. Think about these..

    Figure 2 moves that line through every possible position. A very low decision threshold calls almost everything “signal”: it catches more real signals but also produces more false alarms. A very high threshold does the reverse. Figure 2B plots the resulting hit rate against false-alarm rate. This movable **decision threshold** is distinct from a fixed sensory threshold; the historical evidence and low-threshold alternatives are discussed in [Appendix A3](#a3-why-roc-experiments-mattered).
    """)
    return


@app.cell
def _(auc, intersect_point, loc1, loc2, norm, np, plt, scale1, scale2, x):
    # Calculate theoretical ROC
    fpr_theoretical = []
    tpr_theoretical = []
    thresholds = np.linspace(min(x), max(x), 1000)
    for _threshold in thresholds:
        tpr_theoretical.append(1 - norm.cdf(_threshold, loc2, scale2))
        fpr_theoretical.append(1 - norm.cdf(_threshold, loc1, scale1))
    _percor_theoretical = 0.5 * (np.array(tpr_theoretical) + (1 - np.array(fpr_theoretical)))
    # Add the exact endpoints corresponding to thresholds of -infinity and +infinity.
    _roc_fpr_theoretical = np.r_[1.0, fpr_theoretical, 0.0]
    _roc_tpr_theoretical = np.r_[1.0, tpr_theoretical, 0.0]
    _roc_auc = auc(_roc_fpr_theoretical, _roc_tpr_theoretical)
    print(f'Area Under the Curve (AUC): {_roc_auc:.4f}')
    # Language-independent analytic identities provide internal checks.
    _dprime_theoretical = (loc2 - loc1) / scale1
    _auc_analytic = norm.cdf(_dprime_theoretical / np.sqrt(2))
    assert np.isclose(scale1, scale2)
    assert np.allclose(
        intersect_point,
        (loc1 + loc2) / 2,
        atol=2 * (x[1] - x[0]),
    )
    assert np.isclose(_roc_auc, _auc_analytic, atol=5e-4)
    assert np.isclose(
        np.max(_percor_theoretical),
        norm.cdf(_dprime_theoretical / 2),
        atol=5e-4,
    )
    #calculating percentage correct, assuming 50 % of trials are signal (yes/no experiment)
    n_trials = 10000
    n_half = n_trials // 2
    _rng = np.random.default_rng(602)
    samples_noise = norm.rvs(loc1, scale1, size=n_half, random_state=_rng)
    samples_signal = norm.rvs(loc2, scale2, size=n_half, random_state=_rng)
    labels_noise = np.zeros(n_half)
    # Simulate samples - for simulation-based roc
    labels_signal = np.ones(n_half)
    samples = np.concatenate([samples_noise, samples_signal])  #50 % of trials are signal and 50 % are noise
    labels = np.concatenate([labels_noise, labels_signal])
    fpr_simulation = []
    tpr_simulation = []
    for _threshold in thresholds:
        predictions = samples > _threshold
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        tn = np.sum((predictions == 0) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        tpr_simulation.append(tp / (tp + fn))
    # Calculate simulation-based ROC
        fpr_simulation.append(fp / (fp + tn))  #false-positive rate
    plt.figure(figsize=(10, 3))  #true-positive rate
    plt.subplot(1, 2, 1)
    plt.plot(thresholds, _percor_theoretical)
    plt.axvline(intersect_point, color='black', linestyle='--')
    plt.xlabel('Threshold')
    plt.ylabel('Chance of a correct response')
    plt.title('Figure 2A. Accuracy as the decision threshold moves')
    plt.subplot(1, 2, 2)
    _roc_fpr_simulation = np.r_[1.0, fpr_simulation, 0.0]
    _roc_tpr_simulation = np.r_[1.0, tpr_simulation, 0.0]
    plt.plot(_roc_fpr_theoretical, _roc_tpr_theoretical, label='Theoretical ROC Curve')
    plt.plot(_roc_fpr_simulation, _roc_tpr_simulation, label='Simulation-based ROC Curve', linestyle='--')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='-.', label='Chance level')
    plt.xlabel('False-alarm rate (the opposite of specificity)')
    plt.ylabel('Hit rate (sensitivity / recall)')
    plt.title('Figure 2B. The same thresholds traced in ROC space')
    plt.legend()
    plt.grid(True)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    # Second Column (for the ROC curves)
    # Adjust the layout so plots don't overlap
    plt.show()
    return fpr_theoretical, thresholds, tpr_theoretical


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Standard SDT measures

    The notation $P(x>k\mid\text{signal})$ simply means “the chance that the evidence lands to the right of the decision line on a signal trial.” In Figure 1 that is the blue area to the right of the black line. We call it the hit rate $H$. Replacing “signal” with “noise” gives the red area to the right, the false-alarm rate $F$.

    For the equal-variance Gaussian model,

    \[
    d' = \Phi^{-1}(H)-\Phi^{-1}(F)
       = \frac{\mu_S-\mu_N}{\sigma},
    \]

    **In plain language:** $d'$ is the purple peak-to-peak distance in Figure 1, measured with the bell curves' standard deviation as the ruler. In this illustration the peaks were deliberately placed 20 evidence units apart and the standard deviation was set to 10, so the example has $d'=2$. Other distributions can have other values of $d'$. The symbol $\Phi^{-1}$ only converts each shaded area to that standard-deviation ruler. Larger $d'$ means less overlap and easier discrimination; moving the black criterion line does not move the peaks and therefore does not change $d'$.

    Response bias is commonly summarized by

    \[
    c=-\frac{1}{2}\left[\Phi^{-1}(H)+\Phi^{-1}(F)\right]
      =\frac{k-(\mu_S+\mu_N)/2}{\sigma}.
    \]

    **In plain language:** $c$ describes the location of the black decision line in Figure 1. It is zero when the line is halfway between the two peaks, positive when shifted right (a conservative observer who says “signal” less often), and negative when shifted left (a liberal observer who says “signal” more often). It measures that shift using the same standard-deviation ruler.

    The likelihood ratio at the criterion is

    \[
    \beta=\frac{f_S(k)}{f_N(k)}=e^{d'c}.
    \]

    **In plain language:** $\beta$ compares the height of the blue curve with the height of the red curve exactly at the black line. At the crossing in Figure 1 the heights are equal, so $\beta=1$. With equally common trial types and equally costly errors, that crossing is the accuracy-maximizing location and $c=0$. These short formulas assume equal-variance Gaussian distributions; the idea of comparing how well the evidence agrees with signal versus noise is more general.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. What changes in real observers?

    The ideal model is a baseline, not a claim that every observer literally behaves this way. Real data can violate several assumptions:

    | Ideal assumption | Plausible departure | What it can change |
    |---|---|---|
    | Equal Gaussian variance | Unequal variance or non-Gaussian evidence | ROC shape and the meaning of a single $d'$ |
    | One fixed criterion | Criterion drift, sequential bias, changing payoffs | Apparent slope, bias, and overdispersion |
    | Fixed sensitivity | Learning, fatigue, attention, adaptation | $d'$ across trials, blocks, or conditions |
    | No guesses or lapses | Stimulus-independent responses | Lower and upper asymptotes |
    | One mechanism | Mixtures of strategies, channels, or observer states | Curves that one sigmoid cannot capture |
    | Independent binomial trials | Serial dependence or extra variability | Uncertainty and goodness of fit |

    Each extension may explain structure that the simple model misses, but introduces parameters and potential trade-offs. Add complexity only when the design identifies it: inspect residuals, recover parameters in simulation, compare constrained models, and prefer out-of-sample prediction or principled model comparison over visual fit alone.

    For unequal variances, fit an unequal-variance binormal SDT model or analyze the zROC; a single-threshold rule may no longer be optimal. AUC remains a useful ranking measure. $A'$ is sometimes used as a nonparametric sensitivity index from one hit/false-alarm operating point, while $d_a$ summarizes unequal-variance binormal sensitivity. These indices answer slightly different questions and do not rescue a poorly identified design. Details and examples are in the [appendix](#appendix-selected-extensions).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Simulated yes/no data

    We now simulate the structure of an experiment before replacing it with real data. The next formula is a compact recipe for the simulation:

    \[
    X\mid S,x\sim N(d'(x),1),\qquad d'(x)=g x.
    \]

    **In plain language:** on a signal trial at contrast $x$, draw one internal evidence value $X$ from a bell curve whose peak is at $d'(x)$ and whose standard deviation is one. Increasing contrast moves that peak to the right; $g$ says how much it moves for each unit of contrast. Noise-only catch trials use a bell curve centered at zero. The observer says “signal” when the draw lands to the right of criterion $k$, just as in Figure 1.

    Figure 3 shows the result after many such trials. The black curve is the chance of saying “signal” predicted by the simulation recipe, the blue dots are simulated signal trials, and the red cross is the noise-only catch condition. A small lapse probability produces an occasional random response; ordinary internal and trial-sampling noise supply the remaining variability.
    """)
    return


@app.cell
def _(norm, np, plt):
    _rng = np.random.default_rng(604)
    contrast_levels = np.array([0.25, 0.5, 1, 2, 4, 8, 12, 16])
    n_per_level = 160
    n_catch = 240
    gain_true = 0.25
    criterion_true = 1.0
    lapse_true = 0.02
    true_dprime = gain_true * contrast_levels

    def _simulate_yes_count(mean_response, n_trials):
        _internal_response = _rng.normal(mean_response, 1.0, n_trials)
        _respond_yes = _internal_response > criterion_true
        _lapse = _rng.random(n_trials) < lapse_true
        _respond_yes[_lapse] = _rng.integers(0, 2, np.sum(_lapse)).astype(bool)
        return np.sum(_respond_yes)

    yes_counts = np.array(
        [_simulate_yes_count(_dprime, n_per_level) for _dprime in true_dprime]
    )
    false_alarm_count = _simulate_yes_count(0.0, n_catch)
    false_alarm_rate = false_alarm_count / n_catch

    # psignifit expects: stimulus level, number of target responses, trial count.
    psychometric_data = np.column_stack(
        [contrast_levels, yes_counts, np.full(len(contrast_levels), n_per_level)]
    )

    _contrast_grid = np.linspace(0, max(contrast_levels), 400)
    _p_yes = lapse_true / 2 + (1 - lapse_true) * norm.cdf(
        gain_true * _contrast_grid - criterion_true
    )
    plt.figure(figsize=(8, 3))
    plt.plot(_contrast_grid, _p_yes, color='black', label='Generating model')
    plt.scatter(
        contrast_levels,
        yes_counts / n_per_level,
        color='blue',
        label='Simulated signal trials',
    )
    plt.scatter(
        [0],
        [false_alarm_rate],
        color='red',
        marker='x',
        s=70,
        label='Noise-only catch trials',
    )
    plt.xlabel('Contrast (%)')
    plt.ylabel('Chance of a \"signal\" response')
    plt.title('Figure 3. Simulated yes/no responses across contrast')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.show()

    print(
        f'Noise-only catch trials: {false_alarm_count} of {n_catch} produced '
        'a "signal" response.'
    )
    print('Contrast | signal responses / trials')
    for _contrast, _yes in zip(contrast_levels, yes_counts):
        print(f'{_contrast:8.2f} | {_yes:3d} / {n_per_level}')
    return (
        contrast_levels,
        criterion_true,
        false_alarm_count,
        false_alarm_rate,
        gain_true,
        lapse_true,
        n_catch,
        n_per_level,
        psychometric_data,
        true_dprime,
        yes_counts,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Fit a psychometric function

    We use a specialist toolbox rather than writing an optimizer:

    - [psignifit](https://psignifit.readthedocs.io/en/latest/) provides Bayesian psychometric-function estimation, uncertainty, lapse/guess parameters, and overdispersion in Python. Install with <code>pip install psignifit</code>.
    - [Palamedes](https://www.palamedestoolbox.org/) provides a broad psychophysics toolbox, including maximum-likelihood and Bayesian psychometric fits and joint SDT/psychometric models. Its MATLAB toolbox is currently broader than the growing Python port.

    Common curve choices are:

    - **Cumulative Gaussian (<code>norm</code>):** $P(\text{response})=\Phi(a+bx)$. **In plain language:** the chance of the response is the accumulated area under a bell curve. Parameter $a$ moves the S-shaped curve left or right, while $b$ controls how quickly it rises. This choice follows naturally when Gaussian internal evidence moves linearly with the stimulus and the observer uses a fixed criterion, as in Figure 1.
    - **Logistic:** $\log[p/(1-p)]=a+bx$. **In plain language:** each step in stimulus adds a fixed amount to the log-odds of responding “signal.” The result looks much like a cumulative Gaussian but has somewhat heavier tails; again, $a$ sets horizontal position and $b$ sets steepness.
    - **Weibull:** $P(\text{detect})=1-\exp[-(x/\alpha)^\beta]$. **In plain language:** as positive intensity grows, the chance that every possible detection event fails shrinks. Parameter $\alpha$ sets the intensity scale and $\beta$ controls shape. This curve is common for contrast detection and becomes a Gumbel-shaped sigmoid on log intensity.
    - **Other shapes:** Gumbel variants accommodate asymmetric tails and Student-$t$ links allow heavier tails.

    The fit below uses the signal-trial hit counts; the catch-trial responses enter the subsequent $d'$ calculation. A joint SDT fit would model both together. The next cell selects the curve. For Weibull, psignifit expects the positive stimulus levels to be transformed to log space. Here symmetric stimulus-independent errors were simulated, so <code>equal asymptote</code> is an appropriate constrained yes/no model; use <code>yes/no</code> when the lower and upper asymptotes should vary independently.
    """)
    return


@app.cell
def _():
    # Try "logistic", or "weibull" (which is fit on log10 contrast).
    sigmoid_name = "norm"
    return (sigmoid_name,)


@app.cell
def _(np, ps, psp, psychometric_data, sigmoid_name, plt):
    _fit_data = psychometric_data.copy()
    _x_label = 'Contrast (%)'
    if sigmoid_name == 'weibull':
        _fit_data[:, 0] = np.log10(_fit_data[:, 0])
        _x_label = 'log10 contrast (%)'

    psychometric_fit = ps.psignifit(
        _fit_data,
        experiment_type='equal asymptote',
        sigmoid=sigmoid_name,
    )
    assert np.all(np.isfinite(_fit_data))
    assert np.all((_fit_data[:, 1] >= 0) & (_fit_data[:, 1] <= _fit_data[:, 2]))

    _figure, _axis = plt.subplots(figsize=(8, 3.5))
    psp.plot_psychometric_function(
        psychometric_fit,
        ax=_axis,
        x_label=_x_label,
        y_label='Proportion of signal responses',
    )
    _axis.set_title(f'Figure 4. Psychometric fit ({sigmoid_name} link)')
    _axis.grid(True)
    plt.show()
    print('Posterior parameter estimates:')
    print(psychometric_fit.parameter_estimate)
    return (psychometric_fit,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. From SDT to stimulus-dependent psychometric functions

    Figure 4 fits a smooth curve through the observed responses from Figure 3. At each stimulus level $x$, SDT supplies a hit rate and a false-alarm rate. With unit variance and a fixed criterion,

    \[
    H(x)=\Phi(d'(x)-k),\qquad F=\Phi(-k).
    \]

    **In plain language:** $H(x)$ is the blue area to the right of the decision line in Figure 1 after contrast $x$ has moved the blue curve. $F$ is the red area to the right of that line; in this simple model it stays fixed because the noise curve and criterion do not move. The symbol $\Phi$ converts a distance on the horizontal axis into the area under a bell curve to its left.

    Therefore the psychometric curve is determined jointly by the decision rule and the transducer that maps the physical stimulus to $d'$. If $d'(x)=g x$, every step in contrast moves the signal peak by the same amount and the resulting response curve is cumulative Gaussian. For positive contrast, a common extension is

    \[
    d'(x)=(g x)^p,
    \]

    **In plain language:** $g$ controls the overall horizontal separation of the signal and noise peaks, while $p$ allows that separation to grow faster or slower than a straight line. Figure 5 plots this peak separation, measured in standard-deviation units, against contrast.
    """)
    return


@app.cell
def _(
    contrast_levels,
    false_alarm_count,
    gain_true,
    n_catch,
    n_per_level,
    norm,
    np,
    plt,
    true_dprime,
    yes_counts,
):
    # Log-linear correction avoids infinite z-scores for observed rates of 0 or 1.
    _hit_rate_corrected = (yes_counts + 0.5) / (n_per_level + 1)
    _false_alarm_rate_corrected = (false_alarm_count + 0.5) / (n_catch + 1)
    empirical_dprime = norm.ppf(_hit_rate_corrected) - norm.ppf(
        _false_alarm_rate_corrected
    )
    assert np.all(np.isfinite(empirical_dprime))
    assert len(empirical_dprime) == len(contrast_levels)

    plt.figure(figsize=(8, 3))
    plt.scatter(
        contrast_levels,
        empirical_dprime,
        label="Estimated d' from hit and false-alarm areas",
    )
    plt.plot(
        contrast_levels,
        true_dprime,
        color='black',
        label=f"Generating relation: d' = {gain_true} x",
    )
    plt.xlabel('Contrast (%)')
    plt.ylabel("d'")
    plt.title("Figure 5. Sensitivity grows with contrast")
    plt.grid(True)
    plt.legend()
    plt.show()
    return (empirical_dprime,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In a Carrasco-style multi-condition analysis, the key object can be $d'(x)$ rather than proportion correct alone. Fit the signal and noise responses jointly, then ask whether an experimental manipulation changes gain, exponent, criterion, or lapse rate. Shared-parameter models are often more informative than fitting every condition independently.

    For orientation discrimination, replace contrast with signed or absolute orientation difference and choose a transducer appropriate to that geometry. If false-alarm rate or criterion changes with stimulus level or condition, a curve fitted only to hit rate confounds sensitivity and bias; catch trials, both response categories, confidence ratings, or a joint SDT model help separate them.

    The [appendix](#appendix-selected-extensions) retains useful side paths without interrupting this main sequence.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Appendix: selected extensions

    [Back to the main model](#1-the-ideal-equal-variance-model)

    ### A1. Reading a two-by-two table in plain language

    Imagine a diagnostic study with the following counts:

    | What is actually true? | Test says “positive” | Test says “negative” | Total |
    |---|---:|---:|---:|
    | Disease present | 40 hits | 10 misses | 50 people |
    | Disease absent | 15 false alarms | 135 correct rejections | 150 people |
    | **Total** | **55 positive tests** | **145 negative tests** | **200 people** |

    - **Sensitivity**, also called **recall** or the **true-positive rate**, asks: *if a person has the disease, what is the chance that the test is positive?* In the table, look across the 50 people with disease: the test finds 40 and misses 10.
    - **Specificity**, also called the **true-negative rate**, asks: *if a person does not have the disease, what is the chance that the test is negative?* Look across the 150 people without disease: the test correctly clears 135 and falsely alarms on 15.
    - **Precision**, also called **positive predictive value (PPV)**, reverses the question: *if the test is positive, what is the chance that the person has the disease?* Look down the 55 positive tests: 40 come from people with disease and 15 do not.

    Sensitivity/recall and specificity start with what is **actually true** and look across a row. Precision/PPV starts with what the **test said** and looks down a column. This is why PPV changes when disease becomes more or less common, even if the test's sensitivity and specificity do not change.

    In SDT language, sensitivity/recall is the hit rate, while one minus specificity is the false-alarm rate. Those two quantities locate a point on the ROC plot. The same table applies to a single-interval A-versus-B task once one category is called “positive.” It does not have the same diagnostic meaning in a standard 2I-2AFC task, because every trial contains a signal somewhere.

    There is an important vocabulary trap: clinical **sensitivity** means the hit rate, whereas SDT authors also use “sensitivity” informally for the discriminability index $d'$. They are not the same quantity. Moving the decision threshold changes clinical sensitivity/recall, but it does not change the model's underlying $d'$.

    The code below uses the counts directly. A small finite-sample correction is used only when computing $d'$ and criterion so that an observed “never” or “always” response does not produce an infinite z-score.
    """)
    return


@app.cell
def _(norm, np):
    # Example observed counts: 50 signal trials and 150 noise trials
    _hits, _misses = 40, 10
    _false_alarms, _correct_rejections = 15, 135

    _hit_rate = _hits / (_hits + _misses)
    _false_alarm_rate = _false_alarms / (_false_alarms + _correct_rejections)
    _specificity = _correct_rejections / (
        _false_alarms + _correct_rejections
    )
    _precision = _hits / (_hits + _false_alarms)

    # Log-linear correction gives finite z-scores even when a raw rate is 0 or 1.
    _hit_rate_corrected = (_hits + 0.5) / (_hits + _misses + 1)
    _false_alarm_rate_corrected = (_false_alarms + 0.5) / (
        _false_alarms + _correct_rejections + 1
    )
    _z_hit = norm.ppf(_hit_rate_corrected)
    _z_false_alarm = norm.ppf(_false_alarm_rate_corrected)
    _d_prime_observed = _z_hit - _z_false_alarm
    _criterion_observed = -0.5 * (_z_hit + _z_false_alarm)
    assert np.isfinite(_d_prime_observed)
    assert np.isfinite(_criterion_observed)
    assert _d_prime_observed > 0

    print(
        f'Sensitivity / recall: among {_hits + _misses} people with disease, '
        f'{_hits} test positive and {_misses} test negative.'
    )
    print(
        f'Specificity: among {_false_alarms + _correct_rejections} people '
        f'without disease, {_correct_rejections} test negative and '
        f'{_false_alarms} test positive.'
    )
    print(
        f'Precision / PPV: among {_hits + _false_alarms} positive tests, '
        f'{_hits} come from people with disease and {_false_alarms} do not.'
    )
    print(f"Corrected d': {_d_prime_observed:.3f}; corrected c: {_criterion_observed:.3f}")
    assert np.isclose(_specificity, 1 - _false_alarm_rate)
    assert 0 < _precision < 1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A2. Unequal variance and alternative sensitivity indices

    If $\sigma_S\ne\sigma_N$, $\Phi^{-1}(H)-\Phi^{-1}(F)$ varies with the criterion and is no longer a single criterion-free $d'$. The z-transformed ROC instead obeys

    \[
    \Phi^{-1}(H)=\frac{\mu_S-\mu_N}{\sigma_S}
      +\frac{\sigma_N}{\sigma_S}\Phi^{-1}(F),
    \]

    **In plain language:** $\Phi^{-1}$ redraws each ROC axis using a standard-deviation ruler. On those rulers the ROC becomes a straight line. Its tilt is the noise spread divided by the signal spread, $\sigma_N/\sigma_S$; equal-width bell curves give a tilt of one. If one curve is wider than the other, the simple peak-distance interpretation of $d'$ depends on which spread is used.

    The log-likelihood ratio is then a curved rather than a straight function of evidence, and it may cross the decision cutoff twice. In picture terms, “signal” might be favored only over a middle or outer region instead of everywhere to the right of one vertical line. The ROC, AUC, and general rule of choosing the more plausible source remain valid, but the equal-variance shortcuts above do not.

    With several criteria or confidence ratings, estimate the full ROC or a binormal zROC. With only one $(H,F)$ point, $A'$ is a common nonparametric descriptive index; $d_a$ is a model-based unequal-variance alternative. Neither has exactly the same interpretation as equal-variance $d'$, so report the model and assumptions with the number.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A3. Why ROC experiments mattered

    A single hit/false-alarm pair cannot tell us the shape of an ROC. Classic experiments therefore held the signal strength roughly fixed while moving the observer's decision criterion—for example by changing prior probabilities or payoffs—or asked for confidence ratings, which act like several criteria collected in one session. This separates a change in willingness to say “signal” from a change in sensitivity.

    The continuous-evidence SDT model predicts a smooth, bowed ROC. A simple high-threshold model predicts straight-line structure because observations below its sensory threshold are treated as indistinguishable. Tanner and Swets's visual experiments found that yes/no and forced-choice estimates of detectability were mutually consistent, and later experiments found ROC patterns and above-chance second choices that contradicted the simplest high-threshold account. A rating-scale visual experiment by Nachmias and Steinman gave substantially stronger support to statistical-decision theory than to a two-state low-threshold theory.

    This evidence is important but not a proof that every internal distribution is Gaussian. Luce's **low-threshold** model permits noise alone to cross a sensory threshold and predicts two joined ROC segments; it could explain some results that rejected the simpler high-threshold model. Krantz later showed that two-state low-threshold theory could be rejected by some data, while a three-state low-and-high-threshold model could closely resemble unequal-variance Gaussian SDT. ROC experiments therefore support graded evidence and movable decision criteria over the simplest all-or-none account, while detailed model discrimination requires richer data.

    Sources: [Tanner & Swets (1954)](https://doi.org/10.1037/h0058700); [Swets, Tanner, & Birdsall (1961)](https://doi.org/10.1037/h0040547); [Luce (1963)](https://doi.org/10.1037/h0039723); [Nachmias & Steinman (1963)](https://doi.org/10.1364/JOSA.53.001206); [Krantz (1969)](https://doi.org/10.1037/h0027238).

    ### A4. Two-interval, two-alternative forced choice

    In this task each trial contains two observations: one from A (the signal) and one from B (the noise). The observer knows A is present and chooses its interval or location. An example is choosing which of two brief intervals contained a faint tone. Unlike yes/no detection, “signal absent” is not an option.

    Figure 6 plots the comparison value $D$: observation 1 minus observation 2. The blue curve describes trials with A in interval 1, the red curve describes trials with A in interval 2, and the black dashed line is the boundary between the two choices.
    """)
    return


@app.cell
def _(diff_x, loc1, loc2, math, norm, np, plt, scale1, scale2):
    # Use D = observation in interval 1 minus observation in interval 2.
    # A is the signal distribution and B is the noise distribution.
    _delta = loc2 - loc1
    _difference_sd = math.sqrt(scale1 ** 2 + scale2 ** 2)
    pdf_diff1 = norm.pdf(diff_x, _delta, _difference_sd)
    pdf_diff2 = norm.pdf(diff_x, -_delta, _difference_sd)
    plt.figure(figsize=(10, 4))
    plt.plot(diff_x, pdf_diff1, color='blue', label='A in interval 1: mean +Δ')
    plt.plot(diff_x, pdf_diff2, color='red', label='A in interval 2: mean −Δ')
    _right_of_zero = diff_x >= 0
    _left_of_zero = diff_x <= 0
    plt.fill_between(
        diff_x[_right_of_zero],
        0,
        pdf_diff1[_right_of_zero],
        color='blue',
        alpha=0.18,
        label='Correct area when A is in interval 1',
    )
    plt.fill_between(
        diff_x[_left_of_zero],
        0,
        pdf_diff2[_left_of_zero],
        color='red',
        alpha=0.18,
        label='Correct area when A is in interval 2',
    )
    plt.axvline(0, color='black', linestyle='--', label='Choose interval 1 if D > 0')
    _difference_arrow_height = 1.08 * max(np.max(pdf_diff1), np.max(pdf_diff2))
    plt.annotate(
        '',
        xy=(_delta, _difference_arrow_height),
        xytext=(-_delta, _difference_arrow_height),
        arrowprops={'arrowstyle': '<->', 'color': 'purple', 'linewidth': 2},
    )
    plt.text(
        0,
        1.02 * _difference_arrow_height,
        '2Δ between conditional means',
        color='purple',
        ha='center',
        va='bottom',
    )
    plt.title('Figure 6. 2I-2AFC decision variable: D = X₁ − X₂')
    plt.xlabel('Difference D')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.28 * _difference_arrow_height)
    plt.tight_layout()
    plt.show()
    _percentage_correct = 1 - norm.cdf(0, _delta, _difference_sd)
    print(f'Prob. of being correct in 2-I, 2-AFC task: {_percentage_correct:.4f}')
    _dprime_theoretical = (loc2 - loc1) / scale1
    assert np.isclose(
        _percentage_correct,
        norm.cdf(_dprime_theoretical / np.sqrt(2)),
        atol=1e-12,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let the mean difference between A and B in a single observation be $\Delta=\mu_A-\mu_B$, and let each observation have variance $\sigma^2$. Define the comparison variable as $D=X_1-X_2$.

    - When A is in interval 1, the mean of $D$ is $+\Delta$; when A is in interval 2, it is $-\Delta$. The two distributions of $D$ are therefore separated by $2\Delta$.
    - Because the two observations are independent, the variance of their difference is the sum of their variances: $\sigma^2+\sigma^2=2\sigma^2$. Its standard deviation is therefore $\sqrt{2}\sigma$.
    - The choice boundary is zero. Either mean is only $\Delta$ away from that boundary, so its distance in standard-deviation units is $\Delta/(\sqrt{2}\sigma)=d'/\sqrt{2}$.

    That is the simple origin of the $1/\sqrt{2}$ in proportion correct:

    \[
    P(\text{correct in 2I-2AFC})=\Phi\left(\frac{d'}{\sqrt{2}}\right).
    \]

    **In plain language:** proportion correct is the blue area to the right of the black choice line in Figure 6 (and, by symmetry, the red area to its left). Independent noise from two observations makes the comparison distribution $\sqrt{2}$ times wider than either single-observation distribution.

    Looking instead at the full separation between the “A in interval 1” and “A in interval 2” distributions gives $2\Delta/(\sqrt{2}\sigma)=\sqrt{2}d'$. These are the same geometry viewed in two ways: distance from either mean to the choice boundary versus distance between the two conditional means.

    Under the standard assumptions—independent observations, the same evidence scale in both tasks, and a rule that chooses the larger observation—the **area theorem** described by Green and Swets says that 2I-2AFC proportion correct also equals the area under the yes/no ROC:

    \[
    P(X_A>X_B)=P(\text{correct in 2I-2AFC})=\mathrm{AUC}.
    \]

    **In plain language:** $P(X_A>X_B)$ means the chance that one randomly drawn A observation produces more evidence than one randomly drawn B observation. That is exactly the chance of choosing the correct member of a pair.

    This also explains the connection to the Mann–Whitney test. Empirical AUC counts how often an A observation ranks above a B observation across all A–B pairs (with half credit for ties). The Mann–Whitney $U$ statistic counts the same pairwise orderings. The test uses that count to ask whether the groups differ; AUC uses it to describe discrimination on a chance-to-perfect scale. They are closely connected, but an AUC value is an effect-size description rather than the Mann–Whitney significance test itself ([Green & Swets, 1966](https://books.google.com/books?id=fHR9AAAAMAAJ); [Bamber, 1975](https://doi.org/10.1016/0022-2496(75)90001-2)).

    There is no freely movable yes/no criterion in the ideal comparison rule, but order effects, unequal interval noise, or a preference for one interval can still introduce bias and break the area-theorem correspondence.

    In the balanced yes/no task, accuracy at the optimal criterion is $\Phi(d'/2)$. **In plain language:** the midpoint criterion in Figure 1 lies halfway between the peaks, so it is $d'/2$ standard deviations from either peak; the correct-response chance is the bell-curve area on the correct side of that line. Compare this with Figure 6. Vary the location parameters; vary the scales only after considering the unequal-variance note above.

    ### A5. Unequal priors and error costs

    Next, let us consider a situation where the signal and noise samples do not have an equal chance of appearing on a given trial. For example, many more trials contain the noise sample than the signal sample. Outside of laboratory experiments, unequal proportions is likely much more common. Think of some examples.

    The core idea is simple: say “signal” when the observed evidence is sufficiently more plausible under the signal curve than under the noise curve. “Sufficiently” depends on how often each trial type occurs and how costly each kind of mistake is. The following formula keeps that bookkeeping explicit:

    \[
    \Lambda(x)=\frac{f_S(x)}{f_N(x)}>\beta_{\mathrm{optimal}}
    =\frac{P(N)}{P(S)}
      \frac{C_{FA}-C_{CR}}{C_M-C_H},
    \]

    **In plain language:** $\Lambda(x)$ is the height of the signal curve divided by the height of the noise curve at the observed evidence $x$. The right-hand side is the cutoff: the first ratio reflects how common noise and signal are, and the second reflects the relative consequences of the four outcomes. $C_H$, $C_M$, $C_{FA}$, and $C_{CR}$ denote the costs of a hit, miss, false alarm, and correct rejection.

    With equally costly errors and equally valuable correct decisions, only the base rates remain. In Figure 7A, noise is three times as common as signal, so each curve is scaled by its frequency before finding their crossing. The black point marks the accuracy-maximizing threshold. Figure 7B shows that accuracy peaks at that threshold. It is not generally the crossing of the original unweighted curves or the midpoint between their peaks.
    """)
    return


@app.cell
def _(
    fpr_theoretical,
    loc1,
    loc2,
    np,
    pdf1,
    pdf2,
    plt,
    scale1,
    scale2,
    thresholds,
    tpr_theoretical,
    x,
):
    plt.figure(figsize=(12, 4))
    pdf11 = 3 * pdf1 / 4
    pdf22 = 1 * pdf2 / 4
    _percor_theoretical = (np.array(tpr_theoretical) + 3 * (1 - np.array(fpr_theoretical))) / 4
    plt.subplot(1, 2, 1)
    plt.plot(x, pdf11, label='Noise curve × its 3-in-4 chance', color='red')
    plt.plot(x, pdf22, label='Signal curve × its 1-in-4 chance', color='blue')
    _idx = np.argwhere(np.diff(np.sign(pdf22 - pdf11))).flatten()
    plt.plot(x[_idx], pdf22[_idx], 'ko')
    intersect_point_1 = x[_idx]
    _analytic_intersection = (
        (loc1 + loc2) / 2
        + scale1 ** 2 * np.log(3) / (loc2 - loc1)
    )
    assert np.allclose(
        intersect_point_1,
        _analytic_intersection,
        atol=2 * (x[1] - x[0]),
    )
    assert np.min(
        np.abs(thresholds[np.argmax(_percor_theoretical)] - intersect_point_1)
    ) < 2 * (thresholds[1] - thresholds[0])
    plt.title('Figure 7A. Prior-weighted response densities')
    plt.xlabel('x')
    plt.ylabel('Prior-weighted density')
    plt.legend(fontsize=8, loc='upper right')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(thresholds, _percor_theoretical)
    plt.axvline(intersect_point_1, color='black', linestyle='--')
    plt.xlabel('Threshold')
    plt.ylabel('Chance of a correct response')
    plt.title('Figure 7B. Accuracy as the threshold moves')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Try to understand the plot on the right, and explain why the left and right end of this plot does not lie at 0.5 unlike the situation where the two trial types occurred equally often.

    ### A6. A non-Gaussian example

    Figure 8 replaces the symmetric bell curves in Figure 1 with skewed evidence distributions. See which properties from above are retained and which are affected. In general, each pair of distributions requires a fresh assessment. Here the skew-normal shape parameter <code>a</code> controls which way each curve leans and how strongly it is skewed.
    """)
    return


@app.cell
def _(np, plt, skewnorm):
    # Generate x values
    x_1 = np.linspace(0, 100, 1000)
    # Parameters for the skew-normal noise distribution
    a1 = 2  # shape/skew parameter; its sign determines the skew direction
    loc1_1 = 35  # location parameter (not generally the mean)
    scale1_1 = 10  # scale parameter (not generally the standard deviation)
    # Parameters for the skew-normal signal distribution
    a2 = -2
    loc2_1 = 65
    scale2_1 = 10
    pdf1_1 = skewnorm.pdf(x_1, a1, loc1_1, scale1_1)
    pdf2_1 = skewnorm.pdf(x_1, a2, loc2_1, scale2_1)
    plt.figure(figsize=(10, 6))
    # Compute the PDF values
    plt.plot(x_1, pdf1_1, label=f'Skew-normal noise\na={a1}, loc={loc1_1}, scale={scale1_1}', color='red')
    plt.plot(x_1, pdf2_1, label=f'Skew-normal signal\na={a2}, loc={loc2_1}, scale={scale2_1}', color='blue')
    plt.title('Figure 8. A non-Gaussian pair of evidence distributions')
    # Plot the skew normal distributions
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return a1, a2, loc1_1, loc2_1, scale1_1, scale2_1, x_1


@app.cell
def _(a1, a2, auc, loc1_1, loc2_1, np, scale1_1, scale2_1, skewnorm, x_1):
    numsamples = 10000
    _rng = np.random.default_rng(603)
    simulated_values1 = skewnorm.rvs(
        a1, loc1_1, scale1_1, size=numsamples, random_state=_rng
    )
    simulated_values2 = skewnorm.rvs(
        a2, loc2_1, scale2_1, size=numsamples, random_state=_rng
    )
    correct_choices = np.sum(simulated_values2 > simulated_values1)
    _percentage_correct = correct_choices / numsamples
    print(f'Prob. correct based on the simulation: {_percentage_correct:.4f}')
    fpr_theoretical_1 = []
    tpr_theoretical_1 = []
    thresholds_1 = np.linspace(min(x_1), max(x_1), 100)
    for _threshold in thresholds_1:
        tpr_theoretical_1.append(1 - skewnorm.cdf(_threshold, a2, loc2_1, scale2_1))
        fpr_theoretical_1.append(1 - skewnorm.cdf(_threshold, a1, loc1_1, scale1_1))
    _roc_auc = auc(
        np.r_[1.0, fpr_theoretical_1, 0.0],
        np.r_[1.0, tpr_theoretical_1, 0.0],
    )
    print(f'Area Under the Curve (AUC): {_roc_auc:.4f}')
    assert 0 < _percentage_correct < 1
    assert 0 < _roc_auc < 1
    assert np.isclose(_percentage_correct, _roc_auc, atol=0.02)
    print('All internal consistency checks passed.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A7. Selected references

    - Tanner, W. P., Jr., & Swets, J. A. (1954). [A decision-making theory of visual detection](https://doi.org/10.1037/h0058700). *Psychological Review, 61*, 401–409.
    - Swets, J. A., Tanner, W. P., Jr., & Birdsall, T. G. (1961). [Decision processes in perception](https://doi.org/10.1037/h0040547). *Psychological Review, 68*, 301–340.
    - Green, D. M., & Swets, J. A. (1966). [*Signal Detection Theory and Psychophysics*](https://books.google.com/books?id=fHR9AAAAMAAJ). Wiley.
    - Luce, R. D. (1963). [A threshold theory for simple detection experiments](https://doi.org/10.1037/h0039723). *Psychological Review, 70*, 61–79.
    - Nachmias, J., & Steinman, R. M. (1963). [Study of absolute visual detection by the rating-scale method](https://doi.org/10.1364/JOSA.53.001206). *Journal of the Optical Society of America, 53*, 1206–1213.
    - Krantz, D. H. (1969). [Threshold theories of signal detection](https://doi.org/10.1037/h0027238). *Psychological Review, 76*, 308–324.
    - Bamber, D. (1975). [The area above the ordinal dominance graph and the area below the receiver operating characteristic graph](https://doi.org/10.1016/0022-2496(75)90001-2). *Journal of Mathematical Psychology, 12*, 387–415.
    """)
    return


if __name__ == "__main__":
    app.run()
