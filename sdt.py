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
    # Compute the PDF values
    plt.figure(figsize=(10, 3))
    plt.plot(x, pdf1, label=f'Noise Distribution loc={loc1}, scale={scale1}', color='red')
    plt.plot(x, pdf2, label=f'Signal Distribution loc={loc2}, scale={scale2}', color='blue')
    # Plot the distributions
    _idx = np.argwhere(np.diff(np.sign(pdf2 - pdf1))).flatten()
    plt.plot(x[_idx], pdf2[_idx], 'ko')
    intersect_point = x[_idx]
    plt.title('Normal Distributions of response variable when signal is present/absent (yes/no)')
    plt.xlabel('x')
    plt.ylabel('PDF')
    plt.legend()
    plt.grid(True)
    plt.show()
    return diff_x, intersect_point, loc1, loc2, pdf1, pdf2, scale1, scale2, x


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One way to do this task is to set a value of a threshold. If the response variable is greater than this value, we say the signal is present, and if not, we say it is absent.

    More generally, response variables can be multidimensional (in which case the acceptance region may also be multidimensional). The acceptance region can also be much more complex, even in 1 dimension, if the underlying response variable distributions are. Think about these..

    But for now, let us consider the response variable distributions above, and then see what happens as the threshold is varied. Try to predict what happens when you use a very low threshold, a threshold at the intersection point, and a very high threshold..
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
    plt.ylabel('Prob. correct')
    plt.title('How prob. correct depends on threshold')
    plt.subplot(1, 2, 2)
    _roc_fpr_simulation = np.r_[1.0, fpr_simulation, 0.0]
    _roc_tpr_simulation = np.r_[1.0, tpr_simulation, 0.0]
    plt.plot(_roc_fpr_theoretical, _roc_tpr_theoretical, label='Theoretical ROC Curve')
    plt.plot(_roc_fpr_simulation, _roc_tpr_simulation, label='Simulation-based ROC Curve', linestyle='--')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='-.', label='Chance level')
    plt.xlabel('False Positive Rate = 1 - Specificity')
    plt.ylabel('True Positive Rate = Sensitivity')
    plt.title('Theoretical vs Simulation-based ROC Curve')
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

    For a criterion $k$, let $H=P(x>k\mid\text{signal})$ be the hit rate and $F=P(x>k\mid\text{noise})$ the false-alarm rate. For the equal-variance Gaussian model used above,

    \[
    d' = \Phi^{-1}(H)-\Phi^{-1}(F)
       = \frac{\mu_S-\mu_N}{\sigma},
    \]

    where $d'$ measures sensitivity: larger values mean better separation, independently of the chosen criterion. Response bias is commonly summarized by

    \[
    c=-\frac{1}{2}\left[\Phi^{-1}(H)+\Phi^{-1}(F)\right]
      =\frac{k-(\mu_S+\mu_N)/2}{\sigma}.
    \]

    Thus $c=0$ is unbiased, $c>0$ is conservative (fewer signal responses), and $c<0$ is liberal. The likelihood ratio at the criterion is

    \[
    \beta=\frac{f_S(k)}{f_N(k)}=e^{d'c}.
    \]

    With equal priors and equal error costs, the optimal criterion has $c=0$ and $\beta=1$. These simple formulas for $d'$, $c$, and $\beta=e^{d'c}$ assume equal-variance Gaussian distributions; the likelihood-ratio decision rule itself is more general.
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

    We now simulate the structure of an experiment before replacing it with real data. On signal trials, contrast increases the mean internal response linearly:

    \[
    X\mid S,x\sim N(d'(x),1),\qquad d'(x)=g x.
    \]

    Noise-only catch trials have mean zero. The observer responds "signal" when $X>k$. A small lapse probability produces a random response; ordinary internal and binomial sampling noise supply the remaining variability. The code is visible so every assumption is explicit.
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
    plt.ylabel('Proportion \"signal\" responses')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.show()

    print(f'Catch trials: {false_alarm_count}/{n_catch}; F = {false_alarm_rate:.3f}')
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

    - **Cumulative Gaussian (<code>norm</code>):** follows directly when internal evidence has additive Gaussian noise, its mean changes linearly with the stimulus, and the observer uses a fixed criterion: $P(\text{response})=\Phi(a+bx)$.
    - **Logistic:** follows if the latent decision noise is logistic, or equivalently if response log-odds are linear in stimulus: $\log[p/(1-p)]=a+bx$. It resembles the cumulative Gaussian but has somewhat heavier tails.
    - **Weibull:** for positive intensities, $P(\text{detect})=1-\exp[-(x/\alpha)^\beta]$ follows from a Poisson-like detection or probability-summation account in which detection occurs when at least one effective event succeeds. It is common for contrast detection and becomes a Gumbel-shaped sigmoid on log intensity.
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
        y_label='P(\"signal\" response | signal trial)',
    )
    _axis.set_title(f'psignifit: {sigmoid_name} link')
    _axis.grid(True)
    plt.show()
    print('Posterior parameter estimates:')
    print(psychometric_fit.parameter_estimate)
    return (psychometric_fit,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. From SDT to stimulus-dependent psychometric functions

    At each stimulus level $x$, SDT supplies a hit rate and a false-alarm rate. With unit variance and a fixed criterion,

    \[
    H(x)=\Phi(d'(x)-k),\qquad F=\Phi(-k).
    \]

    Therefore the psychometric curve $H(x)$ is determined jointly by the decision rule and the transducer that maps the physical stimulus to sensitivity. If $d'(x)=g x$, $H(x)$ is cumulative Gaussian in $x$. For positive contrast, a common extension is

    \[
    d'(x)=(g x)^p,
    \]

    where $g$ controls sensitivity/gain and $p$ controls nonlinear growth. This is one route from the elementary SDT picture to a contrast psychometric function.
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
    plt.scatter(contrast_levels, empirical_dprime, label="Estimated from H and F")
    plt.plot(
        contrast_levels,
        true_dprime,
        color='black',
        label=f"Generating relation: d' = {gain_true} x",
    )
    plt.xlabel('Contrast (%)')
    plt.ylabel("d'")
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

    ### A1. Estimating SDT measures from response counts

    | Actual trial | Respond "signal" | Respond "noise" |
    |---|---:|---:|
    | Signal | Hit | Miss |
    | Noise | False alarm | Correct rejection |

    Estimate $H=\text{hits}/(\text{hits}+\text{misses})$ and $F=\text{false alarms}/(\text{false alarms}+\text{correct rejections})$, then substitute them into the formulas above. If an observed rate is 0 or 1, its z-score is infinite. A common finite-sample correction is

    \[
    \widetilde H=\frac{\text{hits}+0.5}{\text{signal trials}+1},\qquad
    \widetilde F=\frac{\text{false alarms}+0.5}{\text{noise trials}+1}.
    \]

    Do not confuse hit rate with precision (positive predictive value):

    \[
    \text{precision}=\frac{\text{hits}}{\text{hits}+\text{false alarms}}.
    \]

    Hit and false-alarm rates condition on the true trial type; precision conditions on the response and therefore changes with the signal base rate.
    """)
    return


@app.cell
def _(norm, np):
    # Example observed counts: 50 signal trials and 150 noise trials
    _hits, _misses = 40, 10
    _false_alarms, _correct_rejections = 15, 135

    _hit_rate = _hits / (_hits + _misses)
    _false_alarm_rate = _false_alarms / (_false_alarms + _correct_rejections)
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

    print(f'Hit rate: {_hit_rate:.3f}; false-alarm rate: {_false_alarm_rate:.3f}')
    print(f"Corrected d': {_d_prime_observed:.3f}; corrected c: {_criterion_observed:.3f}")
    print(f'Precision: {_precision:.3f}')
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

    so its slope is $\sigma_N/\sigma_S$, rather than 1. The log-likelihood ratio is then quadratic in $x$, and the likelihood ratio may cross the optimal $\beta$ twice; consequently, the optimal decision region need not be the simple rule $x>k$. The ROC, AUC, and general likelihood-ratio rule remain valid, but the equal-variance formulas above do not.

    With several criteria or confidence ratings, estimate the full ROC or a binormal zROC. With only one $(H,F)$ point, $A'$ is a common nonparametric descriptive index; $d_a$ is a model-based unequal-variance alternative. Neither has exactly the same interpretation as equal-variance $d'$, so report the model and assumptions with the number.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A3. Two-interval, two-alternative forced choice

    The vertical line above in the left column is drawn at the intersection point of the two distributions from above.

    Next, let us consider a different task, where on each trial, one gets two-samples,  one from the signal and one from the noise, and one has to determine which one is the one from the signal. An example would be a police line-up with only 1 "foil". Or searching for your friend in a crowd. Etc. Any situation where you know the signal is present, but do not know which of the samples has the signal. (Aside: The relationship of this situation to the one above is like the relationship of a two-sample t-test to a single sample t-test).
    """)
    return


@app.cell
def _(diff_x, loc1, loc2, math, norm, np, plt, scale1, scale2):
    #Now consider a two-interval two-alternative forced choice task
    #Each trial gives two samples,  and one has to decide which is signal and which is noise
    #Let us look at the distribtion of the difference for (signal-noise) and for (noise-signal);
    #the latter is just the distribution of -1 multiplied by the former
    pdf_diff1 = norm.pdf(diff_x, loc2 - loc1, math.sqrt(scale1 ** 2 + scale2 ** 2))
    pdf_diff2 = norm.pdf(diff_x, loc1 - loc2, math.sqrt(scale1 ** 2 + scale2 ** 2))  #variance of difference is sum of individual variances
    plt.plot(diff_x, pdf_diff1, color='blue', label='Signal - Noise')
    plt.plot(diff_x, pdf_diff2, color='red', label='Noise - Signal')
    plt.title('Difference of Normal Distributions')
    plt.xlabel('x')
    plt.ylabel('PDF')
    plt.legend()
    plt.grid(True)
    plt.show()
    _percentage_correct = 1 - norm.cdf(0, loc2 - loc1, math.sqrt(scale1 ** 2 + scale2 ** 2))
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
    See how there is no threshold setting here, and no role of response caution. However, there are other confounds and biases, like order effects, tendency to prefer one or other sample etc.

    Under the standard assumptions that the two observations are independent draws from the same signal and noise distributions, and that the observer chooses the larger value of the same decision variable, probability correct in a 2I, 2-AFC task equals the area under the ROC curve from the yes/no experiment. For equal-variance Gaussian distributions,

    \[
    P(\text{correct in 2AFC})=\mathrm{AUC}=\Phi\left(\frac{d'}{\sqrt{2}}\right).
    \]

    In the balanced yes/no task, accuracy at the optimal criterion is $\Phi(d'/2)$. Compare these values above. Vary the location parameters; vary the scales only after considering the unequal-variance note above.

    ### A4. Unequal priors and error costs

    Next, let us consider a situation where the signal and noise samples do not have an equal chance of appearing on a given trial. For example, many more trials contain the noise sample than the signal sample. Outside of laboratory experiments, unequal proportions is likely much more common. Think of some examples.

    More generally, choose "signal" when

    \[
    \Lambda(x)=\frac{f_S(x)}{f_N(x)}>\beta_{\mathrm{optimal}}
    =\frac{P(N)}{P(S)}
      \frac{C_{FA}-C_{CR}}{C_M-C_H},
    \]

    where $C_H$, $C_M$, $C_{FA}$, and $C_{CR}$ are the costs of a hit, miss, false alarm, and correct rejection. With equal error costs and equal correct-decision costs, this reduces to $\beta_{\mathrm{optimal}}=P(N)/P(S)$. Thus the accuracy-maximizing threshold is where the prior-weighted densities intersect; $\beta_{\mathrm{optimal}}=3$ in the example below. At that threshold the two classes have equal posterior probability. It is not generally the intersection of the original conditional densities or the midpoint between their means.
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
    plt.figure(figsize=(10, 3))
    pdf11 = 3 * pdf1 / 4
    pdf22 = 1 * pdf2 / 4
    _percor_theoretical = (np.array(tpr_theoretical) + 3 * (1 - np.array(fpr_theoretical))) / 4
    plt.subplot(1, 2, 1)
    plt.plot(x, pdf11, label=f'Noise: P(noise) f_noise(x), loc={loc1}, scale={scale1}', color='red')
    plt.plot(x, pdf22, label=f'Signal: P(signal) f_signal(x), loc={loc2}, scale={scale2}', color='blue')
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
    plt.title('Prior-weighted response densities')
    plt.xlabel('x')
    plt.ylabel('Prior-weighted density')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(thresholds, _percor_theoretical)
    plt.axvline(intersect_point_1, color='black', linestyle='--')
    plt.xlabel('Threshold')
    plt.ylabel('Prob. correct')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Try to understand the plot on the right, and explain why the left and right end of this plot does not lie at 0.5 unlike the situation where the two trial types occurred equally often.

    ### A5. A non-Gaussian example

    Finally, let us consider a situation where the two response distributions are skewed rather than normal. See which properties from above are retained and which are affected. In general, each pair of distributions requires a fresh assessment. Here the skew-normal shape parameter <code>a</code> controls skew direction and magnitude.
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
    plt.title('Skew Normal Distribution')
    # Plot the skew normal distributions
    plt.xlabel('x')
    plt.ylabel('PDF')
    plt.legend()
    plt.grid(True)
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


if __name__ == "__main__":
    app.run()
