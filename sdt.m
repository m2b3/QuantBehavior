%% Signal detection theory: from an ideal observer to psychometric functions
% Standalone MATLAB/GNU Octave companion to sdt.py.
%
% Run this file as a script. It uses Palamedes for the psychometric fit when
% PAL_PFML_Fit is on the MATLAB path. Otherwise it uses a self-contained
% grouped-binomial maximum-likelihood fit based on fminsearch. The remaining
% analysis does not require the Statistics and Machine Learning Toolbox.
%
% Contents
%   1. The ideal equal-variance model
%   2. What changes in real observers?
%   3. Simulated yes/no data
%   4. Fit a psychometric function
%   5. From SDT to stimulus-dependent psychometric functions
%   Appendix: selected extensions

clearvars;
close all;
clc;

%% 1. The ideal equal-variance model
% Imagine equally common signal and noise trials. Each trial creates a noisy,
% one-dimensional internal response. Signal responses tend to be larger, but
% the distributions overlap. Examples include baggage screening, radiology,
% hearing one's name, or diagnosing a sample from a noisy test.
%
% Our deliberately idealized starting assumptions are Gaussian signal and
% noise distributions with equal variance, independent stationary trials, one
% fixed decision criterion, and no stimulus-independent guesses or lapses.

x = linspace(0, 100, 1000);
diffX = linspace(-100, 100, 1000);
locNoise = 40;
sdNoise = 10;
locSignal = 60;
sdSignal = 10;
pdfNoise = normal_pdf(x, locNoise, sdNoise);
pdfSignal = normal_pdf(x, locSignal, sdSignal);
intersectionPoint = find_curve_crossing(...
    x, pdfSignal - pdfNoise, (locNoise + locSignal) / 2);

figure('Name', 'Equal-variance signal detection model');
plot(x, pdfNoise, 'r-', 'LineWidth', 2);
hold on;
plot(x, pdfSignal, 'b-', 'LineWidth', 2);
plot(intersectionPoint, normal_pdf(intersectionPoint, locSignal, sdSignal), 'ko', ...
    'MarkerFaceColor', 'k');
xlabel('Internal response');
ylabel('Density');
title('Response when signal is present or absent');
legend(...
    sprintf('Noise: mean=%g, SD=%g', locNoise, sdNoise), ...
    sprintf('Signal: mean=%g, SD=%g', locSignal, sdSignal), ...
    'Density intersection', 'Location', 'best');
grid on;

% Respond "signal" when the response exceeds a threshold. Very low thresholds
% create many hits and false alarms; very high thresholds create few of both.
% In multidimensional or non-Gaussian problems, the optimal acceptance region
% can be more complicated than one threshold.
thresholds = linspace(min(x), max(x), 1000);
tprTheoretical = 1 - normal_cdf(thresholds, locSignal, sdSignal);
fprTheoretical = 1 - normal_cdf(thresholds, locNoise, sdNoise);
pCorrectTheoretical = 0.5 .* (tprTheoretical + 1 - fprTheoretical);

rocFprTheoretical = [1, fprTheoretical, 0];
rocTprTheoretical = [1, tprTheoretical, 0];
rocAuc = auc_trapezoid(rocFprTheoretical, rocTprTheoretical);
fprintf('Area under the theoretical ROC curve: %.4f\n', rocAuc);

% Internal checks use analytic identities that do not depend on a random seed.
dprimeTheoretical = (locSignal - locNoise) / sdNoise;
aucAnalytic = normal_cdf(dprimeTheoretical / sqrt(2), 0, 1);
assert(abs(sdSignal - sdNoise) < sqrt(eps), ...
    'The simple d-prime checks require equal variances.');
assert(abs(intersectionPoint - (locNoise + locSignal) / 2) < 2 * (x(2) - x(1)), ...
    'The numerical density intersection is inconsistent with the analytic midpoint.');
assert(abs(rocAuc - aucAnalytic) < 5e-4, ...
    'Numerical ROC AUC is inconsistent with Phi(d-prime/sqrt(2)).');
assert(abs(max(pCorrectTheoretical) - normal_cdf(dprimeTheoretical / 2, 0, 1)) < 5e-4, ...
    'Optimal balanced yes/no accuracy is inconsistent with Phi(d-prime/2).');

rng(602, 'twister');
nTrials = 10000;
nHalf = floor(nTrials / 2);
samplesNoise = locNoise + sdNoise .* randn(1, nHalf);
samplesSignal = locSignal + sdSignal .* randn(1, nHalf);

tprSimulation = zeros(size(thresholds));
fprSimulation = zeros(size(thresholds));
for i = 1:numel(thresholds)
    tprSimulation(i) = mean(samplesSignal > thresholds(i));
    fprSimulation(i) = mean(samplesNoise > thresholds(i));
end

figure('Name', 'Threshold and ROC');
subplot(1, 2, 1);
plot(thresholds, pCorrectTheoretical, 'k-', 'LineWidth', 2);
hold on;
xline_compatible(intersectionPoint, 'k--');
xlabel('Threshold');
ylabel('Probability correct');
title('Accuracy versus threshold');
grid on;

subplot(1, 2, 2);
plot(rocFprTheoretical, rocTprTheoretical, 'k-', 'LineWidth', 2);
hold on;
plot([1, fprSimulation, 0], [1, tprSimulation, 0], 'k--', ...
    'LineWidth', 2);
plot([0, 1], [0, 1], '-.', 'Color', [0, 0, 0.5], 'LineWidth', 1.5);
xlim([0, 1]);
ylim([0, 1]);
xlabel('False-positive rate = 1 - specificity');
ylabel('True-positive rate = sensitivity');
title('Theoretical and simulated ROC');
legend('Theoretical', 'Simulation', 'Chance', 'Location', 'southeast');
grid on;

% Standard SDT measures
% For criterion k, H = P(X > k | signal) and F = P(X > k | noise).
% Under the equal-variance Gaussian model:
%   d'   = Phi^-1(H) - Phi^-1(F) = (mu_signal - mu_noise) / sigma
%   c    = -0.5 * (Phi^-1(H) + Phi^-1(F))
%   beta = f_signal(k) / f_noise(k) = exp(d' * c)
% c = 0 is unbiased, c > 0 conservative, and c < 0 liberal. With equal
% priors and equal error costs, the optimal criterion has c = 0 and beta = 1.
% The likelihood-ratio decision rule is more general than these formulas.

%% 2. What changes in real observers?
% The ideal model is a baseline, not a literal description of every observer.
%
% Ideal assumption             Plausible departure / consequence
% Equal Gaussian variance      Unequal or non-Gaussian evidence changes ROC shape
% One fixed criterion          Drift, sequential bias, or payoff changes add variance
% Fixed sensitivity            Learning, fatigue, attention, and adaptation change d'
% No guesses or lapses         Stimulus-independent responses alter both asymptotes
% One mechanism                Strategy/state mixtures may not form one sigmoid
% Independent binomial trials  Serial dependence causes overdispersion
%
% Add complexity only when the design identifies it. Inspect residuals, recover
% parameters in simulations, compare constrained models, and prefer predictive
% or principled model comparisons over appearance alone. With unequal Gaussian
% variances, use a binormal/zROC analysis. AUC, A', and d_a answer related but
% different questions and do not rescue a poorly identified design.

%% 3. Simulated yes/no data
% On signal trials, contrast x raises the mean internal response linearly:
%   X | signal,x ~ Normal(d'(x), 1), with d'(x) = gain * x.
% Noise-only catch trials have mean zero. The observer says "signal" when X > k.
% On a lapse trial the response is random.

rng(604, 'twister');
contrastLevels = [0.25, 0.5, 1, 2, 4, 8, 12, 16];
nPerLevel = 160;
nCatch = 240;
gainTrue = 0.25;
criterionTrue = 1.0;
lapseTrue = 0.02;
trueDprime = gainTrue .* contrastLevels;

yesCounts = zeros(size(contrastLevels));
for i = 1:numel(contrastLevels)
    yesCounts(i) = simulate_yes_count(...
        trueDprime(i), nPerLevel, criterionTrue, lapseTrue);
end
falseAlarmCount = simulate_yes_count(0, nCatch, criterionTrue, lapseTrue);
falseAlarmRate = falseAlarmCount / nCatch;

% Rows are stimulus level, number of "signal" responses, and trial count.
psychometricData = [contrastLevels(:), yesCounts(:), ...
    repmat(nPerLevel, numel(contrastLevels), 1)];

contrastGrid = linspace(0, max(contrastLevels), 400);
pYesGenerating = lapseTrue / 2 + (1 - lapseTrue) .* ...
    normal_cdf(gainTrue .* contrastGrid - criterionTrue, 0, 1);

figure('Name', 'Simulated yes-no data');
plot(contrastGrid, pYesGenerating, 'k-', 'LineWidth', 2);
hold on;
plot(contrastLevels, yesCounts ./ nPerLevel, 'bo', 'MarkerFaceColor', 'b');
plot(0, falseAlarmRate, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('Contrast (%)');
ylabel('Proportion "signal" responses');
ylim([0, 1]);
title('Simulated yes/no data');
legend('Generating model', 'Signal trials', 'Noise-only catch trials', ...
    'Location', 'southeast');
grid on;

fprintf('Catch trials: %d/%d; F = %.3f\n', ...
    falseAlarmCount, nCatch, falseAlarmRate);
fprintf('Contrast | signal responses / trials\n');
for i = 1:numel(contrastLevels)
    fprintf('%8.2f | %3d / %d\n', ...
        contrastLevels(i), yesCounts(i), nPerLevel);
end

%% 4. Fit a psychometric function
% Palamedes is the natural specialist toolbox in MATLAB. If installed, add its
% folder to the path before running this script, for example:
%   addpath(genpath('path/to/Palamedes'))
% This script then calls PAL_PFML_Fit. With no Palamedes installation it uses
% the local fminsearch fallback below, making the lesson runnable as-is.
%
% Curve choices:
%   norm     additive Gaussian decision noise: Phi(beta * (x - alpha))
%   logistic logistic decision noise: 1/(1 + exp(-beta * (x - alpha)))
%   weibull  positive intensities: 1 - exp(-(x / alpha)^beta)
%
% The lower and upper error asymptotes are constrained to the same q. For a
% random-lapse mechanism, the total lapse probability is 2q. Palamedes calls
% these parameters gamma and lambda and can constrain gamma = lambda.

sigmoidName = 'norm';  % Change to 'logistic' or 'weibull'.
% Set environment variable SDT_FORCE_FALLBACK=1 to test/use the built-in fit
% even when Palamedes is installed.
hasPalamedes = isempty(getenv('SDT_FORCE_FALLBACK')) && ...
    exist('PAL_PFML_Fit', 'file') == 2;
fitGrid = linspace(0, max(contrastLevels), 400);

if hasPalamedes
    switch lower(sigmoidName)
        case 'norm'
            psychometricFunction = @PAL_CumulativeNormal;
        case 'logistic'
            psychometricFunction = @PAL_Logistic;
        case 'weibull'
            psychometricFunction = @PAL_Weibull;
        otherwise
            error('sigmoidName must be norm, logistic, or weibull.');
    end

    % Palamedes uses raw positive stimulus levels for PAL_Weibull. PAL_Gumbel
    % is the corresponding choice when stimulus levels have been log10 transformed.
    searchGrid.alpha = linspace(min(contrastLevels), max(contrastLevels), 41);
    searchGrid.beta = logspace(-2, 1, 41);
    searchGrid.gamma = 0;
    searchGrid.lambda = linspace(0, 0.1, 11);
    paramsFree = [1, 1, 0, 1];

    [fitParams, logLikelihood, fitExitFlag] = PAL_PFML_Fit(...
        contrastLevels, yesCounts, nPerLevel .* ones(size(contrastLevels)), ...
        searchGrid, paramsFree, psychometricFunction, ...
        'gammaEQlambda', true);
    fitProbability = psychometricFunction(fitParams, fitGrid);
    fitAlpha = fitParams(1);
    fitBeta = fitParams(2);
    fitQ = fitParams(4);

    fprintf(['Palamedes fit (%s): alpha = %.4f, beta = %.4f, ', ...
        'equal asymptote q = %.4f (random-lapse probability = %.4f); ', ...
        'log likelihood = %.3f; exit flag = %d\n'], ...
        sigmoidName, fitAlpha, fitBeta, fitQ, 2 * fitQ, ...
        logLikelihood, fitExitFlag);
else
    [fitParams, negativeLogLikelihood, fitExitFlag] = ...
        fit_psychometric(contrastLevels, yesCounts, ...
        nPerLevel .* ones(size(contrastLevels)), sigmoidName);
    fitProbability = psychometric_probability(...
        fitParams, fitGrid, sigmoidName, true);
    fitAlpha = fitParams(1);
    fitBeta = fitParams(2);
    fitQ = fitParams(3);

    fprintf(['Built-in ML fit (%s): alpha = %.4f, beta = %.4f, ', ...
        'equal asymptote q = %.4f (random-lapse probability = %.4f); ', ...
        'negative log likelihood = %.3f; exit flag = %d\n'], ...
        sigmoidName, fitAlpha, fitBeta, fitQ, 2 * fitQ, ...
        negativeLogLikelihood, fitExitFlag);
end

assert(fitExitFlag > 0, 'The psychometric optimizer did not converge.');
assert(all(isfinite(fitProbability)) && ...
    all(fitProbability > 0 & fitProbability < 1), ...
    'The fitted psychometric probabilities are invalid.');
assert(fitBeta > 0 && fitQ >= 0 && fitQ <= 0.1, ...
    'The fitted slope or equal-asymptote parameter is outside its allowed range.');

figure('Name', 'Psychometric fit');
plot(fitGrid, fitProbability, 'k-', 'LineWidth', 2);
hold on;
plot(contrastLevels, yesCounts ./ nPerLevel, 'bo', 'MarkerFaceColor', 'b');
xlabel('Contrast (%)');
ylabel('P("signal" response | signal trial)');
ylim([0, 1]);
title(sprintf('Grouped-binomial fit: %s', sigmoidName));
legend('Fitted curve', 'Observed proportions', 'Location', 'southeast');
grid on;

%% 5. From SDT to stimulus-dependent psychometric functions
% With unit variance and fixed criterion k:
%   H(x) = Phi(d'(x) - k), and F = Phi(-k).
% Thus H(x) depends jointly on the decision rule and the transducer mapping the
% physical stimulus to sensitivity. If d'(x) = gain*x, H(x) is cumulative
% Gaussian in x. A common nonlinear extension is d'(x) = (gain*x)^p.
%
% Fit signal and noise responses jointly when asking whether a manipulation
% changes gain, exponent, criterion, or lapse rate. If false-alarm rate or
% criterion changes with stimulus or condition, fitting hit rates alone
% confounds sensitivity and bias. Palamedes also includes SDT-specific fitting
% routines such as PAL_SDT_PFML_Fit for suitable designs.

% The log-linear correction avoids infinite z-scores for rates of 0 or 1.
hitRateCorrected = (yesCounts + 0.5) ./ (nPerLevel + 1);
falseAlarmRateCorrected = (falseAlarmCount + 0.5) / (nCatch + 1);
empiricalDprime = normal_inv(hitRateCorrected) - ...
    normal_inv(falseAlarmRateCorrected);
assert(all(isfinite(empiricalDprime)) && ...
    numel(empiricalDprime) == numel(contrastLevels), ...
    'The corrected empirical d-prime values are invalid.');

figure('Name', 'Stimulus-dependent sensitivity');
plot(contrastLevels, empiricalDprime, 'bo', 'MarkerFaceColor', 'b');
hold on;
plot(contrastLevels, trueDprime, 'k-', 'LineWidth', 2);
xlabel('Contrast (%)');
ylabel('d''');
title('Stimulus-dependent sensitivity');
legend('Estimated from H and F', ...
    sprintf('Generating d'' = %.2f x', gainTrue), 'Location', 'northwest');
grid on;

%% Appendix A1. Estimating SDT measures from response counts
%                Respond "signal"  Respond "noise"
% Actual signal       Hit               Miss
% Actual noise        False alarm       Correct rejection
%
% H conditions on signal trials; F conditions on noise trials. Precision,
% hits/(hits + false alarms), instead conditions on the response and therefore
% changes with the signal base rate.

hits = 40;
misses = 10;
falseAlarms = 15;
correctRejections = 135;
hitRate = hits / (hits + misses);
falseAlarmRateExample = falseAlarms / (falseAlarms + correctRejections);
precision = hits / (hits + falseAlarms);
hitRateCorrectedExample = (hits + 0.5) / (hits + misses + 1);
faRateCorrectedExample = ...
    (falseAlarms + 0.5) / (falseAlarms + correctRejections + 1);
zHit = normal_inv(hitRateCorrectedExample);
zFalseAlarm = normal_inv(faRateCorrectedExample);
dprimeObserved = zHit - zFalseAlarm;
criterionObserved = -0.5 * (zHit + zFalseAlarm);
assert(isfinite(dprimeObserved) && isfinite(criterionObserved) && ...
    dprimeObserved > 0, 'The observed-count SDT example is inconsistent.');

fprintf('Hit rate: %.3f; false-alarm rate: %.3f\n', ...
    hitRate, falseAlarmRateExample);
fprintf('Corrected d'': %.3f; corrected c: %.3f; precision: %.3f\n', ...
    dprimeObserved, criterionObserved, precision);

%% Appendix A2. Unequal variance and alternative sensitivity indices
% If sigma_signal ~= sigma_noise, Phi^-1(H)-Phi^-1(F) varies with criterion.
% The zROC follows:
%   Phi^-1(H) = (mu_signal-mu_noise)/sigma_signal
%               + (sigma_noise/sigma_signal) * Phi^-1(F).
% Its slope is sigma_noise/sigma_signal, not 1. The log likelihood ratio is
% quadratic, so an optimal decision region can require two crossings. ROC/AUC
% and the general likelihood-ratio rule remain valid. A', d_a, and equal-
% variance d' have different interpretations and should be labelled clearly.

%% Appendix A3. Two-interval, two-alternative forced choice
% Each trial contains one signal and one noise sample; choose the larger. Under
% independent draws from the same decision variable there is no criterion/caution
% term, although order and response biases may remain.

sdDifference = sqrt(sdNoise^2 + sdSignal^2);
pdfSignalMinusNoise = normal_pdf(...
    diffX, locSignal - locNoise, sdDifference);
pdfNoiseMinusSignal = normal_pdf(...
    diffX, locNoise - locSignal, sdDifference);

figure('Name', 'Two-interval two-alternative forced choice');
plot(diffX, pdfSignalMinusNoise, 'b-', 'LineWidth', 2);
hold on;
plot(diffX, pdfNoiseMinusSignal, 'r-', 'LineWidth', 2);
xlabel('Difference');
ylabel('Density');
title('Difference distributions in 2I-2AFC');
legend('Signal - noise', 'Noise - signal', 'Location', 'best');
grid on;

pCorrect2afc = 1 - normal_cdf(0, locSignal - locNoise, sdDifference);
fprintf('Probability correct in 2I-2AFC: %.4f\n', pCorrect2afc);
% Under the standard assumptions, P(correct in 2AFC) = AUC = Phi(d'/sqrt(2)).
% Balanced yes/no accuracy at the optimal criterion is Phi(d'/2).
assert(abs(pCorrect2afc - aucAnalytic) < 1e-12, ...
    'The 2AFC probability is inconsistent with its analytic value.');
assert(abs(pCorrect2afc - rocAuc) < 5e-4, ...
    'The 2AFC probability and numerical ROC AUC should agree.');

%% Appendix A4. Unequal priors and error costs
% Choose signal when the likelihood ratio exceeds
%   beta_optimal = P(noise)/P(signal) *
%                  (C_FA-C_CR)/(C_M-C_H).
% With equal error costs and P(noise)=3/4, P(signal)=1/4, beta_optimal=3.
% The accuracy-maximizing threshold is where prior-weighted densities cross.

priorNoise = 3 / 4;
priorSignal = 1 / 4;
weightedNoise = priorNoise .* pdfNoise;
weightedSignal = priorSignal .* pdfSignal;
weightedIntersection = find_curve_crossing(...
    x, weightedSignal - weightedNoise, (locNoise + locSignal) / 2);
pCorrectUnequal = priorSignal .* tprTheoretical + ...
    priorNoise .* (1 - fprTheoretical);
analyticWeightedIntersection = (locNoise + locSignal) / 2 + ...
    sdNoise^2 * log(priorNoise / priorSignal) / (locSignal - locNoise);
[~, bestUnequalIndex] = max(pCorrectUnequal);
assert(abs(weightedIntersection - analyticWeightedIntersection) < ...
    2 * (x(2) - x(1)), ...
    'The weighted-density intersection is inconsistent with its analytic value.');
assert(abs(thresholds(bestUnequalIndex) - weightedIntersection) < ...
    2 * (thresholds(2) - thresholds(1)), ...
    'The weighted-density intersection should maximize unequal-prior accuracy.');

figure('Name', 'Unequal priors');
subplot(1, 2, 1);
plot(x, weightedNoise, 'r-', 'LineWidth', 2);
hold on;
plot(x, weightedSignal, 'b-', 'LineWidth', 2);
plot(weightedIntersection, ...
    priorSignal .* normal_pdf(weightedIntersection, locSignal, sdSignal), ...
    'ko', 'MarkerFaceColor', 'k');
xlabel('Internal response');
ylabel('Prior-weighted density');
title('Prior-weighted response densities');
legend('P(noise) f_{noise}', 'P(signal) f_{signal}', ...
    'Weighted intersection', 'Location', 'best');
grid on;

subplot(1, 2, 2);
plot(thresholds, pCorrectUnequal, 'k-', 'LineWidth', 2);
hold on;
xline_compatible(weightedIntersection, 'k--');
xlabel('Threshold');
ylabel('Probability correct');
title('Accuracy with P(noise) = 3/4');
grid on;

%% Appendix A5. A non-Gaussian example
% The skew-normal density is 2*phi(z)*Phi(a*z)/scale. Small local functions
% below implement its density, CDF, and random generator so that neither the
% Statistics Toolbox nor another package is required.

xSkew = linspace(0, 100, 1000);
shapeNoise = 2;
locNoiseSkew = 35;
scaleNoiseSkew = 10;
shapeSignal = -2;
locSignalSkew = 65;
scaleSignalSkew = 10;
pdfNoiseSkew = skew_normal_pdf(...
    xSkew, shapeNoise, locNoiseSkew, scaleNoiseSkew);
pdfSignalSkew = skew_normal_pdf(...
    xSkew, shapeSignal, locSignalSkew, scaleSignalSkew);

figure('Name', 'Non-Gaussian example');
plot(xSkew, pdfNoiseSkew, 'r-', 'LineWidth', 2);
hold on;
plot(xSkew, pdfSignalSkew, 'b-', 'LineWidth', 2);
xlabel('Internal response');
ylabel('Density');
title('Skew-normal response distributions');
legend(...
    sprintf('Noise: a=%g, loc=%g, scale=%g', ...
        shapeNoise, locNoiseSkew, scaleNoiseSkew), ...
    sprintf('Signal: a=%g, loc=%g, scale=%g', ...
        shapeSignal, locSignalSkew, scaleSignalSkew), ...
    'Location', 'best');
grid on;

rng(603, 'twister');
nSkewSamples = 10000;
simulatedNoiseSkew = skew_normal_random(...
    nSkewSamples, shapeNoise, locNoiseSkew, scaleNoiseSkew);
simulatedSignalSkew = skew_normal_random(...
    nSkewSamples, shapeSignal, locSignalSkew, scaleSignalSkew);
pCorrectSkew = mean(simulatedSignalSkew > simulatedNoiseSkew);

thresholdsSkew = linspace(min(xSkew), max(xSkew), 100);
tprSkew = 1 - skew_normal_cdf(...
    thresholdsSkew, shapeSignal, locSignalSkew, scaleSignalSkew);
fprSkew = 1 - skew_normal_cdf(...
    thresholdsSkew, shapeNoise, locNoiseSkew, scaleNoiseSkew);
aucSkew = auc_trapezoid([1, fprSkew, 0], [1, tprSkew, 0]);

fprintf('Skew-normal simulated 2AFC probability correct: %.4f\n', ...
    pCorrectSkew);
fprintf('Skew-normal ROC AUC: %.4f\n', aucSkew);
assert(aucSkew > 0 && aucSkew < 1 && ...
    pCorrectSkew > 0 && pCorrectSkew < 1, ...
    'A probability or AUC in the skew-normal example is invalid.');
assert(abs(pCorrectSkew - aucSkew) < 0.02, ...
    'The skew-normal 2AFC simulation and ROC AUC disagree unexpectedly.');
fprintf('All internal consistency checks passed.\n');

%% Local functions
function density = normal_pdf(value, location, scale)
    z = (value - location) ./ scale;
    density = exp(-0.5 .* z.^2) ./ (sqrt(2 * pi) .* scale);
end

function probability = normal_cdf(value, location, scale)
    probability = 0.5 .* (1 + erf((value - location) ./ (scale .* sqrt(2))));
end

function quantile = normal_inv(probability)
    quantile = sqrt(2) .* erfinv(2 .* probability - 1);
end

function area = auc_trapezoid(xValues, yValues)
    [xSorted, ordering] = sort(xValues);
    ySorted = yValues(ordering);
    area = trapz(xSorted, ySorted);
end

function crossing = find_curve_crossing(grid, difference, target)
    crossingIndices = find(diff(sign(difference)) ~= 0);
    assert(~isempty(crossingIndices), ...
        'No curve crossing was found on the supplied grid.');
    crossings = zeros(size(crossingIndices));
    for index = 1:numel(crossingIndices)
        left = crossingIndices(index);
        x1 = grid(left);
        x2 = grid(left + 1);
        y1 = difference(left);
        y2 = difference(left + 1);
        crossings(index) = x1 - y1 .* (x2 - x1) ./ (y2 - y1);
    end
    [~, nearest] = min(abs(crossings - target));
    crossing = crossings(nearest);
end

function yesCount = simulate_yes_count(...
        meanResponse, nTrials, criterion, lapseProbability)
    internalResponse = meanResponse + randn(1, nTrials);
    respondYes = internalResponse > criterion;
    lapseTrials = rand(1, nTrials) < lapseProbability;
    respondYes(lapseTrials) = rand(1, sum(lapseTrials)) < 0.5;
    yesCount = sum(respondYes);
end

function [decodedParams, negativeLogLikelihood, exitFlag] = ...
        fit_psychometric(stimulus, yesCounts, trialCounts, sigmoidName)
    % Optimize unconstrained parameters, then return [alpha, beta, q].
    if strcmpi(sigmoidName, 'weibull')
        thetaStart = [log(median(stimulus)), log(2), log(0.1 / 0.9)];
    else
        thetaStart = [median(stimulus), log(0.25), log(0.1 / 0.9)];
    end
    objective = @(theta) psychometric_nll(...
        theta, stimulus, yesCounts, trialCounts, sigmoidName);
    fitOptions = optimset('MaxIter', 10000, 'MaxFunEvals', 20000, ...
        'TolX', 1e-10, 'TolFun', 1e-10, 'Display', 'off');
    [theta, negativeLogLikelihood, exitFlag] = ...
        fminsearch(objective, thetaStart, fitOptions);

    q = 0.1 ./ (1 + exp(-theta(3)));
    if strcmpi(sigmoidName, 'weibull')
        alpha = exp(theta(1));
    else
        alpha = theta(1);
    end
    beta = exp(theta(2));
    decodedParams = [alpha, beta, q];
end

function value = psychometric_nll(...
        theta, stimulus, yesCounts, trialCounts, sigmoidName)
    probability = psychometric_probability(theta, stimulus, sigmoidName, false);
    probability = min(max(probability, 1e-12), 1 - 1e-12);
    value = -sum(yesCounts .* log(probability) + ...
        (trialCounts - yesCounts) .* log(1 - probability));
end

function probability = psychometric_probability(...
        parameters, stimulus, sigmoidName, parametersAreDecoded)
    if parametersAreDecoded
        alpha = parameters(1);
        beta = parameters(2);
        q = parameters(3);
    else
        q = 0.1 ./ (1 + exp(-parameters(3)));
        if strcmpi(sigmoidName, 'weibull')
            alpha = exp(parameters(1));
        else
            alpha = parameters(1);
        end
        beta = exp(parameters(2));
    end

    switch lower(sigmoidName)
        case 'norm'
            core = normal_cdf(beta .* (stimulus - alpha), 0, 1);
        case 'logistic'
            core = 1 ./ (1 + exp(-beta .* (stimulus - alpha)));
        case 'weibull'
            core = 1 - exp(-(max(stimulus, 0) ./ alpha).^beta);
        otherwise
            error('sigmoidName must be norm, logistic, or weibull.');
    end
    probability = q + (1 - 2 .* q) .* core;
end

function density = skew_normal_pdf(value, shape, location, scale)
    z = (value - location) ./ scale;
    density = 2 .* normal_pdf(z, 0, 1) .* normal_cdf(shape .* z, 0, 1) ./ scale;
end

function probability = skew_normal_cdf(value, shape, location, scale)
    probability = zeros(size(value));
    for index = 1:numel(value)
        integrand = @(sample) skew_normal_pdf(sample, shape, location, scale);
        probability(index) = integral(integrand, -Inf, value(index), ...
            'RelTol', 1e-8, 'AbsTol', 1e-10);
    end
end

function samples = skew_normal_random(n, shape, location, scale)
    delta = shape ./ sqrt(1 + shape.^2);
    samples = location + scale .* (...
        delta .* abs(randn(1, n)) + sqrt(1 - delta.^2) .* randn(1, n));
end

function xline_compatible(xValue, lineSpecification)
    % xline was introduced after many teaching labs standardized MATLAB.
    limits = ylim;
    plot([xValue, xValue], limits, lineSpecification, 'LineWidth', 1.25);
end
