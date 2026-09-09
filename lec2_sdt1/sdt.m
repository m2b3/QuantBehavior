%% Signal detection theory: from an ideal observer to psychometric functions
% Copyright © 2026 Suresh Krishna.
% Licensed as a complete educational work under CC BY-NC-SA 4.0:
% https://creativecommons.org/licenses/by-nc-sa/4.0/
% Attribution details:
% https://github.com/m2b3/QuantBehavior/blob/main/LICENSE-CONTENT.md
%
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
%
% Three tasks that are easy to confuse
% Task                         Trial and response                 Direct summary
% Yes/no detection             Signal present/absent; say yes/no Hits, false alarms,
%                                                                d' and criterion
% Single-interval 2AFC         Category A or B; say A or B       A two-by-two table
% (identification)                                               after naming one
%                                                                category "positive"
% 2I-2AFC detection            Signal in interval 1 or 2;        Proportion correct
%                              choose its interval                and d'
%
% "2AFC" is sometimes shorthand for the two-interval task. Saying 1I-2AFC or
% 2I-2AFC removes the ambiguity. Sensitivity, specificity, recall, precision,
% and PPV apply naturally to yes/no detection and single-interval
% classification. In ordinary 2I-2AFC, a signal occurs somewhere on every
% trial, so catch trials or a separate yes/no task are needed to estimate
% detection specificity and predictive value.

x = linspace(0, 100, 1000);
diffX = linspace(-100, 100, 1000);
locNoise = 40;
sdNoise = 10;
locSignal = 60;
sdSignal = 10;
pdfNoise = normal_pdf(x, locNoise, sdNoise);
pdfSignal = normal_pdf(x, locSignal, sdSignal);
criterion = (locNoise + locSignal) / 2;
intersectionPoint = find_curve_crossing(...
    x, pdfSignal - pdfNoise, (locNoise + locSignal) / 2);

figure('Name', 'Equal-variance signal detection model');
hold on;
rightOfCriterion = x >= criterion;
leftOfCriterion = x < criterion;
arrowHeight = 1.08 * max([pdfNoise, pdfSignal]);
specificityPatch = fill(...
    [x(leftOfCriterion), fliplr(x(leftOfCriterion))], ...
    [pdfNoise(leftOfCriterion), zeros(1, sum(leftOfCriterion))], ...
    'r', 'FaceAlpha', 0.07, 'EdgeColor', 'none', ...
    'HandleVisibility', 'off');
hitPatch = fill(...
    [x(rightOfCriterion), fliplr(x(rightOfCriterion))], ...
    [pdfSignal(rightOfCriterion), zeros(1, sum(rightOfCriterion))], ...
    'b', 'FaceAlpha', 0.20, 'EdgeColor', 'none');
falseAlarmPatch = fill(...
    [x(rightOfCriterion), fliplr(x(rightOfCriterion))], ...
    [pdfNoise(rightOfCriterion), zeros(1, sum(rightOfCriterion))], ...
    'r', 'FaceAlpha', 0.25, 'EdgeColor', 'none');
noiseLine = plot(x, pdfNoise, 'r-', 'LineWidth', 2);
signalLine = plot(x, pdfSignal, 'b-', 'LineWidth', 2);
plot(intersectionPoint, normal_pdf(intersectionPoint, locSignal, sdSignal), 'ko', ...
    'MarkerFaceColor', 'k', 'HandleVisibility', 'off');
criterionLine = plot([criterion, criterion], [0, 1.28 * arrowHeight], ...
    'k--', 'LineWidth', 1.5);
purple = [0.45, 0, 0.55];
plot([locNoise, locSignal], [arrowHeight, arrowHeight], '-', ...
    'Color', purple, 'LineWidth', 2, 'HandleVisibility', 'off');
plot(locNoise, arrowHeight, '<', 'Color', purple, ...
    'MarkerFaceColor', purple, 'HandleVisibility', 'off');
plot(locSignal, arrowHeight, '>', 'Color', purple, ...
    'MarkerFaceColor', purple, 'HandleVisibility', 'off');
text(locNoise + 0.25 * (locSignal - locNoise), 1.02 * arrowHeight, 'd''', ...
    'Color', purple, 'FontSize', 13, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
    'BackgroundColor', 'white', 'Margin', 1);
xlabel('Internal evidence');
ylabel('Density');
ylim([0, 1.28 * arrowHeight]);
title('Figure 1. Equal-variance Gaussian evidence in a yes/no task');
legend([noiseLine, signalLine, criterionLine, hitPatch, falseAlarmPatch], ...
    sprintf('Noise: mean=%g, SD=%g', locNoise, sdNoise), ...
    sprintf('Signal: mean=%g, SD=%g', locSignal, sdSignal), ...
    sprintf('Decision criterion k=%g (c=0)', criterion), ...
    'H: hit area', 'F: false-alarm area', 'Location', 'northwest');
grid on;

% How to read Figure 1: the black dashed line is the decision threshold;
% evidence to its right produces a "signal" response. The purple arrow marks
% d', the peak-to-peak distance measured with the common SD as the ruler. The
% curves drawn here happen to give d'=2 because their peaks are 20 units apart
% and their SD is 10. That is an illustrative choice, not a fixed value of d'.
% Criterion c tells us where the black line lies relative to the midpoint.
% The blue area right of the line is the hit rate H; the red area right of it is
% the false-alarm rate F; the pale red area left of it is specificity.
%
% Figures 2A and 2B move this decision line through every possible position.
% Very low thresholds create many hits and false alarms; very high thresholds
% create few of both. This movable decision threshold differs from a fixed
% sensory threshold; Appendix A3 summarizes the classic ROC evidence.
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
ylabel('Chance of a correct response');
title('Figure 2A. Accuracy as the decision threshold moves');
grid on;

subplot(1, 2, 2);
plot(rocFprTheoretical, rocTprTheoretical, 'k-', 'LineWidth', 2);
hold on;
plot([1, fprSimulation, 0], [1, tprSimulation, 0], 'k--', ...
    'LineWidth', 2);
plot([0, 1], [0, 1], '-.', 'Color', [0, 0, 0.5], 'LineWidth', 1.5);
xlim([0, 1]);
ylim([0, 1]);
xlabel('False-alarm rate (the opposite of specificity)');
ylabel('Hit rate (sensitivity / recall)');
title('Figure 2B. The same thresholds traced in ROC space');
legend('Theoretical', 'Simulation', 'Chance', 'Location', 'southeast');
grid on;

% Standard SDT measures
% P(X > k | signal) means the chance that signal-trial evidence lands to the
% right of the decision line: the blue area in Figure 1. Call this H. Replacing
% signal with noise gives F, the red area to the right of the line.
% Under the equal-variance Gaussian model:
%   d'   = Phi^-1(H) - Phi^-1(F) = (mu_signal - mu_noise) / sigma
% In plain language, d' is the purple peak-to-peak distance in Figure 1, using
% the common SD as the ruler. Phi^-1 merely converts an area to that SD ruler.
% Larger d' means less overlap. Moving the criterion does not move the peaks.
%
%   c    = -0.5 * (Phi^-1(H) + Phi^-1(F))
% In plain language, c locates the black line relative to the midpoint: zero at
% the midpoint, positive when shifted right (conservative), and negative when
% shifted left (liberal), all measured on the same SD ruler.
%
%   beta = f_signal(k) / f_noise(k) = exp(d' * c)
% In plain language, beta compares the heights of the signal and noise curves
% at the black line. At their crossing the heights match, so beta=1. With equal
% priors and costs this crossing maximizes accuracy and c=0. These shortcuts
% assume equal-variance Gaussians; comparing the two sources remains general.

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
% The next formula is a compact recipe for the simulation. On signal trials:
%   X | signal,x ~ Normal(d'(x), 1), with d'(x) = gain * x.
% In plain language, draw an evidence value X from a bell curve centered at
% d'(x) with SD one. Increasing contrast moves its peak right; gain says how
% far per contrast unit. Noise-only catch trials use a curve centered at zero.
% The observer says "signal" when X lands right of k, as in Figure 1. Figure 3
% shows many such trials; lapses occasionally replace this rule with a guess.

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
ylabel('Chance of a "signal" response');
ylim([0, 1]);
title('Figure 3. Simulated yes/no responses across contrast');
legend('Generating model', 'Signal trials', 'Noise-only catch trials', ...
    'Location', 'southeast');
grid on;

fprintf(['Noise-only catch trials: %d of %d produced a ', ...
    '"signal" response.\n'], falseAlarmCount, nCatch);
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
%            In plain language, response chance is accumulated bell-curve area;
%            alpha moves the S-curve and beta controls how quickly it rises.
%   logistic logistic decision noise: 1/(1 + exp(-beta * (x - alpha)))
%            In plain language, every stimulus step adds a fixed amount to the
%            response log-odds. It resembles a Gaussian with heavier tails.
%   weibull  positive intensities: 1 - exp(-(x / alpha)^beta)
%            In plain language, the chance that every possible detection event
%            fails shrinks with intensity; alpha sets scale and beta sets shape.
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
ylabel('Proportion of signal responses');
ylim([0, 1]);
title(sprintf('Figure 4. Psychometric fit (%s link)', sigmoidName));
legend('Fitted curve', 'Observed proportions', 'Location', 'southeast');
grid on;

%% 5. From SDT to stimulus-dependent psychometric functions
% With unit variance and fixed criterion k:
%   H(x) = Phi(d'(x) - k), and F = Phi(-k).
% In plain language, H(x) is the blue area to the right of Figure 1's decision
% line after contrast moves the blue curve. F is the red area right of the line;
% here it stays fixed because the noise curve and criterion do not move. Phi
% converts a horizontal distance into accumulated area under a bell curve.
%
% Thus the psychometric curve depends jointly on the decision rule and the
% transducer mapping stimulus to d'. If d'(x)=gain*x, each contrast step moves
% the signal peak equally and H(x) is cumulative Gaussian. A common extension:
%   d'(x) = (gain*x)^p.
% In plain language, gain controls overall peak separation and p lets that
% separation grow faster or slower than a straight line. Figure 5 plots it.
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
title('Figure 5. Sensitivity grows with contrast');
legend('Estimated d'' from hit and false-alarm areas', ...
    sprintf('Generating d'' = %.2f x', gainTrue), 'Location', 'northwest');
grid on;

%% Appendix A1. Reading a two-by-two table in plain language
% Diagnostic example:
% Actual state       Test positive       Test negative       Total
% Disease present    40 hits             10 misses           50
% Disease absent     15 false alarms     135 correct rejects 150
% Total              55                  145                 200
%
% Sensitivity/recall asks: among the 50 people with disease, how many test
% positive? Look across that row: 40 are found and 10 are missed.
% Specificity asks: among the 150 people without disease, how many test
% negative? Look across that row: 135 are cleared and 15 falsely alarm.
% Precision/PPV reverses the question: among the 55 positive tests, how many
% actually have disease? Look down that column: 40 do and 15 do not.
%
% The same arrangement describes stimulus detection:
% Actual stimulus    Say "I see it"      Say "I don't see it" Total
% Present            40 hits             10 misses            50
% Absent             15 false alarms     135 correct rejects  150
% Total              55                  145                  200
%
% It also describes two-category discrimination when clockwise is designated
% the "positive" category:
% Actual tilt        Say "clockwise"     Say "counterclockwise" Total
% Clockwise          40 hits             10 misses               50
% Counterclockwise   15 false alarms     135 correct rejects     150
% Total              55                  145                     200
% Calling counterclockwise "positive" would swap the outcome labels, not the
% observations. The positive category is bookkeeping, not a special stimulus.
%
% Sensitivity/recall and specificity begin with the actual state and look
% across a row. Precision/PPV begins with the response and looks down a column;
% it therefore changes with the signal base rate. In SDT, sensitivity/recall is
% H and one minus specificity is F. Clinical "sensitivity" (H) is not the same
% as the SDT discriminability index d': moving criterion changes H, not d'.
%
% The log-linear correction below is used only for d' and c, preventing an
% observed "never" or "always" response from producing an infinite z-score.

hits = 40;
misses = 10;
falseAlarms = 15;
correctRejections = 135;
hitRate = hits / (hits + misses);
falseAlarmRateExample = falseAlarms / (falseAlarms + correctRejections);
specificity = correctRejections / (falseAlarms + correctRejections);
precision = hits / (hits + falseAlarms);
hitRateCorrectedExample = (hits + 0.5) / (hits + misses + 1);
faRateCorrectedExample = ...
    (falseAlarms + 0.5) / (falseAlarms + correctRejections + 1);
zHit = normal_inv(hitRateCorrectedExample);
zFalseAlarm = normal_inv(faRateCorrectedExample);
dprimeObserved = zHit - zFalseAlarm;
criterionObserved = -0.5 * (zHit + zFalseAlarm);
assert(isfinite(dprimeObserved) && isfinite(criterionObserved) && ...
    dprimeObserved > 0 && ...
    abs(specificity - (1 - falseAlarmRateExample)) < 1e-12 && ...
    precision > 0 && precision < 1, ...
    'The observed-count SDT example is inconsistent.');

fprintf(['Sensitivity / recall: among %d people with disease, ', ...
    '%d test positive and %d test negative.\n'], hits + misses, hits, misses);
fprintf(['Specificity: among %d people without disease, ', ...
    '%d test negative and %d test positive.\n'], ...
    falseAlarms + correctRejections, correctRejections, falseAlarms);
fprintf(['Precision / PPV: among %d positive tests, ', ...
    '%d come from people with disease and %d do not.\n'], ...
    hits + falseAlarms, hits, falseAlarms);
fprintf('Corrected d'': %.3f; corrected c: %.3f\n', ...
    dprimeObserved, criterionObserved);

%% Appendix A2. Unequal variance and alternative sensitivity indices
% If sigma_signal ~= sigma_noise, Phi^-1(H)-Phi^-1(F) varies with criterion.
% The zROC follows:
%   Phi^-1(H) = (mu_signal-mu_noise)/sigma_signal
%               + (sigma_noise/sigma_signal) * Phi^-1(F).
% In plain language, Phi^-1 redraws each ROC axis with an SD ruler, making the
% ROC a line. Its tilt is sigma_noise/sigma_signal; equal widths give one. If
% widths differ, peak-distance d' depends on which spread is used. The log
% likelihood ratio is curved and can cross the cutoff twice, so "signal" need
% not mean every value right of one line. ROC/AUC and the general rule remain
% valid. A', d_a, and equal-variance d' have different interpretations.

%% Appendix A3. Why ROC experiments mattered
% A single H,F pair cannot reveal ROC shape. Classic experiments held signal
% strength roughly fixed while moving criterion through priors/payoffs, or used
% confidence ratings as several criteria. This separates willingness to say
% "signal" from sensitivity. Continuous-evidence SDT predicts a smooth bowed
% ROC; the simplest high-threshold theory predicts straight-line structure.
% Yes/no and forced-choice detectability agreed, rating ROCs favored graded
% evidence, and above-chance second choices challenged the simplest all-or-none
% account. Luce's low-threshold model and later multi-state threshold models
% show that these results do not prove every internal distribution is Gaussian.
% Rich ROC data are needed for detailed model discrimination.
% Sources: Tanner & Swets (1954), doi:10.1037/h0058700; Swets, Tanner, &
% Birdsall (1961), doi:10.1037/h0040547; Luce (1963), doi:10.1037/h0039723;
% Nachmias & Steinman (1963), doi:10.1364/JOSA.53.001206; Krantz (1969),
% doi:10.1037/h0027238.

%% Appendix A4. Two-interval, two-alternative forced choice
% Each trial contains one A/signal and one B/noise observation. The observer
% knows A is present and chooses its interval; "signal absent" is not an option.
% Figure 6 plots D = interval-1 observation minus interval-2 observation. Blue
% is A in interval 1, red is A in interval 2, and zero separates the choices.

sdDifference = sqrt(sdNoise^2 + sdSignal^2);
deltaDifference = locSignal - locNoise;
pdfSignalMinusNoise = normal_pdf(...
    diffX, deltaDifference, sdDifference);
pdfNoiseMinusSignal = normal_pdf(...
    diffX, -deltaDifference, sdDifference);

figure('Name', 'Two-interval two-alternative forced choice');
hold on;
rightOfZero = diffX >= 0;
leftOfZero = diffX <= 0;
differenceArrowHeight = 1.08 * max(...
    [pdfSignalMinusNoise, pdfNoiseMinusSignal]);
correctInterval1Patch = fill(...
    [diffX(rightOfZero), fliplr(diffX(rightOfZero))], ...
    [pdfSignalMinusNoise(rightOfZero), zeros(1, sum(rightOfZero))], ...
    'b', 'FaceAlpha', 0.18, 'EdgeColor', 'none');
correctInterval2Patch = fill(...
    [diffX(leftOfZero), fliplr(diffX(leftOfZero))], ...
    [pdfNoiseMinusSignal(leftOfZero), zeros(1, sum(leftOfZero))], ...
    'r', 'FaceAlpha', 0.18, 'EdgeColor', 'none');
interval1Line = plot(diffX, pdfSignalMinusNoise, 'b-', 'LineWidth', 2);
interval2Line = plot(diffX, pdfNoiseMinusSignal, 'r-', 'LineWidth', 2);
choiceLine = plot([0, 0], [0, 1.28 * differenceArrowHeight], ...
    'k--', 'LineWidth', 1.5);
plot([-deltaDifference, deltaDifference], ...
    [differenceArrowHeight, differenceArrowHeight], '-', ...
    'Color', purple, 'LineWidth', 2, 'HandleVisibility', 'off');
plot(-deltaDifference, differenceArrowHeight, '<', 'Color', purple, ...
    'MarkerFaceColor', purple, 'HandleVisibility', 'off');
plot(deltaDifference, differenceArrowHeight, '>', 'Color', purple, ...
    'MarkerFaceColor', purple, 'HandleVisibility', 'off');
text(-deltaDifference / 2, 1.02 * differenceArrowHeight, ...
    '2 Delta between conditional means', 'Color', purple, ...
    'FontWeight', 'bold', 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'bottom', 'BackgroundColor', 'white', 'Margin', 1);
xlabel('Difference D');
ylabel('Density');
ylim([0, 1.28 * differenceArrowHeight]);
title('Figure 6. 2I-2AFC decision variable: D = X_1 - X_2');
legend([interval1Line, interval2Line, choiceLine, ...
    correctInterval1Patch, correctInterval2Patch], ...
    'A in interval 1: mean +Delta', 'A in interval 2: mean -Delta', ...
    'Choose interval 1 if D > 0', 'Correct area: A in interval 1', ...
    'Correct area: A in interval 2', 'Location', 'best');
grid on;

pCorrect2afc = 1 - normal_cdf(0, deltaDifference, sdDifference);
fprintf('Probability correct in 2I-2AFC: %.4f\n', pCorrect2afc);
% Why 1/sqrt(2)? When A is in interval 1, mean(D)=+Delta; when it is in
% interval 2, mean(D)=-Delta, so the two D curves are 2*Delta apart.
% Independent observations add their variances: sigma^2+sigma^2=2*sigma^2,
% making SD(D)=sqrt(2)*sigma. Either mean is Delta from the zero boundary, so
% its standardized distance is Delta/(sqrt(2)*sigma)=d'/sqrt(2):
%   P(correct in 2I-2AFC) = Phi(d'/sqrt(2)).
% In plain language, this is the blue area right of zero in Figure 6 (and the
% symmetric red area left). The full distance between D curves is sqrt(2)*d';
% that is the same geometry viewed peak-to-peak instead of peak-to-boundary.
%
% Green and Swets's area theorem says, under the standard assumptions:
%   P(X_A > X_B) = P(correct in 2I-2AFC) = AUC.
% In plain language, this is the chance that a random A observation produces
% more evidence than a random B observation. Empirical AUC and Mann-Whitney U
% count the same A-B orderings (half credit for ties). U tests group difference;
% AUC describes discrimination. See Green & Swets (1966), Signal Detection
% Theory and Psychophysics, and Bamber (1975), doi:10.1016/0022-2496(75)90001-2.
% Order effects, unequal interval noise, or interval preference can add bias and
% break the area-theorem correspondence.
%
% Balanced yes/no accuracy at the midpoint criterion is Phi(d'/2). In plain
% language, Figure 1's line is half the peak separation from either peak, and
% accuracy is the bell-curve area on the correct side of that line.
assert(abs(pCorrect2afc - aucAnalytic) < 1e-12, ...
    'The 2AFC probability is inconsistent with its analytic value.');
assert(abs(pCorrect2afc - rocAuc) < 5e-4, ...
    'The 2AFC probability and numerical ROC AUC should agree.');

%% Appendix A5. Unequal priors and error costs
% Say "signal" when the evidence is sufficiently more plausible under the
% signal curve. "Sufficiently" depends on base rates and mistake costs:
%   beta_optimal = P(noise)/P(signal) *
%                  (C_FA-C_CR)/(C_M-C_H).
% In plain language, the likelihood ratio is signal-curve height divided by
% noise-curve height at the observation. The first ratio on the right describes
% how common each source is; the second describes consequences of the outcomes.
% With equal costs and P(noise)=3/4, P(signal)=1/4, beta_optimal=3. Figure 7A
% scales each curve by its frequency; their crossing maximizes accuracy, as
% Figure 7B shows. It need not be the unweighted crossing or mean midpoint.

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
title('Figure 7A. Prior-weighted response densities');
legend('Noise curve x its 3-in-4 chance', ...
    'Signal curve x its 1-in-4 chance', ...
    'Weighted intersection', 'Location', 'best');
grid on;

subplot(1, 2, 2);
plot(thresholds, pCorrectUnequal, 'k-', 'LineWidth', 2);
hold on;
xline_compatible(weightedIntersection, 'k--');
xlabel('Threshold');
ylabel('Chance of a correct response');
title('Figure 7B. Accuracy as the threshold moves');
grid on;

%% Appendix A6. A non-Gaussian example
% Figure 8 replaces Figure 1's symmetric bell curves with skewed evidence
% distributions. The skew-normal shape parameter a controls the direction and
% degree of leaning. Its density is 2*phi(z)*Phi(a*z)/scale. Small local
% functions below implement the density, CDF, and random generator so neither
% the Statistics Toolbox nor another package is required.

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
title('Figure 8. A non-Gaussian pair of evidence distributions');
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
