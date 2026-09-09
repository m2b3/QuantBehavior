# CSV-to-MAT reconciliation

The MATLAB plotting script does not create the source arrays from trial data. It loads `fig2_psa.mat`, `fig3_V1V2.mat`, and `fig3_FEF.mat` directly. The deposited repository therefore lacks the original trial-selection and d-prime construction script. This audit uses the paper's Methods to define candidate exclusions and treats the `.mat` d-primes as the validation target.

## Stable decoding

For each CSV trial, the stimulus at column 8's queried location is selected from column 6 or 7. Its equality with the response in column 10 reproduces correctness in column 11 for every deposited row. Cue validity and stimulated/test-side mappings also reproduce every Experiment 2 source d-prime. These mappings are therefore retained.

Sensitivity is computed within each participant × experimental cell as:

```text
d' = z(hit rate) - z(false-alarm rate)
```

Following the paper, rates exactly equal to 0 or 1 are replaced by .01 or .99. Reversing which signed orientation is called the signal leaves d-prime unchanged and reverses criterion.

## Paper exclusions

The Methods report online repetition of trials with broken fixation, saccade latency below 150 or above 500 ms, or landing more than 2.5° from the target. Offline, trials required no blink, fixation within 1.75°, an initial saccade within 2.0° of the target, and test-signal presentation within 100 ms before saccade onset.

Those rules cannot all be applied to the CSVs: there are no eye traces, blink flags, fixation deviations, target coordinates, or target-relative landing errors. Column 13 is an absolute landing position, so filtering its raw magnitude or distance from a participant median would not implement the paper's landing-error rule.

## Candidate comparison

The target comprises 36 group-mean Figure 2a cells (3 attention conditions × 2 test/stimulation relationships × 6 contrasts). The MAT fields themselves contain no names, but `Source_Data.xlsx` explicitly labels their ten rows `S 01` through `S 10`. The workbook and MAT values agree to their displayed four-decimal precision (maximum difference 0.00005). Same-numbered rows are therefore the intended participant correspondence.

As a diagnostic—not as a legitimate relabeling—the audit also finds the one-to-one row assignment that minimizes error across all 360 participant-level cells. This reveals possible deposited-file or source-label confusion.

| Experiment 1 rule | Included rows | Group d-prime MAE vs `.mat` |
|---|---:|---:|
| All deposited rows | 24,035 | 0.0604 |
| Saccade latency ≤300 ms | 21,974 | 0.0623 |
| Saccade latency ≤350 ms | 22,945 | 0.0539 |
| Saccade latency <350 ms | 22,930 | **0.0533**; best boundary in the fine-grained individual-cell audit |
| ≤350 ms plus median-centered landing radius 2.5° | 22,680 | 0.0578 |

The last rule happens to come within one row of the paper's 22,679 total but fits d-prime worse and misuses absolute landing position as error. It is rejected.

With latency `<350 ms`, the intended same-ID individual-cell mean absolute error is 0.3030. The diagnostic optimal assignment reduces it to 0.0813 across all 360 cells and to 0.0270 for the nine close pairings. Of those 324 cells, 181 are equal to the source within 0.00005 and 187 within 0.01. Candidate thresholds from 300 through 400 ms have their minimum at 349 ms inclusive, equivalent to `<350 ms` for the integer-valued latency column.

## Participant-identity conflict

The Excel labels say that each CSV should match its same-numbered source row. That works for S01–S06. Instead, the numerical diagnostic gives S08→Excel S10, S09→Excel S08, and S10→Excel S07. Raw S07 does not closely match Excel S09 or any other source subject under signal-sign reversal, test/stimulation-side reversal, or session/block subsets.

This is unlikely to be a Python sorting error: participant identity is read directly from each CSV filename, not inferred from file order. Moreover, Experiment 2a CSV S01–S09 and Experiment 2b CSV S01–S07 each reproduce their same-numbered Excel/MAT rows to rounding precision. The conflict is specific to the later Experiment 1 files. The available artifacts cannot determine whether the CSV filenames, the processed source labels/order, or one exported data version is wrong. The viewer consequently preserves the deposited CSV names and does not silently relabel or exclude anyone.

## Adopted interpretation

The viewer defaults to a transparent **figure-aligned approximation**:

- Experiment 1 neutral/fixation trials: retain all deposited rows because their eye fields are empty.
- Experiment 1 saccade trials: retain rows with deposited latency `<350 ms`.
- Experiments 2a and 2b: retain all deposited rows; these reproduce the corresponding source d-primes exactly.

The viewer also offers **all deposited rows**. Criterion is computed from raw responses under the selected profile and is never presented as a value recovered from the `.mat` files.

## Data-audit addendum: participant reordering and S07

For practical reconciliation, participant rows may be treated as reordered between the Experiment 1 CSV export and the processed source arrays. Under the `<350 ms` profile, the minimum-error one-to-one assignment is:

| CSV participant | Processed source row | Agreement |
|---|---:|---|
| S01–S06 | 1–6, respectively | Close |
| S08 | 10 | Close |
| S09 | 8 | Close |
| S10 | 7 | Close |
| S07 | 9 | Poor; this is the remaining assignment |

With this reordering, nine participants have a mean d-prime absolute error of 0.0270 across their 36 cells. The remaining CSV participant, S07, has a mean error of about 0.57 against its best available source row. It is therefore described as **unreconciled**, not invalid or excluded.

### Is CSV S07 behaviorally unusual?

Not at the aggregate level. Under the adopted profile, S07 retains 2,500 trials and falls within the ordinary range of the other nine CSV participants:

| Summary measure | CSV S07 | Standardized against the other nine |
|---|---:|---:|
| Overall d-prime | 1.334 | +0.24 SD |
| Valid d-prime | 2.408 | +0.92 SD |
| Neutral d-prime | 1.333 | −0.06 SD |
| Invalid d-prime | 0.262 | −0.69 SD |
| Valid − invalid d-prime | 2.146 | +0.91 SD |
| Overall criterion | 0.035 | +0.29 SD |
| Valid − invalid criterion | 0.023 | +0.12 SD |

The detailed profile is somewhat irregular: 6 of its 36 condition-by-contrast d-primes are more than 2 SD from the corresponding values of the other nine participants, with a maximum absolute deviation of 2.53 SD. These deviations occur in both directions rather than reflecting uniformly poor sensitivity or a uniformly shifted response criterion.

### Interpretation

Once row reordering is allowed, the evidence does not show a general Experiment 1 participant-label failure. It shows nine closely reconcilable datasets and one unreconciled dataset. Because the CSV lacks blink flags, fixation traces, target coordinates, and target-relative landing error, additional offline exclusions could plausibly account for the remaining participant's detailed mismatch. The available source files contain only processed d-prime values and cannot test that explanation.

S07 should therefore remain in the primary raw-CSV criterion analysis. A leave-S07-out result can be reported as a sensitivity analysis, but the source mismatch alone is not a principled exclusion criterion.
