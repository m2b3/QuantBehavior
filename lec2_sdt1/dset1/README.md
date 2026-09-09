# Presaccadic attention data viewer

This repository includes a local Marimo companion for the deposited trial data, processed figure-source matrices, and six verified MATLAB plot exports from Hanning, Fernández & Carrasco (2023).

## Run

```powershell
python -m pip install -r requirements.txt
marimo run viewer.py
```

Open the local URL printed by Marimo (normally <http://127.0.0.1:2718>).

To inspect or modify the reactive notebook, run `marimo edit viewer.py` instead.

The publication explorer uses `public/source_data.json`, an exact,
browser-safe snapshot of the deposited compressed MATLAB matrices. The native
test suite verifies every snapshot value against the original `.mat` files;
this avoids a SciPy/Pyodide decompression failure when the viewer runs as WASM.

## What the viewer contains

- **Signal detection:** recomputes d-prime, criterion, accuracy, hit rate, false-alarm rate, and latency summaries from the subject CSVs. Criterion is shown as a dedicated companion plot (with a zero-bias reference) instead of being discoverable only through the metric dropdown. Controls cover experiment, trial-inclusion profile, plotted dimension, condition splits, participants, extreme-rate correction, signal-code convention, minimum cell size, uncertainty, and participant overlays.
- **Publication explorer:** interactive versions of Figures 2, 3, S1, and S2 built from the supplied MATLAB source matrices. Figure 3 uses the scientifically correct 175-to-25 ms-before-saccade axis.
- **Original plots:** a gallery of the six already-generated PNG files in `results/`, embedded directly so it also renders when the viewer output is nested or exported.
- **Methods & data map:** documents the inferred raw-column meanings and every important analysis choice.

## Signal-detection definitions

The paper defines counter-clockwise reports to counter-clockwise gratings as hits and counter-clockwise reports to clockwise gratings as false alarms:

```text
d' = z(hit rate) - z(false-alarm rate)
c  = -0.5 * [z(hit rate) + z(false-alarm rate)]
```

Its extreme-rate rule replaces rates of 0 and 1 with .01 and .99. The viewer also offers log-linear correction and no correction.

The deposited CSVs do not include a data dictionary. Cross-column consistency identifies the test stimulus and response codes exactly, but it does not name which orientation sign is counter-clockwise. The viewer defaults to `-1` as the signal and lets the user reverse that convention. Reversal flips criterion's sign.

The default **figure-aligned approximation** retains all Experiment 1 neutral trials and saccade trials whose deposited latency is below 350 ms. This empirically minimizes d-prime error against `fig2_psa.mat`; it is not a cutoff stated in the paper. It retains 22,930 rows, 251 more than the reported final count, because the CSVs do not encode the blink/fixation traces or target-relative landing errors required to reproduce every paper exclusion. Experiment 2 uses all deposited rows and reproduces its source d-primes exactly. Select **all deposited rows** in the viewer for an unfiltered sensitivity analysis.

See [RECONCILIATION.md](RECONCILIATION.md) for the candidate rules, validation metrics, participant-row provenance issue, and limitations.

Reproduce the reconciliation table with:

```powershell
python reconcile_exp1.py
```

## Test

```powershell
python -m pytest -q
```
