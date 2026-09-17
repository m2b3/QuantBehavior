# Likelihood and neural populations

An interactive marimo tutorial accompanying Jazayeri & Movshon (2006). It
distinguishes response probability, likelihood, and posterior probability,
then develops conditional independence, log likelihood, Poisson population
readouts, expected-count corrections, and noise correlation through figures.
The task examples use an explicit signal-versus-noise likelihood ratio for
detection and explain the assumptions behind estimation and discrimination.
Expanded figure explanations identify the data held fixed, the response model,
and the meaning of each plotting scale. All numerical examples are illustrative.

Run it with:

```powershell
marimo edit likelihood_population_tutorial.py
```

Code starts folded. The project settings make cells instantiate automatically and rerun reactively
when a control changes. Build the editable, pre-executed WebAssembly site with:

```powershell
python build_wasm.py
```

The site is written to
`../docs/lec3/likelihood_population_tutorial/index.html`. The helper also
patches marimo's generated mount configuration so all cells run on first load;
marimo 0.23.x otherwise exports editable WASM with `auto_instantiate` disabled.

The notebook carries its Python dependencies in PEP 723 script metadata, so it
can also be opened with tools that honor inline script dependencies.
