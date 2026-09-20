# Likelihood and neural populations

An interactive marimo tutorial accompanying Jazayeri & Movshon (2006). It
starts with a three-panel coin comparison followed by clickable outcomes,
then checks the multiplication rule for independent events. It introduces three
neurons, three discrete response categories, and an explicit table of response probabilities for
each stimulus. Worked examples build from
one neuron's likelihood to the product for all three neurons, followed by
response tables that students can click directly. Linked panels highlight each
observed response, plot the individual likelihoods, and show their joint product
and likelihood ratio. A step-by-step dot display follows 1,000 expected trials
through the three response probabilities. A collapsed optional panel includes a
prior-odds slider: the posterior changes while the likelihood stays fixed, with
equal posterior probabilities for leftward and rightward at prior odds of
14 to 1. It then develops continuous
measurements, log likelihood, Poisson population readouts, expected-count
corrections, and noise correlation through figures.
The task examples use an explicit signal-versus-noise likelihood ratio for
detection and explain the assumptions behind estimation and discrimination.
Expanded figure explanations identify the data held fixed, the response model,
and the meaning of each plotting scale. All numerical examples are illustrative.

Run it with:

```powershell
marimo edit --sandbox likelihood_population_tutorial.py
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
The introductory graphics use inline `anywidget` views, with no external
JavaScript libraries or asset files. Their controls are keyboard accessible and
respect reduced-motion preferences. The Python response model supplies the
probability tables to the views; widget state is available through the marimo
widget values. For a local build without a sandbox, install the inline Python
dependencies first (including `anywidget` and `traitlets`).
