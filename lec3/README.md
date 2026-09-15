# Likelihood and neural populations

An interactive marimo tutorial inspired by Jazayeri & Movshon (2006). It builds
from the meaning of likelihood to multiplication, log-likelihood accumulation,
weighted population readout, the Poisson link, homogeneous-population
assumptions, correlations, and task-level decisions.

Run it with:

```powershell
marimo edit likelihood_population_tutorial.py
```

The project settings make cells instantiate automatically and rerun reactively
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
