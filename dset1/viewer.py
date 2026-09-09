import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="full",
    app_title="Presaccadic Attention · Data Viewer",
    css_file="assets/marimo.css",
)


@app.cell
def _():
    from html import escape

    import marimo as mo
    import pandas as pd

    from dashboard.data import (
        EXPERIMENT_LABELS,
        ROOT,
        apply_analysis_profile,
        compute_sdt,
        dataset_summary,
        load_trials,
    )
    from dashboard.figures import METRIC_LABELS, publication_figure, sdt_figure

    SUMMARY = dataset_summary()
    TRIALS = load_trials()

    ERROR_OPTIONS = {
        "Standard SEM · SD/√n": "sem",
        "Paper SEM · SD/√(n−1)": "paper_sem",
        "95% normal CI": "ci95",
        "Standard deviation": "sd",
        "None": "none",
    }

    def box(*children, class_name: str):
        """Wrap Marimo outputs in a styleable HTML container."""
        inner = "".join(
            mo.as_html(child).text for child in children if child is not None
        )
        return mo.Html(f'<div class="{escape(class_name, quote=True)}">{inner}</div>')

    def control(component, hint: str | None = None):
        hint_html = f'<div class="control-hint">{escape(hint)}</div>' if hint else ""
        return mo.Html(
            '<div class="control">' + mo.as_html(component).text + hint_html + "</div>"
        )

    def stat_card(value: str, label: str, note: str):
        return mo.Html(
            '<div class="stat-card">'
            f'<div class="stat-value">{escape(value)}</div>'
            f'<div class="stat-label">{escape(label)}</div>'
            f'<div class="stat-note">{escape(note)}</div>'
            "</div>"
        )

    def prepare_sdt(
        experiment,
        metric,
        x_column,
        splits,
        validities,
        stimulation,
        subjects,
        profile,
        correction,
        signal_code,
        min_trials,
    ):
        if not all(
            [
                experiment,
                metric,
                x_column,
                validities,
                stimulation,
                subjects,
            ]
        ):
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        filtered = TRIALS[
            TRIALS["experiment"].eq(experiment)
            & TRIALS["validity"].isin(validities)
            & TRIALS["test_stimulation"].isin(stimulation)
            & TRIALS["participant"].isin(subjects)
        ].copy()
        filtered = apply_analysis_profile(filtered, profile)
        dimensions = [x_column] + list(splits or [])
        metrics = compute_sdt(
            filtered,
            dimensions,
            correction,
            int(signal_code),
            int(min_trials),
        )
        valid = metrics.dropna(subset=[metric]) if not metrics.empty else metrics
        table_groups = [x_column] + [
            dimension for dimension in (splits or []) if dimension != x_column
        ]
        summary_rows = []
        if not valid.empty:
            for keys, cell in valid.groupby(table_groups, observed=True, dropna=False):
                if not isinstance(keys, tuple):
                    keys = (keys,)
                row = dict(zip(table_groups, keys))
                row.update(
                    {
                        "mean": cell[metric].mean(),
                        "sd": cell[metric].std(ddof=1),
                        "participants": cell["participant"].nunique(),
                        "trials": int(cell["n_trials"].sum()),
                    }
                )
                summary_rows.append(row)
        return filtered, metrics, pd.DataFrame(summary_rows)

    def display_summary(summary, x_column, metric):
        display = summary.copy()
        if display.empty:
            return display
        display.rename(
            columns={
                x_column: x_column.replace("_", " ").title(),
                "mean": f"Mean {METRIC_LABELS.get(metric, metric)}",
                "sd": "SD",
                "participants": "N participants",
                "trials": "N trials",
            },
            inplace=True,
        )
        for column in display.select_dtypes(include="number"):
            if not column.lower().startswith("n ") and column != "trials":
                display[column] = display[column].round(4)
        return display

    return (
        ERROR_OPTIONS,
        EXPERIMENT_LABELS,
        METRIC_LABELS,
        ROOT,
        SUMMARY,
        TRIALS,
        box,
        control,
        display_summary,
        escape,
        mo,
        pd,
        prepare_sdt,
        publication_figure,
        sdt_figure,
        stat_card,
    )


@app.cell
def _(box, mo):
    header = box(
        mo.Html(
            """
            <div class="brand">
              <div class="brand-kicker">OPEN DATA COMPANION</div>
              <h1>Presaccadic attention, unpacked</h1>
              <p>A transparent viewer for behavioral trials, signal-detection
              measures, and the figures behind Hanning, Fernández &amp; Carrasco
              (2023).</p>
            </div>
            <div class="local-badge">
              <span>LOCAL</span><div>No upload<br>No telemetry</div>
            </div>
            """
        ),
        class_name="hero",
    )
    footer = box(
        mo.Html(
            """
            <span>Data and analysis remain in this repository.</span>
            <span>Nature Communications 14, 5381 · 2023</span>
            """
        ),
        class_name="viewer-footer",
    )
    return footer, header


@app.cell
def _(SUMMARY, box, mo, stat_card):
    _stats = box(
        stat_card(
            f"{SUMMARY['trials']:,}",
            "deposited trial rows",
            "Across all raw CSV matrices",
        ),
        stat_card(
            str(SUMMARY["participants"]),
            "participant files",
            "10 + 9 + 7 across three experiments",
        ),
        stat_card(
            str(SUMMARY["figures"]),
            "verified plot outputs",
            "Main and supplementary figures",
        ),
        stat_card(
            "2",
            "analysis layers",
            "Raw trials + publication source data",
        ),
        class_name="stats-grid",
    )
    _narrative = box(
        mo.Html(
            """
            <div class="eyebrow">A VIEWER WITH TWO JOBS</div>
            <h2>Explore the inference, then check it against the paper</h2>
            <p>Signal detection recomputes d′ and criterion directly from
            deposited trials. Publication explorer uses the processed source
            matrices behind the supplied figures.</p>
            <div class="steps">
              <div class="step"><span>01</span><div><h3>Signal detection</h3>
                <p>Change experiment, x-axis, condition splits, rate correction,
                and aggregation uncertainty.</p></div></div>
              <div class="step"><span>02</span><div><h3>Publication explorer</h3>
                <p>Inspect group trends, participant trajectories, fitted
                contrast-response functions, and eye controls.</p></div></div>
              <div class="step"><span>03</span><div><h3>Original outputs</h3>
                <p>Compare interactive views with the six already-verified
                MATLAB exports.</p></div></div>
            </div>
            """
        ),
        class_name="card narrative-card",
    )
    _caveat = box(
        mo.Html(
            """
            <div class="eyebrow amber">READ THIS FIRST</div>
            <h2>What is known—and what is inferred</h2>
            <p>The paper explicitly defines d′ = z(hit rate) − z(false-alarm
            rate), with counter-clockwise reports treated as “yes.” Extreme
            rates are replaced by .01 and .99.</p>
            <p>The raw CSVs have 13 unlabeled columns and no deposited codebook.
            Their internal relationships identify the stimulus, response,
            correctness, cue, test, and stimulation codes, but the sign assigned
            to counter-clockwise remains an assumption.</p>
            <p>Accordingly, the SDT tab lets you flip the signal sign. d′ is
            effectively invariant to that relabeling; criterion reverses sign.</p>
            <div class="callout">Publication source values and raw-trial
            recomputations are intentionally kept separate.</div>
            """
        ),
        class_name="card caveat-card",
    )
    overview_page = box(
        _stats,
        box(_narrative, _caveat, class_name="overview-grid"),
        class_name="tab-page",
    )
    return (overview_page,)


@app.cell
def _(ERROR_OPTIONS, EXPERIMENT_LABELS, METRIC_LABELS, mo):
    sdt_experiment = mo.ui.dropdown(
        options={label: value for value, label in EXPERIMENT_LABELS.items()},
        value=EXPERIMENT_LABELS["Exp1"],
        allow_select_none=False,
        label="**Experiment**",
        full_width=True,
    )
    sdt_metric = mo.ui.dropdown(
        options={label: value for value, label in METRIC_LABELS.items()},
        value=METRIC_LABELS["dprime"],
        allow_select_none=False,
        label="**Metric**",
        full_width=True,
    )
    sdt_stimulation = mo.ui.multiselect(
        options=["Stimulated", "Not stimulated"],
        value=["Stimulated", "Not stimulated"],
        label="**Test location**",
        full_width=True,
    )
    sdt_profile = mo.ui.radio(
        options={
            "Figure-aligned · Exp. 1 saccade latency < 350 ms": "figure_aligned",
            "All deposited rows": "all_rows",
        },
        value="Figure-aligned · Exp. 1 saccade latency < 350 ms",
        label="**Trial inclusion**",
    )
    sdt_correction = mo.ui.radio(
        options={
            "Paper · replace 0/1 with .01/.99": "paper",
            "Log-linear · add 0.5": "loglinear",
            "None · omit infinite cells": "none",
        },
        value="Paper · replace 0/1 with .01/.99",
        label="**Extreme-rate correction**",
    )
    sdt_signal = mo.ui.radio(
        options={"−1 (assumed counter-clockwise)": -1, "+1": 1},
        value="−1 (assumed counter-clockwise)",
        label="**Signal / “yes” orientation code**",
    )
    sdt_min_trials = mo.ui.slider(
        start=1,
        stop=20,
        step=1,
        value=1,
        show_value=True,
        include_input=True,
        label="**Minimum trials in each class**",
        full_width=True,
    )
    sdt_error = mo.ui.dropdown(
        options=ERROR_OPTIONS,
        value="Standard SEM · SD/√n",
        allow_select_none=False,
        label="**Group uncertainty**",
        full_width=True,
    )
    sdt_show_subjects = mo.ui.checkbox(
        value=False, label="Show participant trajectories"
    )
    return (
        sdt_correction,
        sdt_error,
        sdt_experiment,
        sdt_metric,
        sdt_min_trials,
        sdt_profile,
        sdt_show_subjects,
        sdt_signal,
        sdt_stimulation,
    )


@app.cell
def _(TRIALS, mo, sdt_experiment):
    _common_x = {
        "Attention condition": "validity",
        "Test location · stimulated or control": "test_stimulation",
        "Participant": "participant",
    }
    if sdt_experiment.value == "Exp1":
        _x_options = {"Contrast (%)": "contrast_pct", **_common_x}
        _x_default = "Contrast (%)"
        _validities = ["Valid", "Neutral", "Invalid"]
    else:
        _x_options = {
            "Time before saccade onset": "time_before_saccade_ms",
            "Planned first TMS pulse": "planned_tms_ms",
            **_common_x,
        }
        _x_default = "Time before saccade onset"
        _validities = ["Valid", "Invalid"]

    sdt_x = mo.ui.dropdown(
        options=_x_options,
        value=_x_default,
        allow_select_none=False,
        label="**X-axis**",
        full_width=True,
    )
    sdt_splits = mo.ui.multiselect(
        options={
            "Attention condition": "validity",
            "Test location · stimulated or control": "test_stimulation",
        },
        value=["Attention condition", "Test location · stimulated or control"],
        label="**Split curves by**",
        full_width=True,
    )
    _subjects = sorted(
        TRIALS.loc[
            TRIALS["experiment"].eq(sdt_experiment.value), "participant"
        ].unique()
    )
    sdt_subjects = mo.ui.multiselect(
        options=_subjects,
        value=_subjects,
        label="**Participants**",
        full_width=True,
    )
    sdt_validity = mo.ui.multiselect(
        options=_validities,
        value=_validities,
        label="**Attention condition**",
        full_width=True,
    )
    return sdt_splits, sdt_subjects, sdt_validity, sdt_x


@app.cell
def _(
    prepare_sdt,
    sdt_correction,
    sdt_experiment,
    sdt_metric,
    sdt_min_trials,
    sdt_profile,
    sdt_signal,
    sdt_splits,
    sdt_stimulation,
    sdt_subjects,
    sdt_validity,
    sdt_x,
):
    sdt_filtered, sdt_metrics, sdt_summary = prepare_sdt(
        sdt_experiment.value,
        sdt_metric.value,
        sdt_x.value,
        sdt_splits.value,
        sdt_validity.value,
        sdt_stimulation.value,
        sdt_subjects.value,
        sdt_profile.value,
        sdt_correction.value,
        sdt_signal.value,
        sdt_min_trials.value,
    )
    return sdt_filtered, sdt_metrics, sdt_summary


@app.cell
def _(
    METRIC_LABELS,
    box,
    control,
    display_summary,
    mo,
    sdt_correction,
    sdt_error,
    sdt_experiment,
    sdt_figure,
    sdt_filtered,
    sdt_metric,
    sdt_metrics,
    sdt_min_trials,
    sdt_profile,
    sdt_show_subjects,
    sdt_signal,
    sdt_splits,
    sdt_stimulation,
    sdt_subjects,
    sdt_summary,
    sdt_validity,
    sdt_x,
):
    _figure = sdt_figure(
        sdt_metrics,
        sdt_metric.value,
        sdt_x.value,
        sdt_splits.value,
        sdt_error.value,
        sdt_show_subjects.value,
    )
    _plot = mo.ui.plotly(
        _figure,
        config={
            "displaylogo": False,
            "responsive": True,
            "toImageButtonOptions": {
                "format": "svg",
                "filename": "signal_detection",
            },
        },
    )
    _display = display_summary(sdt_summary, sdt_x.value, sdt_metric.value)
    _table = (
        mo.ui.table(
            _display,
            pagination=True,
            selection=None,
            page_size=12,
            show_column_summaries=False,
            show_data_types=False,
            show_download=False,
            max_height=450,
        )
        if not _display.empty
        else mo.md("_No finite group-summary cells match these controls._")
    )
    _download = mo.download(
        data=sdt_summary.to_csv(index=False).encode("utf-8"),
        filename="signal_detection_summary.csv",
        mimetype="text/csv",
        disabled=sdt_summary.empty,
        label="Download CSV",
    )
    _total_cells = len(sdt_metrics)
    _usable = (
        int(sdt_metrics[sdt_metric.value].notna().sum()) if not sdt_metrics.empty else 0
    )
    _kpis = box(
        mo.Html(
            f"<div><strong>{len(sdt_filtered):,}</strong><span>selected trials</span></div>"
        ),
        mo.Html(
            f"<div><strong>{sdt_filtered['participant'].nunique() if not sdt_filtered.empty else 0}</strong><span>participants</span></div>"
        ),
        mo.Html(
            f"<div><strong>{_usable}/{_total_cells}</strong><span>finite cells</span></div>"
        ),
        mo.Html(
            f"<div><strong>{METRIC_LABELS.get(sdt_metric.value, sdt_metric.value)}</strong><span>active metric</span></div>"
        ),
        class_name="mini-stats",
    )
    _advanced = mo.accordion(
        {
            "Analysis choices": mo.vstack(
                [
                    control(
                        sdt_profile,
                        "The empirical cutoff improves agreement with Figure 2 source data; Experiment 2 is unchanged.",
                    ),
                    control(sdt_correction),
                    control(
                        sdt_signal,
                        "Changing this reverses the sign of criterion.",
                    ),
                    control(sdt_min_trials),
                    control(sdt_error),
                    control(sdt_show_subjects),
                ],
                gap=0,
            )
        }
    )
    _sidebar = box(
        mo.Html(
            '<div class="eyebrow">RAW TRIAL ANALYSIS</div><h2>Signal detection controls</h2>'
        ),
        control(sdt_experiment),
        control(sdt_metric),
        control(sdt_x),
        control(
            sdt_splits,
            "Metrics are always computed within participant before the group mean.",
        ),
        control(sdt_validity),
        control(sdt_stimulation),
        control(sdt_subjects),
        _advanced,
        class_name="sidebar card",
    )
    _table_heading = box(
        mo.Html(
            "<div><h3>Current group summary</h3><p>Values shown here are the same participant-first means plotted above.</p></div>"
        ),
        _download,
        class_name="table-heading",
    )
    _analysis = box(
        _kpis,
        _plot,
        _table_heading,
        _table,
        class_name="analysis-panel card",
    )
    _method = box(
        mo.Html(
            """
            <h3>Interpretation key</h3>
            <p><strong>d′</strong> measures separation of the two
            orientation-response distributions. Higher is better; response
            preference cancels out.</p>
            <p><strong>Criterion c</strong> = −½[z(H) + z(FA)]. With −1
            designated as signal, c &gt; 0 means fewer −1 responses (a
            conservative bias); c &lt; 0 means more −1 responses.</p>
            <p>Rates are computed inside each participant × displayed cell.
            Group lines average those participant metrics, avoiding trial-count
            weighting.</p>
            """
        ),
        class_name="method-strip",
    )
    sdt_page = box(
        box(_sidebar, _analysis, class_name="analysis-grid"),
        _method,
        class_name="tab-page",
    )
    return (sdt_page,)


@app.cell
def _(ERROR_OPTIONS, mo):
    pub_figure_choice = mo.ui.dropdown(
        options={
            "Figure 2a · presaccadic contrast response": "fig2_psa",
            "Figure 2b · dmax attention effects": "fig2b",
            "Figure 3 · presaccadic time course": "fig3",
            "Figure S1a · exogenous contrast response": "fig2_exo",
            "Figure S1b · endogenous contrast response": "fig2_endo",
            "Figure S2 · eye-movement controls": "figS2",
        },
        value="Figure 2a · presaccadic contrast response",
        allow_select_none=False,
        label="**Figure family**",
        full_width=True,
    )
    pub_error = mo.ui.dropdown(
        options=ERROR_OPTIONS,
        value="Paper SEM · SD/√(n−1)",
        allow_select_none=False,
        label="**Uncertainty**",
        full_width=True,
    )
    pub_show_subjects = mo.ui.checkbox(
        value=False, label="Overlay participant trajectories"
    )
    return pub_error, pub_figure_choice, pub_show_subjects


@app.cell
def _(
    box,
    control,
    mo,
    pub_error,
    pub_figure_choice,
    pub_show_subjects,
    publication_figure,
):
    _publication_plot = mo.ui.plotly(
        publication_figure(
            pub_figure_choice.value,
            pub_error.value,
            pub_show_subjects.value,
        ),
        config={
            "displaylogo": False,
            "responsive": True,
            "toImageButtonOptions": {
                "format": "svg",
                "filename": "publication_figure",
            },
        },
    )
    _publication_sidebar = box(
        mo.Html(
            '<div class="eyebrow">PROCESSED SOURCE DATA</div><h2>Publication explorer</h2>'
        ),
        control(pub_figure_choice),
        control(
            pub_error,
            "“Paper SEM” reproduces the supplied MATLAB denominator.",
        ),
        control(pub_show_subjects),
        mo.Html(
            """
            <div class="callout"><strong>Axis correction</strong><p>Figure 3
            is shown as 175 → 25 ms before saccade onset, matching the
            source-data bins and paper—not the increasing 0 → 200 labels in
            the supplied MATLAB output.</p></div>
            """
        ),
        class_name="sidebar card",
    )
    publication_page = box(
        box(
            _publication_sidebar,
            box(_publication_plot, class_name="analysis-panel card"),
            class_name="analysis-grid",
        ),
        class_name="tab-page",
    )
    return (publication_page,)


@app.cell
def _(ROOT, box, mo):
    _gallery_items = [
        (
            "Figure 2a",
            "Figure_2a.png",
            "Presaccadic contrast-response functions",
        ),
        ("Figure 2b", "Figure_2b.png", "dmax attention effects"),
        (
            "Figure 3",
            "Figure_3.png",
            "Presaccadic time course; note the source-script axis issue",
        ),
        (
            "Figure S1a",
            "Figure_S1a.png",
            "Covert exogenous attention",
        ),
        (
            "Figure S1b",
            "Figure_S1b.png",
            "Covert endogenous attention",
        ),
        (
            "Figure S2",
            "Figure_S2.png",
            "Saccade latency and landing error",
        ),
    ]
    _gallery_cards = [
        box(
            mo.image(
                ROOT / "results" / filename,
                alt=f"Generated {title}",
                width="100%",
            ),
            mo.Html(
                f"<figcaption><strong>{title}</strong><span>{caption}</span></figcaption>"
            ),
            class_name="gallery-card",
        )
        for title, filename, caption in _gallery_items
    ]
    gallery_page = box(
        box(
            mo.Html(
                """
                <div class="eyebrow">MATLAB EXPORTS</div>
                <h2>Original generated outputs</h2>
                <p>These are the six verified figures already present in
                <code>results/</code>. Use the camera control in interactive
                graphs to export new views.</p>
                """
            ),
            class_name="section-intro",
        ),
        box(*_gallery_cards, class_name="gallery-grid"),
        class_name="tab-page",
    )
    return (gallery_page,)


@app.cell
def _(box, mo, pd):
    _rows = [
        ("1", "Block", "Block index"),
        ("2", "Trial", "Trial number within block"),
        (
            "3",
            "Design",
            "Exp. 1: fixation/saccade flag; Exp. 2: planned first-pulse timing code",
        ),
        (
            "4",
            "Analysis level",
            "Exp. 1: contrast code; Exp. 2: TMS-to-saccade time-bin code",
        ),
        ("5", "Cue location", "0 neutral, 1 left, 2 right"),
        (
            "6–7",
            "Stimulus orientations",
            "Signed left and right grating codes",
        ),
        ("8", "Test location", "1 left, 2 right"),
        (
            "9",
            "Stimulated hemifield",
            "Participant-specific left/right code",
        ),
        ("10", "Response", "Signed orientation response"),
        ("11", "Correct", "0 incorrect, 1 correct"),
        ("12", "Saccade latency", "Milliseconds"),
        (
            "13",
            "Landing position",
            "Degrees; not treated as landing error without target coordinates",
        ),
    ]
    _map_table = pd.DataFrame(
        _rows, columns=["Column", "Viewer name", "Interpretation"]
    ).to_html(index=False, escape=True, classes="data-map-table")
    _raw_map = box(
        mo.Html(f"<h3>Raw CSV column map</h3>{_map_table}"),
        class_name="card notes-card",
    )
    _decisions = box(
        mo.Html(
            """
            <h3>Analysis decisions</h3>
            <ul>
              <li>Signal-detection measures use the orientation at the queried
              test location, not the distractor orientation.</li>
              <li>The paper correction changes only rates exactly equal to 0 or
              1; intermediate rates are untouched.</li>
              <li>Criterion is added using the standard equal-variance
              definition c = −½[z(H)+z(FA)]. It was not reported in the paper.</li>
              <li>Raw-data group summaries are participant-first. Publication
              explorer reads the supplied .mat source matrices.</li>
              <li>The default figure-aligned profile retains Experiment 1
              neutral trials and saccade trials with deposited latency &lt;350
              ms. This cutoff was inferred by minimizing d′ error against
              fig2_psa.mat; it is not a stated paper rule.</li>
              <li>That profile leaves 22,930 Experiment 1 rows, versus 22,679
              reported. Blink, fixation, and true landing-error exclusions
              cannot be reconstructed from these CSV columns.</li>
              <li>The Excel workbook labels Figure 2 rows S01–S10, but later
              Experiment 1 CSV identities do not align numerically with those
              labels. Deposited CSV names are preserved; no participant is
              silently relabeled or excluded.</li>
              <li>Column 13 is exposed only as landing position. Figure S2
              landing error comes from the processed source matrix.</li>
            </ul>
            <h3>Run and share</h3>
            <pre>python -m pip install -r requirements.txt
marimo run viewer.py</pre>
            <p>The viewer runs locally and does not transmit data. Plotly graphs
            can be exported as SVG from the camera menu.</p>
            """
        ),
        class_name="card notes-card",
    )
    notes_page = box(
        box(
            mo.Html(
                """
                <div class="eyebrow">REPRODUCIBILITY NOTES</div>
                <h2>Data map and scope</h2>
                <p>The mapping below is inferred from code ranges,
                deterministic cross-column relationships, the experimental
                design, and the processed source matrices. No CSV data
                dictionary was deposited.</p>
                """
            ),
            class_name="section-intro",
        ),
        box(_raw_map, _decisions, class_name="notes-grid"),
        class_name="tab-page",
    )
    return (notes_page,)


@app.cell
def _(
    footer,
    gallery_page,
    header,
    mo,
    notes_page,
    overview_page,
    publication_page,
    sdt_page,
):
    navigation = mo.ui.tabs(
        {
            "Overview": overview_page,
            "Signal detection": sdt_page,
            "Publication explorer": publication_page,
            "Original plots": gallery_page,
            "Methods & data map": notes_page,
        },
        value="Overview",
    )
    mo.Html(
        '<div class="app-shell">'
        + mo.as_html(header).text
        + mo.as_html(navigation).text
        + mo.as_html(footer).text
        + "</div>"
    )
    return (navigation,)


if __name__ == "__main__":
    app.run()
