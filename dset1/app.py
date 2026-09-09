from __future__ import annotations

import os
from io import StringIO

import pandas as pd
from dash import Dash, Input, Output, State, callback, dash_table, dcc, html, no_update
from flask import send_from_directory

from dashboard.data import (
    EXPERIMENT_LABELS,
    ROOT,
    apply_analysis_profile,
    compute_sdt,
    dataset_summary,
    load_trials,
)
from dashboard.figures import METRIC_LABELS, publication_figure, sdt_figure


app = Dash(__name__, title="Presaccadic Attention · Data Viewer", suppress_callback_exceptions=True)
server = app.server


@server.route("/results/<path:filename>")
def serve_result(filename: str):
    return send_from_directory(ROOT / "results", filename)


SUMMARY = dataset_summary()
TRIALS = load_trials()

ERROR_OPTIONS = [
    {"label": "Standard SEM · SD/√n", "value": "sem"},
    {"label": "Paper SEM · SD/√(n−1)", "value": "paper_sem"},
    {"label": "95% normal CI", "value": "ci95"},
    {"label": "Standard deviation", "value": "sd"},
    {"label": "None", "value": "none"},
]


def control(label: str, component, hint: str | None = None):
    children = [html.Label(label, className="control-label"), component]
    if hint:
        children.append(html.Div(hint, className="control-hint"))
    return html.Div(children, className="control")


def stat_card(value: str, label: str, note: str):
    return html.Div(
        [html.Div(value, className="stat-value"), html.Div(label, className="stat-label"), html.Div(note, className="stat-note")],
        className="stat-card",
    )


def overview_tab():
    return html.Div(
        [
            html.Div(
                [
                    stat_card(f"{SUMMARY['trials']:,}", "deposited trial rows", "Across all raw CSV matrices"),
                    stat_card(str(SUMMARY["participants"]), "participant files", "10 + 9 + 7 across three experiments"),
                    stat_card(str(SUMMARY["figures"]), "verified plot outputs", "Main and supplementary figures"),
                    stat_card("2", "analysis layers", "Raw trials + publication source data"),
                ],
                className="stats-grid",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("A viewer with two jobs", className="eyebrow"),
                            html.H2("Explore the inference, then check it against the paper"),
                            html.P(
                                "Signal detection recomputes d′ and criterion directly from deposited trials. "
                                "Publication explorer uses the processed source matrices behind the supplied figures."
                            ),
                            html.Div(
                                [
                                    html.Div([html.Span("01"), html.Div([html.H3("Signal detection"), html.P("Change experiment, x-axis, condition splits, rate correction, and aggregation uncertainty.")])], className="step"),
                                    html.Div([html.Span("02"), html.Div([html.H3("Publication explorer"), html.P("Inspect group trends, participant trajectories, fitted contrast-response functions, and eye controls.")])], className="step"),
                                    html.Div([html.Span("03"), html.Div([html.H3("Original outputs"), html.P("Compare interactive views with the six already-verified MATLAB exports.")])], className="step"),
                                ],
                                className="steps",
                            ),
                        ],
                        className="card narrative-card",
                    ),
                    html.Div(
                        [
                            html.Div("Read this first", className="eyebrow amber"),
                            html.H2("What is known—and what is inferred"),
                            html.P("The paper explicitly defines d′ = z(hit rate) − z(false-alarm rate), with counter-clockwise reports treated as “yes.” Extreme rates are replaced by .01 and .99."),
                            html.P("The raw CSVs have 13 unlabeled columns and no deposited codebook. Their internal relationships identify the stimulus, response, correctness, cue, test, and stimulation codes, but the sign assigned to counter-clockwise remains an assumption."),
                            html.P("Accordingly, the SDT tab lets you flip the signal sign. d′ is effectively invariant to that relabeling; criterion reverses sign."),
                            html.Div("Publication source values and raw-trial recomputations are intentionally kept separate.", className="callout"),
                        ],
                        className="card caveat-card",
                    ),
                ],
                className="overview-grid",
            ),
        ],
        className="tab-page",
    )


def sdt_tab():
    subjects = sorted(TRIALS.loc[TRIALS["experiment"].eq("Exp1"), "participant"].unique())
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("RAW TRIAL ANALYSIS", className="eyebrow"),
                            html.H2("Signal detection controls"),
                            control("Experiment", dcc.Dropdown(id="sdt-experiment", options=[{"label": v, "value": k} for k, v in EXPERIMENT_LABELS.items()], value="Exp1", clearable=False)),
                            control("Metric", dcc.Dropdown(id="sdt-metric", options=[{"label": label, "value": value} for value, label in METRIC_LABELS.items()], value="dprime", clearable=False)),
                            control("X-axis", dcc.Dropdown(id="sdt-x", clearable=False)),
                            control("Split curves by", dcc.Dropdown(id="sdt-splits", multi=True), "Metrics are always computed within participant before the group mean."),
                            control("Attention condition", dcc.Checklist(id="sdt-validity", options=[{"label": x, "value": x} for x in ["Valid", "Neutral", "Invalid"]], value=["Valid", "Neutral", "Invalid"], className="check-grid")),
                            control("Test location", dcc.Checklist(id="sdt-stimulation", options=[{"label": x, "value": x} for x in ["Stimulated", "Not stimulated"]], value=["Stimulated", "Not stimulated"], className="check-grid")),
                            control("Participants", dcc.Dropdown(id="sdt-subjects", options=[{"label": s, "value": s} for s in subjects], value=subjects, multi=True)),
                            html.Details(
                                [
                                    html.Summary("Analysis choices"),
                                    control(
                                        "Trial inclusion",
                                        dcc.RadioItems(
                                            id="sdt-profile",
                                            options=[
                                                {
                                                    "label": "Figure-aligned · Exp. 1 saccade latency < 350 ms",
                                                    "value": "figure_aligned",
                                                },
                                                {
                                                    "label": "All deposited rows",
                                                    "value": "all_rows",
                                                },
                                            ],
                                            value="figure_aligned",
                                        ),
                                        "The empirical cutoff improves agreement with Figure 2 source data; Experiment 2 is unchanged.",
                                    ),
                                    control("Extreme-rate correction", dcc.RadioItems(id="sdt-correction", options=[{"label": "Paper · replace 0/1 with .01/.99", "value": "paper"}, {"label": "Log-linear · add 0.5", "value": "loglinear"}, {"label": "None · omit infinite cells", "value": "none"}], value="paper")),
                                    control("Signal / “yes” orientation code", dcc.RadioItems(id="sdt-signal", options=[{"label": "−1 (assumed counter-clockwise)", "value": -1}, {"label": "+1", "value": 1}], value=-1), "Changing this reverses the sign of criterion."),
                                    control("Minimum trials in each class", dcc.Slider(id="sdt-min-trials", min=1, max=20, step=1, value=1, marks={1: "1", 5: "5", 10: "10", 15: "15", 20: "20"})),
                                    control("Group uncertainty", dcc.Dropdown(id="sdt-error", options=ERROR_OPTIONS, value="sem", clearable=False)),
                                    dcc.Checklist(id="sdt-show-subjects", options=[{"label": "Show participant trajectories", "value": "show"}], value=[]),
                                ],
                                className="advanced",
                            ),
                        ],
                        className="sidebar card",
                    ),
                    html.Div(
                        [
                            html.Div(id="sdt-kpis", className="mini-stats"),
                            dcc.Graph(id="sdt-graph", config={"displaylogo": False, "toImageButtonOptions": {"format": "svg", "filename": "signal_detection"}}),
                            html.Div(
                                [
                                    html.Div([html.H3("Current group summary"), html.P("Values shown here are the same participant-first means plotted above.")]),
                                    html.Button("Download CSV", id="sdt-download-button", className="button"),
                                    dcc.Download(id="sdt-download"),
                                ],
                                className="table-heading",
                            ),
                            dash_table.DataTable(
                                id="sdt-table",
                                page_size=12,
                                sort_action="native",
                                style_table={"overflowX": "auto"},
                                style_cell={"fontFamily": "Inter, Segoe UI, sans-serif", "fontSize": 12, "padding": "9px", "textAlign": "left"},
                                style_header={"backgroundColor": "#eef1f7", "fontWeight": 700, "border": "none"},
                                style_data={"border": "none", "borderBottom": "1px solid #edf0f5"},
                            ),
                        ],
                        className="analysis-panel card",
                    ),
                ],
                className="analysis-grid",
            ),
            html.Div(
                [
                    html.H3("Interpretation key"),
                    html.P([html.Strong("d′"), " measures separation of the two orientation-response distributions. Higher is better; response preference cancels out."]),
                    html.P([html.Strong("Criterion c"), " = −½[z(H) + z(FA)]. With −1 designated as signal, c > 0 means fewer −1 responses (a conservative bias); c < 0 means more −1 responses."]),
                    html.P("Rates are computed inside each participant × displayed cell. Group lines average those participant metrics, avoiding trial-count weighting."),
                ],
                className="method-strip",
            ),
        ],
        className="tab-page",
    )


def publication_tab():
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("PROCESSED SOURCE DATA", className="eyebrow"),
                            html.H2("Publication explorer"),
                            control("Figure family", dcc.Dropdown(id="pub-figure", options=[
                                {"label": "Figure 2a · presaccadic contrast response", "value": "fig2_psa"},
                                {"label": "Figure 2b · dmax attention effects", "value": "fig2b"},
                                {"label": "Figure 3 · presaccadic time course", "value": "fig3"},
                                {"label": "Figure S1a · exogenous contrast response", "value": "fig2_exo"},
                                {"label": "Figure S1b · endogenous contrast response", "value": "fig2_endo"},
                                {"label": "Figure S2 · eye-movement controls", "value": "figS2"},
                            ], value="fig2_psa", clearable=False)),
                            control("Uncertainty", dcc.Dropdown(id="pub-error", options=ERROR_OPTIONS, value="paper_sem", clearable=False), "“Paper SEM” reproduces the supplied MATLAB denominator."),
                            dcc.Checklist(id="pub-show-subjects", options=[{"label": "Overlay participant trajectories", "value": "show"}], value=[]),
                            html.Div(
                                [html.Strong("Axis correction"), html.P("Figure 3 is shown as 175 → 25 ms before saccade onset, matching the source-data bins and paper—not the increasing 0 → 200 labels in the supplied MATLAB output.")],
                                className="callout",
                            ),
                        ],
                        className="sidebar card",
                    ),
                    html.Div([dcc.Loading(dcc.Graph(id="pub-graph", config={"displaylogo": False, "toImageButtonOptions": {"format": "svg", "filename": "publication_figure"}}), type="dot")], className="analysis-panel card"),
                ],
                className="analysis-grid",
            )
        ],
        className="tab-page",
    )


def gallery_tab():
    items = [
        ("Figure 2a", "Figure_2a.png", "Presaccadic contrast-response functions"),
        ("Figure 2b", "Figure_2b.png", "dmax attention effects"),
        ("Figure 3", "Figure_3.png", "Presaccadic time course; note the source-script axis issue"),
        ("Figure S1a", "Figure_S1a.png", "Covert exogenous attention"),
        ("Figure S1b", "Figure_S1b.png", "Covert endogenous attention"),
        ("Figure S2", "Figure_S2.png", "Saccade latency and landing error"),
    ]
    return html.Div(
        [
            html.Div([html.Div("MATLAB EXPORTS", className="eyebrow"), html.H2("Original generated outputs"), html.P("These are the six verified figures already present in results/. Use the camera control in interactive graphs to export new views.")], className="section-intro"),
            html.Div([html.Figure([html.Img(src=f"/results/{file}", alt=f"Generated {title}"), html.Figcaption([html.Strong(title), html.Span(caption)])], className="gallery-card") for title, file, caption in items], className="gallery-grid"),
        ],
        className="tab-page",
    )


def notes_tab():
    rows = [
        ("1", "Block", "Block index"), ("2", "Trial", "Trial number within block"),
        ("3", "Design", "Exp. 1: fixation/saccade flag; Exp. 2: planned first-pulse timing code"),
        ("4", "Analysis level", "Exp. 1: contrast code; Exp. 2: TMS-to-saccade time-bin code"),
        ("5", "Cue location", "0 neutral, 1 left, 2 right"),
        ("6–7", "Stimulus orientations", "Signed left and right grating codes"),
        ("8", "Test location", "1 left, 2 right"), ("9", "Stimulated hemifield", "Participant-specific left/right code"),
        ("10", "Response", "Signed orientation response"), ("11", "Correct", "0 incorrect, 1 correct"),
        ("12", "Saccade latency", "Milliseconds"), ("13", "Landing position", "Degrees; not treated as landing error without target coordinates"),
    ]
    return html.Div(
        [
            html.Div([html.Div("REPRODUCIBILITY NOTES", className="eyebrow"), html.H2("Data map and scope"), html.P("The mapping below is inferred from code ranges, deterministic cross-column relationships, the experimental design, and the processed source matrices. No CSV data dictionary was deposited.")], className="section-intro"),
            html.Div(
                [
                    html.Div([html.H3("Raw CSV column map"), html.Table([html.Thead(html.Tr([html.Th("Column"), html.Th("Viewer name"), html.Th("Interpretation")])), html.Tbody([html.Tr([html.Td(a), html.Td(b), html.Td(c)]) for a, b, c in rows])])], className="card notes-card"),
                    html.Div(
                        [
                            html.H3("Analysis decisions"),
                            html.Ul([
                                html.Li("Signal-detection measures use the orientation at the queried test location, not the distractor orientation."),
                                html.Li("The paper correction changes only rates exactly equal to 0 or 1; intermediate rates are untouched."),
                                html.Li("Criterion is added using the standard equal-variance definition c = −½[z(H)+z(FA)]. It was not reported in the paper."),
                                html.Li("Raw-data group summaries are participant-first. Publication explorer reads the supplied .mat source matrices."),
                                html.Li("The default figure-aligned profile retains Experiment 1 neutral trials and saccade trials with deposited latency <350 ms. This cutoff was inferred by minimizing d′ error against fig2_psa.mat; it is not a stated paper rule."),
                                html.Li("That profile leaves 22,930 Experiment 1 rows, versus 22,679 reported. Blink, fixation, and true landing-error exclusions cannot be reconstructed from these CSV columns."),
                                html.Li("The Excel workbook labels Figure 2 rows S01–S10, but later Experiment 1 CSV identities do not align numerically with those labels. Deposited CSV names are preserved; no participant is silently relabeled or excluded."),
                                html.Li("Column 13 is exposed only as landing position. Figure S2 landing error comes from the processed source matrix."),
                            ]),
                            html.H3("Run and share"),
                            html.Pre("python -m pip install -r requirements.txt\npython app.py"),
                            html.P("The app runs locally and does not transmit data. For a paper-only fallback, Plotly graphs can be exported as SVG from the camera menu and assembled with the existing output audit."),
                        ],
                        className="card notes-card",
                    ),
                ],
                className="notes-grid",
            ),
        ],
        className="tab-page",
    )


app.layout = html.Div(
    [
        html.Header(
            [
                html.Div([html.Div("OPEN DATA COMPANION", className="brand-kicker"), html.H1("Presaccadic attention, unpacked"), html.P("A transparent viewer for behavioral trials, signal-detection measures, and the figures behind Hanning, Fernández & Carrasco (2023).")], className="brand"),
                html.Div([html.Span("LOCAL"), html.Div("No upload\nNo telemetry")], className="local-badge"),
            ],
            className="hero",
        ),
        dcc.Tabs(
            id="tabs",
            value="overview",
            children=[
                dcc.Tab(label="Overview", value="overview"),
                dcc.Tab(label="Signal detection", value="sdt"),
                dcc.Tab(label="Publication explorer", value="publication"),
                dcc.Tab(label="Original plots", value="gallery"),
                dcc.Tab(label="Methods & data map", value="notes"),
            ],
        ),
        html.Main(id="tab-content"),
        html.Footer([html.Span("Data and analysis remain in this repository."), html.Span("Nature Communications 14, 5381 · 2023")]),
        dcc.Store(id="sdt-summary-store"),
    ],
    className="app-shell",
)


@callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    return {"overview": overview_tab, "sdt": sdt_tab, "publication": publication_tab, "gallery": gallery_tab, "notes": notes_tab}[tab]()


@callback(
    Output("sdt-x", "options"), Output("sdt-x", "value"),
    Output("sdt-splits", "options"), Output("sdt-splits", "value"),
    Output("sdt-subjects", "options"), Output("sdt-subjects", "value"),
    Output("sdt-validity", "options"), Output("sdt-validity", "value"),
    Input("sdt-experiment", "value"),
)
def update_sdt_dimensions(experiment):
    common = [
        {"label": "Attention condition", "value": "validity"},
        {"label": "Test location · stimulated or control", "value": "test_stimulation"},
        {"label": "Participant", "value": "participant"},
    ]
    if experiment == "Exp1":
        x_options = [{"label": "Contrast (%)", "value": "contrast_pct"}] + common
        x_value = "contrast_pct"
        validities = ["Valid", "Neutral", "Invalid"]
    else:
        x_options = [
            {"label": "Time before saccade onset", "value": "time_before_saccade_ms"},
            {"label": "Planned first TMS pulse", "value": "planned_tms_ms"},
        ] + common
        x_value = "time_before_saccade_ms"
        validities = ["Valid", "Invalid"]
    split_options = [x for x in common if x["value"] != "participant"]
    subjects = sorted(TRIALS.loc[TRIALS["experiment"].eq(experiment), "participant"].unique())
    subject_options = [{"label": value, "value": value} for value in subjects]
    validity_options = [{"label": value, "value": value} for value in validities]
    return x_options, x_value, split_options, ["validity", "test_stimulation"], subject_options, subjects, validity_options, validities


def _prepare_sdt(experiment, metric, x_column, splits, validities, stimulation, subjects, profile, correction, signal_code, min_trials):
    if not all([experiment, metric, x_column, validities, stimulation, subjects]):
        return pd.DataFrame(), pd.DataFrame(), 0
    filtered = TRIALS[
        TRIALS["experiment"].eq(experiment)
        & TRIALS["validity"].isin(validities)
        & TRIALS["test_stimulation"].isin(stimulation)
        & TRIALS["participant"].isin(subjects)
    ].copy()
    filtered = apply_analysis_profile(filtered, profile)
    dimensions = [x_column] + list(splits or [])
    metrics = compute_sdt(filtered, dimensions, correction, int(signal_code), int(min_trials))
    valid = metrics.dropna(subset=[metric]) if not metrics.empty else metrics
    table_groups = [x_column] + [x for x in (splits or []) if x != x_column]
    summary_rows = []
    if not valid.empty:
        for keys, cell in valid.groupby(table_groups, observed=True, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(table_groups, keys))
            row.update({
                "mean": cell[metric].mean(),
                "sd": cell[metric].std(ddof=1),
                "participants": cell["participant"].nunique(),
                "trials": int(cell["n_trials"].sum()),
            })
            summary_rows.append(row)
    return filtered, metrics, pd.DataFrame(summary_rows)


@callback(
    Output("sdt-graph", "figure"), Output("sdt-kpis", "children"),
    Output("sdt-table", "data"), Output("sdt-table", "columns"), Output("sdt-summary-store", "data"),
    Input("sdt-experiment", "value"), Input("sdt-metric", "value"), Input("sdt-x", "value"), Input("sdt-splits", "value"),
    Input("sdt-validity", "value"), Input("sdt-stimulation", "value"), Input("sdt-subjects", "value"),
    Input("sdt-profile", "value"), Input("sdt-correction", "value"), Input("sdt-signal", "value"), Input("sdt-min-trials", "value"), Input("sdt-error", "value"), Input("sdt-show-subjects", "value"),
)
def update_sdt(experiment, metric, x_column, splits, validities, stimulation, subjects, profile, correction, signal_code, min_trials, error, show_subjects):
    filtered, metrics, summary = _prepare_sdt(experiment, metric, x_column, splits, validities, stimulation, subjects, profile, correction, signal_code, min_trials)
    fig = sdt_figure(metrics, metric, x_column, splits or [], error, "show" in (show_subjects or []))
    total_cells = len(metrics)
    usable = int(metrics[metric].notna().sum()) if not metrics.empty else 0
    kpis = [
        html.Div([html.Strong(f"{len(filtered):,}"), html.Span("selected trials")]),
        html.Div([html.Strong(str(filtered["participant"].nunique() if not filtered.empty else 0)), html.Span("participants")]),
        html.Div([html.Strong(f"{usable}/{total_cells}"), html.Span("finite cells")]),
        html.Div([html.Strong(METRIC_LABELS.get(metric, metric)), html.Span("active metric")]),
    ]
    display = summary.copy()
    if not display.empty:
        display.rename(columns={x_column: x_column.replace("_", " ").title(), "mean": f"Mean {METRIC_LABELS.get(metric, metric)}", "sd": "SD", "participants": "N participants", "trials": "N trials"}, inplace=True)
        for column in display.select_dtypes(include="number"):
            if not column.lower().startswith("n ") and column != "trials":
                display[column] = display[column].round(4)
    records = display.to_dict("records")
    columns = [{"name": col, "id": col} for col in display.columns]
    return fig, kpis, records, columns, summary.to_json(orient="split")


@callback(
    Output("sdt-download", "data"),
    Input("sdt-download-button", "n_clicks"),
    State("sdt-summary-store", "data"),
    prevent_initial_call=True,
)
def download_sdt(_, payload):
    if not payload:
        return no_update
    frame = pd.read_json(StringIO(payload), orient="split")
    return dcc.send_data_frame(frame.to_csv, "signal_detection_summary.csv", index=False)


@callback(
    Output("pub-graph", "figure"),
    Input("pub-figure", "value"), Input("pub-error", "value"), Input("pub-show-subjects", "value"),
)
def update_publication_figure(key, error, show_subjects):
    return publication_figure(key, error, "show" in (show_subjects or []))


if __name__ == "__main__":
    app.run(
        host=os.environ.get("DASH_HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8050")),
        debug=os.environ.get("DASH_DEBUG", "").lower() in {"1", "true", "yes"},
    )
