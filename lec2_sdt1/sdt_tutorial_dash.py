"""Responsive Dash companion for the equal-variance SDT tutorial."""

from __future__ import annotations

from statistics import NormalDist
from typing import Any

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, dcc, html, no_update


NORMAL = NormalDist()
NOISE_COLOR = "#2b6cb0"
SIGNAL_COLOR = "#d97706"
CRITERION_COLOR = "#7c3aed"
INK_COLOR = "#182338"
GRID_COLOR = "rgba(95, 107, 124, 0.14)"
FIXED_THRESHOLDS = np.linspace(-2.4, 2.4, 9)
THRESHOLD_COLORS = [
    "#482878",
    "#3e4989",
    "#31688e",
    "#26828e",
    "#1f9e89",
    "#35b779",
    "#6ece58",
    "#b5de2b",
    "#fde725",
]


def normal_pdf(x: np.ndarray, mean: float) -> np.ndarray:
    return np.exp(-0.5 * (x - mean) ** 2) / np.sqrt(2.0 * np.pi)


def evidence_means(dprime_magnitude: float, polarity: int) -> tuple[float, float]:
    signed_dprime = polarity * dprime_magnitude
    return signed_dprime / 2.0, -signed_dprime / 2.0


def rates(
    dprime_magnitude: float, polarity: int, threshold: float
) -> dict[str, float]:
    mu_signal, mu_noise = evidence_means(dprime_magnitude, polarity)
    hit_rate = NORMAL.cdf(mu_signal - threshold)
    false_alarm_rate = NORMAL.cdf(mu_noise - threshold)
    accuracy = 0.5 * (hit_rate + 1.0 - false_alarm_rate)
    z_hit = NORMAL.inv_cdf(hit_rate)
    z_false_alarm = NORMAL.inv_cdf(false_alarm_rate)
    criterion_right = -0.5 * (z_hit + z_false_alarm)
    return {
        "mu_signal": mu_signal,
        "mu_noise": mu_noise,
        "hit_rate": hit_rate,
        "false_alarm_rate": false_alarm_rate,
        "accuracy": accuracy,
        "z_hit": z_hit,
        "z_false_alarm": z_false_alarm,
        "dprime_signed": z_hit - z_false_alarm,
        "criterion_right": criterion_right,
        "criterion_oriented": polarity * criterion_right,
    }


def base_layout(title: str, height: int) -> dict[str, Any]:
    return {
        "title": {"text": title, "x": 0.02, "xanchor": "left"},
        "height": height,
        "margin": {"l": 55, "r": 25, "t": 70, "b": 55},
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "font": {"family": "Segoe UI, Arial, sans-serif", "color": INK_COLOR},
        "hovermode": "closest",
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "left",
            "x": 0,
        },
        "xaxis": {
            "gridcolor": GRID_COLOR,
            "zeroline": False,
            "linecolor": "#aab3c2",
        },
        "yaxis": {
            "gridcolor": GRID_COLOR,
            "zeroline": False,
            "linecolor": "#aab3c2",
        },
        "uirevision": "keep-view",
    }


def distribution_traces(
    mu_signal: float,
    mu_noise: float,
    threshold: float | None = None,
) -> list[go.Scatter]:
    x = np.linspace(-4.5, 4.5, 650)
    signal_pdf = normal_pdf(x, mu_signal)
    noise_pdf = normal_pdf(x, mu_noise)
    traces: list[go.Scatter] = []

    if threshold is not None:
        tail_x = np.linspace(threshold, 4.5, 320)
        traces.extend(
            [
                go.Scatter(
                    x=tail_x,
                    y=normal_pdf(tail_x, mu_noise),
                    mode="lines",
                    line={"width": 0},
                    fill="tozeroy",
                    fillcolor="rgba(43,108,176,0.20)",
                    hoverinfo="skip",
                    showlegend=False,
                ),
                go.Scatter(
                    x=tail_x,
                    y=normal_pdf(tail_x, mu_signal),
                    mode="lines",
                    line={"width": 0},
                    fill="tozeroy",
                    fillcolor="rgba(217,119,6,0.20)",
                    hoverinfo="skip",
                    showlegend=False,
                ),
            ]
        )

    traces.extend(
        [
            go.Scatter(
                x=x,
                y=noise_pdf,
                mode="lines",
                name="Noise",
                line={"color": NOISE_COLOR, "width": 3},
                hovertemplate="Noise density: %{y:.3f}<extra></extra>",
            ),
            go.Scatter(
                x=x,
                y=signal_pdf,
                mode="lines",
                name="Signal",
                line={"color": SIGNAL_COLOR, "width": 3},
                hovertemplate="Signal density: %{y:.3f}<extra></extra>",
            ),
        ]
    )
    return traces


def make_draggable_sdt(
    dprime_magnitude: float, polarity: int, threshold: float
) -> go.Figure:
    result = rates(dprime_magnitude, polarity, threshold)
    figure = go.Figure(
        distribution_traces(
            result["mu_signal"], result["mu_noise"], threshold=threshold
        )
    )
    figure.update_layout(
        **base_layout("Drag the purple threshold line", height=480),
        xaxis_title="evidence x",
        yaxis_title="density",
        xaxis_range=[-4.5, 4.5],
        yaxis_range=[0, 0.47],
        shapes=[
            {
                "type": "line",
                "x0": threshold,
                "x1": threshold,
                "y0": 0,
                "y1": 0.45,
                "line": {"color": CRITERION_COLOR, "width": 5, "dash": "dash"},
                "editable": True,
                "name": "criterion",
            }
        ],
        annotations=[
            {
                "x": threshold,
                "y": 0.455,
                "text": f"λ = {threshold:+.2f}",
                "showarrow": False,
                "font": {"color": CRITERION_COLOR, "size": 13},
                "xanchor": "center",
            },
            {
                "xref": "paper",
                "yref": "paper",
                "x": 0.99,
                "y": 0.96,
                "xanchor": "right",
                "yanchor": "top",
                "align": "left",
                "showarrow": False,
                "bordercolor": "#cbd3df",
                "borderwidth": 1,
                "borderpad": 8,
                "bgcolor": "rgba(255,255,255,0.94)",
                "text": (
                    f"<b>Hit rate</b> = {result['hit_rate']:.3f}<br>"
                    f"<b>False-alarm rate</b> = {result['false_alarm_rate']:.3f}<br>"
                    f"<b>Equal-prior accuracy</b> = {result['accuracy']:.3f}<br>"
                    f"<b>c<sub>R</sub></b> = {result['criterion_right']:+.3f}"
                ),
            },
        ],
    )
    return figure


def make_fixed_threshold_sdt(dprime_magnitude: float, polarity: int) -> go.Figure:
    mu_signal, mu_noise = evidence_means(dprime_magnitude, polarity)
    figure = go.Figure(distribution_traces(mu_signal, mu_noise))
    figure.data[0].showlegend = False
    figure.data[1].showlegend = False
    figure.add_annotation(
        x=mu_noise,
        y=0.41,
        text="Noise",
        showarrow=False,
        font={"color": NOISE_COLOR, "size": 11},
    )
    figure.add_annotation(
        x=mu_signal,
        y=0.41,
        text="Signal",
        showarrow=False,
        font={"color": SIGNAL_COLOR, "size": 11},
    )
    for threshold, color in zip(FIXED_THRESHOLDS, THRESHOLD_COLORS, strict=True):
        figure.add_trace(
            go.Scatter(
                x=[threshold, threshold],
                y=[0, 0.43],
                mode="lines",
                line={"color": color, "width": 1.6, "dash": "dot"},
                name=f"λ={threshold:+.1f}",
                legendgroup="thresholds",
                showlegend=False,
                hovertemplate=f"threshold λ={threshold:+.1f}<extra></extra>",
            )
        )
        figure.add_annotation(
            x=threshold,
            y=0.438,
            text=f"{threshold:+.1f}",
            textangle=-90,
            showarrow=False,
            font={"color": color, "size": 9},
        )
    figure.update_layout(
        **base_layout("Fixed criteria on the evidence distributions", height=470),
        xaxis_title="evidence x",
        yaxis_title="density",
        xaxis_range=[-4.5, 4.5],
        yaxis_range=[0, 0.47],
    )
    return figure


def make_roc(dprime_magnitude: float, polarity: int) -> go.Figure:
    mu_signal, mu_noise = evidence_means(dprime_magnitude, polarity)
    dense_thresholds = np.linspace(-5.5, 5.5, 650)
    false_alarm_rates = np.array(
        [NORMAL.cdf(mu_noise - float(value)) for value in dense_thresholds]
    )
    hit_rates = np.array(
        [NORMAL.cdf(mu_signal - float(value)) for value in dense_thresholds]
    )

    sampled_false_alarms = np.array(
        [NORMAL.cdf(mu_noise - float(value)) for value in FIXED_THRESHOLDS]
    )
    sampled_hits = np.array(
        [NORMAL.cdf(mu_signal - float(value)) for value in FIXED_THRESHOLDS]
    )

    signed_dprime = polarity * dprime_magnitude
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance",
            line={"color": "#aab3c2", "width": 1.5, "dash": "dash"},
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=false_alarm_rates,
            y=hit_rates,
            mode="lines",
            name=f"ROC (d′={signed_dprime:+.1f})",
            line={"color": INK_COLOR, "width": 3},
            hovertemplate="FA=%{x:.3f}<br>H=%{y:.3f}<extra></extra>",
        )
    )
    for threshold, false_alarm, hit, color in zip(
        FIXED_THRESHOLDS,
        sampled_false_alarms,
        sampled_hits,
        THRESHOLD_COLORS,
        strict=True,
    ):
        figure.add_trace(
            go.Scatter(
                x=[false_alarm],
                y=[hit],
                mode="markers+text",
                marker={"symbol": "x", "size": 11, "color": color, "line": {"width": 2}},
                text=[f"λ={threshold:+.1f}"],
                textposition="top right",
                textfont={"color": color, "size": 9},
                name=f"λ={threshold:+.1f}",
                showlegend=False,
                hovertemplate=(
                    f"λ={threshold:+.1f}<br>FA={false_alarm:.3f}"
                    f"<br>H={hit:.3f}<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        **base_layout("The same criteria traced on the ROC", height=470),
        xaxis_title="false-alarm rate",
        yaxis_title="hit rate",
        xaxis_range=[-0.03, 1.03],
        yaxis_range=[-0.03, 1.03],
    )
    figure.update_yaxes(scaleanchor="x", scaleratio=1)
    return figure


def metric_cards(result: dict[str, float]) -> list[html.Div]:
    entries = [
        ("Hit rate", result["hit_rate"], "signal trials called ‘yes’"),
        ("False-alarm rate", result["false_alarm_rate"], "noise trials called ‘yes’"),
        ("Equal-prior accuracy", result["accuracy"], "½[H + (1 − FA)]"),
        ("Right-tail criterion cᵣ", result["criterion_right"], "−½[z(H) + z(FA)]"),
    ]
    return [
        html.Div(
            [
                html.Strong(f"{value:+.3f}" if "criterion" in label else f"{value:.3f}"),
                html.Span(label),
                html.Small(note),
            ],
            className="metric-card",
        )
        for label, value, note in entries
    ]


def threshold_from_relayout(payload: dict[str, Any] | None, current: float) -> float:
    if not payload:
        return current
    candidates: list[float] = []
    for key in ("shapes[0].x0", "shapes[0].x1"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            candidates.append(float(value))
    shapes = payload.get("shapes")
    if isinstance(shapes, list) and shapes:
        for key in ("x0", "x1"):
            value = shapes[0].get(key)
            if isinstance(value, (int, float)):
                candidates.append(float(value))
    if not candidates:
        return current
    return float(np.clip(np.mean(candidates), -3.5, 3.5))


app = Dash(__name__, title="Interactive Gaussian SDT")
server = app.server

GRAPH_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
    "editable": True,
    "edits": {
        "shapePosition": True,
        "annotationPosition": False,
        "annotationTail": False,
        "annotationText": False,
        "axisTitleText": False,
        "colorbarPosition": False,
        "colorbarTitleText": False,
        "legendPosition": False,
        "legendText": False,
        "titleText": False,
    },
    "scrollZoom": False,
}

app.layout = html.Main(
    [
        dcc.Store(id="threshold-store", data=0.3),
        html.Header(
            [
                html.H1("Why is z(H) − z(FA) equal to d′?"),
                html.P(
                    "A direct-manipulation view of equal-variance Gaussian "
                    "signal detection. Drag the criterion, then change d′ and "
                    "watch the distributions and ROC move together."
                ),
            ],
            className="hero",
        ),
        html.Section(
            [
                html.Div(
                    [
                        html.Label("Sensitivity |d′| = |μS − μN| / σ"),
                        dcc.Slider(
                            id="dprime-slider",
                            min=0.2,
                            max=3.5,
                            step=0.1,
                            value=1.6,
                            marks={value: str(value) for value in [0.5, 1, 2, 3]},
                            tooltip={"placement": "bottom", "always_visible": True},
                            updatemode="mouseup",
                        ),
                    ],
                    className="control",
                ),
                html.Div(
                    [
                        html.Label("Evidence ordering"),
                        dcc.RadioItems(
                            id="polarity-radio",
                            options=[
                                {"label": "Signal right of noise", "value": 1},
                                {"label": "Signal left of noise", "value": -1},
                            ],
                            value=1,
                            inline=True,
                            className="radio-row",
                        ),
                    ],
                    className="control",
                ),
            ],
            className="controls",
        ),
        html.Section(
            [
                html.H2("1. Move the decision criterion directly"),
                html.P(
                    "Grab the purple dashed line and drag it left or right. "
                    "Both shaded right tails, the rates, accuracy, and criterion update."
                ),
                dcc.Graph(
                    id="sdt-drag",
                    figure=make_draggable_sdt(1.6, 1, 0.3),
                    config=GRAPH_CONFIG,
                    responsive=True,
                    className="graph-card",
                ),
                html.Div(id="metric-cards", className="metric-grid"),
                dcc.Markdown(
                    r"""
                    The displayed accuracy assumes equally common signal and noise
                    trials. Moving the threshold changes $H$, $FA$, accuracy, and
                    $c_R=-\tfrac12[z(H)+z(FA)]$, but it does not change
                    $d'_R=z(H)-z(FA)=(\mu_S-\mu_N)/\sigma$.
                    """,
                    mathjax=True,
                    className="formula-note",
                ),
            ],
            className="section-card",
        ),
        html.Section(
            [
                html.H2("2. One d′, many criteria: the ROC"),
                html.P(
                    "The colored fixed thresholds at left are the matching × marks "
                    "on the ROC at right. Change d′ above to reshape both plots."
                ),
                html.Div(
                    [
                        dcc.Graph(
                            id="sdt-thresholds",
                            figure=make_fixed_threshold_sdt(1.6, 1),
                            config={"displayModeBar": False, "responsive": True},
                            responsive=True,
                            className="graph-card",
                        ),
                        dcc.Graph(
                            id="roc-graph",
                            figure=make_roc(1.6, 1),
                            config={"displayModeBar": False, "responsive": True},
                            responsive=True,
                            className="graph-card",
                        ),
                    ],
                    className="paired-plots",
                ),
            ],
            className="section-card",
        ),
        html.Footer(
            [
                html.Strong("Deployment choice: "),
                "use this Dash version when direct shape dragging matters; use the "
                "Marimo WASM export when static, serverless hosting matters most.",
            ]
        ),
    ],
    className="app-shell",
)


@callback(
    Output("threshold-store", "data"),
    Input("sdt-drag", "relayoutData"),
    State("threshold-store", "data"),
    prevent_initial_call=True,
)
def remember_dragged_threshold(
    relayout_data: dict[str, Any] | None, current_threshold: float
) -> float | Any:
    updated = threshold_from_relayout(relayout_data, current_threshold)
    if abs(updated - current_threshold) < 1e-9:
        return no_update
    return updated


@callback(
    Output("sdt-drag", "figure"),
    Output("metric-cards", "children"),
    Input("dprime-slider", "value"),
    Input("polarity-radio", "value"),
    Input("threshold-store", "data"),
)
def update_draggable_sdt(
    dprime_magnitude: float, polarity: int, threshold: float
) -> tuple[go.Figure, list[html.Div]]:
    current_rates = rates(dprime_magnitude, polarity, threshold)
    return (
        make_draggable_sdt(dprime_magnitude, polarity, threshold),
        metric_cards(current_rates),
    )


@callback(
    Output("sdt-thresholds", "figure"),
    Output("roc-graph", "figure"),
    Input("dprime-slider", "value"),
    Input("polarity-radio", "value"),
)
def update_fixed_threshold_views(
    dprime_magnitude: float, polarity: int
) -> tuple[go.Figure, go.Figure]:
    return (
        make_fixed_threshold_sdt(dprime_magnitude, polarity),
        make_roc(dprime_magnitude, polarity),
    )


if __name__ == "__main__":
    app.run(debug=True)
