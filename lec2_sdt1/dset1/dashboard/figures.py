from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import least_squares

from .data import load_mat_struct


COLORS = {
    "Valid": "#7568b0",
    "Neutral": "#777a83",
    "Invalid": "#47a47f",
    "Presaccadic": "#7568b0",
    "Exogenous": "#2599ad",
    "Endogenous": "#df925b",
    "Stimulated": "#d7a900",
    "Not stimulated": "#495166",
}

METRIC_LABELS = {
    "dprime": "Sensitivity (d′)",
    "criterion": "Criterion (c)",
    "accuracy": "Accuracy",
    "hit_rate": "Hit rate",
    "false_alarm_rate": "False-alarm rate",
    "mean_saccade_latency_ms": "Saccade latency (ms)",
}


def base_layout(figure: go.Figure, *, height: int = 560) -> go.Figure:
    figure.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=62, r=28, t=62, b=58),
        font=dict(family="Inter, Segoe UI, sans-serif", color="#253047", size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        hoverlabel=dict(bgcolor="#17223b", font_color="white"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    figure.update_xaxes(gridcolor="#edf0f5", zerolinecolor="#aeb7c8")
    figure.update_yaxes(gridcolor="#edf0f5", zerolinecolor="#aeb7c8")
    return figure


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16, color="#657089"),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return base_layout(fig)


def error_size(values: pd.Series | np.ndarray, method: str) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if method == "none" or n < 2:
        return 0.0
    sd = float(np.std(values, ddof=1))
    if method == "sd":
        return sd
    if method == "paper_sem":
        return sd / np.sqrt(n - 1)
    if method == "ci95":
        return 1.96 * sd / np.sqrt(n)
    return sd / np.sqrt(n)


def sdt_figure(
    metrics: pd.DataFrame,
    metric: str,
    x_column: str,
    split_columns: list[str],
    error: str,
    show_participants: bool,
) -> go.Figure:
    if metrics.empty or metric not in metrics:
        return empty_figure("No analyzable cells match these controls.")

    split_columns = [c for c in split_columns if c != x_column and c in metrics.columns]
    trace_groups = split_columns or [None]
    grouped = [((), metrics)] if trace_groups == [None] else metrics.groupby(
        trace_groups, observed=True, dropna=False
    )
    fig = go.Figure()

    for index, (keys, subset) in enumerate(grouped):
        if trace_groups == [None]:
            keys = ()
        elif not isinstance(keys, tuple):
            keys = (keys,)
        labels = dict(zip(split_columns, keys))
        name = " · ".join(str(value) for value in keys) or "All selected trials"
        color = COLORS.get(str(keys[0]), ["#5969c9", "#24a187", "#dc7a58", "#73819a"][index % 4])
        dash = "dash" if "Not stimulated" in keys else "solid"

        if show_participants:
            for _, person in subset.groupby("participant", observed=True):
                person = person.sort_values(x_column, ascending=True)
                fig.add_trace(
                    go.Scatter(
                        x=person[x_column],
                        y=person[metric],
                        mode="lines+markers",
                        line=dict(color=color, width=1),
                        marker=dict(size=4),
                        opacity=0.17,
                        showlegend=False,
                        hoverinfo="skip",
                        legendgroup=name,
                    )
                )

        summary_rows = []
        for x_value, cell in subset.groupby(x_column, observed=True, dropna=False):
            values = cell[metric].dropna()
            if values.empty:
                continue
            summary_rows.append(
                {
                    "x": x_value,
                    "mean": values.mean(),
                    "error": error_size(values, error),
                    "n": len(values),
                }
            )
        summary = pd.DataFrame(summary_rows)
        if summary.empty:
            continue
        summary.sort_values("x", inplace=True)
        custom = np.column_stack((summary["n"],))
        fig.add_trace(
            go.Scatter(
                x=summary["x"],
                y=summary["mean"],
                error_y=dict(
                    type="data",
                    array=summary["error"],
                    visible=error != "none",
                    thickness=1.5,
                    width=4,
                ),
                customdata=custom,
                mode="lines+markers",
                name=name,
                legendgroup=name,
                line=dict(color=color, width=3, dash=dash),
                marker=dict(size=9, color=color, line=dict(width=1.5, color="white")),
                hovertemplate=(
                    "%{x}<br>Mean: %{y:.3f}<br>Participants: %{customdata[0]:.0f}<extra>"
                    + name
                    + "</extra>"
                ),
            )
        )

    x_titles = {
        "contrast_pct": "Contrast (%)",
        "time_before_saccade_ms": "Time before saccade onset (ms)",
        "planned_tms_ms": "First TMS pulse after cue onset (ms)",
        "validity": "Attention condition",
        "test_stimulation": "Test location",
        "participant": "Participant",
    }
    title = METRIC_LABELS.get(metric, metric)
    fig.update_layout(title=title, hovermode="x unified")
    fig.update_xaxes(title=x_titles.get(x_column, x_column))
    fig.update_yaxes(title=title)
    if x_column == "contrast_pct":
        fig.update_xaxes(type="log", tickvals=[2, 7, 13, 24, 46, 85])
    if x_column == "time_before_saccade_ms":
        fig.update_xaxes(autorange="reversed", tickvals=[175, 125, 75, 25])
    if metric in {"accuracy", "hit_rate", "false_alarm_rate"}:
        fig.update_yaxes(range=[0, 1.03], tickformat=".0%")
    return base_layout(fig)


def _naka_rushton(log_contrast: np.ndarray, c50: float, dmax: float, slope: float) -> np.ndarray:
    """Match the supplied MATLAB fit, which operates on log10 contrast."""
    return dmax * log_contrast**slope / (log_contrast**slope + c50**slope)


def _fit_curve(x: np.ndarray, y: np.ndarray, slope: float, dmax_bounds: tuple[float, float]):
    log_x = np.log10(x)
    c50_upper_index = -1 if len(x) == 8 else 4
    fit = least_squares(
        lambda p: _naka_rushton(log_x, p[0], p[1], slope) - y,
        x0=[log_x[2], 2.0],
        bounds=([log_x[0], dmax_bounds[0]], [log_x[c50_upper_index], dmax_bounds[1]]),
    )
    dense = np.logspace(np.log10(1), np.log10(x[-1]), 300)
    return dense, _naka_rushton(np.log10(dense), fit.x[0], fit.x[1], slope)


def contrast_response_figure(stem: str, error: str, show_participants: bool) -> go.Figure:
    data = load_mat_struct(stem)
    if stem == "fig2_endo":
        x = np.array([2, 3, 6, 10, 17, 29, 50, 85], dtype=float)
        title, slope, bounds = "Covert endogenous attention · Figure S1b", 4.0, (1.0, 5.0)
    else:
        x = np.array([2, 7, 13, 24, 46, 85], dtype=float)
        title = "Presaccadic attention · Figure 2a" if stem == "fig2_psa" else "Covert exogenous attention · Figure S1a"
        slope = 5.0 if stem == "fig2_psa" else 4.0
        bounds = (0.3, 5.0) if stem == "fig2_psa" else (1.0, 4.0)

    fig = go.Figure()
    for validity, stimulation in product(
        ["Valid", "Neutral", "Invalid"], ["Stimulated", "Not stimulated"]
    ):
        suffix = "T" if stimulation == "Stimulated" else "n"
        values = data[f"{validity.lower()}_{suffix}"]
        color = COLORS[validity]
        dash = "solid" if stimulation == "Stimulated" else "dash"
        name = f"{validity} · {stimulation}"
        if show_participants:
            for row in values:
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=row,
                        mode="lines",
                        line=dict(color=color, width=1),
                        opacity=0.1,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
        mean = np.nanmean(values, axis=0)
        err = [error_size(values[:, i], error) for i in range(values.shape[1])]
        dense_x, dense_y = _fit_curve(x, mean, slope, bounds)
        fig.add_trace(
            go.Scatter(
                x=dense_x,
                y=dense_y,
                mode="lines",
                line=dict(color=color, width=2.5, dash=dash),
                name=name,
                legendgroup=name,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=mean,
                error_y=dict(type="data", array=err, visible=error != "none"),
                mode="markers",
                marker=dict(
                    size=9,
                    color="#f3c623" if stimulation == "Stimulated" else "white",
                    line=dict(color=color, width=2),
                ),
                name=name + " observations",
                legendgroup=name,
                showlegend=False,
                hovertemplate="Contrast %{x}%<br>d′ %{y:.3f}<extra>" + name + "</extra>",
            )
        )
    fig.update_layout(title=title)
    fig.update_xaxes(title="Contrast (%)", type="log", tickvals=x)
    fig.update_yaxes(title="Sensitivity (d′)")
    return base_layout(fig, height=620)


def dmax_figure(error: str, show_participants: bool) -> go.Figure:
    fig = go.Figure()
    for stem, label in [
        ("fig2_psa", "Presaccadic"),
        ("fig2_exo", "Exogenous"),
        ("fig2_endo", "Endogenous"),
    ]:
        data = load_mat_struct(stem)
        x = data["diff_T_dmax"].ravel()
        y = data["diff_n_dmax"].ravel()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers" if show_participants else "markers",
                name=label,
                marker=dict(size=10, color=COLORS[label], line=dict(color="white", width=1)),
                opacity=1 if show_participants else 0.25,
                hovertemplate="Stimulated: %{x:.3f}<br>Not stimulated: %{y:.3f}<extra>" + label + "</extra>",
            )
        )
        if not show_participants:
            fig.add_trace(
                go.Scatter(
                    x=[np.mean(x)],
                    y=[np.mean(y)],
                    error_x=dict(type="data", array=[error_size(x, error)], visible=error != "none"),
                    error_y=dict(type="data", array=[error_size(y, error)], visible=error != "none"),
                    mode="markers",
                    marker=dict(size=13, color=COLORS[label], line=dict(color="white", width=2)),
                    name=label + " mean",
                    showlegend=False,
                )
            )
    fig.add_shape(type="line", x0=-0.5, x1=3.6, y0=-0.5, y1=3.6, line=dict(color="#263048"))
    fig.add_hline(y=0, line_dash="dash", line_color="#8a93a6")
    fig.add_vline(x=0, line_dash="dash", line_color="#8a93a6")
    fig.update_layout(title="Attention effect on fitted dmax · Figure 2b")
    fig.update_xaxes(title="Test stimulated · dmax(valid − invalid)", range=[-0.5, 3.6])
    fig.update_yaxes(title="Test not stimulated · dmax(valid − invalid)", range=[-0.5, 3.6])
    return base_layout(fig, height=620)


def timecourse_figure(error: str, show_participants: bool) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("V1/V2 sensitivity", "V1/V2 TMS effect", "rFEF+ sensitivity", "rFEF+ TMS effect"),
        horizontal_spacing=0.12,
        vertical_spacing=0.18,
    )
    x = np.array([175, 125, 75, 25])
    for row, stem in [(1, "fig3_V1V2"), (2, "fig3_FEF")]:
        data = load_mat_struct(stem)
        for col, difference in [(1, False), (2, True)]:
            for validity in ["Valid", "Invalid"]:
                color = COLORS[validity]
                fields = [f"{validity.lower()}_diff"] if difference else [
                    f"{validity.lower()}_T",
                    f"{validity.lower()}_n",
                ]
                for field in fields:
                    values = data[field]
                    stim = "TMS effect" if difference else ("Stimulated" if field.endswith("_T") else "Not stimulated")
                    name = f"{validity} · {stim}"
                    if show_participants:
                        for person in values:
                            fig.add_trace(
                                go.Scatter(x=x, y=person, mode="lines", line=dict(color=color, width=1), opacity=0.1, showlegend=False, hoverinfo="skip"),
                                row=row,
                                col=col,
                            )
                    mean = np.nanmean(values, axis=0)
                    err = [error_size(values[:, i], error) for i in range(values.shape[1])]
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=mean,
                            error_y=dict(type="data", array=err, visible=error != "none"),
                            mode="lines+markers",
                            line=dict(color=color, width=2.5, dash="dash" if field.endswith("_n") else "solid"),
                            marker=dict(size=7, color=color),
                            name=name,
                            legendgroup=name,
                            showlegend=row == 1,
                            hovertemplate="%{x} ms before<br>d′ %{y:.3f}<extra>" + name + "</extra>",
                        ),
                        row=row,
                        col=col,
                    )
    fig.update_xaxes(title_text="Time before saccade onset (ms)", autorange="reversed", tickvals=x)
    fig.update_yaxes(title_text="Sensitivity (d′)", col=1)
    fig.update_yaxes(title_text="d′ difference", col=2, zeroline=True)
    fig.update_layout(title="Presaccadic time course · Figure 3")
    return base_layout(fig, height=760)


def ocular_figure(error: str, show_participants: bool) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("V1/V2 · latency", "V1/V2 · landing error", "rFEF+ · latency", "rFEF+ · landing error"),
        horizontal_spacing=0.13,
        vertical_spacing=0.19,
    )
    x_latency = np.array([0, 50, 100, 150, 200])
    x_error = np.array([225, 175, 125, 75, 25])
    lat, err_data = load_mat_struct("figS2_lat"), load_mat_struct("figS2_err")
    for row, site in [(1, "V1V2"), (2, "FEF")]:
        for col, source in [(1, lat), (2, err_data)]:
            for suffix, label in [("stim", "Stimulated"), ("not", "Not stimulated")]:
                values = source[f"{site}_{suffix}"]
                x = x_latency if col == 1 else x_error
                color = COLORS[label]
                if show_participants:
                    for person in values:
                        fig.add_trace(
                            go.Scatter(x=x, y=person, mode="lines", line=dict(color=color, width=1), opacity=0.12, showlegend=False, hoverinfo="skip"),
                            row=row,
                            col=col,
                        )
                mean = np.nanmean(values, axis=0)
                errors = [error_size(values[:, i], error) for i in range(values.shape[1])]
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=mean,
                        error_y=dict(type="data", array=errors, visible=error != "none"),
                        mode="lines+markers",
                        line=dict(color=color, width=2.5),
                        marker=dict(size=7),
                        name=label,
                        legendgroup=label,
                        showlegend=row == 1 and col == 1,
                    ),
                    row=row,
                    col=col,
                )
    fig.update_xaxes(title_text="First TMS pulse after cue (ms)", col=1)
    fig.update_xaxes(title_text="Time before saccade (ms)", autorange="reversed", col=2)
    fig.update_yaxes(title_text="Latency (ms)", col=1)
    fig.update_yaxes(title_text="Landing error (°)", col=2)
    fig.update_layout(title="Eye-movement controls · Figure S2")
    return base_layout(fig, height=760)


def publication_figure(key: str, error: str, show_participants: bool) -> go.Figure:
    if key in {"fig2_psa", "fig2_exo", "fig2_endo"}:
        return contrast_response_figure(key, error, show_participants)
    if key == "fig2b":
        return dmax_figure(error, show_participants)
    if key == "fig3":
        return timecourse_figure(error, show_participants)
    return ocular_figure(error, show_participants)
