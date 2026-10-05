"""Plotly-Abbildungen der Go-Explore-Demo: Archiv (Zustände nach Schritten gefärbt, mit der gefundenen Fahrt), berührte Zellen der Vergleichsverfahren, Verläufe von Zellen und Restweg über den Schritten, Anteil erfolgreicher Läufe über dem Budget,
Erfolg und Schritte je Stufe und Verfahren, Einstellungen von Go-Explore, Entgleisung durch Rauschen. Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen. Das Raster `grid[i, j]` wird mit i als x-Achse gezeichnet."""

import numpy as np
import plotly.graph_objects as go

import ge_constants as C
import ge_evaluation as E

WALL_COLOR = "#3a3f47"
START_COLOR, GOAL_COLOR, PATH_COLOR = "#54a24b", "#e45756", "#e45756"
TOUCH_COLOR = "#9ecae1"
METHOD_COLORS = {"fitness": "#9a9a9a", "novelty": "#4c78a8", "ge_free": "#e45756", "ge_replay": "#f2a541"}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height, legend_y=-0.25, top=10):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=top, b=10), legend=dict(orientation="h", y=legend_y), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _arena_axes(fig):
    """Beide Achsen 0 bis N mit gleichem Maßstab (quadratische Zellen), Zeichenfläche passt sich der kleineren Seite an (constrain=domain)."""
    fig.update_xaxes(range=[0, C.N], showgrid=False, zeroline=False, title_text="x (Zellen)", constrain="domain")
    fig.update_yaxes(range=[0, C.N], showgrid=False, zeroline=False, title_text="y (Zellen)", scaleanchor="x", scaleratio=1, constrain="domain")


def _walls(fig, grid):
    z = np.where(grid, 1.0, np.nan)
    fig.add_trace(go.Heatmap(z=z.T, x0=0.5, dx=1, y0=0.5, dy=1, zmin=0, zmax=1, colorscale=[[0, WALL_COLOR], [1, WALL_COLOR]], showscale=False, hoverinfo="skip", xgap=0, ygap=0))


def _markers(fig):
    fig.add_trace(go.Scatter(x=[C.START[0]], y=[C.START[1]], mode="markers", marker=dict(color=START_COLOR, size=12, symbol="square", line=dict(color="white", width=1.5)), name="Start"))
    fig.add_trace(go.Scatter(x=[C.GOAL[0]], y=[C.GOAL[1]], mode="markers", marker=dict(color=GOAL_COLOR, size=15, symbol="star", line=dict(color="white", width=1.5)), name="Ziel"))
    fig.add_shape(type="circle", xref="x", yref="y", x0=C.GOAL[0] - C.SUCCESS_RADIUS, x1=C.GOAL[0] + C.SUCCESS_RADIUS, y0=C.GOAL[1] - C.SUCCESS_RADIUS, y1=C.GOAL[1] + C.SUCCESS_RADIUS,
                  line=dict(color=GOAL_COLOR, dash="dot", width=1.5))


def build_archive_map(grid, xy, steps, path=None):
    """Archiv von Go-Explore: je Zelle der Zustand, mit dem sie am schnellsten erreicht wurde, nach den Schritten bis dorthin gefärbt (dunkel = früh); darüber die gefundene Fahrt, wenn es eine gibt."""
    fig = go.Figure()
    _walls(fig, grid)
    fig.add_trace(go.Scatter(x=xy[:, 0], y=xy[:, 1], mode="markers", name="Archiv (ein Zustand je Zelle)", hoverinfo="skip",
                             marker=dict(color=steps, colorscale="Viridis", cmin=0, cmax=max(float(np.max(steps)), 1.0), size=5, line=dict(color="white", width=0.3),
                                         colorbar=dict(title=dict(text="Schritte bis zur Zelle", side="top"), orientation="h", thickness=10, len=0.9, y=-0.2, yanchor="top"))))
    if path is not None:
        fig.add_trace(go.Scatter(x=path[:, 0], y=path[:, 1], mode="lines", line=dict(color=PATH_COLOR, width=2.5), name="gefundene Fahrt", hoverinfo="skip"))
    _markers(fig)
    _arena_axes(fig)
    return _base(fig, 470, legend_y=-0.55, top=0)


def build_touch_map(grid, touched, path=None):
    """Berührte Zellen eines Vergleichsverfahrens (jede Zelle, die irgendeine Fahrt berührt hat) und die Fahrt, die dem Ziel am nächsten kam."""
    fig = go.Figure()
    _walls(fig, grid)
    z = np.where(touched, 1.0, np.nan)
    fig.add_trace(go.Heatmap(z=z.T, x0=0.5, dx=1, y0=0.5, dy=1, zmin=0, zmax=1, colorscale=[[0, TOUCH_COLOR], [1, TOUCH_COLOR]], showscale=False, hoverinfo="skip", xgap=1, ygap=1, name="berührte Zellen"))
    if path is not None:
        fig.add_trace(go.Scatter(x=path[:, 0], y=path[:, 1], mode="lines", line=dict(color=PATH_COLOR, width=2.5), name="Fahrt nächst dem Ziel", hoverinfo="skip"))
    _markers(fig)
    _arena_axes(fig)
    return _base(fig, 470, legend_y=-0.3, top=0)


def _pair_curve(a, b, y_title, a_label, b_label, a_color, b_color, y_range=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a[0], y=a[1], mode="lines", line=dict(color=a_color, width=2.5), name=a_label))
    fig.add_trace(go.Scatter(x=b[0], y=b[1], mode="lines", line=dict(color=b_color, width=2.5), name=b_label))
    fig.update_xaxes(title_text="verbrauchte Schritte")
    fig.update_yaxes(title_text=y_title, rangemode="tozero", range=y_range)
    return _base(fig, 320, legend_y=-0.4)


def build_cells_curve(ge, other, other_method):
    """Berührte Zellen (1 × 1) über den verbrauchten Schritten: Go-Explore gegen das Vergleichsverfahren. Jeweils (Schritte, Zellen)."""
    return _pair_curve(ge, other, "Zellen im Archiv bzw. berührte Zellen", C.METHOD_LABELS["ge_free"], C.BASELINE_LABELS[other_method], METHOD_COLORS["ge_free"], METHOD_COLORS[other_method])


def build_progress_curve(ge, other, other_method):
    """Kleinster wahrer Restweg (Zellen) über den verbrauchten Schritten: Go-Explore gegen das Vergleichsverfahren. Jeweils (Schritte, Restweg)."""
    return _pair_curve(ge, other, "kleinster Restweg zum Ziel (Zellen)", C.METHOD_LABELS["ge_free"], C.BASELINE_LABELS[other_method], METHOD_COLORS["ge_free"], METHOD_COLORS[other_method])


def build_success_curve(pre, level):
    """Anteil der Läufe, die bis zum Budget (x) im Zielkreis waren, je Verfahren (20 Läufe, Budget bis 5.6 Mio Schritte)."""
    fig = go.Figure()
    for m in C.METHODS:
        fig.add_trace(go.Scatter(x=list(C.CURVE_MARKS), y=E.method_cell(pre, level, m)["solved_by"], mode="lines", line=dict(color=METHOD_COLORS[m], width=2.5), name=C.METHOD_LABELS[m]))
    fig.update_xaxes(title_text="Budget (simulierte Schritte)")
    fig.update_yaxes(title_text="Anteil erfolgreicher Läufe", tickformat=".0%", range=[0, 1.03])
    return _base(fig, 360, legend_y=-0.5)


def build_success_bars(pre):
    """Erfolgreiche Läufe von 20 je Stufe und Verfahren nach dem vollen Budget (5.6 Mio Schritte)."""
    fig = go.Figure()
    for m in C.METHODS:
        fig.add_trace(go.Bar(x=[C.LEVEL_LABELS[lv] for lv in C.LEVELS], y=[E.success_count(E.method_cell(pre, lv, m)["solved"]) for lv in C.LEVELS], name=C.METHOD_LABELS[m], marker_color=METHOD_COLORS[m]))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text=f"erfolgreiche Läufe von {C.STUDY_SEEDS}", range=[0, C.STUDY_SEEDS])
    fig.update_xaxes(tickangle=-20)
    return _base(fig, 400, legend_y=-0.45)


def build_steps_chart(pre):
    """Median der Schritte bis zum Erfolg (nur erfolgreiche Läufe) je Stufe und Verfahren, logarithmische Achse; fehlt ein Balken, hat kein Lauf das Ziel erreicht."""
    fig = go.Figure()
    for m in C.METHODS:
        meds = [E.median_solved(E.method_cell(pre, lv, m)["solved"]) for lv in C.LEVELS]
        fig.add_trace(go.Bar(x=[C.LEVEL_LABELS[lv] for lv in C.LEVELS], y=meds, name=C.METHOD_LABELS[m], marker_color=METHOD_COLORS[m]))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Median der Schritte bis zum Erfolg", type="log")
    fig.update_xaxes(tickangle=-20)
    return _base(fig, 400, legend_y=-0.45)


def build_sweep_chart(pre, level):
    """Median der Schritte bis zum Erfolg je Einstellung von Go-Explore (Rückkehr kostenlos, wenn nicht anders gewählt), eine Änderung gegenüber der Voreinstellung je Balken."""
    fig = go.Figure()
    meds = [E.median_solved(E.sweep_cell(pre, level, cfg)["solved"]) for cfg in C.SWEEP_CONFIGS]
    fig.add_trace(go.Bar(x=[C.SWEEP_LABELS[cfg] for cfg in C.SWEEP_CONFIGS], y=meds, marker_color=[METHOD_COLORS["ge_free"] if cfg == "default" else "#f2b1b0" for cfg in C.SWEEP_CONFIGS], showlegend=False))
    fig.update_yaxes(title_text="Median der Schritte bis zum Erfolg", type="log")
    fig.update_xaxes(tickangle=-35)
    return _base(fig, 480)


def build_noise_chart(pre):
    """Anteil der 200 Wiederholungen der gefundenen Fahrt ohne Zurücksetzen, die am Ende im Zielkreis stehen (Mittel über die 20 Läufe), über dem Rauschen; je Stufe ein Linienzug."""
    fig = go.Figure()
    for lv, color in zip(C.SWEEP_LEVELS, ("#4c78a8", "#e45756")):
        ys = [float(np.mean([r for r in E.noise_cell(pre, lv, nz)["replay"] if r is not None])) for nz in C.NOISE_OPTIONS]
        fig.add_trace(go.Scatter(x=[f"{nz:g}" for nz in C.NOISE_OPTIONS], y=ys, mode="lines+markers", line=dict(color=color, width=2.5), name=C.LEVEL_LABELS[lv]))
    fig.update_xaxes(title_text="Rauschen auf der Verschiebung je Schritt (Zellen)", type="category")
    fig.update_yaxes(title_text="Wiederholungen der Fahrt, die ins Ziel kommen", tickformat=".0%", range=[0, 1.03])
    return _base(fig, 360, legend_y=-0.4)


def build_replay_figure(grid, states, goal_hits):
    """Wiederholungen der gefundenen Fahrt (ein dünner Linienzug je Wiederholung) in einer Welt mit Rauschen; Endpunkte im Zielkreis grün, sonst rot. `states`: (Wiederholungen, Länge + 1, 3)."""
    fig = go.Figure()
    _walls(fig, grid)
    xs, ys = [], []
    for s in states:
        xs += list(s[:, 0]) + [None]
        ys += list(s[:, 1]) + [None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#4c78a8", width=1), opacity=0.35, name="Wiederholungen", hoverinfo="skip"))
    ends = states[:, -1]
    fig.add_trace(go.Scatter(x=ends[:, 0], y=ends[:, 1], mode="markers", name="Endpunkte", hoverinfo="skip", marker=dict(color=["#54a24b" if h else "#e45756" for h in goal_hits], size=5, line=dict(color="white", width=0.4))))
    _markers(fig)
    _arena_axes(fig)
    return _base(fig, 470, legend_y=-0.3, top=0)
