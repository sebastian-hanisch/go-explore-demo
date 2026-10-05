"""Abbildungen: gesperrte Achsen, Spuren, Beschriftungen, Farben."""

import numpy as np
import pytest

import ge_constants as C
import ge_evaluation as E
import ge_maze as M
import ge_search as S
import ge_visualization as V

PRE = E.load_precomputed()


def _locked(fig):
    return all(ax.fixedrange for ax in fig.select_xaxes()) and all(ax.fixedrange for ax in fig.select_yaxes())


def test_all_charts_lock_their_axes():
    grid = M.make_maze(3)
    ge = E.ge_run(3, 1, 100000, 10, 140, "free", "pos1", "count")
    bl = E.baseline_run(3, 1, 300000, "novelty")
    states = S.replay_actions(grid, ge.actions, 0.05, 20)
    figs = [V.build_archive_map(grid, ge.archive_xy, ge.archive_steps, E.solution_path(3, ge.actions)), V.build_touch_map(grid, bl.touched), V.build_replay_figure(grid, states, np.ones(20, bool)),
            V.build_cells_curve((ge.curve_steps, ge.curve_cells), (bl.curve_steps, bl.curve_cells), "novelty"), V.build_progress_curve((ge.curve_steps, ge.curve_progress), (bl.curve_steps, bl.curve_progress), "fitness"),
            *[V.build_success_curve(PRE, lv) for lv in C.LEVELS], V.build_success_bars(PRE), V.build_steps_chart(PRE), *[V.build_sweep_chart(PRE, lv) for lv in C.SWEEP_LEVELS], V.build_noise_chart(PRE)]
    assert all(_locked(f) for f in figs)


def test_archive_map_shows_every_archive_state_coloured_by_steps_and_the_found_ride():
    grid = M.make_maze(4)
    ge = E.ge_run(4, 1, 300000, 10, 140, "free", "pos1", "count")
    path = E.solution_path(4, ge.actions)
    fig = V.build_archive_map(grid, ge.archive_xy, ge.archive_steps, path)
    arch = next(t for t in fig.data if t.name == "Archiv (ein Zustand je Zelle)")
    ride = next(t for t in fig.data if t.name == "gefundene Fahrt")
    assert len(arch.x) == ge.cells and list(arch.marker.color) == list(ge.archive_steps) and len(ride.x) == ge.solution_len + 1
    walls = np.array(fig.data[0].z).T
    assert (walls[grid] == 1.0).all() and np.isnan(walls[~grid]).all()
    assert not any(t.name == "gefundene Fahrt" for t in V.build_archive_map(grid, ge.archive_xy, ge.archive_steps).data)


def test_touch_map_marks_the_touched_cells_and_draws_the_closest_ride_when_given():
    grid = M.make_maze(3)
    bl = E.baseline_run(3, 1, 140000, "fitness")
    fig = V.build_touch_map(grid, bl.touched)
    heat = next(t for t in fig.data if t.name == "berührte Zellen")
    z = np.array(heat.z).T
    assert (z[bl.touched] == 1.0).all() and np.isnan(z[~bl.touched]).all()
    assert any(t.name == "Fahrt nächst dem Ziel" for t in V.build_touch_map(grid, bl.touched, np.zeros((5, 2))).data)


def test_pair_curves_name_both_methods_and_the_colours_are_fixed():
    fig = V.build_cells_curve(([1, 2], [3, 4]), ([1, 2], [5, 6]), "novelty")
    assert [t.name for t in fig.data] == [C.METHOD_LABELS["ge_free"], C.BASELINE_LABELS["novelty"]] and list(fig.data[0].y) == [3, 4] and list(fig.data[1].y) == [5, 6]
    assert fig.data[0].line.color == V.METHOD_COLORS["ge_free"] and fig.data[1].line.color == V.METHOD_COLORS["novelty"]
    prog = V.build_progress_curve(([1], [2]), ([1], [3]), "fitness")
    assert prog.data[1].line.color == V.METHOD_COLORS["fitness"]


def test_method_colours_cover_every_method():
    assert set(V.METHOD_COLORS) == set(C.METHODS) and len(set(V.METHOD_COLORS.values())) == len(C.METHODS)


def test_success_curve_has_one_line_per_method_over_the_budget_marks():
    fig = V.build_success_curve(PRE, 3)
    assert [t.name for t in fig.data] == [C.METHOD_LABELS[m] for m in C.METHODS] and all(list(t.x) == list(C.CURVE_MARKS) for t in fig.data)
    assert fig.data[0].y[-1] == pytest.approx(11 / 20)


def test_success_bars_and_steps_chart_have_one_series_per_method_over_all_levels():
    bars, steps = V.build_success_bars(PRE), V.build_steps_chart(PRE)
    assert [t.name for t in bars.data] == [t.name for t in steps.data] == [C.METHOD_LABELS[m] for m in C.METHODS] and all(len(t.x) == len(C.LEVELS) for t in bars.data)
    assert list(bars.data[0].y) == [20, 20, 20, 11, 0] and list(bars.data[2].y) == [20] * 5
    assert steps.data[0].y[4] is None and steps.data[2].y[4] == 216500.0 and steps.layout.yaxis.type == "log"


def test_sweep_chart_has_one_bar_per_setting_with_the_default_highlighted():
    fig = V.build_sweep_chart(PRE, 4)
    t = fig.data[0]
    assert list(t.x) == [C.SWEEP_LABELS[c] for c in C.SWEEP_CONFIGS] and t.y[0] == 216500.0 and t.marker.color[0] == V.METHOD_COLORS["ge_free"] and len(set(t.marker.color)) == 2


def test_noise_chart_has_a_line_per_level_over_the_noise_values():
    fig = V.build_noise_chart(PRE)
    assert [t.name for t in fig.data] == [C.LEVEL_LABELS[3], C.LEVEL_LABELS[4]] and all(list(t.x) == [f"{z:g}" for z in C.NOISE_OPTIONS] for t in fig.data)
    assert fig.data[0].y[0] == 1.0 and fig.data[1].y[0] == 1.0 and fig.data[1].y[-1] == 0.0


def test_replay_figure_has_one_line_per_replay_and_one_end_marker_each():
    grid = M.make_maze(3)
    states = S.replay_actions(grid, np.tile([[0.0, 0.5]], (10, 1)), 0.05, 12)
    fig = V.build_replay_figure(grid, states, np.arange(12) % 2 == 0)
    lines = next(t for t in fig.data if t.name == "Wiederholungen")
    ends = next(t for t in fig.data if t.name == "Endpunkte")
    assert sum(1 for v in lines.x if v is None) == 12 and len(ends.x) == 12 and list(ends.marker.color[:2]) == ["#54a24b", "#e45756"]
