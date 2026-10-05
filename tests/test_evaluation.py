"""Auswertung: Täuschungsmaße je Stufe, Lösbarkeit, Läufe, Hilfsfunktionen der Studie, Vollständigkeit und Stimmigkeit der vorgerechneten Datei (inklusive Neurechnung einzelner Läufe)."""

import numpy as np
import pytest
from scipy.stats import mannwhitneyu

import ge_baselines as B
import ge_constants as C
import ge_evaluation as E
import ge_maze as M
import ge_search as S

PRE = E.load_precomputed()


def test_level_metrics_by_hand():
    m0, m3 = E.level_metrics(0), E.level_metrics(3)
    assert m0["path"] == 17 and m0["straight"] == pytest.approx(17.0) and m0["detour"] == pytest.approx(1.0) and m0["minima"] == 0 and m0["trap"] is None and m0["free"] == 22 * 22
    assert m3["path"] == 49 and m3["detour"] == pytest.approx(49 / 17) and m3["minima"] == 1 and m3["trap"] == pytest.approx(9.0) and m3["free"] == 22 * 22 - 19


@pytest.mark.parametrize("level", list(C.LEVELS))
def test_every_level_is_solvable_within_the_step_budget(level):
    dist, steps = E.solvability(level)
    assert dist < 0.5 and steps <= C.STEPS


def test_the_live_runs_use_the_settings_and_the_baseline_budget_in_generations():
    ge = E.ge_run(3, 1, 100000, 10, 140, "free", "pos1", "count")
    assert ge.solved is not None and ge.used >= ge.solved
    bl = E.baseline_run(3, 1, 140000, "novelty")
    assert bl.evaluations <= 1000 and bl.curve_steps[-1] <= 140000
    assert len(E.baseline_run(3, 1, 1400, "fitness").curve_steps) == 1                      # mindestens eine Generation


def test_the_live_run_equals_a_direct_run():
    a = E.ge_run(4, 2, 300000, 10, 140, "free", "pos1", "count")
    b = S.run_go_explore(M.make_maze(4), 2, budget=300000)
    assert a.solved == b.solved and a.curve_cells == b.curve_cells
    c = E.baseline_run(3, 2, 280000, "novelty")
    d = B.run_baseline(M.make_maze(3), 2, "novelty", gens=20)
    assert c.curve_cells == d.curve_cells and c.first_solved == d.first_solved


def test_the_solution_path_has_one_point_more_than_actions_and_ends_in_the_goal_circle():
    ge = E.ge_run(0, 1, 100000, 10, 140, "free", "pos1", "count")
    path = E.solution_path(0, ge.actions)
    assert path.shape == (ge.solution_len + 1, 2) and tuple(path[0]) == C.START and np.hypot(*(path[-1] - np.array(C.GOAL))) < C.SUCCESS_RADIUS and E.solution_path(0, None) is None


def test_success_helpers_by_hand():
    runs = [100, None, 500, 50, None]
    assert E.success_count(runs) == 3 and E.success_count(runs, 100) == 2 and E.median_solved(runs) == 100.0 and E.median_solved([None, None]) is None


def test_mann_whitney_agrees_with_scipy_and_unsolved_runs_count_as_worst():
    x, y = [1.0, 2.0, 3.0, 4.0, 5.0], [3.5, 4.5, 5.5, 6.5, 7.5]
    assert E.mannwhitney_p(x, y) == pytest.approx(mannwhitneyu(x, y, alternative="two-sided").pvalue) and E.mannwhitney_p(x, x) == pytest.approx(1.0)
    assert E.mannwhitney_steps_p([1, 2, 3, None], [1, 2, 3, None]) == pytest.approx(1.0)
    assert E.mannwhitney_steps_p([1, 2, 3, 4, 5, 6], [None] * 6) < 0.01 and E.mannwhitney_steps_p([1, 2, 3, None], [1, 2, 3, 4]) == pytest.approx(mannwhitneyu([1, 2, 3, np.inf], [1, 2, 3, 4]).pvalue)


def test_the_study_covers_every_level_method_setting_and_noise():
    assert PRE["budget"] == C.STUDY_BUDGET == 5600000 and set(PRE["levels"]) == {str(lv) for lv in C.LEVELS} == set(PRE["methods"]) and set(PRE["sweep"]) == set(PRE["noise"]) == {str(lv) for lv in C.SWEEP_LEVELS}
    for lv in C.LEVELS:
        assert set(PRE["methods"][str(lv)]) == set(C.METHODS)
        for m in C.METHODS:
            c = E.method_cell(PRE, lv, m)
            assert len(c["solved"]) == len(c["length"]) == len(c["cells"]) == len(c["progress"]) == C.STUDY_SEEDS and len(c["mean_cells"]) == len(c["solved_by"]) == len(C.CURVE_MARKS) == 56
            assert m in C.GENOME_METHODS or all((s is None) == (n is None) for s, n in zip(c["solved"], c["length"]))             # Go-Explore: Fahrtlänge genau bei Erfolg
            assert all(b >= a for a, b in zip(c["solved_by"], c["solved_by"][1:])) and c["solved_by"][-1] == E.success_count(c["solved"]) / C.STUDY_SEEDS and all(b >= a - 1e-9 for a, b in zip(c["mean_cells"], c["mean_cells"][1:]))
            assert (m not in C.GENOME_METHODS) or all(n is None for n in c["length"])
    for lv in C.SWEEP_LEVELS:
        assert set(PRE["sweep"][str(lv)]) == set(C.SWEEP_CONFIGS) and set(PRE["noise"][str(lv)]) == {str(z) for z in C.NOISE_OPTIONS}
        for cfg in C.SWEEP_CONFIGS:
            c = E.sweep_cell(PRE, lv, cfg)
            assert len(c["solved"]) == len(c["length"]) == len(c["cells"]) == C.STUDY_SEEDS
        for z in C.NOISE_OPTIONS:
            c = E.noise_cell(PRE, lv, z)
            assert len(c["solved"]) == len(c["length"]) == len(c["replay"]) == C.STUDY_SEEDS and all((r is None) == (s is None) for r, s in zip(c["replay"], c["solved"])) and all(r is None or 0.0 <= r <= 1.0 for r in c["replay"])


def test_the_stored_level_metrics_equal_the_exact_ones():
    for lv in C.LEVELS:
        stored, fresh = E.level_info(PRE, lv), E.level_metrics(lv)
        for key in ("path", "straight", "detour", "fdc", "minima", "free"):
            assert stored[key] == pytest.approx(fresh[key])


@pytest.mark.parametrize("level,method,seed", [(0, "ge_free", 0), (3, "ge_free", 4), (4, "ge_free", 9), (3, "ge_replay", 2), (4, "ge_replay", 5), (1, "ge_free", 7)])
def test_a_go_explore_study_run_is_reproduced_exactly_by_a_fresh_run_with_the_same_seed(level, method, seed):
    """Go-Explore braucht nur NumPy-Zufall und elementare Rechnungen: alle gespeicherten Werte eines Laufs stimmen mit einem frischen Lauf überein."""
    res = S.run_go_explore(M.make_maze(level), seed, ret="free" if method == "ge_free" else "replay")
    c = E.method_cell(PRE, level, method)
    assert res.solved == c["solved"][seed] and res.solution_len == c["length"][seed] and res.cells == c["cells"][seed] and res.best_progress == pytest.approx(c["progress"][seed])


@pytest.mark.parametrize("level,method,seed", [(0, "fitness", 0), (3, "novelty", 3), (3, "fitness", 6), (2, "novelty", 11), (4, "novelty", 1)])
def test_a_baseline_study_run_is_reproduced_exactly_by_a_fresh_run_with_the_same_seed(level, method, seed):
    res = B.run_baseline(M.make_maze(level), seed, method)
    c = E.method_cell(PRE, level, method)
    assert res.solved_steps == c["solved"][seed] and int(res.touched.sum()) == c["cells"][seed] and res.curve_progress[-1] == pytest.approx(c["progress"][seed])


@pytest.mark.parametrize("level,config,seed", [(3, "posdir", 1), (4, "explore3", 2), (4, "nocap", 3), (3, "uniform", 4), (4, "pos2", 5)])
def test_a_sweep_run_is_reproduced_exactly(level, config, seed):
    kw = C.SWEEP_CONFIGS[config]
    res = S.run_go_explore(M.make_maze(level), seed, explore=kw.get("explore", C.DEFAULT_EXPLORE), cap=kw.get("cap", C.DEFAULT_CAP), ret=kw.get("ret", C.DEFAULT_RETURN), cell=kw.get("cell", C.DEFAULT_CELL), select=kw.get("select", C.DEFAULT_SELECT))
    c = E.sweep_cell(PRE, level, config)
    assert res.solved == c["solved"][seed] and res.solution_len == c["length"][seed] and res.cells == c["cells"][seed]


@pytest.mark.parametrize("level,noise,seed", [(3, 0.02, 1), (4, 0.05, 2), (4, 0.0, 3), (3, 0.1, 4)])
def test_a_noise_run_is_reproduced_exactly_including_the_replay_rate(level, noise, seed):
    grid = M.make_maze(level)
    res = S.run_go_explore(grid, seed, noise=noise)
    c = E.noise_cell(PRE, level, noise)
    assert res.solved == c["solved"][seed] and res.solution_len == c["length"][seed]
    assert round(S.replay_success_rate(grid, res.actions, noise), 4) == c["replay"][seed]


def test_the_mean_curves_are_the_mean_over_the_runs():
    c = E.method_cell(PRE, 3, "fitness")
    assert c["solved_by"][-1] == pytest.approx(11 / 20) and c["mean_cells"][-1] == pytest.approx(np.mean(c["cells"]), abs=1.0)
