"""Go-Explore: Zellen, Schritt der Kinematik, Archiv, Rückkehr, Kosten, Lösung und Wiederabspielen von Hand und an kleinen Instanzen."""

import numpy as np
import pytest

import ge_constants as C
import ge_maze as M
import ge_robot as R
import ge_search as S

FREE = M.make_maze(0)
TRAP = M.make_maze(3)
SNAKE = M.make_maze(4)


def test_cell_counts_and_ids_by_hand():
    assert [S.n_cells(k) for k in C.CELL_KINDS] == [576, 144, 4608] and [S.cells_per_axis(k) for k in C.CELL_KINDS] == [24, 12, 24]
    assert int(S.cell_ids("pos1", 3.5, 3.5, 0.0)) == 3 * 24 + 3 and int(S.cell_ids("pos2", 3.5, 3.5, 0.0)) == 1 * 12 + 1 and int(S.cell_ids("pos2", 4.0, 4.0, 0.0)) == 2 * 12 + 2
    assert int(S.cell_ids("pos1", 23.9, 23.9, 0.0)) == 23 * 24 + 23 and int(S.cell_ids("pos1", 30.0, -2.0, 0.0)) == 23 * 24 + 0


def test_the_direction_sector_by_hand():
    """Sektor = Richtung / 2π · 8 abgerundet: 0.8 rad → 1, −0.1 rad (= 6.18) → 7, 0 → 0, 2π + 0.1 → 0; die Zelle ist 8 · Position + Sektor."""
    base = (3 * 24 + 3) * 8
    assert [int(S.cell_ids("posdir", 3.5, 3.5, h)) - base for h in (0.0, 0.8, -0.1, 2 * np.pi + 0.1, 3.2)] == [0, 1, 7, 0, 4]


def test_cell_ids_are_arrays_for_arrays():
    ids = S.cell_ids("pos1", np.array([[3.5, 4.5]]), np.array([[3.5, 3.5]]), np.zeros((1, 2)))
    assert ids.shape == (1, 2) and list(ids[0]) == [75, 99]


def test_one_step_agrees_with_the_robot_simulation():
    """Schritt für Schritt dieselbe Kinematik wie `simulate` (mit Wänden): 30 zufällige Steuerfolgen im Labyrinth der Stufe 3."""
    genomes = R.random_genomes(np.random.default_rng(1), 30, 60)
    _, track = R.simulate(genomes, TRAP, trajectories=True)
    x, y, h = np.full(30, C.START[0]), np.full(30, C.START[1]), np.full(30, C.START_HEADING)
    for t in range(60):
        x, y, h = S.step_states(TRAP, x, y, h, genomes[:, t, 0], genomes[:, t, 1])
        assert np.stack([x, y], axis=1) == pytest.approx(track[:, t + 1], abs=1e-12)


def test_a_blocked_step_keeps_the_position_and_changes_the_heading_by_hand():
    x, y, h = S.step_states(FREE, np.array([22.9]), np.array([10.5]), np.array([0.0]), np.array([0.5]), np.array([1.0]))
    # Richtung 0.3 rad: neuer Punkt x = 22.9 + 0.7 cos 0.3 = 23.57 liegt im Rand (Zelle 23): bleibt stehen, Richtung ist 0.3
    assert (x[0], y[0], h[0]) == (22.9, 10.5, pytest.approx(0.3))


def test_noise_moves_the_state_and_needs_a_generator():
    rng = np.random.default_rng(0)
    x, y, _ = S.step_states(FREE, np.array([10.5]), np.array([10.5]), np.array([0.0]), np.array([0.0]), np.array([0.0]), rng, 0.1)
    assert (x[0], y[0]) != (10.5, 10.5) and abs(x[0] - 10.5) < 0.6
    x0, y0, _ = S.step_states(FREE, np.array([10.5]), np.array([10.5]), np.array([0.0]), np.array([0.0]), np.array([0.0]))
    assert (x0[0], y0[0]) == (10.5, 10.5)


def test_a_run_is_reproduced_exactly_by_a_second_run_with_the_same_seed_and_differs_for_another_seed():
    a, b, c = (S.run_go_explore(TRAP, s, budget=200000) for s in (3, 3, 4))
    assert a.curve_cells == b.curve_cells and a.solved == b.solved and (a.actions == b.actions).all() and a.curve_cells != c.curve_cells


def test_the_curves_have_one_point_per_round_cells_never_shrink_and_the_progress_never_gets_worse():
    res = S.run_go_explore(SNAKE, 2, budget=100000, stop_when_solved=False)
    assert len(res.curve_steps) == len(res.curve_cells) == len(res.curve_progress) == res.rounds
    assert all(b >= a for a, b in zip(res.curve_cells, res.curve_cells[1:])) and all(b <= a for a, b in zip(res.curve_progress, res.curve_progress[1:]))
    assert res.curve_steps == [(i + 1) * C.N_EXPLORERS * C.DEFAULT_EXPLORE for i in range(res.rounds)] and res.cells == res.curve_cells[-1] == len(res.archive_xy) == len(res.archive_steps)


def test_the_budget_is_used_up_to_one_round():
    res = S.run_go_explore(SNAKE, 1, budget=10000, stop_when_solved=False)
    assert 10000 <= res.used < 10000 + C.N_EXPLORERS * C.DEFAULT_EXPLORE and res.rounds == 10 and res.solved is None and res.actions is None


def test_returning_by_replay_costs_the_steps_of_the_chosen_ways():
    """Je Runde 100 · 10 Schritte Erkunden plus die Schritte der gewählten Zellen; in der ersten Runde ist alles der Start (0 Schritte): gleiche Kosten wie frei."""
    free = S.run_go_explore(SNAKE, 1, budget=5000, stop_when_solved=False)
    replay = S.run_go_explore(SNAKE, 1, budget=5000, ret="replay", stop_when_solved=False)
    assert free.curve_steps[0] == replay.curve_steps[0] == 1000 and replay.curve_steps[1] > free.curve_steps[1] and replay.curve_cells[:2] == free.curve_cells[:2]
    assert replay.rounds < free.rounds


def test_the_length_cap_limits_the_archive_and_the_solution():
    capped = S.run_go_explore(SNAKE, 1, cap=140, budget=C.STUDY_BUDGET)
    assert capped.solved is not None and capped.solution_len <= 140 and capped.archive_steps.max() <= 140 + 0
    free = S.run_go_explore(SNAKE, 1, cap=0, budget=C.STUDY_BUDGET)
    assert free.solved is not None and free.solution_len > 140 and free.solved < capped.solved


def test_the_archive_keeps_the_start_with_zero_steps_and_every_state_is_inside_the_maze():
    res = S.run_go_explore(TRAP, 1, budget=20000, stop_when_solved=False)
    assert res.archive_steps.min() == 0 and (res.archive_xy >= 0).all() and (res.archive_xy < C.N).all()
    assert (~TRAP[res.archive_xy[:, 0].astype(int), res.archive_xy[:, 1].astype(int)]).all()


def test_the_solution_is_an_action_sequence_that_replays_into_the_goal_circle():
    """Ohne Rauschen ist die Welt deterministisch: die gefundene Aktionsfolge fährt, vom Start abgespielt, ins Ziel; als Steuerfolge der Länge 140 (mit Stillstand aufgefüllt) auch über `simulate`."""
    for level in (0, 3, 4):
        grid = M.make_maze(level)
        res = S.run_go_explore(grid, 1)
        assert res.solved is not None and len(res.actions) == res.solution_len and (res.actions[:, 0] >= -1).all() and (res.actions[:, 0] <= 1).all() and (res.actions[:, 1] >= 0).all() and (res.actions[:, 1] <= 1).all()
        final = S.replay_actions(grid, res.actions)[0, -1]
        assert np.hypot(final[0] - C.GOAL[0], final[1] - C.GOAL[1]) < C.SUCCESS_RADIUS
        genome = S.solution_genome(res.actions)
        assert genome.shape == (C.STEPS, 2) and R.distance_to_goal(R.simulate(genome[None], grid))[0] < C.SUCCESS_RADIUS
        assert S.replay_success_rate(grid, res.actions, 0.0) == 1.0


def test_a_solution_longer_than_the_genome_has_no_genome():
    assert S.solution_genome(np.zeros((141, 2))) is None and S.solution_genome(None) is None and S.solution_genome(np.ones((3, 2)))[3:].sum() == 0


def test_replaying_a_found_ride_in_a_noisy_world_often_misses_the_goal():
    """Entgleisung: dieselbe Aktionsfolge mit Rauschen 0.05 erreicht das Ziel nur noch selten (Stufe 4: die Fahrt ist lang und führt dicht an Wänden vorbei)."""
    res = S.run_go_explore(SNAKE, 1)
    assert S.replay_success_rate(SNAKE, res.actions, 0.0) == 1.0 and S.replay_success_rate(SNAKE, res.actions, 0.05) < 0.3
    assert S.replay_actions(SNAKE, res.actions, 0.05, reps=7).shape == (7, len(res.actions) + 1, 3)


def test_exploring_in_a_noisy_world_still_finds_the_goal():
    res = S.run_go_explore(TRAP, 1, noise=0.05)
    assert res.solved is not None and res.solved < 200000


def test_the_archive_does_not_use_the_goal(monkeypatch):
    """Das Archiv kennt das Ziel nicht: ein verschobenes Ziel ändert das Wachstum der Zellen nicht; nur Erfolg und Restweg ändern sich."""
    a = S.run_go_explore(TRAP, 2, budget=30000, stop_when_solved=False)
    monkeypatch.setattr(C, "GOAL", (20.5, 3.5))
    b = S.run_go_explore(TRAP, 2, budget=30000, stop_when_solved=False)
    assert a.curve_cells == b.curve_cells and (a.archive_xy == b.archive_xy).all() and a.curve_progress != b.curve_progress


def test_continuing_after_the_success_keeps_the_first_solution():
    stopped = S.run_go_explore(TRAP, 5)
    longer = S.run_go_explore(TRAP, 5, budget=stopped.solved + 50000, stop_when_solved=False)
    assert longer.solved == stopped.solved and (longer.actions == stopped.actions).all() and longer.rounds > stopped.rounds and longer.cells >= stopped.cells


def test_the_selection_rule_and_the_cell_kind_change_the_run():
    base = S.run_go_explore(TRAP, 1, budget=30000, stop_when_solved=False)
    assert S.run_go_explore(TRAP, 1, budget=30000, select="uniform", stop_when_solved=False).curve_cells != base.curve_cells
    assert S.run_go_explore(TRAP, 1, budget=30000, cell="pos2", stop_when_solved=False).cells < base.cells < S.run_go_explore(TRAP, 1, budget=30000, cell="posdir", stop_when_solved=False).cells


def test_unknown_settings_are_rejected():
    for kw in ({"ret": "teleport"}, {"cell": "hex"}, {"select": "best"}):
        with pytest.raises(ValueError):
            S.run_go_explore(FREE, 1, budget=1000, **kw)


def test_the_first_hit_with_the_fewest_steps_wins_by_hand():
    hit = np.array([[False, True, True], [True, False, False]])
    steps = np.array([[1, 5, 4], [6, 7, 8]])
    assert S._first_min(hit, steps) == (0, 2)
    assert S._first_min(np.array([[True, True]]), np.array([[3, 3]])) == (0, 0)
