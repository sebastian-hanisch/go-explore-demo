"""GA mit Fitness und Novelty Search im Schrittbudget: Auswahl, Novelty von Hand und gegen SciPy, berührte Zellen, Schritte."""

import numpy as np
import pytest
from scipy.spatial import cKDTree

import ge_baselines as B
import ge_constants as C
import ge_maze as M
import ge_robot as R

FREE = M.make_maze(0)
TRAP = M.make_maze(3)


def test_novelty_by_hand():
    """Punkte 0, 1, 3 auf einer Linie, k = 1: Abstand zum nächsten Nachbarn 1, 1, 2."""
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    assert B.novelty_scores(pts, 1) == pytest.approx([1.0, 1.0, 2.0])
    assert B.novelty_scores(pts, 2) == pytest.approx([2.0, 1.5, 2.5])


def test_novelty_agrees_with_scipys_nearest_neighbours():
    pts = np.random.default_rng(1).uniform(0, 24, (60, 2))
    d, _ = cKDTree(pts).query(pts, k=C.NOVELTY_K + 1)
    assert B.novelty_scores(pts) == pytest.approx(d[:, 1:].mean(axis=1))


def test_a_duplicate_point_counts_with_distance_zero_but_the_point_itself_does_not():
    pts = np.array([[1.0, 1.0], [1.0, 1.0], [5.0, 1.0]])
    assert B.novelty_scores(pts, 1) == pytest.approx([0.0, 0.0, 4.0])


def test_the_next_generation_keeps_the_elite_unchanged_and_mutates_the_rest():
    rng = np.random.default_rng(2)
    genomes = R.random_genomes(rng, 20)
    score = np.arange(20, dtype=float)
    nxt = B.next_generation(rng, genomes, score)
    assert nxt.shape == genomes.shape and (nxt[0] == genomes[19]).all() and (nxt[1] == genomes[18]).all() and not (nxt[2] == genomes[17]).all()
    assert (nxt[..., 0] >= -1).all() and (nxt[..., 0] <= 1).all() and (nxt[..., 1] >= 0).all() and (nxt[..., 1] <= 1).all()


def test_a_run_counts_evaluations_and_steps_and_is_reproducible():
    a, b = (B.run_baseline(TRAP, 4, "novelty", gens=5, stop_when_solved=False) for _ in range(2))
    assert a.evaluations == 500 and a.curve_steps == [(g + 1) * 100 * 140 for g in range(5)] and a.curve_cells == b.curve_cells and a.curve_progress == b.curve_progress


def test_cells_touched_and_progress_never_shrink_or_get_worse():
    res = B.run_baseline(TRAP, 1, "fitness", gens=20, stop_when_solved=False)
    assert all(b >= a for a, b in zip(res.curve_cells, res.curve_cells[1:])) and all(b <= a for a, b in zip(res.curve_progress, res.curve_progress[1:])) and res.touched.sum() == res.curve_cells[-1]


def test_the_first_generation_touches_the_cells_of_its_trajectories_by_hand_and_the_first_success_is_counted():
    """Eine Generation (die Zufallspopulation des Seeds): berührte Zellen aus einer skalaren Neuberechnung der Fahrten."""
    res = B.run_baseline(FREE, 3, "fitness", gens=1, stop_when_solved=False)
    genomes = R.random_genomes(np.random.default_rng(3), C.POP)
    _, track = R.simulate(genomes, FREE, trajectories=True)
    cells = {(int(x), int(y)) for t in track for x, y in t}
    assert int(res.touched.sum()) == len(cells) and all(res.touched[i, j] for i, j in cells)
    near = R.distance_to_goal(R.simulate(genomes, FREE)) < C.SUCCESS_RADIUS
    assert res.first_solved == (int(np.argmax(near)) + 1 if near.any() else None)


def test_solved_steps_are_evaluations_times_the_genome_length():
    res = B.run_baseline(FREE, 1, "fitness")
    assert res.first_solved is not None and res.solved_steps == res.first_solved * 140
    none = B.run_baseline(M.make_maze(4), 1, "fitness", gens=3)
    assert none.first_solved is None and none.solved_steps is None


def test_fitness_and_novelty_choose_differently_and_unknown_methods_are_rejected():
    a = B.run_baseline(TRAP, 1, "fitness", gens=10, stop_when_solved=False)
    b = B.run_baseline(TRAP, 1, "novelty", gens=10, stop_when_solved=False)
    assert a.curve_cells != b.curve_cells
    with pytest.raises(ValueError):
        B.run_baseline(TRAP, 1, "ge_free", gens=2)


def test_the_best_genome_is_the_one_closest_to_the_goal():
    res = B.run_baseline(TRAP, 1, "fitness", gens=10, stop_when_solved=False)
    best = R.distance_to_goal(R.simulate(res.best_genome[None], TRAP))[0]
    assert best < 20.0 and best == pytest.approx(min(best, best))
