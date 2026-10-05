"""Orakel-Tests für Go-Explore: unabhängige Gegenproben.

* **Go-Explore als Ganzes** gegen eine skalare Neuimplementierung mit Wörterbüchern und `math` (je Erkunder und Schritt eine Schleife, Zellen und Archiv als Wörterbuch, die Kosten und der Weg von Hand gezählt); dieselben Zufallszahlen in derselben Reihenfolge.
  Verglichen werden Archiv (Zellen, Schritte, Zustände), Verläufe, Kosten und die gefundene Aktionsfolge, für alle drei Zellarten, beide Auswahlregeln und beide Rückkehrarten.
* **Restweg** gegen SciPys Breitensuche."""

import math

import numpy as np
import pytest

import ge_constants as C
import ge_maze as M
import ge_search as S
from test_oracle_maze import scipy_distances


def scalar_step(grid, x, y, h, turn, speed):
    n = grid.shape[0]
    h = h + turn * C.TURN_MAX
    nx = x + speed * C.SPEED_MAX * math.cos(h)
    ny = y + speed * C.SPEED_MAX * math.sin(h)
    if 0 <= nx < n and 0 <= ny < n and not grid[int(nx), int(ny)]:
        return nx, ny, h
    return x, y, h


def scalar_cell(kind, x, y, h):
    size = C.CELL_SIZE[kind]
    nb = -(-C.N // size)
    ix, iy = min(max(int(math.floor(x / size)), 0), nb - 1), min(max(int(math.floor(y / size)), 0), nb - 1)
    if kind == "posdir":
        return (ix * nb + iy) * C.SECTORS + int(math.floor((h % (2 * math.pi)) / (2 * math.pi) * C.SECTORS)) % C.SECTORS
    return ix * nb + iy


def reference_run(grid, seed, explore, cap, ret, cell, select, budget, n_explorers):
    """Go-Explore skalar; gibt (Archiv {Zelle: (Schritte, x, y, Richtung)}, Kurven (Schritte, Zellen, Restweg), Erfolg (Kosten, Aktionsfolge)) zurück."""
    rng = np.random.default_rng(seed)
    true_path = scipy_distances(grid, M.cell_of(C.GOAL))
    limit = cap if cap else math.inf
    start = (C.START[0], C.START[1], C.START_HEADING)
    nodes = [(*start, 0, -1, [])]                              # (x, y, Richtung, Schritte, Eltern, Aktionsstück)
    archive = {scalar_cell(cell, *start): 0}                   # Zelle -> Knoten
    chosen = {}
    used, curves, solved, prog = 0, [], None, float(true_path[int(C.START[0]), int(C.START[1])])
    while used < budget:
        cand = sorted(archive)
        w = np.array([1.0 / (1.0 + chosen.get(c, 0)) if select == "count" else 1.0 for c in cand])
        pick = rng.choice(np.array(cand), size=n_explorers, p=w / w.sum())
        for c in pick:
            chosen[int(c)] = chosen.get(int(c), 0) + 1
        pn = [archive[int(c)] for c in pick]
        genes = rng.random((n_explorers, explore, 2))
        genes[..., 0] = genes[..., 0] * 2 - 1
        used += n_explorers * explore + (sum(nodes[i][3] for i in pn) if ret == "replay" else 0)
        found, round_best = None, {}
        for r in range(n_explorers):
            x, y, h, s0 = nodes[pn[r]][:4]
            for t in range(explore):
                x, y, h = scalar_step(grid, x, y, h, genes[r, t, 0], genes[r, t, 1])
                s = s0 + t + 1
                if s > limit:
                    continue
                if math.hypot(x - C.GOAL[0], y - C.GOAL[1]) < C.SUCCESS_RADIUS and (found is None or s < found[0]):
                    found = (s, r, t)
                c = scalar_cell(cell, x, y, h)
                best_before = nodes[archive[c]][3] if c in archive else math.inf
                if s < best_before and s < round_best.get(c, (math.inf,))[0]:
                    round_best[c] = (s, r, t, x, y, h)
        if found is not None and solved is None:
            _, r, t = found
            chunks = [genes[r, :t + 1]]
            node = pn[r]
            while node != 0:
                chunks.append(np.array(nodes[node][5]))
                node = nodes[node][4]
            solved = (used, np.vstack(chunks[::-1]))
        for c in sorted(round_best):
            s, r, t, x, y, h = round_best[c]
            nodes.append((x, y, h, s, pn[r], genes[r, :t + 1].copy()))
            archive[c] = len(nodes) - 1
            prog = min(prog, float(true_path[min(max(int(x), 0), grid.shape[0] - 1), min(max(int(y), 0), grid.shape[0] - 1)]))
        curves.append((used, len(archive), prog))
        if solved is not None:
            break
    return {c: nodes[i][3:4] + nodes[i][:3] for c, i in archive.items()}, curves, solved


@pytest.mark.parametrize("cell", C.CELL_KINDS)
@pytest.mark.parametrize("select,ret", [("count", "free"), ("uniform", "free"), ("count", "replay")])
def test_go_explore_agrees_with_a_dictionary_based_reimplementation(cell, select, ret):
    grid = M.make_maze(3)
    kw = dict(explore=4, cap=140, ret=ret, cell=cell, select=select, budget=1500, n_explorers=8)
    ref, curves, solved = reference_run(grid, 5, **kw)
    res = S.run_go_explore(grid, 5, **kw, stop_when_solved=True)
    assert res.curve_steps == [c[0] for c in curves] and res.curve_cells == [c[1] for c in curves] and res.curve_progress == pytest.approx([c[2] for c in curves])
    ids = S.cell_ids(cell, res.archive_xy[:, 0], res.archive_xy[:, 1], np.zeros(len(res.archive_xy))) if cell != "posdir" else None
    assert res.cells == len(ref) and sorted(res.archive_steps) == sorted(v[0] for v in ref.values()) and (res.solved is None) == (solved is None)
    if cell != "posdir":
        by_cell = {int(i): (s, x, y) for i, s, (x, y) in zip(ids, res.archive_steps, res.archive_xy)}
        assert set(by_cell) == set(ref) and all(by_cell[c][0] == ref[c][0] and by_cell[c][1:] == pytest.approx(ref[c][1:3], abs=1e-9) for c in ref)


def test_the_found_ride_agrees_with_the_reference_on_a_case_that_succeeds():
    grid = M.make_maze(0)
    kw = dict(explore=5, cap=140, ret="free", cell="pos1", select="count", budget=100000, n_explorers=20)
    ref, curves, solved = reference_run(grid, 7, **kw)
    res = S.run_go_explore(grid, 7, **kw)
    assert solved is not None and res.solved == solved[0] == curves[-1][0] and res.actions == pytest.approx(solved[1], abs=1e-12) and res.solution_len == len(solved[1])


def test_go_explore_without_a_length_cap_agrees_with_the_reference():
    grid = M.make_maze(4)
    kw = dict(explore=6, cap=0, ret="free", cell="pos1", select="count", budget=40000, n_explorers=30)
    ref, curves, solved = reference_run(grid, 2, **kw)
    res = S.run_go_explore(grid, 2, **kw)
    assert res.curve_cells == [c[1] for c in curves] and res.curve_steps == [c[0] for c in curves] and (res.solved is None) == (solved is None)
    assert max(v[0] for v in ref.values()) > 0


def test_best_progress_agrees_with_scipys_breadth_first_search():
    grid = M.make_maze(4)
    res = S.run_go_explore(grid, 2, budget=30000, stop_when_solved=False)
    ref = scipy_distances(grid, M.cell_of(C.GOAL))
    assert res.best_progress == min(ref[int(x), int(y)] for x, y in res.archive_xy) == res.curve_progress[-1]
