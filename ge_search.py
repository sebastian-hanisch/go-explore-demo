"""Go-Explore (Ecoffet, Huizinga, Lehman, Stanley, Clune): erst zu einem vielversprechenden Zwischenstand **zurückkehren**, dann von dort **erkunden**.

**Archiv:** je **Zelle** (Position im 1 × 1- oder 2 × 2-Raster, optional mit Richtungssektor) der Zwischenstand (x, y, Richtung), mit dem die Zelle am schnellsten (mit den wenigsten Schritten seit dem Start) erreicht wurde. Dazu der Weg dorthin als Baum von Aktionsstücken.
**Eine Runde:** 100 Erkunder; jeder wählt eine Zelle (Gewicht 1 / (1 + bisherige Wahlen), oder gleichverteilt), **kehrt zurück** (kostenlos: der Simulator wird auf den gespeicherten Zustand gesetzt; oder durch Wiederabspielen der Aktionen vom Start, das die Schritte des Wegs kostet)
und fährt `explore` zufällige Schritte (Drehgen gleichverteilt in [−1, 1], Tempogen in [0, 1]). **Jeder** dabei erreichte Zustand wird seiner Zelle zugeordnet; ist die Zelle neu oder mit weniger Schritten erreicht (und der Weg nicht länger als die Längenbegrenzung `cap`), ersetzt er den Eintrag.
Kommt ein Zustand in den Zielkreis, ist der Lauf erfolgreich; die Aktionsfolge von Start bis dorthin ist die gefundene Fahrt (eine Steuerfolge, die nach `solution_len` Schritten stillsteht).

**Kosten** in simulierten Schritten: Runde = 100 · explore (+ bei Wiederabspielen die Summe der Wege der gewählten Zellen). Die Genauigkeit des Budgets ist eine Runde.
**Rauschen** (nur für das Entgleisungs-Experiment): Gauß-Rauschen auf der Verschiebung je Schritt; die Welt ist dann nicht mehr deterministisch, das Zurücksetzen hilft aber weiter.

Zufall nur über `numpy.random.default_rng(seed)` (PCG64, plattformstabil)."""

from dataclasses import dataclass, field

import numpy as np

import ge_constants as C
import ge_maze as M


def cells_per_axis(kind):
    return -(-C.N // C.CELL_SIZE[kind])


def n_cells(kind):
    return cells_per_axis(kind) ** 2 * (C.SECTORS if kind == "posdir" else 1)


def cell_ids(kind, x, y, h):
    """Zellnummer der Zustände (x, y, Richtung): Position im Raster der Kantenlänge CELL_SIZE (außerhalb auf den Rand begrenzt), bei `posdir` mal 8 plus Richtungssektor ([0, 2π) in 8 gleiche Teile)."""
    nb, size = cells_per_axis(kind), C.CELL_SIZE[kind]
    ix = np.clip(np.floor(np.asarray(x) / size).astype(int), 0, nb - 1)
    iy = np.clip(np.floor(np.asarray(y) / size).astype(int), 0, nb - 1)
    base = ix * nb + iy
    if kind == "posdir":
        sector = np.floor(np.mod(np.asarray(h), 2 * np.pi) / (2 * np.pi) * C.SECTORS).astype(int) % C.SECTORS
        return base * C.SECTORS + sector
    return base


def step_states(grid, x, y, h, turn, speed, rng=None, noise=0.0):
    """Ein Schritt der Kinematik für viele Zustände: Richtung += Drehgen · TURN_MAX, Fahrt um Tempogen · SPEED_MAX; bei Rauschen kommt Gauß-Rauschen auf die Verschiebung. Landet der neue Punkt in einer belegten Zelle, bleibt der Roboter stehen (die Richtung bleibt geändert)."""
    n = grid.shape[0]
    h = h + turn * C.TURN_MAX
    nx = x + speed * C.SPEED_MAX * np.cos(h)
    ny = y + speed * C.SPEED_MAX * np.sin(h)
    if noise:
        nx = nx + rng.normal(0.0, noise, np.shape(nx))
        ny = ny + rng.normal(0.0, noise, np.shape(ny))
    idx = np.clip(np.stack([nx, ny], axis=-1).astype(int), 0, n - 1)
    blocked = grid[idx[..., 0], idx[..., 1]] | (nx < 0) | (ny < 0) | (nx >= n) | (ny >= n)
    return np.where(blocked, x, nx), np.where(blocked, y, ny), h


@dataclass
class GoExploreResult:
    solved: object                           # verbrauchte Schritte bis zum Erfolg (Ende der Runde), None: nie
    solution_len: object                     # Zahl der Schritte der gefundenen Fahrt (None: nie)
    actions: object                          # gefundene Aktionsfolge (Länge, 2): Drehgen, Tempogen (None: nie)
    used: int                                # verbrauchte Schritte insgesamt
    rounds: int
    curve_steps: list = field(default_factory=list)       # verbrauchte Schritte nach jeder Runde
    curve_cells: list = field(default_factory=list)       # Zellen im Archiv nach jeder Runde
    curve_progress: list = field(default_factory=list)    # kleinster wahrer Restweg (Zellen) aller Archivzustände nach jeder Runde
    archive_xy: object = None                # Positionen der Archivzustände (E, 2)
    archive_steps: object = None             # Schritte bis zu diesen Zuständen (E,)
    cells: int = 0                           # Zellen im Archiv am Ende
    best_progress: float = np.inf            # kleinster wahrer Restweg am Ende


def run_go_explore(grid, seed, explore=C.DEFAULT_EXPLORE, cap=C.DEFAULT_CAP, ret=C.DEFAULT_RETURN, cell=C.DEFAULT_CELL, select=C.DEFAULT_SELECT, budget=C.STUDY_BUDGET,
                   n_explorers=C.N_EXPLORERS, noise=0.0, stop_when_solved=True):
    """Ein Lauf von Go-Explore. `cap` = 0: keine Längenbegrenzung. Bei `stop_when_solved=False` läuft das Budget weiter und das Archiv wächst über den Erfolg hinaus (die erste gefundene Fahrt bleibt die Lösung)."""
    if ret not in C.RETURNS or cell not in C.CELL_KINDS or select not in C.SELECTS:
        raise ValueError("unbekannte Einstellung")
    rng = np.random.default_rng(seed)
    n = grid.shape[0]
    true_path = M.bfs_distances(grid, M.cell_of(C.GOAL))
    best = np.full(n_cells(cell), np.inf)                      # wenigste Schritte je Zelle
    node_of = np.full(n_cells(cell), -1)                       # Knoten des Archivs je Zelle
    chosen = np.zeros(n_cells(cell))
    nodes = [(C.START[0], C.START[1], C.START_HEADING, 0, -1, np.zeros((0, 2)))]        # (x, y, Richtung, Schritte, Elternknoten, Aktionsstück)
    c0 = int(cell_ids(cell, C.START[0], C.START[1], C.START_HEADING))
    best[c0], node_of[c0] = 0, 0
    res = GoExploreResult(solved=None, solution_len=None, actions=None, used=0, rounds=0)
    prog = float(true_path[int(C.START[0]), int(C.START[1])])
    limit = cap if cap else np.inf
    while res.used < budget:
        cand = np.flatnonzero(node_of >= 0)
        w = 1.0 / (1.0 + chosen[cand]) if select == "count" else np.ones(len(cand))
        pick = rng.choice(cand, size=n_explorers, p=w / w.sum())
        np.add.at(chosen, pick, 1)
        pn = node_of[pick]
        x = np.array([nodes[i][0] for i in pn])
        y = np.array([nodes[i][1] for i in pn])
        h = np.array([nodes[i][2] for i in pn])
        s0 = np.array([nodes[i][3] for i in pn])
        genes = rng.random((n_explorers, explore, 2))
        genes[..., 0] = genes[..., 0] * 2 - 1
        X, Y, H = (np.empty((n_explorers, explore)) for _ in range(3))
        for t in range(explore):
            x, y, h = step_states(grid, x, y, h, genes[:, t, 0], genes[:, t, 1], rng, noise)
            X[:, t], Y[:, t], H[:, t] = x, y, h
        res.used += n_explorers * explore + (int(s0.sum()) if ret == "replay" else 0)
        res.rounds += 1
        steps = s0[:, None] + np.arange(1, explore + 1)[None, :]
        valid = steps <= limit
        hit = valid & (np.hypot(X - C.GOAL[0], Y - C.GOAL[1]) < C.SUCCESS_RADIUS)
        if hit.any() and res.solved is None:
            r, t = _first_min(hit, steps)
            chunks = [genes[r, :t + 1].copy()]
            node = int(pn[r])
            while node != 0:
                chunks.append(nodes[node][5])
                node = nodes[node][4]
            res.actions = np.vstack(chunks[::-1])
            res.solution_len, res.solved = len(res.actions), res.used
        ids = cell_ids(cell, X, Y, H)
        flat_c, flat_s, flat_v = ids.ravel(), steps.ravel(), valid.ravel()
        improved = np.flatnonzero(flat_v & (flat_s < best[flat_c]))
        if len(improved):
            order = improved[np.lexsort((flat_s[improved], flat_c[improved]))]
            first = np.ones(len(order), bool)
            first[1:] = flat_c[order][1:] != flat_c[order][:-1]
            for k in order[first]:
                r, t = divmod(int(k), explore)
                nodes.append((X[r, t], Y[r, t], H[r, t], int(flat_s[k]), int(pn[r]), genes[r, :t + 1].copy()))
                best[int(flat_c[k])], node_of[int(flat_c[k])] = flat_s[k], len(nodes) - 1
                prog = min(prog, float(true_path[min(max(int(X[r, t]), 0), n - 1), min(max(int(Y[r, t]), 0), n - 1)]))
        res.curve_steps.append(res.used)
        res.curve_cells.append(int((node_of >= 0).sum()))
        res.curve_progress.append(prog)
        if res.solved is not None and stop_when_solved:
            break
    arch = node_of[node_of >= 0]
    res.archive_xy = np.array([[nodes[i][0], nodes[i][1]] for i in arch])
    res.archive_steps = np.array([nodes[i][3] for i in arch])
    res.cells, res.best_progress = len(arch), prog
    return res


def _first_min(hit, steps):
    """Der Treffer mit den wenigsten Schritten (bei Gleichstand der erste in Zeilen-, dann Spaltenreihenfolge): (Erkunder, Schritt)."""
    idx = np.argwhere(hit)
    s = steps[hit]
    r, t = idx[int(np.argmin(s))]
    return int(r), int(t)


def solution_genome(actions):
    """Die gefundene Aktionsfolge als Steuerfolge der Länge STEPS (mit Gen (0, 0), also Stillstand, aufgefüllt); None, wenn die Fahrt länger als STEPS ist."""
    if actions is None or len(actions) > C.STEPS:
        return None
    g = np.zeros((C.STEPS, 2))
    g[:len(actions)] = actions
    return g


def replay_actions(grid, actions, noise=0.0, reps=1, seed=C.REPLAY_SEED):
    """Spielt die Aktionsfolge vom Start aus `reps`-mal ab (bei Rauschen jedes Mal anders). Gibt die Zustände (reps, Länge + 1, 3) zurück: x, y, Richtung."""
    rng = np.random.default_rng(seed)
    x, y, h = np.full(reps, C.START[0]), np.full(reps, C.START[1]), np.full(reps, C.START_HEADING)
    out = [np.stack([x, y, h], axis=1)]
    for turn, speed in actions:
        x, y, h = step_states(grid, x, y, h, turn, speed, rng, noise)
        out.append(np.stack([x, y, h], axis=1))
    return np.stack(out, axis=1)


def replay_success_rate(grid, actions, noise, reps=C.REPLAYS, seed=C.REPLAY_SEED):
    """Anteil der Wiederholungen, deren Fahrt am Ende der Aktionsfolge im Zielkreis steht."""
    final = replay_actions(grid, actions, noise, reps, seed)[:, -1]
    return float((np.hypot(final[:, 0] - C.GOAL[0], final[:, 1] - C.GOAL[1]) < C.SUCCESS_RADIUS).mean())
