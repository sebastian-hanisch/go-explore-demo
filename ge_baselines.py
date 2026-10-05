"""Die Vergleichsverfahren aus Stück 1 und 2 im selben Schrittbudget: derselbe Genetische Algorithmus (Turnier 3, die zwei Besten bleiben, Mutation ohne Kreuzung), Auswahl nach Fitness (−Abstand der Endposition zum Ziel)
oder nach Novelty (mittlerer Abstand der Endposition zu den 15 nächsten Endpositionen der Population, ohne Archiv). Eine Evaluation kostet STEPS = 140 Schritte; zusätzlich werden alle Zellen gezählt, die irgendeine Fahrt berührt hat,
und der kleinste wahre Restweg (Zellen) aller berührten Zellen, damit die Verfahren vergleichbar sind.

Zufall nur über `numpy.random.default_rng(seed)` (PCG64, plattformstabil)."""

from dataclasses import dataclass, field

import numpy as np

import ge_constants as C
import ge_maze as M
import ge_robot as R


@dataclass
class BaselineResult:
    first_solved: object                     # Zahl der Evaluationen bis zum ersten Individuum im Zielkreis (None: nie)
    curve_steps: list = field(default_factory=list)       # verbrauchte Schritte nach jeder Generation
    curve_cells: list = field(default_factory=list)       # berührte Zellen (1 × 1) nach jeder Generation
    curve_progress: list = field(default_factory=list)    # kleinster wahrer Restweg aller berührten Zellen nach jeder Generation
    touched: object = None                   # bool-Raster der berührten Zellen
    best_genome: object = None               # Steuerfolge mit dem kleinsten Abstand zum Ziel über den ganzen Lauf
    evaluations: int = 0

    @property
    def solved_steps(self):
        """Verbrauchte Schritte bis zum Erfolg (Evaluationen mal Länge der Steuerfolge), None: nie."""
        return None if self.first_solved is None else self.first_solved * C.STEPS


def next_generation(rng, genomes, score):
    """Neue Population: die ELITE höchsten `score` bleiben unverändert, der Rest sind Mutationen von Turnier-Siegern. Größer ist besser."""
    pop = len(genomes)
    order = np.argsort(-score, kind="stable")
    nxt = [genomes[i].copy() for i in order[:C.ELITE]]
    while len(nxt) < pop:
        cand = rng.integers(0, pop, C.TOURNAMENT)
        parent = cand[np.argmax(score[cand])]
        child = genomes[parent] + rng.normal(0.0, C.MUT_SIGMA, genomes[parent].shape) * (rng.random(genomes[parent].shape) < C.MUT_RATE)
        nxt.append(R.clip_genomes(child))
    return np.array(nxt)


def novelty_scores(points, k=C.NOVELTY_K):
    """Novelty jedes Punkts: mittlerer Abstand zu den `k` nächsten anderen Punkten der Population (der Punkt selbst zählt nicht, ein anderer am selben Ort zählt mit Abstand 0)."""
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    d.sort(axis=1)
    return d[:, 1:k + 1].mean(axis=1)


def run_baseline(grid, seed, method, gens=C.STUDY_GENS, pop=C.POP, stop_when_solved=True):
    """Ein Lauf von GA (`fitness`) oder Novelty Search (`novelty`). Zählt Evaluationen und Schritte bis die erste Endposition im Zielkreis liegt."""
    if method not in C.GENOME_METHODS:
        raise ValueError(f"unbekanntes Verfahren: {method}")
    rng = np.random.default_rng(seed)
    n = grid.shape[0]
    genomes = R.random_genomes(rng, pop)
    true_path = M.bfs_distances(grid, M.cell_of(C.GOAL))
    touched = np.zeros(grid.shape, bool)
    res = BaselineResult(first_solved=None)
    best_d, best_p = np.inf, np.inf
    for gen in range(gens):
        end, track = R.simulate(genomes, grid, trajectories=True)
        dist = R.distance_to_goal(end)
        res.evaluations += pop
        if res.first_solved is None and dist.min() < C.SUCCESS_RADIUS:
            res.first_solved = gen * pop + int(np.argmax(dist < C.SUCCESS_RADIUS)) + 1
        idx = np.clip(track.reshape(-1, 2).astype(int), 0, n - 1)
        touched[idx[:, 0], idx[:, 1]] = True
        best_p = min(best_p, float(true_path[idx[:, 0], idx[:, 1]].min()))
        res.curve_steps.append(res.evaluations * C.STEPS)
        res.curve_cells.append(int(touched.sum()))
        res.curve_progress.append(best_p)
        if dist.min() < best_d:
            best_d = float(dist.min())
            res.best_genome = genomes[int(np.argmin(dist))].copy()
        if stop_when_solved and res.first_solved is not None:
            break
        genomes = next_generation(rng, genomes, -dist if method == "fitness" else novelty_scores(end))
    res.touched = touched
    return res
