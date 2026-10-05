"""Auswertung: Täuschungsmaße je Stufe (exakt), Lösbarkeitsnachweis, Live-Läufe (Go-Explore neben GA oder Novelty im selben Schrittbudget), Lesen und Aggregieren der vorgerechneten Studie, Mann-Whitney-Test.
Die Studie (Verfahren je Stufe, Einstellungen von Go-Explore, Rauschen) wird beim Bau gerechnet (`generate_precomputed.py`) und hier nur gelesen."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

import ge_baselines as B
import ge_constants as C
import ge_maze as M
import ge_robot as R
import ge_search as S

PRECOMPUTED_PATH = Path(__file__).resolve().parent / "precomputed_sweep.json"


def level_metrics(level):
    """Exakte Täuschungsmaße der Stufe: kürzester Weg (Zellen), Luftlinie Start–Ziel, Umwegfaktor, Fitness-Distanz-Korrelation, lokale Minima (Zahl), Abstand des tiefsten Minimums zum Ziel, freie Zellen."""
    g = M.make_maze(level)
    path = M.shortest_path(g)
    return {"path": len(path) - 1, "straight": float(np.hypot(C.START[0] - C.GOAL[0], C.START[1] - C.GOAL[1])), "detour": M.detour_ratio(g), "fdc": M.fitness_distance_correlation(g),
            "minima": len(M.local_minima(g)), "trap": M.trap_distance(g), "free": int((~g).sum())}


def solvability(level):
    """Lösbarkeitsnachweis: eine aus dem kürzesten Pfad abgeleitete Steuerfolge fährt in diesem Labyrinth ins Ziel. Gibt (Abstand am Ende, benötigte Schritte) zurück."""
    g = M.make_maze(level)
    genome = R.controller_from_path(M.shortest_path(g))
    end = R.simulate(genome[None], g)
    return float(R.distance_to_goal(end)[0]), int((np.abs(genome).sum(axis=1) > 0).sum())


def ge_run(level, seed, budget, explore, cap, ret, cell, select):
    """Ein Go-Explore-Lauf mit den gewählten Einstellungen; läuft bis zum Erfolg oder bis das Budget verbraucht ist."""
    return S.run_go_explore(M.make_maze(level), seed, explore=explore, cap=cap, ret=ret, cell=cell, select=select, budget=budget)


def baseline_run(level, seed, budget, method):
    """Ein Lauf von GA oder Novelty mit dem Budget `budget` (Schritte): so viele Generationen, wie hineinpassen (eine Generation kostet POP · STEPS Schritte)."""
    return B.run_baseline(M.make_maze(level), seed, method, gens=max(1, budget // (C.POP * C.STEPS)))


def solution_path(level, actions):
    """Die Fahrt der gefundenen Aktionsfolge als Punkte (Länge + 1, 2); None, wenn es keine gibt."""
    if actions is None:
        return None
    return S.replay_actions(M.make_maze(level), actions)[0][:, :2]


@lru_cache(maxsize=1)
def load_precomputed():
    return json.loads(PRECOMPUTED_PATH.read_text(encoding="utf-8"))


def success_count(runs, budget=None):
    """Zahl der Läufe, die den Zielkreis erreichten (höchstens `budget` Schritte, wenn angegeben)."""
    return sum(1 for r in runs if r is not None and (budget is None or r <= budget))


def median_solved(runs):
    """Median der Schritte bis zum Erfolg über die erfolgreichen Läufe (None, wenn kein Lauf erfolgreich war)."""
    ok = [r for r in runs if r is not None]
    return float(np.median(ok)) if ok else None


def mannwhitney_p(values_a, values_b):
    """Zweiseitiger Mann-Whitney-Test auf zwei Zahlenlisten."""
    return float(mannwhitneyu(values_a, values_b, alternative="two-sided").pvalue)


def mannwhitney_steps_p(runs_a, runs_b):
    """Zweiseitiger Mann-Whitney-Test auf die Schritte bis zum Erfolg zweier Verfahren; ein Lauf ohne Erfolg zählt als schlechter als jeder erfolgreiche (Schritte unendlich)."""
    cens = lambda runs: [np.inf if r is None else r for r in runs]
    return mannwhitney_p(cens(runs_a), cens(runs_b))


def method_cell(pre, level, method):
    """Die Läufe (20 Seeds) eines Verfahrens auf einer Stufe."""
    return pre["methods"][str(level)][method]


def sweep_cell(pre, level, config):
    return pre["sweep"][str(level)][config]


def noise_cell(pre, level, noise):
    return pre["noise"][str(level)][str(noise)]


def level_info(pre, level):
    return pre["levels"][str(level)]
