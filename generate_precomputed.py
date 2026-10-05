"""Rechnet die Studie beim Bau vor (nicht in der App): `python generate_precomputed.py [Prozesse]` schreibt `precomputed_sweep.json`.

  methods  je Stufe (0 bis 4) × Verfahren (GA mit Fitness, Novelty, Go-Explore mit kostenloser Rückkehr, Go-Explore mit Rückkehr durch Wiederabspielen) 20 Läufe im Budget von 5.6 Mio Schritten (40 000 Evaluationen):
           Schritte bis zum Erfolg (None = nie), Länge der gefundenen Fahrt (Go-Explore), berührte Zellen, kleinster Restweg; mittlerer Verlauf der Zellen und Anteil erfolgreicher Läufe je 100 000 Schritte
  sweep    Stufe 3 und 4: Go-Explore (kostenlose Rückkehr) mit je einer geänderten Einstellung (Erkundungslänge, Zelle, Auswahl, Längenbegrenzung, Rückkehr), 20 Läufe
  noise    Stufe 3 und 4: Go-Explore (Voreinstellung) in einer Welt mit Rauschen 0 / 0.02 / 0.05 / 0.1 Zellen je Schritt, 20 Läufe; Anteil der 200 Wiederholungen der gefundenen Fahrt ohne Zurücksetzen, die am Ende im Zielkreis stehen
  levels   exakte Täuschungsmaße der Stufen"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import ge_constants as C
import ge_evaluation as E
import ge_maze as M
import ge_search as S


def _curve_at(steps, values, mark):
    """Wert der Verlaufskurve bei `mark` Schritten: der letzte Punkt mit höchstens `mark` Schritten (vor dem ersten Punkt 0; nach dem Ende des Laufs der Endwert)."""
    k = int(np.searchsorted(np.asarray(steps), mark, side="right")) - 1
    return 0 if k < 0 else values[k]


def _method_task(args):
    level, method, seed = args
    grid = M.make_maze(level)
    if method in C.GENOME_METHODS:
        res = E.B.run_baseline(grid, seed, method)
        return level, method, seed, res.solved_steps, None, int(res.touched.sum()), res.curve_progress[-1], res.curve_steps, res.curve_cells
    res = S.run_go_explore(grid, seed, ret="free" if method == "ge_free" else "replay")
    return level, method, seed, res.solved, res.solution_len, res.cells, res.best_progress, res.curve_steps, res.curve_cells


def _sweep_task(args):
    level, config, seed = args
    kw = dict(C.SWEEP_CONFIGS[config])
    res = S.run_go_explore(M.make_maze(level), seed, explore=kw.get("explore", C.DEFAULT_EXPLORE), cap=kw.get("cap", C.DEFAULT_CAP), ret=kw.get("ret", C.DEFAULT_RETURN), cell=kw.get("cell", C.DEFAULT_CELL),
                           select=kw.get("select", C.DEFAULT_SELECT))
    return level, config, seed, res.solved, res.solution_len, res.cells


def _noise_task(args):
    level, noise, seed = args
    grid = M.make_maze(level)
    res = S.run_go_explore(grid, seed, noise=noise)
    rate = None if res.actions is None else S.replay_success_rate(grid, res.actions, noise)
    return level, noise, seed, res.solved, res.solution_len, rate


def main(workers):
    t0 = time.time()
    m_tasks = [(lv, m, s) for lv in C.LEVELS for m in C.METHODS for s in range(C.STUDY_SEEDS)]
    s_tasks = [(lv, cfg, s) for lv in C.SWEEP_LEVELS for cfg in C.SWEEP_CONFIGS for s in range(C.STUDY_SEEDS)]
    n_tasks = [(lv, nz, s) for lv in C.SWEEP_LEVELS for nz in C.NOISE_OPTIONS for s in range(C.STUDY_SEEDS)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        m_res = list(ex.map(_method_task, m_tasks, chunksize=2))
        print(f"Verfahren fertig nach {time.time() - t0:.0f} s", flush=True)
        s_res = list(ex.map(_sweep_task, s_tasks, chunksize=2))
        n_res = list(ex.map(_noise_task, n_tasks, chunksize=2))
    methods = {}
    for lv in C.LEVELS:
        methods[str(lv)] = {}
        for m in C.METHODS:
            rows = sorted([r for r in m_res if r[0] == lv and r[1] == m], key=lambda r: r[2])
            cells_curve = [round(float(np.mean([_curve_at(r[7], r[8], mark) for r in rows])), 3) for mark in C.CURVE_MARKS]
            solved_curve = [round(float(np.mean([r[3] is not None and r[3] <= mark for r in rows])), 3) for mark in C.CURVE_MARKS]
            methods[str(lv)][m] = {"solved": [r[3] for r in rows], "length": [r[4] for r in rows], "cells": [r[5] for r in rows], "progress": [r[6] for r in rows], "mean_cells": cells_curve, "solved_by": solved_curve}
    sweep = {}
    for lv in C.SWEEP_LEVELS:
        sweep[str(lv)] = {}
        for cfg in C.SWEEP_CONFIGS:
            rows = sorted([r for r in s_res if r[0] == lv and r[1] == cfg], key=lambda r: r[2])
            sweep[str(lv)][cfg] = {"solved": [r[3] for r in rows], "length": [r[4] for r in rows], "cells": [r[5] for r in rows]}
    noise = {}
    for lv in C.SWEEP_LEVELS:
        noise[str(lv)] = {}
        for nz in C.NOISE_OPTIONS:
            rows = sorted([r for r in n_res if r[0] == lv and r[1] == nz], key=lambda r: r[2])
            noise[str(lv)][str(nz)] = {"solved": [r[3] for r in rows], "length": [r[4] for r in rows], "replay": [None if r[5] is None else round(r[5], 4) for r in rows]}
    data = {"budget": C.STUDY_BUDGET, "levels": {str(lv): E.level_metrics(lv) for lv in C.LEVELS}, "methods": methods, "sweep": sweep, "noise": noise}
    E.PRECOMPUTED_PATH.write_text(json.dumps(data), encoding="utf-8")
    print(f"fertig in {time.time() - t0:.0f} s -> {E.PRECOMPUTED_PATH}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else min(14, os.cpu_count() or 1))
