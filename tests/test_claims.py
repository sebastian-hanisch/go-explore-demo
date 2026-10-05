"""JEDE Zahl aus README und App-Texten wird hier nachgerechnet: Täuschungsmaße exakt aus dem Labyrinth, Studienzahlen aus der vorgerechneten Datei (je Zelle 20 Läufe, Seeds 0 bis 19, Budget 5.6 Mio simulierte Schritte).
Die Datei ist fest; ändern sich die Zahlen nach einer neuen Rechnung, müssen README und Hilfetexte nachgezogen werden."""

from pathlib import Path

import numpy as np

import ge_constants as C
import ge_evaluation as E

PRE = E.load_precomputed()


def med(level, method):
    return E.median_solved(E.method_cell(PRE, level, method)["solved"])


def count(level, method):
    return E.success_count(E.method_cell(PRE, level, method)["solved"])


def sweep_med(level, config):
    return E.median_solved(E.sweep_cell(PRE, level, config)["solved"])


def length_med(cell):
    return float(np.median([x for x in cell["length"] if x is not None]))


def p_methods(level, a, b):
    return E.mannwhitney_steps_p(E.method_cell(PRE, level, a)["solved"], E.method_cell(PRE, level, b)["solved"])


def p_sweep(level, a, b):
    return E.mannwhitney_steps_p(E.sweep_cell(PRE, level, a)["solved"], E.sweep_cell(PRE, level, b)["solved"])


def replay_mean(level, noise):
    return 100 * float(np.mean([r for r in E.noise_cell(PRE, level, noise)["replay"] if r is not None]))


def test_exact_deception_measures_and_study_design():
    """README: Stufe 4: kürzester Weg 77 Zellen, Umwegfaktor 4.53, Korrelation 0.68, 2 Fallen; die kürzeste Lösung per Pfad-Regler braucht 136 Schritte; Studie: 5 Stufen × 4 Verfahren × 20 Läufe = 400, Einstellungen 2 × 8 × 20 = 320,
    Rauschen 2 × 4 × 20 = 160, zusammen 880 Läufe; Budget 5.6 Mio Schritte = 40 000 Evaluationen × 140; 100 Erkunder je Runde."""
    m4 = E.level_metrics(4)
    assert (m4["path"], round(m4["detour"], 2), round(m4["fdc"], 2), m4["minima"]) == (77, 4.53, 0.68, 2) and E.solvability(4)[1] == 136
    assert len(C.LEVELS) * len(C.METHODS) * C.STUDY_SEEDS == 400 and len(C.SWEEP_LEVELS) * len(C.SWEEP_CONFIGS) * C.STUDY_SEEDS == 320 and len(C.SWEEP_LEVELS) * len(C.NOISE_OPTIONS) * C.STUDY_SEEDS == 160
    assert C.STUDY_BUDGET == 5600000 == 40000 * 140 and C.N_EXPLORERS == 100 and C.NOVELTY_K == 15 and C.REPLAYS == 200


def test_stage_four_is_solved_by_go_explore_only():
    """README: Stufe 4: GA mit Fitness 0 von 20, Novelty 0 von 20 (kleinster Restweg 34.60 und 26.20 Zellen), Go-Explore 20 von 20: Median 216 500 Schritte (kostenlose Rückkehr), 1 677 411.5 (Wiederabspielen); Fahrtlänge 140 Schritte."""
    assert [count(4, m) for m in C.METHODS] == [0, 0, 20, 20]
    assert round(float(np.mean(E.method_cell(PRE, 4, "fitness")["progress"])), 2) == 34.60 and round(float(np.mean(E.method_cell(PRE, 4, "novelty")["progress"])), 2) == 26.20
    assert (med(4, "ge_free"), med(4, "ge_replay")) == (216500.0, 1677411.5) and length_med(E.method_cell(PRE, 4, "ge_free")) == 140.0 and length_med(E.method_cell(PRE, 4, "ge_replay")) == 140.0
    assert C.STUDY_BUDGET / 216500 > 25 and round(216500 / 140) == 1546 and round(100 * 216500 / C.STUDY_BUDGET, 1) == 3.9 and round(1677411.5 / C.STUDY_BUDGET, 2) == 0.30


def test_stage_three_go_explore_is_ten_times_cheaper_than_novelty():
    """README: Stufe 3: Median 14 000 Schritte (Go-Explore, kostenlos), 58 499 (Wiederabspielen) gegen 138 390 (Novelty, 20 von 20) und 177 380 (Fitness, 11 von 20); p < 0.001 gegen Novelty und gegen Fitness."""
    assert [med(3, m) for m in C.METHODS] == [177380.0, 138390.0, 14000.0, 58499.0] and [count(3, m) for m in C.METHODS] == [11, 20, 20, 20]
    assert p_methods(3, "ge_free", "novelty") < 0.001 and p_methods(3, "ge_free", "fitness") < 0.001 and p_methods(3, "ge_replay", "novelty") < 0.001 and p_methods(3, "ge_replay", "fitness") < 0.001
    assert 138390 / 14000 > 9.8 and 138390 / 58499 > 2.3


def test_the_easy_stages_show_a_smaller_or_no_advantage():
    """README: Stufe 0 / 1 / 2: Median Fitness 8 890 / 17 640 / 26 670, Novelty 8 890 / 20 440 / 29 050, Go-Explore kostenlos 5 000 / 6 000 / 10 000, mit Wiederabspielen 11 177 / 14 149.5 / 31 545.5; gegen Novelty kostenlos p = 0.32 (Stufe 0), p < 0.001 (Stufe 1),
    p = 0.0019 (Stufe 2); mit Wiederabspielen p = 0.32 (0), 0.034 (1), 0.44 (2). Alle 20 von 20."""
    assert [med(lv, "fitness") for lv in (0, 1, 2)] == [8890.0, 17640.0, 26670.0] and [med(lv, "novelty") for lv in (0, 1, 2)] == [8890.0, 20440.0, 29050.0]
    assert [med(lv, "ge_free") for lv in (0, 1, 2)] == [5000.0, 6000.0, 10000.0] and [med(lv, "ge_replay") for lv in (0, 1, 2)] == [11177.0, 14149.5, 31545.5]
    assert [round(p_methods(lv, "ge_free", "novelty"), 2) for lv in (0,)] == [0.32] and p_methods(1, "ge_free", "novelty") < 0.001 and round(p_methods(2, "ge_free", "novelty"), 4) == 0.0019
    assert [round(p_methods(0, "ge_replay", "novelty"), 2), round(p_methods(1, "ge_replay", "novelty"), 3), round(p_methods(2, "ge_replay", "novelty"), 2)] == [0.32, 0.034, 0.44]
    assert all(count(lv, m) == 20 for lv in (0, 1, 2) for m in C.METHODS)


def test_the_found_ride_lengths():
    """README: Median der Fahrtlänge der Go-Explore-Läufe (kostenlos) auf Stufe 0 bis 4: 41 / 39 / 64 / 99 / 140 Schritte."""
    assert [length_med(E.method_cell(PRE, lv, "ge_free")) for lv in C.LEVELS] == [41.0, 39.0, 64.0, 99.0, 140.0]


def test_the_settings_on_stage_four():
    """README: Stufe 4, Median der Schritte: Voreinstellung 216 500; Erkunden mit 3 / 40 Schritten 77 400 / 848 000; Zelle 2 × 2 267 500; mit Richtung 1 003 000; gleichverteilt 199 500 (p = 0.27); ohne Längenbegrenzung 28 000 (Fahrtlänge 189);
    Wiederabspielen 1 677 411.5; alle anderen p < 0.01; Zellen im Archiv 406.6 (Voreinstellung), 137.2 (2 × 2), 3 167.1 (mit Richtung)."""
    assert [sweep_med(4, c) for c in C.SWEEP_CONFIGS] == [216500.0, 77400.0, 848000.0, 267500.0, 1003000.0, 199500.0, 28000.0, 1677411.5]
    assert round(p_sweep(4, "uniform", "default"), 2) == 0.27 and all(p_sweep(4, c, "default") < 0.01 for c in C.SWEEP_CONFIGS if c not in ("default", "uniform"))
    assert length_med(E.sweep_cell(PRE, 4, "nocap")) == 189.0 and all(E.success_count(E.sweep_cell(PRE, 4, c)["solved"]) == 20 for c in C.SWEEP_CONFIGS)
    cells = lambda c: round(float(np.mean(E.sweep_cell(PRE, 4, c)["cells"])), 1)
    assert (cells("default"), cells("pos2"), cells("posdir")) == (406.6, 137.2, 3167.1)
    assert 1677411.5 / 216500 > 7 and 1003000 / 216500 > 4 and 216500 / 28000 > 7.7 and 848000 / 216500 > 3.9 and 216500 / 77400 > 2.7


def test_the_settings_on_stage_three():
    """README: Stufe 3, Median: Voreinstellung 14 000; Erkunden mit 3 / 40 Schritten 12 600 / 20 000; Zelle 2 × 2 13 000; mit Richtung 35 000; gleichverteilt 22 000 (p < 0.001, hier belegt langsamer); ohne Längenbegrenzung 14 000 (gleiche Läufe, p = 1);
    Wiederabspielen 58 499."""
    assert [sweep_med(3, c) for c in C.SWEEP_CONFIGS] == [14000.0, 12600.0, 20000.0, 13000.0, 35000.0, 22000.0, 14000.0, 58499.0]
    assert p_sweep(3, "uniform", "default") < 0.001 and p_sweep(3, "nocap", "default") == 1.0 and E.sweep_cell(PRE, 3, "nocap")["solved"] == E.sweep_cell(PRE, 3, "default")["solved"]


def test_the_default_sweep_row_equals_the_method_row():
    """Die Voreinstellung im Sweep und Go-Explore mit kostenloser Rückkehr in der Methodenstudie sind dieselben Läufe; ebenso das Wiederabspielen."""
    for lv in C.SWEEP_LEVELS:
        assert E.sweep_cell(PRE, lv, "default")["solved"] == E.method_cell(PRE, lv, "ge_free")["solved"] and E.sweep_cell(PRE, lv, "replay")["solved"] == E.method_cell(PRE, lv, "ge_replay")["solved"]
        assert E.noise_cell(PRE, lv, 0.0)["solved"] == E.method_cell(PRE, lv, "ge_free")["solved"]


def test_the_noise_study():
    """README: Anteil der 200 Wiederholungen ohne Zurücksetzen im Zielkreis (Mittel über die Läufe), Rauschen 0 / 0.02 / 0.05 / 0.1: Stufe 3 100.00 / 55.30 / 51.78 / 14.75 %, Stufe 4 100.00 / 9.42 / 0.97 / 0.00 %;
    Go-Explore löst alle 160 Läufe; Median Stufe 3 14 000 / 15 000 / 15 000 / 15 000, Stufe 4 216 500 / 227 500 / 194 500 / 149 000."""
    assert [round(replay_mean(3, z), 2) for z in C.NOISE_OPTIONS] == [100.00, 55.30, 51.78, 14.75] and [round(replay_mean(4, z), 2) for z in C.NOISE_OPTIONS] == [100.00, 9.42, 0.97, 0.00]
    assert all(E.success_count(E.noise_cell(PRE, lv, z)["solved"]) == 20 for lv in C.SWEEP_LEVELS for z in C.NOISE_OPTIONS)
    assert [E.median_solved(E.noise_cell(PRE, 3, z)["solved"]) for z in C.NOISE_OPTIONS] == [14000.0, 15000.0, 15000.0, 15000.0] and [E.median_solved(E.noise_cell(PRE, 4, z)["solved"]) for z in C.NOISE_OPTIONS] == [216500.0, 227500.0, 194500.0, 149000.0]


def test_the_preset_help_texts_quote_the_study():
    """PRESET_HELP: 0 von 20 / 20 von 20 / 216 500; 1 677 411.5; 1 003 000 und 3 167.1 gegen 406.6; 51.78 %."""
    h = C.PRESET_HELP
    assert "0 von 20" in h["Die Falle"] and "20 von 20" in h["Die Falle"] and "216 500" in h["Die Falle"] and "5.6 Mio" in h["Die Falle"]
    assert "1 677 411.5" in h["Rückkehr kostet"] and "216 500" in h["Rückkehr kostet"]
    assert "1 003 000" in h["Zu feine Zellen"] and "3 167.1" in h["Zu feine Zellen"] and "406.6" in h["Zu feine Zellen"]
    assert "51.78 %" in h["Entgleisung"] and "0.05" in h["Entgleisung"]


def test_the_app_info_formats_match_the_readme_numbers():
    """Der Infotext der App rundet die Anteile der Wiederholungen auf eine Stelle (14.75 → 14.8, ein Gleichstand rundet auf gerade)."""
    assert f"{replay_mean(3, 0.1):.1f}" in ("14.7", "14.8") and f"{replay_mean(4, 0.02):.1f}" == "9.4" and f"{replay_mean(4, 0.05):.1f}" == "1.0"


def test_every_row_of_the_readme_tables_matches_the_stored_study():
    """Die drei großen Tabellen der README (20 + 16 + 8 Zeilen) werden Zeile für Zeile aus der vorgerechneten Datei neu gebildet."""
    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
    n = C.STUDY_SEEDS
    for lv in C.LEVELS:
        for m in C.METHODS:
            c = E.method_cell(PRE, lv, m)
            ln = [x for x in c["length"] if x is not None]
            row = f"| {C.LEVEL_LABELS[lv]} | {C.METHOD_LABELS[m]} | {E.success_count(c['solved'])} von {n} | {C.fmt_med(E.median_solved(c['solved']))} | {C.fmt_med(float(np.median(ln))) if ln else '–'} | {np.mean(c['cells']):.1f} | {np.mean(c['progress']):.2f} |"
            assert row in readme, row
    for lv in C.SWEEP_LEVELS:
        for cfg in C.SWEEP_CONFIGS:
            c = E.sweep_cell(PRE, lv, cfg)
            ln = [x for x in c["length"] if x is not None]
            row = f"| {C.LEVEL_LABELS[lv]} | {C.SWEEP_LABELS[cfg]} | {E.success_count(c['solved'])} von {n} | {C.fmt_med(E.median_solved(c['solved']))} | {C.fmt_med(float(np.median(ln))) if ln else '–'} | {np.mean(c['cells']):.1f} |"
            assert row in readme, row
        for z in C.NOISE_OPTIONS:
            c = E.noise_cell(PRE, lv, z)
            row = f"| {C.LEVEL_LABELS[lv]} | {z:g} | {E.success_count(c['solved'])} von {n} | {C.fmt_med(E.median_solved(c['solved']))} | {replay_mean(lv, z):.2f} % |"
            assert row in readme, row


def test_the_replay_costs_on_stage_three_are_more_than_four_times_the_free_return():
    """README: Stufe 3: Wiederabspielen gut das Vierfache (58 499 gegen 14 000)."""
    assert 58499 / 14000 > 4.1 and 58499 / 14000 < 4.2
