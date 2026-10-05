"""AppTest-Rauchtests: Voreinstellung, jedes Preset, alle Stufen, Einstellungen, Randwerte, Studien-Regler, Würfel-Knopf, Permalink-Grenzen, Abschnitte, Footer."""

import random
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import ge_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")
CHARTS = 10                   # Archiv, berührte Zellen, zwei Kurven, Wiederholungen; Erfolgsbalken, Schrittebalken, Erfolgsverlauf, Einstellungen, Rauschen


def _run(**state):
    at = AppTest.from_file(APP, default_timeout=600)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def _n_charts(solved=True):
    return CHARTS if solved else CHARTS - 1


def test_default_run_has_no_exception_and_shows_the_reference_values():
    at = _run()
    _ok(at)
    assert _metric(at, "Kürzester Weg") == "77 Zellen" and _metric(at, "Fallen (lokale Minima)") == "2" and _metric(at, "Freie Zellen") == "427"
    assert _metric(at, "Go-Explore: Länge der Fahrt").endswith("Schritte") and _metric(at, "Vergleich: Schritte bis zum Erfolg") == "–" and len(at.get("plotly_chart")) == _n_charts()


def test_all_sections_are_present():
    at = _run()
    assert [s.value for s in at.subheader] == ["📐 Ein Lauf: Go-Explore gegen ein Genom-Verfahren im selben Schrittbudget", "🌫️ Entgleisung: dieselbe Fahrt in einer Welt mit Rauschen", "🔬 Löst das die Falle? Die Studie über 20 Läufe",
                                              "🔬 Was kostet welche Einstellung? Stufe 3 und 4", "🔬 Wie robust ist die gefundene Fahrt? Rauschen in der ganzen Studie", "🚧 Wo die Annahmen enden"]
    assert any(m.value.startswith("## 🔗 Wie täuschend ist diese Stufe?") for m in at.markdown)
    assert any("Diese Demo ist Teil des Portfolios" in c.value for c in at.caption)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = C.PRESETS[name]
    assert all(at.session_state[k] == p[key] for key, k in (("level", "level_select"), ("budget", "budget_slider"), ("compare", "compare_select"), ("ret", "return_select"), ("cell", "cell_select"),
                                                              ("explore", "explore_select"), ("cap", "cap_select"), ("select", "select_select"), ("noise", "noise_select")))
    assert at.metric


@pytest.mark.parametrize("level", list(C.LEVELS))
def test_every_level_runs(level):
    at = _run(level_select=level, budget_slider=300000)
    _ok(at)
    assert _metric(at, "Go-Explore: Zellen im Archiv")


@pytest.mark.parametrize("compare", C.GENOME_METHODS)
def test_both_comparison_methods_run(compare):
    at = _run(compare_select=compare, budget_slider=300000)
    _ok(at)
    assert _metric(at, "Vergleich: Evaluationen") == "2 100" and any(compare == "fitness" and "GA mit Fitness" in m.value or compare == "novelty" and "Novelty Search" in m.value for m in at.markdown)


@pytest.mark.parametrize("kw", [dict(return_select="replay"), dict(cell_select="pos2"), dict(cell_select="posdir"), dict(explore_select=3), dict(explore_select=40), dict(cap_select=0), dict(select_select="uniform"),
                                dict(noise_select=0.0), dict(noise_select=0.1)])
def test_every_go_explore_setting_runs(kw):
    _ok(_run(level_select=3, budget_slider=300000, **kw))


@pytest.mark.parametrize("kw", [dict(budget_slider=100000), dict(budget_slider=2000000, level_select=0), dict(level_select=4, budget_slider=100000, seed_input=0), dict(seed_input=999999, budget_slider=100000, return_select="replay", cell_select="posdir")])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def test_an_unsolved_run_says_so_and_has_no_replay_experiment():
    at = _run(level_select=4, budget_slider=100000, cell_select="posdir")
    _ok(at)
    assert _metric(at, "Go-Explore: Schritte bis zum Erfolg") == "–" and any("keine Fahrt gefunden" in c.value for c in at.caption) and len(at.get("plotly_chart")) == _n_charts(False)
    assert any("erreicht das Ziel in diesem Budget nicht" in i.value for i in at.info)


def test_the_replay_experiment_shows_the_share_in_the_goal_circle():
    at = _run(level_select=3, budget_slider=500000, noise_select=0.0)
    assert _metric(at, "Wiederholungen im Zielkreis") == "100.0 %" and _metric(at, "Rauschen je Schritt") == "0 Zellen"
    at2 = _run(level_select=3, budget_slider=500000, noise_select=0.1)
    assert float(_metric(at2, "Wiederholungen im Zielkreis").split()[0]) < 60.0


def test_the_live_info_names_both_methods():
    at = _run(level_select=3, budget_slider=500000)
    text = next(i.value for i in at.info if i.value.startswith("Go-Explore "))
    assert "erreicht das Ziel nach" in text and "Novelty Search" in text and "Das ist ein Lauf" in text


def test_the_solvability_line_names_the_steps_used():
    assert any("136 von 140 Schritten benutzt" in c.value for c in _run(level_select=4, budget_slider=100000).caption)


def test_the_study_table_has_one_row_per_method_and_the_selector_switches_it():
    at = _run(study_level_select=3)
    _ok(at)
    table = next(m.value for m in at.markdown if m.value.startswith(f"**{C.LEVEL_LABELS[3]}**\n\n| Verfahren"))
    assert table.count("\n| ") == len(C.METHODS) + 1 and all(C.METHOD_LABELS[m] in table for m in C.METHODS)


def test_the_sweep_table_has_one_row_per_setting_and_the_selector_switches_it():
    at = _run(sweep_level_select=3)
    _ok(at)
    table = next(m.value for m in at.markdown if m.value.startswith(f"**{C.LEVEL_LABELS[3]}**\n\n| Einstellung"))
    assert table.count("\n| ") == len(C.SWEEP_CONFIGS) + 1 and all(C.SWEEP_LABELS[c] in table for c in C.SWEEP_CONFIGS)


def test_the_study_infos_name_the_main_findings():
    at = _run()
    info = next(i.value for i in at.info if i.value.startswith("**Stufe 4:** GA mit Fitness"))
    assert "0 und 0 von 20" in info and "216 500" in info and "1 677 411.5" in info and "genau an der Grenze von 140" in info and "177 380" in info and "138 390" in info
    sweep = next(i.value for i in at.info if i.value.startswith("**Stufe 4:** die Voreinstellung"))
    assert "77 400" in sweep and "848 000" in sweep and "1 003 000" in sweep and "ohne Längenbegrenzung nur 28 000" in sweep and "kein belegter Unterschied" in sweep
    noise = next(i.value for i in at.info if i.value.startswith("**Das Zurücksetzen verdeckt das Rauschen"))
    import numpy as np
    import ge_evaluation as E
    pre = E.load_precomputed()
    share = lambda lv: " / ".join(f"{100 * np.mean([r for r in E.noise_cell(pre, lv, z)['replay'] if r is not None]):.1f}" for z in C.NOISE_OPTIONS) + " %"
    assert share(3) in noise and share(4) in noise and share(3).startswith("100.0 / 55.3 / 51.8 / 14.") and share(4) == "100.0 / 9.4 / 1.0 / 0.0 %"


def test_dice_button_changes_the_seed_and_the_run(monkeypatch):
    """Der Würfel zieht sonst einen unseeded Zufalls-Seed; deshalb ist der gewürfelte Seed im Test fest."""
    monkeypatch.setattr(random, "randint", lambda a, b: 508145)
    at = _run(level_select=3, budget_slider=500000)
    old_seed = at.session_state["seed_input"]
    old = (_metric(at, "Go-Explore: Schritte bis zum Erfolg"), _metric(at, "Vergleich: berührte Zellen"))
    next(b for b in at.button if b.label == "🎲 Neuen Lauf würfeln").click().run()
    _ok(at)
    assert at.session_state["seed_input"] == 508145 != old_seed
    assert (_metric(at, "Go-Explore: Schritte bis zum Erfolg"), _metric(at, "Vergleich: berührte Zellen")) != old


def test_permalink_values_are_clamped_and_snapped():
    at = AppTest.from_file(APP, default_timeout=600)
    at.query_params["level"] = "9"
    at.query_params["budget"] = "340000"
    at.query_params["explore"] = "25"
    at.query_params["cap"] = "50"
    at.query_params["noise"] = "0.04"
    at.query_params["cell"] = "posdir"
    at.query_params["ret"] = "replay"
    at.query_params["compare"] = "fitness"
    at.query_params["select"] = "uniform"
    at.run()
    _ok(at)
    s = at.session_state
    assert (s["level_select"], s["budget_slider"], s["explore_select"], s["cap_select"], s["noise_select"]) == (4, 300000, 10, 0, 0.05)
    assert (s["cell_select"], s["return_select"], s["compare_select"], s["select_select"]) == ("posdir", "replay", "fitness", "uniform")


def test_permalink_ignores_garbage():
    at = AppTest.from_file(APP, default_timeout=600)
    at.query_params["level"] = "viele"
    at.query_params["seed"] = "nan"
    at.query_params["compare"] = "ge_free"
    at.query_params["cell"] = "hex"
    at.query_params["ret"] = "teleport"
    at.run()
    _ok(at)
    s = at.session_state
    assert (s["level_select"], s["seed_input"], s["compare_select"], s["cell_select"], s["return_select"]) == (C.DEFAULT_LEVEL, C.DEFAULT_SEED, C.DEFAULT_COMPARE, C.DEFAULT_CELL, C.DEFAULT_RETURN)


def test_no_sentence_wide_comma_replacement_in_the_app_source():
    """Regressionsschutz: `.replace(",", ".")` auf einem ganzen (verketteten) Satz macht aus Kommas im Fließtext Punkte; Tausender nur über `fmt_int`."""
    assert '.replace(",", ".")' not in Path(APP).read_text(encoding="utf-8")
