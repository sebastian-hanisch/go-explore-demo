"""Presets: Vollständigkeit, gültige Werte, Permalink-Konstanten, Formatierer."""

import pytest

import ge_constants as C
import ge_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_ORDER) and len(C.PRESETS) == 4
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS) and C.PRESET_HELP[name]


def test_preset_values_are_valid_and_match_the_setting_specs():
    for preset in C.PRESETS.values():
        assert preset["level"] in C.LEVELS and C.BUDGET_MIN <= preset["budget"] <= C.BUDGET_MAX and P.snap_budget(preset["budget"]) == preset["budget"]
        assert preset["compare"] in C.GENOME_METHODS and preset["ret"] in C.RETURNS and preset["cell"] in C.CELL_KINDS and preset["explore"] in C.EXPLORE_OPTIONS and preset["cap"] in C.CAP_OPTIONS
        assert preset["select"] in C.SELECTS and preset["noise"] in C.NOISE_OPTIONS
        for key, state_key in P.PRESET_KEYS.items():
            P.SETTING_SPECS[state_key].caster(preset[key])


def test_preset_names_state_the_values_they_set():
    assert C.PRESETS["Die Falle"]["level"] == 4 and C.PRESETS["Rückkehr kostet"]["ret"] == "replay" and C.PRESETS["Zu feine Zellen"]["cell"] == "posdir" and C.PRESETS["Entgleisung"]["level"] == 3 and C.PRESETS["Entgleisung"]["noise"] > 0


def test_the_default_preset_equals_the_default_settings():
    p = C.PRESETS["Die Falle"]
    assert (p["level"], p["budget"], p["seed"], p["compare"], p["ret"], p["cell"], p["explore"], p["cap"], p["select"], p["noise"]) == (C.DEFAULT_LEVEL, C.DEFAULT_BUDGET, C.DEFAULT_SEED, C.DEFAULT_COMPARE, C.DEFAULT_RETURN,
                                                                                                                                     C.DEFAULT_CELL, C.DEFAULT_EXPLORE, C.DEFAULT_CAP, C.DEFAULT_SELECT, C.DEFAULT_NOISE)


def test_bounds_and_url_params():
    assert P.bounds("seed_input") == (0, C.SEED_MAX) and P.bounds("budget_slider") == (100000, 2000000) and P.bounds("level_select") == (0, 4)
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS) == 10


@pytest.mark.parametrize("value,expected", [(0, 100000), (140000, 100000), (160000, 200000), (1040000, 1000000), (1060000, 1100000), (9999999, 2000000)])
def test_the_budget_snaps_to_the_step_from_the_lower_bound(value, expected):
    assert P.snap_budget(value) == expected


@pytest.mark.parametrize("key,value,expected", [("explore_select", 2, 3), ("explore_select", 6, 3), ("explore_select", 7, 10), ("explore_select", 25, 10), ("explore_select", 26, 40), ("cap_select", 50, 0), ("cap_select", 71, 140), ("cap_select", 70, 0),
                                                ("noise_select", 0.03, 0.02), ("noise_select", 0.04, 0.05), ("noise_select", 0.0751, 0.1), ("noise_select", 0.5, 0.1)])
def test_values_snap_to_the_nearest_option_and_ties_go_down(key, value, expected):
    assert P.snap_to_option(key, value) == expected


@pytest.mark.parametrize("key,options", [("compare_select", C.GENOME_METHODS), ("return_select", C.RETURNS), ("cell_select", C.CELL_KINDS), ("select_select", C.SELECTS)])
def test_choice_casters_accept_their_options_and_refuse_everything_else(key, options):
    for o in options:
        assert P.SETTING_SPECS[key].caster(o) == o
    for bad in ("ge_free", "", 3, None, ["x"]):
        with pytest.raises(ValueError):
            P.SETTING_SPECS[key].caster(bad)


def test_the_comparison_cannot_be_go_explore_itself():
    with pytest.raises(ValueError):
        P.SETTING_SPECS["compare_select"].caster("ge_replay")


def test_method_and_setting_tables_are_consistent():
    assert set(C.METHOD_LABELS) == set(C.METHODS) and C.GENOME_METHODS == C.METHODS[:2] and set(C.BASELINE_LABELS) == set(C.GENOME_METHODS) and set(C.CELL_LABELS) == set(C.CELL_SIZE) == set(C.CELL_KINDS)
    assert set(C.RETURN_LABELS) == set(C.RETURNS) and set(C.SELECT_LABELS) == set(C.SELECTS) and set(C.CAP_LABELS) == set(C.CAP_OPTIONS) and set(C.SWEEP_CONFIGS) == set(C.SWEEP_LABELS) and C.DEFAULT_EXPLORE in C.EXPLORE_OPTIONS
    assert C.STUDY_BUDGET == 100 * 400 * 140 == 5600000 and len(C.CURVE_MARKS) == 56 and C.CURVE_MARKS[-1] == C.STUDY_BUDGET and C.DEFAULT_CAP == C.STEPS and C.DEFAULT_NOISE in C.NOISE_OPTIONS


def test_the_sweep_configurations_change_one_setting_each():
    allowed = {"explore", "cap", "ret", "cell", "select"}
    assert C.SWEEP_CONFIGS["default"] == {} and all(len(kw) == 1 and set(kw) <= allowed for cfg, kw in C.SWEEP_CONFIGS.items() if cfg != "default")


def test_formatters():
    assert C.fmt_int(40000) == "40 000" and C.fmt_med(None) == "–" and C.fmt_med(988.5) == "988.5" and C.fmt_med(1195.0) == "1 195" and C.fmt_med(1677411.5) == "1 677 411.5"
