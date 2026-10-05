"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).
Alle Regler sind immer sichtbar und wirken immer: Go-Explore läuft im Live-Lauf mit allen gewählten Einstellungen neben dem Vergleichsverfahren (GA oder Novelty) mit demselben Schrittbudget; es gibt keinen ausblendbaren Regler und kein KEPT-Muster."""

import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import ge_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


def _choice(options):
    """Permalink-Wert aus einer festen Liste von Namen; alles andere ist ein ValueError (der Wert wird dann ignoriert)."""
    def cast(value):
        if not isinstance(value, str) or value not in options:
            raise ValueError(f"unbekannter Wert: {value!r}")
        return value
    return cast


SETTING_SPECS = {
    "level_select": SettingSpec("level", int, C.DEFAULT_LEVEL, C.LEVELS[0], C.LEVELS[-1]),
    "budget_slider": SettingSpec("budget", int, C.DEFAULT_BUDGET, C.BUDGET_MIN, C.BUDGET_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "compare_select": SettingSpec("compare", _choice(C.GENOME_METHODS), C.DEFAULT_COMPARE),
    "return_select": SettingSpec("ret", _choice(C.RETURNS), C.DEFAULT_RETURN),
    "cell_select": SettingSpec("cell", _choice(C.CELL_KINDS), C.DEFAULT_CELL),
    "explore_select": SettingSpec("explore", int, C.DEFAULT_EXPLORE, C.EXPLORE_OPTIONS[0], C.EXPLORE_OPTIONS[-1]),
    "cap_select": SettingSpec("cap", int, C.DEFAULT_CAP, 0, C.CAP_OPTIONS[0]),
    "select_select": SettingSpec("select", _choice(C.SELECTS), C.DEFAULT_SELECT),
    "noise_select": SettingSpec("noise", float, C.DEFAULT_NOISE, C.NOISE_OPTIONS[0], C.NOISE_OPTIONS[-1]),
}
PRESET_KEYS = {"level": "level_select", "budget": "budget_slider", "seed": "seed_input", "compare": "compare_select", "ret": "return_select", "cell": "cell_select", "explore": "explore_select", "cap": "cap_select", "select": "select_select", "noise": "noise_select"}
OPTIONS = {"explore_select": C.EXPLORE_OPTIONS, "cap_select": (0, 140), "noise_select": C.NOISE_OPTIONS}       # Regler, die nur feste Werte kennen


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def snap_to_option(state_key, value):
    """Regler mit festen Werten: ein Permalink-Wert dazwischen rastet auf den nächsten Wert ein (bei Gleichstand auf den kleineren)."""
    return min(OPTIONS[state_key], key=lambda o: (abs(o - value), o))


def snap_budget(value):
    """Das Budget rastet auf das nächste Vielfache der Schrittweite (100 000, von 100 000 an) innerhalb der Grenzen ein."""
    snapped = C.BUDGET_MIN + round((value - C.BUDGET_MIN) / C.BUDGET_STEP) * C.BUDGET_STEP
    return int(min(C.BUDGET_MAX, max(C.BUDGET_MIN, snapped)))


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if value != value:                     # NaN
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                if state_key in OPTIONS:
                    value = snap_to_option(state_key, value)
                if state_key == "budget_slider":
                    value = snap_budget(value)
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
