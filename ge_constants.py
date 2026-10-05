"""Konstanten der Go-Explore-Demo: Labyrinth und Roboter wie in Stück 1 bis 8, Verfahren, Einstellungen von Go-Explore, Studien-Achsen, Regler, Voreinstellungen.
Längen in Zellen (Zellraster N × N, Zelle (i, j) belegt den Bereich [i, i + 1) × [j, j + 1)); Budget in **simulierten Schritten** (eine Evaluation einer Steuerfolge kostet 140 Schritte)."""


def fmt_int(n):
    """Ganzzahl mit Leerzeichen als Tausendertrenner (10000 -> 10 000), wie in der README."""
    return f"{n:,}".replace(",", " ")


def fmt_med(x):
    """Median: ganze Zahl mit Leerzeichen als Tausendertrenner, ein halber Wert behält seine Stelle (988.5, 1 308.5); None = kein Lauf war erfolgreich."""
    if x is None:
        return "–"
    return fmt_int(int(x)) if x == int(x) else f"{x:,.1f}".replace(",", " ")


N = 24                                       # Kantenlänge des Labyrinths in Zellen (Rand ist Wand)
START = (3.5, 3.5)                           # Mitte der Startzelle (3, 3)
GOAL = (3.5, 20.5)                           # Mitte der Zielzelle (3, 20)
SUCCESS_RADIUS = 2.0                         # Erfolg: Endposition höchstens so weit vom Ziel
START_HEADING = 0.7853981633974483           # Anfangsrichtung π/4
STEPS = 140                                  # Länge der Steuerfolge (Genom) und Längenbegrenzung der Go-Explore-Lösung
TURN_MAX = 0.6                               # Drehung je Schritt (rad), Gen ∈ [−1, 1]
SPEED_MAX = 0.7                              # Fahrstrecke je Schritt (Zellen), Gen ∈ [0, 1]

POP = 100                                    # Kinder je Generation (GA und Novelty)
MUT_SIGMA = 0.25                             # Standardabweichung der Mutation je verändertem Gen
MUT_RATE = 0.15                              # Anteil der Gene, die je Kind mutieren
ELITE = 2                                    # die besten Individuen, die unverändert bleiben
TOURNAMENT = 3                               # Turniergröße der Auswahl
NOVELTY_K = 15                               # Nachbarn der Novelty (wie in Stück 2)

LEVELS = (0, 1, 2, 3, 4)
LEVEL_LABELS = {0: "0: freies Feld", 1: "1: Öffnung links", 2: "2: Öffnung in der Mitte", 3: "3: Öffnung rechts", 4: "4: Schlangenlinie"}
GAP = 3                                      # Breite einer Öffnung in der Wand (Zellen)

# Verfahren der Studie: GA mit Fitness (Stück 1), Novelty ohne Archiv (Stück 2), Go-Explore mit kostenloser Rückkehr (Zurücksetzen des Simulators) und mit Rückkehr durch Wiederabspielen der Aktionen
METHODS = ("fitness", "novelty", "ge_free", "ge_replay")
METHOD_LABELS = {"fitness": "GA mit Fitness (Stück 1)", "novelty": "Novelty Search (Stück 2)", "ge_free": "Go-Explore, Rückkehr kostenlos", "ge_replay": "Go-Explore, Rückkehr durch Wiederabspielen"}
GENOME_METHODS = ("fitness", "novelty")
BASELINE_LABELS = {"fitness": "GA mit Fitness", "novelty": "Novelty Search"}

# Einstellungen von Go-Explore
N_EXPLORERS = 100                            # Erkunder je Runde (parallel)
EXPLORE_OPTIONS = (3, 10, 40)                # zufällige Schritte je Erkundung
DEFAULT_EXPLORE = 10
CAP_OPTIONS = (140, 0)                       # Längenbegrenzung der Lösung (Schritte); 0 = keine
CAP_LABELS = {140: "140 Schritte (wie die Genome)", 0: "keine"}
DEFAULT_CAP = 140
RETURNS = ("free", "replay")
RETURN_LABELS = {"free": "kostenlos (Simulator zurücksetzen)", "replay": "durch Wiederabspielen der Aktionen"}
DEFAULT_RETURN = "free"
CELL_KINDS = ("pos1", "pos2", "posdir")
CELL_LABELS = {"pos1": "Position (1 × 1 Zellen)", "pos2": "Position (2 × 2 Zellen)", "posdir": "Position und Richtung (1 × 1, 8 Sektoren)"}
CELL_SIZE = {"pos1": 1, "pos2": 2, "posdir": 1}
SECTORS = 8                                  # Richtungssektoren bei `posdir`
DEFAULT_CELL = "pos1"
SELECTS = ("count", "uniform")
SELECT_LABELS = {"count": "selten gewählte Zellen bevorzugt (Gewicht 1 / (1 + Wahlen))", "uniform": "gleichverteilt"}
DEFAULT_SELECT = "count"

# Studie: Budget in simulierten Schritten wie bei GA und Novelty (40 000 Evaluationen × 140 Schritte)
STUDY_SEEDS = 20
STUDY_GENS = 400                             # Generationen von GA und Novelty
STUDY_BUDGET = POP * STUDY_GENS * STEPS      # 5 600 000 Schritte
CURVE_STEP = 100000                          # Schrittweite der Verlaufskurven (Mittel über die Läufe, je 100 000 Schritte)
CURVE_MARKS = tuple(range(CURVE_STEP, STUDY_BUDGET + 1, CURVE_STEP))
SWEEP_LEVELS = (3, 4)
SWEEP_CONFIGS = {                            # Einstellungen von Go-Explore (Rückkehr kostenlos) je Zeile: eine Änderung gegenüber der Voreinstellung
    "default": {},
    "explore3": {"explore": 3}, "explore40": {"explore": 40}, "pos2": {"cell": "pos2"}, "posdir": {"cell": "posdir"}, "uniform": {"select": "uniform"}, "nocap": {"cap": 0}, "replay": {"ret": "replay"},
}
SWEEP_LABELS = {"default": "Voreinstellung (10 Schritte, Position 1 × 1, Zähler, Deckel 140)", "explore3": "Erkunden mit 3 Schritten", "explore40": "Erkunden mit 40 Schritten", "pos2": "Zelle 2 × 2", "posdir": "Zelle mit Richtung",
                "uniform": "gleichverteilte Auswahl", "nocap": "ohne Längenbegrenzung", "replay": "Rückkehr durch Wiederabspielen"}
NOISE_OPTIONS = (0.0, 0.02, 0.05, 0.1)       # Gauß-Rauschen auf der Verschiebung je Schritt (Zellen): Entgleisung beim Wiederabspielen
REPLAYS = 200                                # Wiederholungen beim Wiederabspielen einer gefundenen Fahrt
REPLAY_SEED = 999

BUDGET_MIN, BUDGET_MAX, BUDGET_STEP, DEFAULT_BUDGET = 100000, 2000000, 100000, 1000000      # Regler: Budget des Live-Laufs in Schritten
DEFAULT_LEVEL = 4
SEED_MAX = 999999
DEFAULT_SEED = 35
DEFAULT_COMPARE = "novelty"
DEFAULT_NOISE = 0.05

PRESET_ORDER = ("Die Falle", "Rückkehr kostet", "Zu feine Zellen", "Entgleisung")


def _preset(level=DEFAULT_LEVEL, budget=DEFAULT_BUDGET, compare=DEFAULT_COMPARE, ret=DEFAULT_RETURN, cell=DEFAULT_CELL, explore=DEFAULT_EXPLORE, cap=DEFAULT_CAP, select=DEFAULT_SELECT, noise=DEFAULT_NOISE):
    return {"level": level, "budget": budget, "seed": DEFAULT_SEED, "compare": compare, "ret": ret, "cell": cell, "explore": explore, "cap": cap, "select": select, "noise": noise}


PRESETS = {
    "Die Falle": _preset(),
    "Rückkehr kostet": _preset(ret="replay", budget=2000000),
    "Zu feine Zellen": _preset(cell="posdir", budget=2000000),
    "Entgleisung": _preset(level=3, budget=500000, noise=0.05),
}
PRESET_HELP = {
    "Die Falle": "Stufe 4: GA mit Fitness und Novelty lösen sie in der Studie in 0 von 20 Läufen (5.6 Mio Schritte), Go-Explore in 20 von 20 mit einem Median von 216 500 Schritten.",
    "Rückkehr kostet": "Rückkehr durch Wiederabspielen statt Zurücksetzen: Stufe 4 braucht im Median 1 677 411.5 statt 216 500 Schritte (Studie).",
    "Zu feine Zellen": "Zelle mit Richtung statt Position: Stufe 4 braucht im Median 1 003 000 statt 216 500 Schritte, das Archiv hat 3 167.1 statt 406.6 Zellen (Studie).",
    "Entgleisung": "Stufe 3: die gefundene Fahrt ohne Zurücksetzen in einer Welt mit Rauschen 0.05 wiederholt, kommt im Mittel nur in 51.78 % der Wiederholungen ins Ziel (Studie, 20 Läufe).",
}                                                      # jede Zahl in tests/test_claims.py
