"""Go-Explore – erst zurückkehren, dann erkunden - Stück 9 der Konzepte-Linie "Novelty Search und Quality-Diversity"
Sebastian Hanisch - Operations Research und Machine Learning

Derselbe Hof-Roboter und dasselbe Labyrinth wie in Stück 1 bis 8 - aber die Suche arbeitet nicht mehr mit ganzen 140-Schritte-Steuerfolgen, die gemutiert werden, sondern baut aus Zwischenständen weiter: ein Archiv von Zellen, Rückkehr zu einer Zelle,
zufälliges Erkunden von dort. Gemessen wird in simulierten Schritten, ob das die Falle löst, an der alle früheren Verfahren scheiterten, und was es dafür braucht. Siehe README für die Einordnung in die Linie.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import ge_constants as C
import ge_evaluation as E
import ge_maze as M
import ge_robot as R
import ge_search as S
from ge_presets import (apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sync_query_params)
from ge_visualization import (build_archive_map, build_cells_curve, build_noise_chart, build_progress_curve, build_replay_figure, build_success_bars, build_success_curve, build_steps_chart, build_sweep_chart, build_touch_map)

st.set_page_config(page_title="Go-Explore – Sebastian Hanisch", layout="wide")

PRE = E.load_precomputed()
N = C.STUDY_SEEDS


@st.cache_data(show_spinner=False)
def _ge(level, seed, budget, explore, cap, ret, cell, select):
    return E.ge_run(level, seed, budget, explore, cap, ret, cell, select)


@st.cache_data(show_spinner=False)
def _baseline(level, seed, budget, method):
    return E.baseline_run(level, seed, budget, method)


@st.cache_data(show_spinner=False)
def _solvability(level):
    return E.solvability(level)


def _mc(level, method):
    return E.method_cell(PRE, level, method)


def _med(level, method):
    return E.median_solved(_mc(level, method)["solved"])


def _count(level, method):
    return E.success_count(_mc(level, method)["solved"])


def _sweep_med(level, config):
    return E.median_solved(E.sweep_cell(PRE, level, config)["solved"])


def _pstr(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


st.title("🧗 Go-Explore – erst zurückkehren, dann erkunden")
st.markdown(
    """
Bisher wurde in dieser Linie **eine ganze Steuerfolge** von 140 Schritten bewertet und gemutiert, ob nach Fitness, Neuheit oder Zellen. **Go-Explore** (Ecoffet, Huizinga, Lehman, Stanley und Clune) macht es anders: es merkt sich **Zwischenstände**, **kehrt zu einem von ihnen zurück**
und **erkundet von dort aus** mit ein paar zufälligen Schritten. So wächst eine Fahrt Stück für Stück, statt dass jedes Mal ein ganzer Weg neu erraten werden muss. Dieses Stück misst, **ob das die Falle löst**, an der Mutation, Fitness und Neuheit gescheitert sind (Stufe 4), und **was es dafür voraussetzt**:
einen Simulator, den man auf einen gespeicherten Zustand zurücksetzen kann, eine Zelleneinteilung, die den Ort kennt, und eine Welt ohne Zufall.
"""
)
st.caption(
    "Stück 9 der Linie „Novelty Search und Quality-Diversity“; Labyrinth, Roboter und Erfolg (Endposition im Zielkreis) wie in [Stück 1](https://sebastianhanisch-deception-maze-demo.streamlit.app/) bis [Stück 8](https://sebastianhanisch-aurora-demo.streamlit.app/). "
    "Die Umsetzung (Archiv, Rückkehr, Auswahl) ist eigen und enthält nur die erste Phase des Verfahrens (Erkunden); die zweite Phase der Quelle (Robustifizieren durch Nachahmung) fehlt."
)

with st.expander("So arbeitet Go-Explore", expanded=True):
    st.markdown(
        """
- **Archiv:** je **Zelle** der Zustand (Position und Richtung), mit dem sie am schnellsten erreicht wurde (die wenigsten Schritte seit dem Start), und der Weg dorthin als Aktionsfolge. Die Zelle ist ein Stück Position (1 × 1 oder 2 × 2 Zellen des Labyrinths), wahlweise mit einem von 8 Richtungssektoren.
- **Eine Runde:** 100 Erkunder; jeder wählt eine Zelle (selten gewählte bevorzugt: Gewicht 1 / (1 + bisherige Wahlen), oder gleichverteilt), **kehrt zurück** und fährt 3, 10 oder 40 **zufällige** Schritte (Drehung und Tempo zufällig). **Jeder** dabei erreichte Zustand kommt in seine Zelle, wenn die Zelle neu ist oder
  mit weniger Schritten erreicht wurde.
- **Rückkehr:** *kostenlos* heißt: der Simulator wird auf den gespeicherten Zustand gesetzt (das setzt einen deterministischen, rücksetzbaren Simulator voraus). *Durch Wiederabspielen* heißt: die Aktionen des Wegs werden vom Start an noch einmal gefahren, und diese Schritte zählen zum Budget.
- **Längenbegrenzung:** wie die Genome der früheren Stücke darf die gefundene Fahrt höchstens 140 Schritte lang sein (oder unbegrenzt). Die kürzeste Lösung der Stufe 4 braucht 136 Schritte.
- **Budget:** in **simulierten Schritten**, damit der Vergleich mit GA und Novelty stimmt: eine Evaluation einer Steuerfolge kostet 140 Schritte, die Studie gibt allen 5.6 Mio Schritte (40 000 Evaluationen). Erfolg heißt bei allen: ein Zustand im Zielkreis (Radius 2).
- **Vergleichsverfahren:** **GA mit Fitness** (Stück 1) und **Novelty Search** (Stück 2, ohne Archiv, k = 15), mit denselben Operatoren wie dort.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_cols = st.columns(len(C.PRESET_ORDER))
for i, name in enumerate(C.PRESET_ORDER):
    with preset_cols[i]:
        st.button(name, key=f"preset_{name}", width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name] or None)

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    level = st.selectbox("Labyrinth (Stufe)", options=C.LEVELS, key="level_select", format_func=lambda v: C.LEVEL_LABELS[v], help="0 freies Feld bis 4 Schlangenlinie; je höher, desto länger der Umweg.")
    budget = st.slider("Budget (simulierte Schritte)", *bounds("budget_slider"), step=C.BUDGET_STEP, key="budget_slider", help="Gilt für beide Verfahren. Eine Evaluation einer Steuerfolge kostet 140 Schritte, 1 000 000 Schritte sind also etwa 7 143 Evaluationen.")
    compare = st.selectbox("Vergleichsverfahren", options=C.GENOME_METHODS, key="compare_select", format_func=lambda v: C.BASELINE_LABELS[v], help="Läuft neben Go-Explore mit gleichem Seed und gleichem Schrittbudget.")
    st.markdown("**Go-Explore**")
    ret = st.selectbox("Rückkehr", options=C.RETURNS, key="return_select", format_func=lambda v: C.RETURN_LABELS[v], help="Kostenlos: der Simulator wird zurückgesetzt. Wiederabspielen: die Schritte des Wegs zählen zum Budget.")
    cell = st.selectbox("Zelle", options=C.CELL_KINDS, key="cell_select", format_func=lambda v: C.CELL_LABELS[v], help="Wie fein das Archiv einteilt: gröbere Zellen haben weniger Einträge, feinere (mit Richtung) viel mehr.")
    explore = st.selectbox("Erkunden mit … Schritten", options=C.EXPLORE_OPTIONS, key="explore_select", help="Zufällige Schritte je Erkundung nach der Rückkehr.")
    cap = st.selectbox("Längenbegrenzung der Fahrt", options=C.CAP_OPTIONS, key="cap_select", format_func=lambda v: C.CAP_LABELS[v], help="Höchstzahl der Schritte seit dem Start, bis zu der Zustände ins Archiv kommen.")
    select = st.selectbox("Auswahl der Zelle", options=C.SELECTS, key="select_select", format_func=lambda v: C.SELECT_LABELS[v])
    noise = st.select_slider("Rauschen im Entgleisungs-Experiment", options=C.NOISE_OPTIONS, key="noise_select", format_func=lambda v: f"{v:g}", help="Gauß-Rauschen auf der Verschiebung je Schritt (Zellen), wenn die gefundene Fahrt ohne Zurücksetzen wiederholt wird (Abschnitt unter dem Lauf).")
    seed = st.number_input("Zufalls-Seed", min_value=bounds("seed_input")[0], max_value=bounds("seed_input")[1], step=1, key="seed_input", help="Bestimmt alle Zufallsentscheidungen; beide Verfahren starten gleich.")
    st.button("🎲 Neuen Lauf würfeln", on_click=randomize_seed)

level, budget, compare, ret, cell, explore, cap, select, noise, seed = int(level), int(budget), str(compare), str(ret), str(cell), int(explore), int(cap), str(select), float(noise), int(seed)
sync_query_params({"level_select": level, "budget_slider": budget, "compare_select": compare, "return_select": ret, "cell_select": cell, "explore_select": explore, "cap_select": cap, "select_select": select, "noise_select": noise, "seed_input": seed})

grid = M.make_maze(level)
lm = E.level_metrics(level)

st.markdown("---")
st.markdown("## 🔗 Wie täuschend ist diese Stufe?")
st.caption(f"{C.LEVEL_LABELS[level]}. Die Maße stammen aus Stück 1. Weder Go-Explore noch die Vergleichsverfahren sehen das Labyrinth: sie spüren nur, ob ein Schritt an einer Wand endet.")
m1 = st.columns(4)
m1[0].metric("Kürzester Weg", f"{lm['path']} Zellen", delta=f"{lm['detour']:.2f}-facher Umweg", delta_color="off")
m1[1].metric("Fitness-Distanz-Korrelation", f"{lm['fdc']:.2f}", help="1 = die Luftlinie führt zum Ziel; je kleiner, desto täuschender.")
m1[2].metric("Fallen (lokale Minima)", f"{lm['minima']}", help="Zellen, in denen die Luftlinie zum Ziel kleiner ist als in allen freien Nachbarzellen.")
m1[3].metric("Freie Zellen", f"{lm['free']}", help="So viele Zellen gibt es im Archiv bei Zellen von 1 × 1 höchstens.")
dist_ok, steps_used = _solvability(level)
st.caption(f"Lösbar? Ja: eine aus dem kürzesten Pfad abgeleitete Steuerfolge fährt ins Ziel (Abstand am Ende {dist_ok:.2f} Zellen, {steps_used} von {C.STEPS} Schritten benutzt).")

st.markdown("---")
st.subheader("📐 Ein Lauf: Go-Explore gegen ein Genom-Verfahren im selben Schrittbudget")
with st.spinner(f"Lasse beide Verfahren je {C.fmt_int(budget)} Schritte laufen …"):
    ge = _ge(level, seed, budget, explore, cap, ret, cell, select)
    bl = _baseline(level, seed, budget, compare)
path = E.solution_path(level, ge.actions)
bl_path = R.simulate(bl.best_genome[None], grid, trajectories=True)[1][0] if bl.best_genome is not None else None
st.caption(f"Beide mit höchstens {C.fmt_int(budget)} simulierten Schritten ({C.fmt_int(budget // C.STEPS)} Evaluationen von 140 Schritten bei {C.BASELINE_LABELS[compare]}); die Läufe enden beim ersten Erfolg.")
c_ge, c_bl = st.columns(2)
with c_ge:
    st.markdown("**Go-Explore**")
    a1 = st.columns(3)
    a1[0].metric("Go-Explore: Schritte bis zum Erfolg", C.fmt_int(ge.solved) if ge.solved is not None else "–", help="Verbrauchte Schritte am Ende der Runde, in der ein Zustand im Zielkreis lag (Genauigkeit eine Runde).")
    a1[1].metric("Go-Explore: Länge der Fahrt", f"{ge.solution_len} Schritte" if ge.solution_len is not None else "–")
    a1[2].metric("Go-Explore: Zellen im Archiv", C.fmt_int(ge.cells))
    a2 = st.columns(2)
    a2[0].metric("Go-Explore: verbrauchte Schritte", C.fmt_int(ge.used))
    a2[1].metric("Go-Explore: kleinster Restweg", f"{ge.best_progress:g} Zellen", help="Wahrer Weg (Zellen, um die Wände herum) von der besten erreichten Zelle zum Ziel. Der Zielkreis hat Radius 2, deshalb kann der Wert auch bei Erfolg größer als 0 sein.")
    st.plotly_chart(build_archive_map(grid, ge.archive_xy, ge.archive_steps, path), width="stretch", key=f"map_ge_{level}_{seed}_{budget}_{explore}_{cap}_{ret}_{cell}_{select}")
with c_bl:
    st.markdown(f"**{C.BASELINE_LABELS[compare]}**")
    b1 = st.columns(3)
    b1[0].metric("Vergleich: Schritte bis zum Erfolg", C.fmt_int(bl.solved_steps) if bl.solved_steps is not None else "–", help="Evaluationen bis zur ersten Endposition im Zielkreis mal 140 Schritte.")
    b1[1].metric("Vergleich: Evaluationen", C.fmt_int(bl.evaluations))
    b1[2].metric("Vergleich: berührte Zellen", C.fmt_int(int(bl.touched.sum())))
    b2 = st.columns(2)
    b2[0].metric("Vergleich: verbrauchte Schritte", C.fmt_int(bl.curve_steps[-1]))
    b2[1].metric("Vergleich: kleinster Restweg", f"{bl.curve_progress[-1]:g} Zellen", help="Wahrer Weg (Zellen, um die Wände herum) von der besten erreichten Zelle zum Ziel. Der Zielkreis hat Radius 2, deshalb kann der Wert auch bei Erfolg größer als 0 sein.")
    st.plotly_chart(build_touch_map(grid, bl.touched, bl_path), width="stretch", key=f"map_bl_{level}_{seed}_{budget}_{compare}")
st.caption("Links: je Zelle ein Zustand des Archivs, nach den Schritten bis dorthin gefärbt (dunkel = früh); rot die gefundene Fahrt. Rechts: jede Zelle, die irgendeine Fahrt des Vergleichsverfahrens berührt hat, und die Fahrt, die dem Ziel am nächsten kam.")
col_c, col_d = st.columns(2)
with col_c:
    st.markdown("**Zellen über den Schritten**")
    st.plotly_chart(build_cells_curve((ge.curve_steps, ge.curve_cells), (bl.curve_steps, bl.curve_cells), compare), width="stretch", key=f"cells_{level}_{seed}_{budget}_{explore}_{cap}_{ret}_{cell}_{select}_{compare}")
with col_d:
    st.markdown("**Kleinster Restweg über den Schritten**")
    st.plotly_chart(build_progress_curve((ge.curve_steps, ge.curve_progress), (bl.curve_steps, bl.curve_progress), compare), width="stretch", key=f"progress_{level}_{seed}_{budget}_{explore}_{cap}_{ret}_{cell}_{select}_{compare}")
st.info(
    f"Go-Explore {'erreicht das Ziel nach ' + C.fmt_int(ge.solved) + ' Schritten mit einer Fahrt von ' + str(ge.solution_len) + ' Schritten' if ge.solved is not None else 'erreicht das Ziel in diesem Budget nicht (kleinster Restweg ' + format(ge.best_progress, 'g') + ' Zellen)'}; "
    f"{C.BASELINE_LABELS[compare]} {'nach ' + C.fmt_int(bl.solved_steps) + ' Schritten' if bl.solved_steps is not None else 'erreicht es nicht (kleinster Restweg ' + format(bl.curve_progress[-1], 'g') + ' Zellen)'}. Das ist ein Lauf; die Studie unten vergleicht {N} Läufe."
)

st.markdown("---")
st.subheader("🌫️ Entgleisung: dieselbe Fahrt in einer Welt mit Rauschen")
st.markdown(
    "Go-Explore findet die Fahrt, weil es **zurücksetzen** kann. Fährt man die gefundene Aktionsfolge aber ohne Zurücksetzen noch einmal, und die Welt hat Rauschen (Gauß auf der Verschiebung je Schritt), weicht jede Wiederholung ein wenig ab, "
    "und eine Fahrt, die dicht an Wänden vorbeiführt, **entgleist**. Die zweite Phase der Quelle (Robustifizieren) soll das beheben; sie ist hier nicht gebaut."
)
if ge.actions is None:
    st.caption("Go-Explore hat in diesem Budget keine Fahrt gefunden, es gibt nichts zu wiederholen.")
else:
    states = S.replay_actions(grid, ge.actions, noise, C.REPLAYS)
    final = states[:, -1]
    hits = np.hypot(final[:, 0] - C.GOAL[0], final[:, 1] - C.GOAL[1]) < C.SUCCESS_RADIUS
    r1 = st.columns(3)
    r1[0].metric("Wiederholungen", f"{C.REPLAYS}")
    r1[1].metric("Rauschen je Schritt", f"{noise:g} Zellen")
    r1[2].metric("Wiederholungen im Zielkreis", f"{100 * hits.mean():.1f} %")
    st.plotly_chart(build_replay_figure(grid, states[:60], hits[:60]), width="stretch", key=f"replay_{level}_{seed}_{budget}_{explore}_{cap}_{ret}_{cell}_{select}_{noise}")
    st.caption(f"Gezeichnet sind die ersten 60 der {C.REPLAYS} Wiederholungen; grün ein Endpunkt im Zielkreis, rot daneben. Bei Rauschen 0 ist die Welt deterministisch und jede Wiederholung gleich.")

st.markdown("---")
st.subheader("🔬 Löst das die Falle? Die Studie über 20 Läufe")
st.markdown(
    f"Je Stufe und Verfahren {N} Läufe mit unterschiedlichem Seed, jeweils mit höchstens {C.fmt_int(PRE['budget'])} simulierten Schritten ({C.fmt_int(PRE['budget'] // C.STEPS)} Evaluationen von 140 Schritten). Go-Explore mit Voreinstellung "
    "(Zelle 1 × 1, 10 Erkundungsschritte, Zählerauswahl, Längenbegrenzung 140), einmal mit kostenloser Rückkehr und einmal mit Wiederabspielen."
)
col_g, col_h = st.columns(2)
with col_g:
    st.markdown("**Erfolgreiche Läufe je Stufe**")
    st.plotly_chart(build_success_bars(PRE), width="stretch", key="success_bars")
with col_h:
    st.markdown("**Median der Schritte bis zum Erfolg**")
    st.plotly_chart(build_steps_chart(PRE), width="stretch", key="steps_chart")
study_level = st.selectbox("Stufe für die Einzelheiten", options=C.LEVELS, index=C.LEVELS.index(4), key="study_level_select", format_func=lambda v: C.LEVEL_LABELS[v])
study_level = int(study_level)
rows = []
for m in C.METHODS:
    c = _mc(study_level, m)
    ln = [x for x in c["length"] if x is not None]
    rows.append(f"| {C.METHOD_LABELS[m]} | {E.success_count(c['solved'])} von {N} | {C.fmt_med(E.median_solved(c['solved']))} | {C.fmt_med(float(np.median(ln))) if ln else '–'} | {np.mean(c['cells']):.1f} | {np.mean(c['progress']):.2f} |")
st.markdown(f"**{C.LEVEL_LABELS[study_level]}**\n\n| Verfahren | Erfolg | Median der Schritte bis zum Erfolg | Median der Fahrtlänge (Schritte) | Zellen (Mittel) | Restweg (Mittel) |\n|---|---|---|---|---|---|\n" + "\n".join(rows))
st.markdown("**Anteil erfolgreicher Läufe über dem Budget**")
st.plotly_chart(build_success_curve(PRE, study_level), width="stretch", key=f"success_curve_{study_level}")

p_gn3 = E.mannwhitney_steps_p(_mc(3, "ge_free")["solved"], _mc(3, "novelty")["solved"])
p_free = {lv: E.mannwhitney_steps_p(_mc(lv, "ge_free")["solved"], _mc(lv, "novelty")["solved"]) for lv in (0, 1, 2)}
p_replay = {lv: E.mannwhitney_steps_p(_mc(lv, "ge_replay")["solved"], _mc(lv, "novelty")["solved"]) for lv in (0, 1, 2)}
st.info(
    f"**Stufe 4:** GA mit Fitness und Novelty lösen sie in {_count(4, 'fitness')} und {_count(4, 'novelty')} von {N} Läufen (kleinster Restweg {np.mean(_mc(4, 'fitness')['progress']):.2f} und {np.mean(_mc(4, 'novelty')['progress']):.2f} Zellen nach {C.fmt_int(PRE['budget'])} Schritten), "
    f"Go-Explore in {_count(4, 'ge_free')} von {N}: Median {C.fmt_med(_med(4, 'ge_free'))} Schritte mit kostenloser Rückkehr, {C.fmt_med(_med(4, 'ge_replay'))} mit Wiederabspielen. Die gefundene Fahrt ist {C.fmt_med(float(np.median([x for x in _mc(4, 'ge_free')['length'] if x is not None])))} Schritte lang, "
    f"also genau an der Grenze von {C.STEPS} (die kürzeste Lösung der Stufe braucht {_solvability(4)[1]}). **Stufe 3:** Median {C.fmt_med(_med(3, 'ge_free'))} Schritte gegen {C.fmt_med(_med(3, 'novelty'))} (Novelty, {_count(3, 'novelty')} von {N}) und {C.fmt_med(_med(3, 'fitness'))} (Fitness, {_count(3, 'fitness')} von {N}; {_pstr(p_gn3)} gegen Novelty). "
    f"**Stufe 0 bis 2:** mit kostenloser Rückkehr {C.fmt_med(_med(0, 'ge_free'))} / {C.fmt_med(_med(1, 'ge_free'))} / {C.fmt_med(_med(2, 'ge_free'))} Schritte gegen {C.fmt_med(_med(0, 'novelty'))} / {C.fmt_med(_med(1, 'novelty'))} / {C.fmt_med(_med(2, 'novelty'))} bei Novelty; "
    f"das ist gegen Novelty auf Stufe 1 bis 3 belegt schneller, auf Stufe 0 nicht ({_pstr(p_free[0])}); mit Wiederabspielen {C.fmt_med(_med(0, 'ge_replay'))} / {C.fmt_med(_med(1, 'ge_replay'))} / {C.fmt_med(_med(2, 'ge_replay'))}, "
    f"auf Stufe 0 und 2 ohne belegten Unterschied zu Novelty ({_pstr(p_replay[0])} / {_pstr(p_replay[2])})."
)

st.markdown("---")
st.subheader("🔬 Was kostet welche Einstellung? Stufe 3 und 4")
sweep_level = st.selectbox("Stufe für die Einstellungen", options=C.SWEEP_LEVELS, index=1, key="sweep_level_select", format_func=lambda v: C.LEVEL_LABELS[v])
sweep_level = int(sweep_level)
st.markdown(f"Go-Explore mit kostenloser Rückkehr, je eine geänderte Einstellung gegenüber der Voreinstellung, je {N} Läufe, Median der Schritte bis zum Erfolg (logarithmische Achse).")
st.plotly_chart(build_sweep_chart(PRE, sweep_level), width="stretch", key=f"sweep_chart_{sweep_level}")
rows = []
for cfg in C.SWEEP_CONFIGS:
    c = E.sweep_cell(PRE, sweep_level, cfg)
    ln = [x for x in c["length"] if x is not None]
    rows.append(f"| {C.SWEEP_LABELS[cfg]} | {E.success_count(c['solved'])} von {N} | {C.fmt_med(E.median_solved(c['solved']))} | {C.fmt_med(float(np.median(ln))) if ln else '–'} | {np.mean(c['cells']):.1f} |")
st.markdown(f"**{C.LEVEL_LABELS[sweep_level]}**\n\n| Einstellung | Erfolg | Median der Schritte | Median der Fahrtlänge | Zellen im Archiv (Mittel) |\n|---|---|---|---|---|\n" + "\n".join(rows))
p_uni = E.mannwhitney_steps_p(E.sweep_cell(PRE, 4, "uniform")["solved"], E.sweep_cell(PRE, 4, "default")["solved"])
st.info(
    f"**Stufe 4:** die Voreinstellung braucht {C.fmt_med(_sweep_med(4, 'default'))} Schritte; Erkunden mit 3 Schritten {C.fmt_med(_sweep_med(4, 'explore3'))}, mit 40 Schritten {C.fmt_med(_sweep_med(4, 'explore40'))}; Zellen von 2 × 2 {C.fmt_med(_sweep_med(4, 'pos2'))}, "
    f"Zellen mit Richtung {C.fmt_med(_sweep_med(4, 'posdir'))} (im Mittel {np.mean(E.sweep_cell(PRE, 4, 'posdir')['cells']):.1f} statt {np.mean(E.sweep_cell(PRE, 4, 'default')['cells']):.1f} Zellen im Archiv); gleichverteilte Auswahl {C.fmt_med(_sweep_med(4, 'uniform'))} "
    f"(gegen die Zählerauswahl {_pstr(p_uni)}: kein belegter Unterschied); **ohne Längenbegrenzung nur {C.fmt_med(_sweep_med(4, 'nocap'))} Schritte**, aber mit einer Fahrt von {C.fmt_med(float(np.median([x for x in E.sweep_cell(PRE, 4, 'nocap')['length'] if x is not None])))} Schritten, "
    f"die als 140-Schritte-Steuerfolge nicht mehr passt; Wiederabspielen {C.fmt_med(_sweep_med(4, 'replay'))}. Die Längenbegrenzung kostet also viel, die Auswahlregel nichts Messbares."
)

st.markdown("---")
st.subheader("🔬 Wie robust ist die gefundene Fahrt? Rauschen in der ganzen Studie")
st.markdown(
    f"Stufe 3 und 4, je {N} Läufe in einer Welt mit Rauschen (Gauß auf der Verschiebung je Schritt). Go-Explore findet das Ziel weiter mit dem Zurücksetzen; die gefundene Fahrt wird dann {C.REPLAYS}-mal **ohne** Zurücksetzen wiederholt (selbes Rauschen). "
    "Gezeigt ist der Anteil der Wiederholungen, die am Ende im Zielkreis stehen."
)
st.plotly_chart(build_noise_chart(PRE), width="stretch", key="noise_chart")
nz = {(lv, z): E.noise_cell(PRE, lv, z) for lv in C.SWEEP_LEVELS for z in C.NOISE_OPTIONS}
rep = {k: float(np.mean([r for r in v["replay"] if r is not None])) for k, v in nz.items()}
st.info(
    f"**Das Zurücksetzen verdeckt das Rauschen:** Go-Explore löst beide Stufen in jedem Lauf (Stufe 4: Median {C.fmt_med(E.median_solved(nz[(4, 0.0)]['solved']))} Schritte ohne und {C.fmt_med(E.median_solved(nz[(4, 0.1)]['solved']))} mit Rauschen 0.1). "
    f"**Die Fahrt hält es nicht aus:** Anteil der Wiederholungen im Ziel bei Rauschen 0 / 0.02 / 0.05 / 0.1: Stufe 3 {100 * rep[(3, 0.0)]:.1f} / {100 * rep[(3, 0.02)]:.1f} / {100 * rep[(3, 0.05)]:.1f} / {100 * rep[(3, 0.1)]:.1f} %, "
    f"Stufe 4 {100 * rep[(4, 0.0)]:.1f} / {100 * rep[(4, 0.02)]:.1f} / {100 * rep[(4, 0.05)]:.1f} / {100 * rep[(4, 0.1)]:.1f} %. Das ist die Entgleisung der Quelle; ihre zweite Phase (Robustifizieren) ist hier nicht gebaut."
)

st.markdown("---")
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Der Simulator lässt sich zurücksetzen** | Ohne Zurücksetzen kostet die Rückkehr das Wiederabspielen: auf Stufe 4 mehr als das Siebenfache der Schritte. Mit Rauschen hilft auch das Wiederabspielen nicht: die Fahrt entgleist. | kein Folgestück |
| **Die Zelle kennt den Ort** | Die Einteilung nach Position ist Vorwissen (wie die Endposition in Stück 8). Feinere Zellen (mit Richtung) kosten auf Stufe 4 mehr als das Vierfache; eine gelernte Einteilung wurde nicht gemessen. | kein Folgestück |
| **Die Welt ist deterministisch** | Die gefundene Fahrt ist nur dann eine Lösung. Die zweite Phase der Quelle (Robustifizieren durch Nachahmung) fehlt hier. | kein Folgestück |
| **Eine Fahrt kürzer als 140 Schritte reicht** | Ohne Längenbegrenzung wird die Stufe 4 viel schneller gelöst, die Fahrt ist dann aber länger als eine Steuerfolge der früheren Stücke. | kein Folgestück |
| **Eine Aufgabe, ein Labyrinth** | Die Umgebungen sind fest; dass sich die Aufgabe mit dem Können verändert, misst erst das letzte Stück. | **POET** (Folgestück) |
"""
)
st.caption(
    "Verwandt im Portfolio: [deception-maze-demo](https://sebastianhanisch-deception-maze-demo.streamlit.app/) (Stück 1), [novelty-search-demo](https://sebastianhanisch-novelty-search-demo.streamlit.app/) (Stück 2), "
    "[behavior-space-demo](https://sebastianhanisch-behavior-space-demo.streamlit.app/) (Stück 3), [novelty-objective-demo](https://sebastianhanisch-novelty-objective-demo.streamlit.app/) (Stück 4), "
    "[map-elites-demo](https://sebastianhanisch-map-elites-demo.streamlit.app/) (Stück 5), [cma-me-demo](https://sebastianhanisch-cma-me-demo.streamlit.app/) (Stück 6), "
    "[cvt-map-elites-demo](https://sebastianhanisch-cvt-map-elites-demo.streamlit.app/) (Stück 7), [aurora-demo](https://sebastianhanisch-aurora-demo.streamlit.app/) (Stück 8)."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Zustand und Schritt:** $(x, y, \theta)$; je Schritt $\theta \leftarrow \theta + a\,\theta_{\max}$, $p \leftarrow p + v\,s_{\max}(\cos\theta, \sin\theta)$ mit Drehgen $a \in [-1, 1]$ und Tempogen $v \in [0, 1]$; liegt $p$ in einer Wand, bleibt der Roboter stehen. Mit Rauschen kommt $\mathcal{N}(0, \sigma^2 I)$ auf $p$.

**Zelle:** $c(x, y, \theta) = (\lfloor x/g\rfloor, \lfloor y/g\rfloor)$ mit Kantenlänge $g \in \{1, 2\}$, bei Richtung zusätzlich $\lfloor 8\,(\theta \bmod 2\pi)/2\pi\rfloor$.

**Archiv:** $\mathcal{A}(c) = (\text{Zustand}, \text{Schritte } t_c, \text{Weg})$ mit kleinstem $t_c$ über alle bisher erreichten Zustände der Zelle, höchstens $T = 140$ (oder unbegrenzt).

**Runde:** Wahl der Zelle $c$ mit Wahrscheinlichkeit $\propto 1/(1 + n_c)$ ($n_c$ = bisherige Wahlen), Rückkehr (Kosten 0 oder $t_c$), $E$ zufällige Schritte; Kosten der Runde $100\,E$ (plus $\sum t_c$ bei Wiederabspielen). Erfolg, sobald ein Zustand höchstens 2 Zellen vom Ziel liegt.

Implementiert in `ge_search.py` (Go-Explore), `ge_baselines.py` (GA, Novelty) und `ge_evaluation.py` (Studie).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html))."
)
