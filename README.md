# Go-Explore – erst zurückkehren, dann erkunden (Streamlit-Demo)

---

Interaktive Demo zu **Go-Explore**. **Neuntes Stück der Konzepte-Linie „Novelty Search und Quality-Diversity“** im Portfolio von
[Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning): ein Verfahren, ein wachsendes Beispiel, jedes Folgestück hebt genau eine Annahme auf.

Derselbe Hof-Roboter, dasselbe Labyrinth und dasselbe Erfolgsmaß (Endposition im Zielkreis) wie in [`deception-maze-demo`](https://github.com/sebastian-hanisch/deception-maze-demo) (Stück 1) bis [`aurora-demo`](https://github.com/sebastian-hanisch/aurora-demo) (Stück 8). Bisher wurde eine **ganze Steuerfolge**
von 140 Schritten bewertet und gemutiert; die Stufe 4 (Schlangenlinie) löste in allen Stücken keines der Verfahren. **Go-Explore** (Ecoffet, Huizinga, Lehman, Stanley und Clune) merkt sich stattdessen **Zwischenstände**, **kehrt zu einem von ihnen zurück** und **erkundet von dort aus** mit ein paar zufälligen Schritten:
eine Fahrt wächst Stück für Stück. Das Stück misst **in simulierten Schritten**, ob das die Falle löst, und **was es dafür voraussetzt**.

## Kernfrage

Löst „erst zurückkehren, dann erkunden“ die Falle, an der Mutation, Fitness, Neuheit und Zellen scheitern, und welche Annahmen stecken in dem Erfolg (rücksetzbarer Simulator, Zelleneinteilung nach dem Ort, deterministische Welt)?

## Modell und Methodik

- **Labyrinth, Roboter** (`ge_maze.py`, `ge_robot.py`): wie Stück 1 bis 8, Stufen 0 bis 4. Zustand (x, y, Richtung); je Schritt Richtung += Drehgen · 0.6 rad, Fahrt um Tempogen · 0.7 Zellen; eine Wand lässt den Roboter stehen. **Erfolg** bei allen Verfahren: ein Zustand höchstens 2 Zellen vom Ziel.
- **Go-Explore** (`ge_search.py`): **Archiv** je Zelle (Position im 1 × 1- oder 2 × 2-Raster, wahlweise mit 8 Richtungssektoren) der Zustand, mit dem sie mit den wenigsten Schritten seit dem Start erreicht wurde, und der Weg dorthin als Baum von Aktionsstücken. **Eine Runde:** 100 Erkunder; jeder wählt eine Zelle
  (Gewicht 1 / (1 + bisherige Wahlen) oder gleichverteilt), **kehrt zurück** und fährt 3, 10 oder 40 zufällige Schritte (Drehgen gleichverteilt in [−1, 1], Tempogen in [0, 1]). **Jeder** dabei erreichte Zustand ersetzt den Eintrag seiner Zelle, wenn die Zelle neu ist oder mit weniger Schritten erreicht wurde
  und der Weg die Längenbegrenzung (140 Schritte oder keine) nicht überschreitet.
- **Rückkehr:** *kostenlos* (der Simulator wird auf den gespeicherten Zustand gesetzt) oder *durch Wiederabspielen* (die Aktionen des Wegs werden vom Start an noch einmal gefahren, ihre Schritte zählen zum Budget).
- **Kosten:** in **simulierten Schritten**: je Runde 100 · Erkundungsschritte (bei Wiederabspielen plus die Schritte der gewählten Wege). Die Genauigkeit ist eine Runde. Eine Evaluation einer Steuerfolge kostet 140 Schritte; das Budget der Studie sind 5.6 Mio Schritte (40 000 Evaluationen), wie in den früheren Stücken.
- **Vergleichsverfahren** (`ge_baselines.py`): **GA mit Fitness** (−Abstand der Endposition zum Ziel) und **Novelty Search** (mittlerer Abstand zu den 15 nächsten Endpositionen, ohne Archiv), mit den Operatoren aus Stück 1 und 2 (Turnier 3, die zwei Besten bleiben, Mutation Gauß 0.25 auf 15 % der Gene, Population 100).
- **Die gefundene Fahrt:** die Aktionsfolge vom Start bis zum Zustand im Zielkreis; sie ist, mit Stillstand aufgefüllt, eine Steuerfolge der Länge 140 und fährt in der deterministischen Welt über die Simulation der früheren Stücke ins Ziel.
- **Entgleisung** (Derailment, so heißt es in der Quelle): in einer Welt mit **Rauschen** (Gauß auf der Verschiebung je Schritt) findet Go-Explore das Ziel weiter, weil es zurücksetzen kann; die gefundene Fahrt wird dann 200-mal **ohne** Zurücksetzen wiederholt, gemessen wird der Anteil, der am Ende im Zielkreis steht.
- **Eigene Umsetzung:** nur die erste Phase der Quelle (Erkunden); Archiv, Auswahl und Rückkehr sind eigener NumPy-Code. Die zweite Phase der Quelle (Robustifizieren der Fahrt durch Nachahmung) fehlt.
- **Gegenproben:** (1) die Zellen von Hand (Raster 1 × 1 und 2 × 2, Randfall, die Richtungssektoren); (2) der Schritt der Kinematik gegen die Simulation der früheren Stücke, auch mit Wänden, und ein blockierter Schritt von Hand; (3) **Orakel:** Go-Explore **als Ganzes** gegen eine skalare Neuimplementierung mit Wörterbüchern und `math`
  (je Erkunder und Schritt eine Schleife, Archiv als Wörterbuch, Kosten und Wege von Hand gezählt, gleiche Zufallszahlen) für alle drei Zellarten, beide Auswahlregeln und beide Rückkehrarten, mit und ohne Längenbegrenzung: gleiche Zellen, Schritte, Zustände, Verläufe, Kosten und gefundene Aktionsfolge; (4) die gefundene
  Fahrt wird vom Start abgespielt und endet im Zielkreis, auch als 140-Schritte-Steuerfolge über `simulate`; (5) der Restweg gegen SciPys Breitensuche; (6) die Novelty gegen SciPys nächste Nachbarn und von Hand, die berührten Zellen von GA und Novelty gegen eine skalare Neuberechnung der Fahrten; (7) das Archiv benutzt das Ziel nicht
  (verschobenes Ziel ändert das Wachstum der Zellen nicht); (8) Läufe der Studie aus allen drei Teilen (Verfahren, Einstellungen, Rauschen einschließlich der Wiederholungsquote) werden mit demselben Seed bitgleich reproduziert.
- **Vorgerechnete Studie** (`generate_precomputed.py` → `precomputed_sweep.json`, rund 30 Sekunden parallel): **Verfahren:** 5 Stufen × 4 Verfahren × 20 Läufe = 400; **Einstellungen** (Stufe 3 und 4): Go-Explore mit je einer geänderten Einstellung (8 Zeilen) × 20 = 320; **Rauschen** (Stufe 3 und 4): 4 Werte × 20 = 160; zusammen 880 Läufe.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`. 20 Läufe je Zelle (Seeds 0 bis 19), Budget 5.6 Mio simulierte Schritte; „Median“ über die erfolgreichen Läufe (ein Lauf ohne Erfolg hat keine Schritte); „p“ der Mann-Whitney-Test auf die Schritte, ein Lauf ohne Erfolg zählt als schlechter als jeder erfolgreiche.
Zellen: bei Go-Explore die Zellen im Archiv (1 × 1), bei GA und Novelty die Zellen, die irgendeine Fahrt berührt hat; sie sind nicht vergleichbar, weil die Läufe beim ersten Erfolg enden. Restweg: der kleinste wahre Weg (um die Wände herum) vom besten erreichten Zustand zum Ziel; der Zielkreis hat Radius 2, deshalb ist er auch bei Erfolg nicht immer 0.

| Stufe | Verfahren | Erfolg | Median der Schritte bis zum Erfolg | Median der Fahrtlänge (Schritte) | Zellen (Mittel) | Restweg (Mittel) |
|---|---|---|---|---|---|---|
| 0: freies Feld | GA mit Fitness (Stück 1) | 20 von 20 | 8 890 | – | 476.9 | 0.00 |
| 0: freies Feld | Novelty Search (Stück 2) | 20 von 20 | 8 890 | – | 477.3 | 0.00 |
| 0: freies Feld | Go-Explore, Rückkehr kostenlos | 20 von 20 | 5 000 | 41 | 280.8 | 1.50 |
| 0: freies Feld | Go-Explore, Rückkehr durch Wiederabspielen | 20 von 20 | 11 177 | 41 | 280.8 | 1.50 |
| 1: Öffnung links | GA mit Fitness (Stück 1) | 20 von 20 | 17 640 | – | 402.1 | 0.05 |
| 1: Öffnung links | Novelty Search (Stück 2) | 20 von 20 | 20 440 | – | 417.2 | 0.05 |
| 1: Öffnung links | Go-Explore, Rückkehr kostenlos | 20 von 20 | 6 000 | 39 | 255.8 | 1.10 |
| 1: Öffnung links | Go-Explore, Rückkehr durch Wiederabspielen | 20 von 20 | 14 149.5 | 39 | 255.8 | 1.10 |
| 2: Öffnung in der Mitte | GA mit Fitness (Stück 1) | 20 von 20 | 26 670 | – | 414.4 | 0.40 |
| 2: Öffnung in der Mitte | Novelty Search (Stück 2) | 20 von 20 | 29 050 | – | 421.2 | 0.45 |
| 2: Öffnung in der Mitte | Go-Explore, Rückkehr kostenlos | 20 von 20 | 10 000 | 64 | 384.6 | 1.25 |
| 2: Öffnung in der Mitte | Go-Explore, Rückkehr durch Wiederabspielen | 20 von 20 | 31 545.5 | 64 | 384.6 | 1.25 |
| 3: Öffnung rechts | GA mit Fitness (Stück 1) | 11 von 20 | 177 380 | – | 355.1 | 5.30 |
| 3: Öffnung rechts | Novelty Search (Stück 2) | 20 von 20 | 138 390 | – | 411.3 | 0.95 |
| 3: Öffnung rechts | Go-Explore, Rückkehr kostenlos | 20 von 20 | 14 000 | 99 | 420.6 | 0.95 |
| 3: Öffnung rechts | Go-Explore, Rückkehr durch Wiederabspielen | 20 von 20 | 58 499 | 99 | 420.6 | 0.95 |
| 4: Schlangenlinie | GA mit Fitness (Stück 1) | 0 von 20 | – | – | 233.6 | 34.60 |
| 4: Schlangenlinie | Novelty Search (Stück 2) | 0 von 20 | – | – | 272.0 | 26.20 |
| 4: Schlangenlinie | Go-Explore, Rückkehr kostenlos | 20 von 20 | 216 500 | 140 | 406.6 | 1.80 |
| 4: Schlangenlinie | Go-Explore, Rückkehr durch Wiederabspielen | 20 von 20 | 1 677 411.5 | 140 | 406.6 | 1.80 |

| Frage | Befund |
|---|---|
| Löst Go-Explore die Falle (Stufe 4)? | **Ja, in jedem Lauf.** GA mit Fitness und Novelty lösen sie in 0 von 20 Läufen (kleinster Restweg 34.60 und 26.20 Zellen nach 5.6 Mio Schritten), Go-Explore in 20 von 20: Median **216 500 Schritte** mit kostenloser Rückkehr (3.9 % des Budgets, etwa 1 546 Evaluationen-Äquivalente), 1 677 411.5 mit Wiederabspielen (30 % des Budgets). |
| Wie lang ist die gefundene Fahrt? | **140 Schritte, genau die Grenze.** Median der Fahrtlänge auf Stufe 0 bis 4: 41 / 39 / 64 / 99 / 140 Schritte; die kürzeste Lösung der Stufe 4 braucht 136 (Pfad-Regler). Ohne Längenbegrenzung löst Go-Explore Stufe 4 in 28 000 Schritten, die Fahrt ist dann 189 Schritte lang und passt nicht mehr in eine Steuerfolge der früheren Stücke. |
| Und die Stufe 3? | **Knapp zehnmal weniger Schritte als Novelty.** Median 14 000 (kostenlos) und 58 499 (Wiederabspielen) gegen 138 390 (Novelty, 20 von 20) und 177 380 (Fitness, 11 von 20); alle vier Vergleiche p < 0.001. |
| Und die einfachen Stufen? | **Kein oder kleiner Vorteil.** Stufe 0 / 1 / 2: Median Fitness 8 890 / 17 640 / 26 670, Novelty 8 890 / 20 440 / 29 050, Go-Explore kostenlos 5 000 / 6 000 / 10 000 (gegen Novelty p = 0.32 auf Stufe 0, p < 0.001 auf Stufe 1, p = 0.0019 auf Stufe 2), mit Wiederabspielen 11 177 / 14 149.5 / 31 545.5 (p = 0.32 / 0.034 / 0.44): auf Stufe 0 und 2 ohne belegten Unterschied. |
| Was kostet die Rückkehr? | **Viel, sobald man nicht zurücksetzen kann.** Auf Stufe 4 mehr als das Siebenfache (1 677 411.5 gegen 216 500, p < 0.001), auf Stufe 3 gut das Vierfache (58 499 gegen 14 000). |
| Wie fein sollen die Zellen sein? | **Nicht zu fein.** Stufe 4: Zellen von 2 × 2 267 500 Schritte (137.2 Zellen im Archiv), Position 1 × 1 216 500 (406.6 Zellen), mit Richtung 1 003 000 (3 167.1 Zellen): mehr als das Vierfache; auf Stufe 3 35 000 gegen 14 000. |
| Wie lang soll erkundet werden? | **Kurz.** Stufe 4: 3 Schritte 77 400, 10 Schritte 216 500, 40 Schritte 848 000 (p < 0.001); Stufe 3: 12 600, 14 000, 20 000. |
| Hilft die Auswahl selten gewählter Zellen? | **Auf Stufe 4 nicht belegt, auf Stufe 3 ja.** Gleichverteilt 199 500 gegen 216 500 Schritte auf Stufe 4 (p = 0.27), auf Stufe 3 22 000 gegen 14 000 (p < 0.001). |
| Entgleist die gefundene Fahrt? | **Ja, sobald die Welt Rauschen hat.** Go-Explore löst alle 160 Läufe mit Rauschen (Median Stufe 4: 216 500 / 227 500 / 194 500 / 149 000 Schritte bei Rauschen 0 / 0.02 / 0.05 / 0.1). Wird die gefundene Fahrt aber ohne Zurücksetzen wiederholt, steht nur noch ein Teil der Wiederholungen im Ziel: Stufe 3 100.00 / 55.30 / 51.78 / 14.75 %, Stufe 4 100.00 / 9.42 / 0.97 / 0.00 % bei Rauschen 0 / 0.02 / 0.05 / 0.1. |

### Einstellungen von Go-Explore (Stufe 3 und 4)

Go-Explore mit kostenloser Rückkehr, je eine Einstellung gegenüber der Voreinstellung geändert; alle 20 von 20.

| Stufe | Einstellung | Erfolg | Median der Schritte | Median der Fahrtlänge | Zellen im Archiv (Mittel) |
|---|---|---|---|---|---|
| 3: Öffnung rechts | Voreinstellung (10 Schritte, Position 1 × 1, Zähler, Deckel 140) | 20 von 20 | 14 000 | 99 | 420.6 |
| 3: Öffnung rechts | Erkunden mit 3 Schritten | 20 von 20 | 12 600 | 88 | 401.9 |
| 3: Öffnung rechts | Erkunden mit 40 Schritten | 20 von 20 | 20 000 | 118.5 | 413.1 |
| 3: Öffnung rechts | Zelle 2 × 2 | 20 von 20 | 13 000 | 97 | 132.2 |
| 3: Öffnung rechts | Zelle mit Richtung | 20 von 20 | 35 000 | 117.5 | 2554.2 |
| 3: Öffnung rechts | gleichverteilte Auswahl | 20 von 20 | 22 000 | 97.5 | 411.2 |
| 3: Öffnung rechts | ohne Längenbegrenzung | 20 von 20 | 14 000 | 99 | 420.6 |
| 3: Öffnung rechts | Rückkehr durch Wiederabspielen | 20 von 20 | 58 499 | 99 | 420.6 |
| 4: Schlangenlinie | Voreinstellung (10 Schritte, Position 1 × 1, Zähler, Deckel 140) | 20 von 20 | 216 500 | 140 | 406.6 |
| 4: Schlangenlinie | Erkunden mit 3 Schritten | 20 von 20 | 77 400 | 140 | 404.3 |
| 4: Schlangenlinie | Erkunden mit 40 Schritten | 20 von 20 | 848 000 | 140 | 405.6 |
| 4: Schlangenlinie | Zelle 2 × 2 | 20 von 20 | 267 500 | 140 | 137.2 |
| 4: Schlangenlinie | Zelle mit Richtung | 20 von 20 | 1 003 000 | 140 | 3167.1 |
| 4: Schlangenlinie | gleichverteilte Auswahl | 20 von 20 | 199 500 | 139.5 | 405.9 |
| 4: Schlangenlinie | ohne Längenbegrenzung | 20 von 20 | 28 000 | 189 | 406.2 |
| 4: Schlangenlinie | Rückkehr durch Wiederabspielen | 20 von 20 | 1 677 411.5 | 140 | 406.6 |

### Rauschen und Entgleisung (Stufe 3 und 4)

Go-Explore (Voreinstellung) in einer Welt mit Gauß-Rauschen auf der Verschiebung je Schritt; die gefundene Fahrt wird 200-mal ohne Zurücksetzen wiederholt (selbes Rauschen).

| Stufe | Rauschen (Zellen je Schritt) | Erfolg | Median der Schritte bis zum Erfolg | Wiederholungen im Zielkreis (Mittel über die Läufe) |
|---|---|---|---|---|
| 3: Öffnung rechts | 0 | 20 von 20 | 14 000 | 100.00 % |
| 3: Öffnung rechts | 0.02 | 20 von 20 | 15 000 | 55.30 % |
| 3: Öffnung rechts | 0.05 | 20 von 20 | 15 000 | 51.78 % |
| 3: Öffnung rechts | 0.1 | 20 von 20 | 15 000 | 14.75 % |
| 4: Schlangenlinie | 0 | 20 von 20 | 216 500 | 100.00 % |
| 4: Schlangenlinie | 0.02 | 20 von 20 | 227 500 | 9.42 % |
| 4: Schlangenlinie | 0.05 | 20 von 20 | 194 500 | 0.97 % |
| 4: Schlangenlinie | 0.1 | 20 von 20 | 149 000 | 0.00 % |

## Befunde und Korrekturen gegenüber der Vorab-Messreihe

- **Die Vorab-Messreihe (20 Seeds, Skript `ge_probe.py`) zeigte dieselben Zahlen** für die Voreinstellung (Stufe 4: 216 500, Stufe 3: 14 000) und die Reihenfolge der Einstellungen; die Studie baut auf demselben Verfahren auf und ergänzt GA und Novelty im selben Schrittbudget sowie die Stufen 0 bis 2.
- **Das Rauschen wirkt in der Vorab-Messreihe (nur 6 Seeds) stärker als in der Studie:** Stufe 4 3.5 % / 0 % / 0 % gegen 9.42 % / 0.97 % / 0.00 % bei Rauschen 0.02 / 0.05 / 0.1, Stufe 3 59 % gegen 55.30 % bei 0.02. Die Richtung ist gleich, mit 20 Läufen sind die Quoten aber etwas höher.
- **Erwartet war ein großer Vorteil auch auf den einfachen Stufen;** gemessen ist er dort klein oder nicht belegt (Stufe 0: p = 0.32). Der Vorteil ist ein Vorteil in der Falle.
- **Die Auswahl selten gewählter Zellen ist kein allgemeiner Gewinn:** auf Stufe 4 ist sie von der gleichverteilten Auswahl nicht zu unterscheiden (p = 0.27).

## Ehrliche Grenzen

- **Ein rücksetzbarer, deterministischer Simulator.** Der Vorteil beruht darauf, dass der Roboter auf einen gespeicherten Zustand gesetzt werden kann. Das Wiederabspielen als Ersatz kostet auf Stufe 4 mehr als das Siebenfache, funktioniert aber nur in einer Welt ohne Rauschen.
- **Die Zelle kennt den Ort.** Position (und Richtung) als Zelle ist Vorwissen über die Aufgabe; GA und Novelty benutzen es nur indirekt (Endposition als Beschreibung bei Novelty). Eine gelernte oder grobe Zelleneinteilung wurde nicht gemessen (nur 2 × 2 und mit Richtung).
- **Die Längenbegrenzung 140 ist knapp.** Die kürzeste Lösung braucht 136 Schritte, Go-Explore findet 140; ohne Begrenzung geht es mehr als siebenmal schneller, die Fahrt ist dann länger als jedes Genom der früheren Stücke. Der Vergleich in Schritten gilt für die Begrenzung auf 140.
- **Vergleich in Schritten, nicht in Evaluationen.** Eine Evaluation kostet 140 Schritte, Go-Explore zahlt nur, was es fährt. GA und Novelty speichern keine Zwischenstände: jede Evaluation fährt die ganze Steuerfolge.
- **Nur die erste Phase der Quelle.** Die gefundene Fahrt ist nur in der deterministischen Welt eine Lösung. Die Robustifizierung der Quelle (Nachahmung in der verrauschten Welt) fehlt; die Entgleisungsquote zeigt, was sie beheben müsste.
- **Eigene, einfache Einstellungen.** 100 Erkunder, gleichverteilte zufällige Aktionen, Gewicht 1 / (1 + Wahlen): die Regeln der Quelle für Auswahl und Erkunden wurden nicht nachgebaut und nicht verglichen.
- **Die Genauigkeit des Budgets ist eine Runde** (1 000 Schritte bei 10 Erkundungsschritten), bei Wiederabspielen mehr. Die Median-Schritte liegen deshalb auf Tausendern.
- **Kleine, aber „signifikante“ Unterschiede.** Bei 20 Läufen sind auch kleine Unterschiede belegt; „kein belegter Unterschied“ meint praktisch gleich (z. B. Auswahl auf Stufe 4, p = 0.27). Das Rauschen ist in der Studie nur auf zwei Stufen und vier Werten gemessen.
- **Das Ziel kennt nur der Erfolg.** Archiv und Auswahl benutzen das Ziel nicht (Test); es beendet den Lauf und misst den Restweg.

## Verwandte Demos im Portfolio

- [`deception-maze-demo`](https://github.com/sebastian-hanisch/deception-maze-demo): Stück 1, das täuschende Labyrinth (GA mit Fitness).
- [`novelty-search-demo`](https://github.com/sebastian-hanisch/novelty-search-demo): Stück 2, Novelty Search mit und ohne Archiv.
- [`behavior-space-demo`](https://github.com/sebastian-hanisch/behavior-space-demo): Stück 3, zwölf Verhaltensräume.
- [`novelty-objective-demo`](https://github.com/sebastian-hanisch/novelty-objective-demo): Stück 4, Novelty plus Ziel.
- [`map-elites-demo`](https://github.com/sebastian-hanisch/map-elites-demo): Stück 5, MAP-Elites.
- [`cma-me-demo`](https://github.com/sebastian-hanisch/cma-me-demo): Stück 6, CMA-ME und CMA-MAE.
- [`cvt-map-elites-demo`](https://github.com/sebastian-hanisch/cvt-map-elites-demo): Stück 7, CVT-MAP-Elites.
- [`aurora-demo`](https://github.com/sebastian-hanisch/aurora-demo): Stück 8, gelernte Verhaltensbeschreibung.

## Bewusst nicht umgesetzt

Jede dieser Annahmen hebt ein Folgestück der Linie auf:

| Annahme | Folgestück |
|---|---|
| Der Simulator lässt sich zurücksetzen | kein Folgestück |
| Die Zelle kennt den Ort | kein Folgestück |
| Die Welt ist deterministisch | kein Folgestück (die zweite Phase der Quelle fehlt) |
| Eine Fahrt kürzer als 140 Schritte reicht | kein Folgestück |
| Eine Aufgabe, ein Labyrinth | POET (Folgestück) |

## Tests

218 Tests, rund eine Minute: Labyrinth und Roboter (aus Stück 1), Zellen, Schritt der Kinematik, Go-Explore (Reproduzierbarkeit, Kurven, Kosten, Längenbegrenzung, Lösung und Wiederabspielen, Rauschen, Ziel-Unabhängigkeit), GA und Novelty, **Orakel** (Go-Explore als Ganzes gegen eine Wörterbuch-Buchführung mit skalarer Kinematik,
Restweg gegen SciPy), Auswertung und Vollständigkeit der vorgerechneten Datei (inklusive Neurechnung einzelner Läufe aller drei Teile), Presets und Permalink, Diagramme (gesperrte Achsen), AppTest-Rauchtests mit festem Würfel-Seed, der Smoke-Test der Portfolio-Vorlage, ein Quelltext-Test gegen Satz-Komma-Fehler
und `test_claims.py` für jede Zahl dieser README.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `ge_maze.py`, `ge_robot.py` | Labyrinth, Breitensuche, Täuschungsmaße; Kinematik, Population, Pfad-Regler (wie Stück 1) |
| `ge_search.py` | Go-Explore: Zellen, Schritt, Archiv, Rückkehr, Kosten, gefundene Fahrt, Wiederabspielen mit Rauschen |
| `ge_baselines.py` | GA mit Fitness und Novelty Search im Schrittbudget |
| `ge_evaluation.py` | Maße je Stufe, Lösbarkeit, Läufe, Zugriff auf die Studie, Mann-Whitney-Hilfen |
| `generate_precomputed.py` | rechnet die Studie vor → `precomputed_sweep.json` |
| `ge_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `ge_presets.py`, `ge_constants.py` | Presets, Permalink, Grenzen |
| `tests/` | siehe oben |

## Literatur

- Ecoffet, A., Huizinga, J., Lehman, J., Stanley, K. O., Clune, J. (2021): First return, then explore. *Nature* 590, 580–586; arXiv:2004.12919.
- Ecoffet, A., Huizinga, J., Lehman, J., Stanley, K. O., Clune, J. (2019): Go-Explore: a new approach for hard-exploration problems. arXiv:1901.10995.
- Lehman, J., Stanley, K. O. (2011): Abandoning objectives: evolution through the search for novelty alone. *Evolutionary Computation* 19(2), 189–223.

## Lokal ausführen

```
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` und `python -m pytest tests/ -v`. Studie neu rechnen: `python generate_precomputed.py` (rund 30 Sekunden auf 14 Prozessen).

Gebaut mit Streamlit und Plotly (die Simulation rechnet NumPy, der Mann-Whitney-Test SciPy).
