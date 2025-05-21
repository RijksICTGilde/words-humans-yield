# Words Humans Yield game


Een interactieve workshop waarin deelnemers zelf een Large Language Model (LLM) spelen om te leren hoe AI-taalmodellen werken. Spelers vormen samen een neuraal netwerk dat tekst voorspelt, waardoor ze inzicht krijgen in de werking van neurale netwerken, training versus inference, en hoe taalmodellen leren van data.

[Zie het hele spel hier](game.md)


## Simulatie

```sh
$ uv run game.py

```

Dit is de output van het runnen van dit spel:

```text
Strategische gewichten geïnitialiseerd voor 'kat zit op' → 'mat' en 'kind speelt' → 'bal'

===== VOORBEELD VOORSPELLING =====
Input: 'de kat zit op de ___'
Voorspelling: 'mat'

--- NETWERKSTATUS ---

INPUT LAAG:
Actieve woorden: 'de', 'op', 'kat', 'zit'

VERBORGEN LAAG 1:
Neuron VL1-1: ACTIEF (1)
Neuron VL1-2: ACTIEF (1)
Neuron VL1-3: ACTIEF (1)
Neuron VL1-4: INACTIEF (0)
Neuron VL1-5: INACTIEF (0)

VERBORGEN LAAG 2:
Neuron VL2-1: ACTIEF (1)
Neuron VL2-2: ACTIEF (1)
Neuron VL2-3: ACTIEF (1)
Neuron VL2-4: INACTIEF (0)
Neuron VL2-5: ACTIEF (1)

OUTPUT LAAG:
'mat': Score 2.9
'tafel': Score -0.3
'stoel': Score 0.8
'vloer': Score -0.2
'deur': Score 0.6
'tuin': Score 0.2
'boek': Score 0.5
'bal': Score 2.4
'kast': Score 0.2
'lamp': Score 0.7

VOORSPELLING: 'mat' (Score: 2.9)


===== SIMULATIE VAN SPELRONDEN =====

===== WORDS HUMANS YIELD SPELSIMULATIE =====

Trainingsronden:

----- RONDE 1 -----
Trainingszin: 'de kat zit op de ___' (Correct: 'mat')

Stap 1: Input-processors activeren woorden
Actieve woorden: de, op, kat, zit

Stap 2: Verborgen Laag 1 berekent activaties
VL1-1: Som = 7.0 → ACTIEF (1)
VL1-2: Som = 0.3 → ACTIEF (1)
VL1-3: Som = 0.1 → ACTIEF (1)
VL1-4: Som = -0.0 → INACTIEF (0)
VL1-5: Som = -0.3 → INACTIEF (0)

Stap 3: Verborgen Laag 2 berekent activaties
VL2-1: Som = 2.6 → ACTIEF (1)
VL2-2: Som = 3.0 → ACTIEF (1)
VL2-3: Som = 1.0 → ACTIEF (1)
VL2-4: Som = -0.3 → INACTIEF (0)
VL2-5: Som = 0.4 → ACTIEF (1)

Stap 4: Output-processors berekenen scores
'mat': Score 2.9
'tafel': Score -0.3
'stoel': Score 0.8
'vloer': Score -0.2
'deur': Score 0.6
'tuin': Score 0.2
'boek': Score 0.5
'bal': Score 2.4
'kast': Score 0.2
'lamp': Score 0.7

Voorspelling: 'mat'
Correct antwoord: 'mat'

Voorspelling is CORRECT! Geen gewichtsaanpassingen nodig.

==================================================

----- RONDE 2 -----
Trainingszin: 'het kind speelt met de ___' (Correct: 'bal')

Stap 1: Input-processors activeren woorden
Actieve woorden: de, het, kind, speelt

Stap 2: Verborgen Laag 1 berekent activaties
VL1-1: Som = -0.2 → INACTIEF (0)
VL1-2: Som = 4.8 → ACTIEF (1)
VL1-3: Som = 0.2 → ACTIEF (1)
VL1-4: Som = 0.1 → ACTIEF (1)
VL1-5: Som = 0.5 → ACTIEF (1)

Stap 3: Verborgen Laag 2 berekent activaties
VL2-1: Som = -0.2 → INACTIEF (0)
VL2-2: Som = 3.7 → ACTIEF (1)
VL2-3: Som = 1.3 → ACTIEF (1)
VL2-4: Som = 0.5 → ACTIEF (1)
VL2-5: Som = -0.1 → INACTIEF (0)

Stap 4: Output-processors berekenen scores
'mat': Score 0.7
'tafel': Score -0.5
'stoel': Score 0.2
'vloer': Score -0.2
'deur': Score 0.3
'tuin': Score -0.2
'boek': Score -0.3
'bal': Score 2.4
'kast': Score -0.2
'lamp': Score 0.3

Voorspelling: 'bal'
Correct antwoord: 'bal'

Voorspelling is CORRECT! Geen gewichtsaanpassingen nodig.

==================================================

----- RONDE 3 -----
Trainingszin: 'de hond ligt op de ___' (Correct: 'vloer')

Stap 1: Input-processors activeren woorden
Actieve woorden: de, op, hond, ligt

Stap 2: Verborgen Laag 1 berekent activaties
VL1-1: Som = 1.7 → ACTIEF (1)
VL1-2: Som = 0.3 → ACTIEF (1)
VL1-3: Som = -0.3 → INACTIEF (0)
VL1-4: Som = -0.2 → INACTIEF (0)
VL1-5: Som = -0.9 → INACTIEF (0)

Stap 3: Verborgen Laag 2 berekent activaties
VL2-1: Som = 3.1 → ACTIEF (1)
VL2-2: Som = 2.7 → ACTIEF (1)
VL2-3: Som = 0.7 → ACTIEF (1)
VL2-4: Som = -0.6 → INACTIEF (0)
VL2-5: Som = -0.1 → INACTIEF (0)

Stap 4: Output-processors berekenen scores
'mat': Score 3.2
'tafel': Score -0.3
'stoel': Score 0.8
'vloer': Score -0.3
'deur': Score 0.3
'tuin': Score 0.3
'boek': Score 0.5
'bal': Score 2.5
'kast': Score -0.1
'lamp': Score 0.2

Voorspelling: 'mat'
Correct antwoord: 'vloer'

Voorspelling is INCORRECT. Backpropagation-trainers passen gewichten aan:
  VL2-1 → vloer: -0.5 → 0.5
  VL2-1 → mat: 3.0 → 2.0
  VL2-2 → vloer: 0.2 → 1.2
  VL2-2 → mat: -0.1 → -1.1
  VL2-3 → vloer: 0.0 → 1.0
  VL2-3 → mat: 0.4 → -0.6

Na gewichtsaanpassingen:
Nieuwe voorspelling zou zijn: 'vloer'

==================================================

----- RONDE 4 -----
Trainingszin: 'de plant staat op de ___' (Correct: 'tafel')

Stap 1: Input-processors activeren woorden
Actieve woorden: de, op, plant, staat

Stap 2: Verborgen Laag 1 berekent activaties
VL1-1: Som = 1.2 → ACTIEF (1)
VL1-2: Som = -0.3 → INACTIEF (0)
VL1-3: Som = 0.1 → ACTIEF (1)
VL1-4: Som = -0.3 → INACTIEF (0)
VL1-5: Som = -0.4 → INACTIEF (0)

Stap 3: Verborgen Laag 2 berekent activaties
VL2-1: Som = 2.5 → ACTIEF (1)
VL2-2: Som = 0.0 → ACTIEF (1)
VL2-3: Som = 0.7 → ACTIEF (1)
VL2-4: Som = 0.1 → ACTIEF (1)
VL2-5: Som = 0.5 → ACTIEF (1)

Stap 4: Output-processors berekenen scores
'mat': Score 0.3
'tafel': Score -0.3
'stoel': Score 0.5
'vloer': Score 2.5
'deur': Score 0.9
'tuin': Score 0.1
'boek': Score 0.0
'bal': Score 2.2
'kast': Score 0.2
'lamp': Score 0.8

Voorspelling: 'vloer'
Correct antwoord: 'tafel'

Voorspelling is INCORRECT. Backpropagation-trainers passen gewichten aan:
  VL2-1 → tafel: 0.1 → 1.1
  VL2-1 → vloer: 0.5 → -0.5
  VL2-2 → tafel: -0.0 → 1.0
  VL2-2 → vloer: 1.2 → 0.2
  VL2-3 → tafel: -0.4 → 0.6
  VL2-3 → vloer: 1.0 → 0.0
  VL2-4 → tafel: -0.1 → 0.9
  VL2-4 → vloer: -0.4 → -1.4
  VL2-5 → tafel: 0.1 → 1.1
  VL2-5 → vloer: 0.1 → -0.9

Na gewichtsaanpassingen:
Nieuwe voorspelling zou zijn: 'tafel'

==================================================

===== TESTEN VAN HET NETWERK =====
Nu testen we het netwerk op nieuwe zinnen die het nog niet heeft gezien:

Input: 'de kleine kat ligt op de ___'
Voorspelling: 'tafel' (Correct: 'mat') ✗ INCORRECT

Relevante gewichten voor 'kat+zit+op→mat' patroon:
Gewicht 'kat'→VL1-1: 3.0
Gewicht VL1-1→VL2-1: 3.0
Gewicht VL2-1→'mat': 2.0

Input: 'het kind staat op de ___'
Voorspelling: 'tafel' (Correct: 'stoel') ✗ INCORRECT

Input: 'de rode auto staat naast de ___'
Voorspelling: 'tafel' (Correct: 'deur') ✗ INCORRECT

Input: 'de hond speelt met de ___'
Voorspelling: 'tafel' (Correct: 'bal') ✗ INCORRECT

```