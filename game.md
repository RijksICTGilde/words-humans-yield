# Words Humans Yield game

## Spelconcept

Een interactieve workshop waarin deelnemers zelf een Large Language Model (LLM) spelen om te leren hoe AI-taalmodellen
werken. Spelers vormen samen een neuraal netwerk dat tekst voorspelt, waardoor ze inzicht krijgen in de werking van
neurale netwerken, training versus inference, en hoe taalmodellen leren van data.

**Tijdsduur:** 75 minuten **Spelers:** 10-25 personen

## Leerdoelen

* Begrijpen hoe een neuraal netwerk werkt op basaal niveau
* Onderscheid tussen trainen en inference
* Inzicht in gewichten en backpropagation
* Belang van trainingsdata
* Evaluatie van modelperformance

## Benodigdheden

### Kernmaterialen

* 20 input woordkaarten (A5-formaat)
* 10 output woordkaarten (A5-formaat)
* 250 gewichtenkaartjes (getallen van -5 tot +5, meerdere van elk)
* 175 verbindingskoorden (touwtjes van 1-2 meter, verschillende kleuren)
* 20 rolkaarten met instructies
* 10 trainingsdata-kaarten met voorbeeldzinnen
* 1 groot whiteboard of flipchart voor netwerkstatus en scores

### Aanvullende materialen

* Groene en rode stippen (om actieve/inactieve status van input woorden aan te duiden)
* Kleine whiteboards of clipboards met papier voor berekeningen
* Plakband of tape om verbindingen vast te maken
* Standaards om woordkaartjes rechtop te plaatsen
* Naamstickers voor deelnemers
* Markeerstiften en pennen
* Klembord voor de evaluator

## Vocabulaire

### Input Vocabulaire (20 woorden)

1. de
2. het
3. een
4. op
5. in
6. onder
7. naast
8. kat
9. hond
10. kind
11. plant
12. auto
13. zit
14. staat
15. ligt
16. speelt
17. loopt
18. groot
19. klein
20. rood

### Output Vocabulaire (10 woorden)

1. mat
2. tafel
3. stoel
4. vloer
5. deur
6. tuin
7. boek
8. bal
9. kast
10. lamp

## Netwerkarchitectuur

* **Input laag:** 20 woordkaartjes met verbindingen naar VL1
* **Verborgen Laag 1 (VL1):** 3-5 neuronen (menselijke spelers)
* **Verborgen Laag 2 (VL2):** 3-5 neuronen (menselijke spelers)
* **Output laag:** 10 woordkaartjes met verbindingen naar VL2

Het netwerk is **fully connected**, wat betekent dat:

* Elk input woord is verbonden (met draden) met ALLE neuronen in Verborgen Laag 1
* Elk neuron in Verborgen Laag 1 is verbonden met ALLE neuronen in Verborgen Laag 2
* Elk neuron in Verborgen Laag 2 is verbonden met ALLE output woorden

Zie <https://claude.ai/public/artifacts/d60b1f6d-d31d-4429-b19a-d18423cd0e77>

## Rolverdeling (10-25 spelers)

|                          |              |              |                                                      |
| ------------------------ | ------------ | ------------ | ---------------------------------------------------- |
| Rol                      | Min. spelers | Max. spelers | Beschrijving                                         |
| Input-processors         | 1            | 2            | Activeren inputwoorden op basis van trainingszinnen  |
| Verborgen Laag 1         | 3            | 5            | Verwerken signalen van de inputlaag                  |
| Verborgen Laag 2         | 3            | 5            | Verwerken signalen van Verborgen Laag 1              |
| Output-processors        | 1            | 2            | Berekenen scores voor outputwoorden                  |
| Trainingsdata-providers  | 1            | 2            | Leveren voorbeeldzinnen                              |
| Evaluatoren              | 1            | 1            | Beoordelen voorspellingen en houden scores bij       |
| Backpropagation-trainers | 1            | 2            | Passen gewichten aan na foute voorspellingen         |
| Netwerk-beheerders       | 1            | 2            | Houden alle gewichten bij en helpen bij berekeningen |
| Spelleiders              | 1            | 2            | Leggen uit, houden tijd bij, faciliteren discussie   |

*Opmerking: Bij minder dan 13 spelers kunnen rollen worden gecombineerd, bijvoorbeeld input-processors en
trainingsdata-providers, of backpropagation-trainers en netwerk-beheerders.*

## Instructiekaarten

### Input-processor

```javascript
ROL: INPUT-PROCESSOR

Je taak is het activeren van de juiste inputwoorden op basis van de trainingszinnen.

INSTRUCTIES:
1. Luister naar de zin van de trainingsdata-provider
2. Identificeer welke woorden uit het input-vocabulaire voorkomen in de zin
3. Markeer deze woordkaartjes als ACTIEF (1) met een groene stip
4. Alle andere woordkaartjes blijven INACTIEF (0) (geen stip)
5. Kondig duidelijk aan welke woorden actief zijn:
   "De volgende woorden zijn actief: [woord1], [woord2]..."
6. Zorg dat alle verborgen laag 1 neuronen kunnen zien welke inputwoorden actief zijn

VOORBEELD:
Bij de zin "De kat zit op de mat":
- Actieve woorden: "de", "kat", "zit", "op"
- Inactieve woorden: alle andere

Voor elke nieuwe trainingszin: Zet eerst alle woorden terug naar inactief (0).
```

### Verborgen Laag 1 Neuron

```javascript
ROL: VERBORGEN LAAG 1 NEURON

Je bent een verwerkingsneuron in de eerste verborgen laag van het netwerk.

INSTRUCTIES:
1. Ontvang signalen van alle ACTIEVE input woorden waarmee je verbonden bent
2. Voor elk actief woord:
   a. Noteer het gewicht op de verbinding tussen jou en dat woord
   b. Vermenigvuldig de activatie (altijd 1) met dat gewicht
3. Tel al deze gewogen signalen op
4. Als je totale som > 0: Je bent ACTIEF (1)
   Als je totale som ≤ 0: Je bent INACTIEF (0)
5. Toon je status (ACTIEF/INACTIEF) met een groene of rode kaart
6. Kondig je status aan: "Verborgen Neuron 1-[X] is [actief/inactief]"
7. Stuur je activatie door naar alle neuronen in Verborgen Laag 2

Gebruik het whiteboard om je berekening te laten zien:
Input woord 1 (1) × gewicht (+2) = +2
Input woord 2 (1) × gewicht (-1) = -1
...
Totaal: +1 → ACTIEF (1)
```

### Verborgen Laag 2 Neuron

```javascript
ROL: VERBORGEN LAAG 2 NEURON

Je bent een verwerkingsneuron in de tweede verborgen laag van het netwerk.

INSTRUCTIES:
1. Ontvang signalen van alle ACTIEVE neuronen in Verborgen Laag 1
2. Voor elk actief VL1-neuron:
   a. Noteer het gewicht op de verbinding tussen jou en dat neuron
   b. Vermenigvuldig de activatie (altijd 1) met dat gewicht
3. Tel al deze gewogen signalen op
4. Als je totale som > 0: Je bent ACTIEF (1)
   Als je totale som ≤ 0: Je bent INACTIEF (0)
5. Toon je status (ACTIEF/INACTIEF) met een groene of rode kaart
6. Kondig je status aan: "Verborgen Neuron 2-[X] is [actief/inactief]"
7. Stuur je activatie door naar de output-laag

Gebruik het whiteboard om je berekening te laten zien:
VL1-Neuron 1 (1) × gewicht (+3) = +3
VL1-Neuron 2 (0) × gewicht (+1) = 0
...
Totaal: +3 → ACTIEF (1)
```

### Output-processor

```javascript
ROL: OUTPUT-PROCESSOR

Je berekent voorspellingsscores voor outputwoorden op basis van signalen van Verborgen Laag 2.

INSTRUCTIES:
1. Wacht tot de Verborgen Laag 2 neuronen hun activaties hebben doorgegeven
2. Voor elk outputwoord, bereken de totale score:
   a. Kijk welke VL2-neuronen ACTIEF zijn (waarde 1)
   b. Voor elk actief VL2-neuron, noteer het gewicht naar het outputwoord
   c. Tel alle gewichten van actieve VL2-neuronen op
3. Schrijf de totaalscore naast elk outputwoord
4. Identificeer het outputwoord met de hoogste score
5. Kondig duidelijk aan: "Het netwerk voorspelt het woord: [outputwoord] met score [score]"
6. Bij gelijke hoogste scores, kies het woord dat alfabetisch eerst komt

VOORBEELD:
"mat": VL2-1(1)×(+3) + VL2-2(0)×(+1) + VL2-3(1)×(-2) = +1
"tafel": VL2-1(1)×(-1) + VL2-2(0)×(+4) + VL2-3(1)×(+5) = +4
Voorspelling: "tafel" met score +4
```

### Trainingsdata-provider

```javascript
ROL: TRAININGSDATA-PROVIDER

Je levert voorbeeldzinnen om het netwerk te trainen of te testen.

INSTRUCTIES:
1. Kies een trainingskaart met een onvolledige zin
2. Kondig aan: "Nieuwe trainingsronde" (of "Nieuwe testfase" tijdens inference)
3. Lees de zin duidelijk voor, zonder het laatste woord te onthullen
   Bijv. "De kat zit op de ___"
4. Wacht tot de input-processors alle actieve woorden hebben gemarkeerd
5. Wacht tot het model een voorspelling heeft gemaakt
6. Onthul het correcte antwoord: "Het juiste antwoord is: [woord]"
7. Geef het resultaat door aan de evaluator

TIP: Tijdens de trainingsfase, begin met eenvoudige zinnen waarbij het patroon duidelijk is.
Tijdens de testfase, kun je variëren om te testen of het netwerk heeft gegeneraliseerd.
```

### Evaluator

```javascript
ROL: EVALUATOR

Je beoordeelt hoe goed het netwerk presteert en houdt de scores bij.

INSTRUCTIES:
1. Luister naar de voorspelling van de output-processor
2. Vergelijk deze met het correcte antwoord van de trainingsdata-provider
3. Bepaal of de voorspelling correct is
4. Kondig het resultaat aan: "De voorspelling is [CORRECT/INCORRECT]"
5. Houd de scores bij op het scorebord:
   a. Noteer de trainingszin (bijv. "De kat zit op de ___")
   b. Noteer het voorspelde woord
   c. Noteer het correcte woord
   d. Markeer als correct (✓) of incorrect (✗)
6. Bij incorrecte voorspellingen, geef een seintje aan de backpropagation-trainers
   om de gewichten aan te passen

TIP: Na meerdere trainingsronden, houd je ook de totaalscore bij:
"Het netwerk presteert nu [X] uit [Y] correct ([Z]%)."
```

### Backpropagation-trainer

```javascript
ROL: BACKPROPAGATION-TRAINER

Je past gewichten aan om het netwerk te verbeteren na incorrecte voorspellingen.

INSTRUCTIES:
1. Na een incorrecte voorspelling, analyseer wat er mis ging:
   a. Welk woord was het juiste antwoord?
   b. Welk woord werd incorrect voorspeld?
   c. Welke neuronen in VL2 waren actief of inactief?

2. Kies strategisch 3-5 gewichten om aan te passen:
   a. VERSTERK verbindingen tussen actieve VL2-neuronen en het juiste output-woord
      (verhoog gewicht met +1 of +2)
   b. VERZWAK verbindingen tussen actieve VL2-neuronen en het fout voorspelde output-woord
      (verlaag gewicht met -1 of -2)
   c. Indien nodig, pas ook gewichten aan tussen VL1 en VL2

3. Kondig je aanpassingen aan:
   "Ik pas de volgende gewichten aan:
   - Verbinding van VL2-[X] naar [juiste woord]: van [oud] naar [nieuw]
   - Verbinding van VL2-[Y] naar [fout woord]: van [oud] naar [nieuw]"

4. Werk samen met de netwerk-beheerder om de gewichtenmatrix bij te werken

TIP: Focus op de meest invloedrijke verbindingen. Drastische wijzigingen kunnen de training verstoren.
```

### Netwerk-beheerder

```javascript
ROL: NETWERK-BEHEERDER

Je houdt alle netwerkverbindingen en gewichten bij en zorgt voor correcte informatieoverdracht.

INSTRUCTIES:
1. Bij aanvang:
   a. Zorg dat alle verbindingen tussen lagen duidelijk zichtbaar zijn
   b. Ken initiële gewichten toe aan alle verbindingen (tussen -2 en +2)
   c. Houd een matrix bij van alle gewichten (bijv. op een groot vel papier)

2. Tijdens training:
   a. Assisteer neuronen bij berekeningen indien nodig
   b. Controleer of informatie correct wordt doorgegeven
   c. Let op activatiepatronen

3. Na backpropagation:
   a. Update de gewichtenkaartjes op de relevante verbindingen
   b. Werk de gewichtenmatrix bij
   c. Kondig belangrijke veranderingen aan: "Let op, deze verbindingen zijn nu versterkt/verzwakt..."

4. Tussen trainingsronden:
   a. Reset alle activaties naar inactief (0)
   b. Zorg dat alles klaar is voor de volgende ronde

TIP: Gebruik een tabel/matrix met rijen voor bronnen en kolommen voor doelen om alle gewichten overzichtelijk te houden.
```

### Spelleider

```javascript
ROL: SPELLEIDER

Je begeleidt het hele proces, legt uit, houdt de tijd bij en faciliteert de discussie.

INSTRUCTIES:
1. Begin met uitleg van het concept (5-10 min):
   a. Wat is een neuraal netwerk en hoe werkt het?
   b. Wat zijn de verschillende rollen?
   c. Hoe verloopt de informatie door het netwerk?

2. Begeleid de netwerkopbouw (5-10 min):
   a. Help deelnemers hun positie te vinden
   b. Zorg dat alle verbindingen worden gemaakt
   c. Leg uit hoe gewichten werken

3. Leid de trainingsfase (30-35 min):
   a. Kondig elke nieuwe trainingszin aan
   b. Pauzeer regelmatig om concepten uit te leggen
   c. Stel gerichte vragen: "Waarom denken jullie dat deze voorspelling fout was?"

4. Leid de testfase (10-15 min):
   a. Leg uit hoe inference verschilt van training
   b. Test met nieuwe zinnen
   c. Bespreek resultaten

5. Faciliteer afsluitende discussie (10 min):
   a. Vraag naar inzichten van deelnemers
   b. Leg verbanden met echte LLMs
   c. Beantwoord vragen

TIP: Houd het tempo hoog, maar neem tijd voor leermomenten. Zorg voor een goede balans tussen spel en uitleg.
```

## Voorbeeldzinnen voor Training en Test

### Trainingszinnen (8)

1. "De kat zit op de ___" (mat)
2. "Het kind speelt met de ___" (bal)
3. "De hond ligt op de ___" (vloer)
4. "De plant staat op de ___" (tafel)
5. "Het kind staat naast de ___" (deur)
6. "De kat speelt in de ___" (tuin)
7. "De grote hond zit onder de ___" (tafel)
8. "Het kind leest een ___" (boek)

### Testzinnen (4)

1. "De kleine kat ligt op de ___" (mat)
2. "Het kind staat op de ___" (stoel)
3. "De rode auto staat naast de ___" (deur)
4. "De hond speelt met de ___" (bal)

## Spelverloop en Tijdschema (75 minuten)

### 1. Introductie en Uitleg (10 min)

* Welkom en overzicht van het spelconcept
* Uitleg neuraal netwerk, LLMs en leerdoelen
* Toewijzing van rollen aan deelnemers

### 2. Netwerkopbouw (10 min)

* Positioneren van alle spelers en kaartjes
* Leggen van verbindingen tussen lagen
* Toekennen van initiële gewichten (willekeurig!)

### 3. Eerste Trainingsfase (25 min)

* Trainen met 4 voorbeeldzinnen
* Na elke zin: activatie, berekening, voorspelling, evaluatie
* Bij fouten: backpropagation en gewichtsaanpassingen

### 4. Tweede Trainingsfase (15 min)

* Trainen met 4 meer complexe voorbeeldzinnen
* Versterken van patronen die werken
* Verfijnen van gewichten

### 5. Testfase (10 min)

* Testen met 4 nieuwe zinnen
* Geen gewichtsaanpassingen meer
* Evaluatie van prestaties

### 6. Reflectie en Discussie (5 min)

* Bespreken van leermomenten
* Verbanden met echte taalmodellen
* Vragen en antwoorden

## Tips voor Facilitators

1. **Voorbereiding**:

    * Test het spel vooraf met een kleine groep
    * Bereid de fysieke ruimte zo in dat het netwerkdiagram duidelijk zichtbaar is
    * Maak kleurgecodeerde verbindingen voor betere visualisatie

2. **Tijdens het Spel**:

    * Begin met éénvoudige voorbeelden
    * Leg regelmatig pauzes in om concepten te verduidelijken
    * Help bij berekeningen waar nodig
    * Betrek alle deelnemers door gerichte vragen te stellen

3. **Simplificaties voor Beginners**:

    * Begin met kleinere input en output vocabulaires (bijv. 10 input, 5 output)
    * Start met vooraf ingestelde gewichten die werken voor de eerste zin
    * Vereenvoudig het backpropagation-proces

4. **Uitdagingen voor Gevorderden**:

    * Voeg complexere zinnen toe met meerdere subjecten
    * Introduceer een validatieset om overfitting te demonstreren
    * Laat spelers zelf nieuwe trainingsvoorbeelden verzinnen

5. **Afsluiting**:

    * Vat de belangrijkste leerpunten samen
    * Vergelijk met de schaal van echte LLMs (miljarden parameters)
    * Wijs op de beperkingen van het model en hoe echte LLMs anders werken

## Technische Specificatie

Het geïmplementeerde model is een discrete feed-forward MLP met binaire stap-activatiefunctie (Heaviside),
gestructureerd als 20-5-5-10 architectuur zonder bias-termen die directe, heuristische gewichtsaanpassingen gebruikt in
plaats van gradiëntgebaseerde backpropagation. Conceptueel volgt het Rosenblatt's perceptron-principes (1958) met
McCulloch-Pitts neuronen en representeert tekst via een bag-of-words benadering zonder sequentiële informatie. De
simplistische gewichtsaanpassing omzeilt differentieerbare functies en stocastische gradiëntafdaling die in moderne
netwerken gebruikelijk zijn (Rumelhart et al., 1986). Cruciale limitaties omvatten het ontbreken van contextuele
verbanden (versus Transformers; Vaswani et al., 2017), parameter-efficiënte backpropagation, en regularisatietechnieken
zoals gewichtsverval of dropout (Srivastava et al., 2014), wat het model gevoelig maakt voor overfitting zoals
gedemonstreerd in de simulatieresultaten. Voor moderne NLP-implementaties, zie Jurafsky & Martin (web.stanford.edu/~
jurafsky/slp3), en voor visuele neurale netwerkdemonstraties: playground.tensorflow.org.
