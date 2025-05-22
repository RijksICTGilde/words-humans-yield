# Words Humans Yield game

An interactive workshop where participants physically embody a neural network to learn how language models work. This
educational game demonstrates neural network concepts through a simplified "human neural network" that predicts words,
with participants playing neurons, connections, and various network components. Includes complete workshop materials and
a Python simulation to test and visualize the learning process.

_The name "Words Humans Yield" is a playful nod to [WHY2025](https://why2025.org/) (What Hackers Yearn), the
international non-profit outdoor hacker camp/conference in The Netherlands where this workshop will be presented._

[See the whole game here](game.md).

## Simulation

This simulation is to illustrate the process of the game. It's meant for facilitators, to understand the workings of
the game better.

```sh
uv run game.py

```

This is the output of the simulation:

```text
Strategic weights initialized for 'cat sits on' → 'mat' and 'child plays' → 'ball'

===== EXAMPLE PREDICTION =====
Input: 'the cat sits on the ___'
Prediction: 'mat'

--- NETWORK STATUS ---

INPUT LAYER:
Active words: 'the', 'on', 'cat', 'sits'

HIDDEN LAYER 1:
Neuron HL1-1: ACTIVE (1)
Neuron HL1-2: INACTIVE (0)
Neuron HL1-3: ACTIVE (1)
Neuron HL1-4: INACTIVE (0)
Neuron HL1-5: INACTIVE (0)

HIDDEN LAYER 2:
Neuron HL2-1: ACTIVE (1)
Neuron HL2-2: ACTIVE (1)
Neuron HL2-3: ACTIVE (1)
Neuron HL2-4: INACTIVE (0)
Neuron HL2-5: INACTIVE (0)

OUTPUT LAYER:
'mat': Score 3.2
'table': Score 0.4
'chair': Score -0.3
'floor': Score 0.8
'door': Score -0.4
'garden': Score -0.8
'book': Score 0.0
'ball': Score 3.0
'cabinet': Score 0.0
'lamp': Score 0.1

PREDICTION: 'mat' (Score: 3.2)


===== SIMULATION OF GAME ROUNDS =====

===== WORDS HUMANS YIELD GAME SIMULATION =====

Training rounds:

----- ROUND 1 -----
Training sentence: 'the cat sits on the ___' (Correct: 'mat')

Step 1: Input processors activate words
Active words: the, on, cat, sits

Step 2: Hidden Layer 1 calculates activations
HL1-1: Sum = 7.0 → ACTIVE (1)
HL1-2: Sum = -0.5 → INACTIVE (0)
HL1-3: Sum = 0.5 → ACTIVE (1)
HL1-4: Sum = -0.2 → INACTIVE (0)
HL1-5: Sum = -0.3 → INACTIVE (0)

Step 3: Hidden Layer 2 calculates activations
HL2-1: Sum = 3.1 → ACTIVE (1)
HL2-2: Sum = 0.4 → ACTIVE (1)
HL2-3: Sum = 0.2 → ACTIVE (1)
HL2-4: Sum = -0.3 → INACTIVE (0)
HL2-5: Sum = -0.1 → INACTIVE (0)

Step 4: Output processors calculate scores
'mat': Score 3.2
'table': Score 0.4
'chair': Score -0.3
'floor': Score 0.8
'door': Score -0.4
'garden': Score -0.8
'book': Score 0.0
'ball': Score 3.0
'cabinet': Score 0.0
'lamp': Score 0.1

Prediction: 'mat'
Correct answer: 'mat'

Prediction is CORRECT! No weight adjustments needed.

==================================================

----- ROUND 2 -----
Training sentence: 'the child plays with the ___' (Correct: 'ball')

Step 1: Input processors activate words
Active words: the, child, plays

Step 2: Hidden Layer 1 calculates activations
HL1-1: Sum = -0.1 → INACTIVE (0)
HL1-2: Sum = 4.8 → ACTIVE (1)
HL1-3: Sum = 0.9 → ACTIVE (1)
HL1-4: Sum = -0.8 → INACTIVE (0)
HL1-5: Sum = -1.0 → INACTIVE (0)

Step 3: Hidden Layer 2 calculates activations
HL2-1: Sum = -0.3 → INACTIVE (0)
HL2-2: Sum = 3.0 → ACTIVE (1)
HL2-3: Sum = -0.2 → INACTIVE (0)
HL2-4: Sum = 0.5 → ACTIVE (1)
HL2-5: Sum = 0.1 → ACTIVE (1)

Step 4: Output processors calculate scores
'mat': Score -0.2
'table': Score -0.8
'chair': Score 0.2
'floor': Score 0.8
'door': Score -0.6
'garden': Score -0.6
'book': Score -0.0
'ball': Score 2.1
'cabinet': Score -0.4
'lamp': Score -0.2

Prediction: 'ball'
Correct answer: 'ball'

Prediction is CORRECT! No weight adjustments needed.

==================================================

----- ROUND 3 -----
Training sentence: 'the dog lies on the ___' (Correct: 'floor')

Step 1: Input processors activate words
Active words: the, on, dog, lies

Step 2: Hidden Layer 1 calculates activations
HL1-1: Sum = 1.2 → ACTIVE (1)
HL1-2: Sum = -0.8 → INACTIVE (0)
HL1-3: Sum = 1.0 → ACTIVE (1)
HL1-4: Sum = -0.1 → INACTIVE (0)
HL1-5: Sum = -0.5 → INACTIVE (0)

Step 3: Hidden Layer 2 calculates activations
HL2-1: Sum = 3.1 → ACTIVE (1)
HL2-2: Sum = 0.4 → ACTIVE (1)
HL2-3: Sum = 0.2 → ACTIVE (1)
HL2-4: Sum = -0.3 → INACTIVE (0)
HL2-5: Sum = -0.1 → INACTIVE (0)

Step 4: Output processors calculate scores
'mat': Score 3.2
'table': Score 0.4
'chair': Score -0.3
'floor': Score 0.8
'door': Score -0.4
'garden': Score -0.8
'book': Score 0.0
'ball': Score 3.0
'cabinet': Score 0.0
'lamp': Score 0.1

Prediction: 'mat'
Correct answer: 'floor'

Prediction is INCORRECT. Backpropagation trainers adjust weights:
  HL2-1 → floor: 0.1 → 1.1
  HL2-1 → mat: 3.0 → 2.0
  HL2-2 → floor: 0.5 → 1.5
  HL2-2 → mat: 0.1 → -0.9
  HL2-3 → floor: 0.2 → 1.2
  HL2-3 → mat: 0.1 → -0.9

After weight adjustments:
New prediction would be: 'floor'

==================================================

----- ROUND 4 -----
Training sentence: 'the plant stands on the ___' (Correct: 'table')

Step 1: Input processors activate words
Active words: the, on, plant, stands

Step 2: Hidden Layer 1 calculates activations
HL1-1: Sum = 1.9 → ACTIVE (1)
HL1-2: Sum = -0.5 → INACTIVE (0)
HL1-3: Sum = 0.4 → ACTIVE (1)
HL1-4: Sum = -0.6 → INACTIVE (0)
HL1-5: Sum = 0.2 → ACTIVE (1)

Step 3: Hidden Layer 2 calculates activations
HL2-1: Sum = 3.3 → ACTIVE (1)
HL2-2: Sum = 0.0 → ACTIVE (1)
HL2-3: Sum = 0.7 → ACTIVE (1)
HL2-4: Sum = -0.6 → INACTIVE (0)
HL2-5: Sum = -0.2 → INACTIVE (0)

Step 4: Output processors calculate scores
'mat': Score 0.2
'table': Score 0.4
'chair': Score -0.3
'floor': Score 3.8
'door': Score -0.4
'garden': Score -0.8
'book': Score 0.0
'ball': Score 3.0
'cabinet': Score 0.0
'lamp': Score 0.1

Prediction: 'floor'
Correct answer: 'table'

Prediction is INCORRECT. Backpropagation trainers adjust weights:
  HL2-1 → table: 0.5 → 1.5
  HL2-1 → floor: 1.1 → 0.1
  HL2-2 → table: -0.3 → 0.7
  HL2-2 → floor: 1.5 → 0.5
  HL2-3 → table: 0.3 → 1.3
  HL2-3 → floor: 1.2 → 0.2

After weight adjustments:
New prediction would be: 'table'

==================================================

===== TESTING THE NETWORK =====
Now we test the network on new sentences it hasn't seen before:

Input: 'the small cat lies on the ___'
Prediction: 'table' (Correct: 'mat') ✗ INCORRECT

Relevant weights for 'cat+sits+on→mat' pattern:
Weight 'cat'→HL1-1: 3.0
Weight HL1-1→HL2-1: 3.0
Weight HL2-1→'mat': 2.0

Input: 'the child stands on the ___'
Prediction: 'table' (Correct: 'chair') ✗ INCORRECT

Input: 'the red car stands next to the ___'
Prediction: 'ball' (Correct: 'door') ✗ INCORRECT

Input: 'the dog plays with the ___'
Prediction: 'ball' (Correct: 'ball') ✓ CORRECT
```
