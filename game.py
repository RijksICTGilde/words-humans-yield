import numpy as np
import random

class WordsHumansYieldSimulation:
    def __init__(self):
        # Definieer de vocabulaires
        self.input_vocab = [
            "de", "het", "een", "op", "in",
            "onder", "naast", "kat", "hond", "kind",
            "plant", "auto", "zit", "staat", "ligt",
            "speelt", "loopt", "groot", "klein", "rood"
        ]

        self.output_vocab = [
            "mat", "tafel", "stoel", "vloer", "deur",
            "tuin", "boek", "bal", "kast", "lamp"
        ]

        # Netwerkdimensies
        self.input_size = len(self.input_vocab)
        self.hidden1_size = 5  # Verborgen Laag 1
        self.hidden2_size = 5  # Verborgen Laag 2
        self.output_size = len(self.output_vocab)

        # Initialiseer gewichten met willekeurige kleine waarden tussen -2 en +2
        self.weights_input_hidden1 = np.random.uniform(-2, 2, (self.input_size, self.hidden1_size))
        self.weights_hidden1_hidden2 = np.random.uniform(-2, 2, (self.hidden1_size, self.hidden2_size))
        self.weights_hidden2_output = np.random.uniform(-2, 2, (self.hidden2_size, self.output_size))

        # Voor het opslaan van activaties tijdens het netwerk gebruik
        self.input_activations = np.zeros(self.input_size)
        self.hidden1_activations = np.zeros(self.hidden1_size)
        self.hidden2_activations = np.zeros(self.hidden2_size)
        self.output_scores = np.zeros(self.output_size)

    def activation_function(self, x):
        # Eenvoudige drempelactivatie: 1 als > 0, anders 0
        return np.where(x > 0, 1, 0)

    def process_input(self, sentence):
        """Activeer input woorden op basis van de zin (simuleer Input-processors)"""
        # Reset activaties
        self.input_activations = np.zeros(self.input_size)

        # Activeer input neuronen op basis van woorden in de zin
        for word in sentence.lower().split():
            if word in self.input_vocab:
                idx = self.input_vocab.index(word)
                self.input_activations[idx] = 1

        return self.input_activations

    def process_hidden1(self):
        """Bereken activaties voor Verborgen Laag 1 (simuleer VL1 neuronen)"""
        # Bereken inputs voor verborgen laag 1
        hidden1_inputs = np.dot(self.input_activations, self.weights_input_hidden1)

        # Pas activatiefunctie toe
        self.hidden1_activations = self.activation_function(hidden1_inputs)

        return self.hidden1_activations, hidden1_inputs

    def process_hidden2(self):
        """Bereken activaties voor Verborgen Laag 2 (simuleer VL2 neuronen)"""
        # Bereken inputs voor verborgen laag 2
        hidden2_inputs = np.dot(self.hidden1_activations, self.weights_hidden1_hidden2)

        # Pas activatiefunctie toe
        self.hidden2_activations = self.activation_function(hidden2_inputs)

        return self.hidden2_activations, hidden2_inputs

    def process_output(self):
        """Bereken scores voor output woorden (simuleer Output-processors)"""
        # Bereken output scores (geen activatiefunctie voor de output)
        self.output_scores = np.dot(self.hidden2_activations, self.weights_hidden2_output)

        # Zoek het woord met de hoogste score
        predicted_idx = np.argmax(self.output_scores)
        predicted_word = self.output_vocab[predicted_idx]

        return predicted_word, self.output_scores

    def forward_pass(self, sentence):
        """Voer een volledige forward pass uit door het netwerk"""
        self.process_input(sentence)
        self.process_hidden1()
        self.process_hidden2()
        return self.process_output()

    def backpropagation(self, correct_word, learning_rate=1):
        """Pas gewichten aan met backpropagation (simuleer Backpropagation-trainers)"""
        # Als de voorspelling correct is, hoeven de gewichten niet aangepast te worden
        predicted_idx = np.argmax(self.output_scores)
        predicted_word = self.output_vocab[predicted_idx]

        if predicted_word == correct_word:
            return True, []  # Correct, geen aanpassingen

        # Als incorrect, pas gewichten aan
        correct_idx = self.output_vocab.index(correct_word)

        # Lijst om bij te houden welke gewichten zijn aangepast
        weight_adjustments = []

        # Eenvoudige gewichtsaanpassing strategie:
        # 1. Versterk verbindingen van actieve VL2 neuronen naar correcte output
        # 2. Verzwak verbindingen van actieve VL2 neuronen naar incorrecte output
        for i in range(self.hidden2_size):
            if self.hidden2_activations[i] == 1:
                # Versterk verbinding naar correcte output
                old_weight = self.weights_hidden2_output[i, correct_idx]
                self.weights_hidden2_output[i, correct_idx] += learning_rate
                weight_adjustments.append((f"VL2-{i+1} → {correct_word}",
                                           old_weight,
                                           self.weights_hidden2_output[i, correct_idx]))

                # Verzwak verbinding naar incorrecte output
                old_weight = self.weights_hidden2_output[i, predicted_idx]
                self.weights_hidden2_output[i, predicted_idx] -= learning_rate
                weight_adjustments.append((f"VL2-{i+1} → {predicted_word}",
                                           old_weight,
                                           self.weights_hidden2_output[i, predicted_idx]))

        return False, weight_adjustments  # Incorrect, gewichten aangepast

    def print_network_state(self):
        """Print de huidige toestand van het netwerk (simuleer wat deelnemers zien)"""
        print("\n--- NETWERKSTATUS ---")

        # Input laag
        print("\nINPUT LAAG:")
        active_inputs = [f"'{self.input_vocab[i]}'" for i in range(self.input_size)
                       if self.input_activations[i] == 1]
        if active_inputs:
            print(f"Actieve woorden: {', '.join(active_inputs)}")
        else:
            print("Geen actieve woorden")

        # Verborgen laag 1
        print("\nVERBORGEN LAAG 1:")
        for i in range(self.hidden1_size):
            state = "ACTIEF (1)" if self.hidden1_activations[i] == 1 else "INACTIEF (0)"
            print(f"Neuron VL1-{i+1}: {state}")

        # Verborgen laag 2
        print("\nVERBORGEN LAAG 2:")
        for i in range(self.hidden2_size):
            state = "ACTIEF (1)" if self.hidden2_activations[i] == 1 else "INACTIEF (0)"
            print(f"Neuron VL2-{i+1}: {state}")

        # Output laag
        print("\nOUTPUT LAAG:")
        for i in range(self.output_size):
            print(f"'{self.output_vocab[i]}': Score {self.output_scores[i]:.1f}")

        predicted_idx = np.argmax(self.output_scores)
        print(f"\nVOORSPELLING: '{self.output_vocab[predicted_idx]}' (Score: {self.output_scores[predicted_idx]:.1f})")

    def set_strategic_weights(self):
        """Initialiseer enkele strategische gewichten om het leren te versnellen"""
        # Reset gewichten naar kleine waarden
        self.weights_input_hidden1 = np.random.uniform(-0.5, 0.5, self.weights_input_hidden1.shape)
        self.weights_hidden1_hidden2 = np.random.uniform(-0.5, 0.5, self.weights_hidden1_hidden2.shape)
        self.weights_hidden2_output = np.random.uniform(-0.5, 0.5, self.weights_hidden2_output.shape)

        # Patroon: "kat" + "zit" + "op" moet "mat" activeren
        kat_idx = self.input_vocab.index("kat")
        zit_idx = self.input_vocab.index("zit")
        op_idx = self.input_vocab.index("op")
        mat_idx = self.output_vocab.index("mat")

        # Maak "kat", "zit" en "op" sterke activeerders van VL1-1
        self.weights_input_hidden1[kat_idx, 0] = 3
        self.weights_input_hidden1[zit_idx, 0] = 2
        self.weights_input_hidden1[op_idx, 0] = 2

        # Maak VL1-1 een sterke activeerder van VL2-1
        self.weights_hidden1_hidden2[0, 0] = 3

        # Maak VL2-1 een sterke activeerder van "mat"
        self.weights_hidden2_output[0, mat_idx] = 3

        # Patroon: "kind" + "speelt" moet "bal" activeren
        kind_idx = self.input_vocab.index("kind")
        speelt_idx = self.input_vocab.index("speelt")
        bal_idx = self.output_vocab.index("bal")

        # Maak "kind" en "speelt" sterke activeerders van VL1-2
        self.weights_input_hidden1[kind_idx, 1] = 3
        self.weights_input_hidden1[speelt_idx, 1] = 2

        # Maak VL1-2 een sterke activeerder van VL2-2
        self.weights_hidden1_hidden2[1, 1] = 3

        # Maak VL2-2 een sterke activeerder van "bal"
        self.weights_hidden2_output[1, bal_idx] = 3

        print("Strategische gewichten geïnitialiseerd voor 'kat zit op' → 'mat' en 'kind speelt' → 'bal'")

    def simulate_game_rounds(self, training_data, learning_rate=1):
        """Simuleer een aantal ronden van het spel, zoals in de workshop"""
        print("\n===== WORDS HUMANS YIELD SPELSIMULATIE =====")
        print("\nTrainingsronden:")

        for round_num, (sentence, correct_word) in enumerate(training_data, 1):
            print(f"\n----- RONDE {round_num} -----")
            print(f"Trainingszin: '{sentence} ___' (Correct: '{correct_word}')")

            # Stap 1: Input-processors activeren woorden
            print("\nStap 1: Input-processors activeren woorden")
            self.process_input(sentence)
            active_words = [self.input_vocab[i] for i in range(self.input_size)
                           if self.input_activations[i] == 1]
            print(f"Actieve woorden: {', '.join(active_words)}")

            # Stap 2: Verborgen Laag 1 berekent activaties
            print("\nStap 2: Verborgen Laag 1 berekent activaties")
            activations, raw_inputs = self.process_hidden1()
            for i in range(self.hidden1_size):
                status = "ACTIEF (1)" if activations[i] == 1 else "INACTIEF (0)"
                print(f"VL1-{i+1}: Som = {raw_inputs[i]:.1f} → {status}")

            # Stap 3: Verborgen Laag 2 berekent activaties
            print("\nStap 3: Verborgen Laag 2 berekent activaties")
            activations, raw_inputs = self.process_hidden2()
            for i in range(self.hidden2_size):
                status = "ACTIEF (1)" if activations[i] == 1 else "INACTIEF (0)"
                print(f"VL2-{i+1}: Som = {raw_inputs[i]:.1f} → {status}")

            # Stap 4: Output-processors berekenen scores
            print("\nStap 4: Output-processors berekenen scores")
            predicted_word, scores = self.process_output()
            for i in range(self.output_size):
                print(f"'{self.output_vocab[i]}': Score {scores[i]:.1f}")

            print(f"\nVoorspelling: '{predicted_word}'")
            print(f"Correct antwoord: '{correct_word}'")

            # Stap 5: Backpropagation (indien nodig)
            is_correct, adjustments = self.backpropagation(correct_word, learning_rate)

            if is_correct:
                print("\nVoorspelling is CORRECT! Geen gewichtsaanpassingen nodig.")
            else:
                print("\nVoorspelling is INCORRECT. Backpropagation-trainers passen gewichten aan:")
                for connection, old_weight, new_weight in adjustments:
                    print(f"  {connection}: {old_weight:.1f} → {new_weight:.1f}")

                # Toon nieuwe voorspelling na aanpassingen
                print("\nNa gewichtsaanpassingen:")
                new_pred, _ = self.forward_pass(sentence)
                print(f"Nieuwe voorspelling zou zijn: '{new_pred}'")

            print("\n" + "=" * 50)

# Definieer trainingsdata
training_data = [
    ("de kat zit op de", "mat"),
    ("het kind speelt met de", "bal"),
    ("de hond ligt op de", "vloer"),
    ("de plant staat op de", "tafel"),
    ("het kind staat naast de", "deur"),
    ("de kat speelt in de", "tuin"),
    ("de grote hond zit onder de", "tafel"),
    ("het kind leest een", "boek")
]

# Definieer testdata
test_data = [
    ("de kleine kat ligt op de", "mat"),
    ("het kind staat op de", "stoel"),
    ("de rode auto staat naast de", "deur"),
    ("de hond speelt met de", "bal")
]

# Start de simulatie
def run_simulation():
    # Maak een nieuwe instantie van de simulatie
    simulation = WordsHumansYieldSimulation()

    # Optioneel: stel strategische gewichten in voor sneller leren
    simulation.set_strategic_weights()

    # 1. Toon voorbeeld van één voorspelling
    print("\n===== VOORBEELD VOORSPELLING =====")
    sentence = "de kat zit op de"
    prediction, _ = simulation.forward_pass(sentence)
    print(f"Input: '{sentence} ___'")
    print(f"Voorspelling: '{prediction}'")
    simulation.print_network_state()

    # 2. Simuleer een aantal spelronden
    print("\n\n===== SIMULATIE VAN SPELRONDEN =====")
    simulation.simulate_game_rounds(training_data[:4])  # Gebruik eerste 4 trainingsvoorbeelden

    # 3. Test het netwerk op nieuwe zinnen
    print("\n===== TESTEN VAN HET NETWERK =====")
    print("Nu testen we het netwerk op nieuwe zinnen die het nog niet heeft gezien:")

    for sentence, correct_word in test_data:
        predicted_word, _ = simulation.forward_pass(sentence)
        is_correct = "✓ CORRECT" if predicted_word == correct_word else "✗ INCORRECT"

        print(f"\nInput: '{sentence} ___'")
        print(f"Voorspelling: '{predicted_word}' (Correct: '{correct_word}') {is_correct}")

        # Toon gewichten en activaties voor kat+zit+op → mat voorbeeld indien aanwezig
        if "kat" in sentence and ("zit" in sentence or "ligt" in sentence):
            kat_idx = simulation.input_vocab.index("kat")
            vl1_1_idx = 0  # VL1-1
            vl2_1_idx = 0  # VL2-1
            mat_idx = simulation.output_vocab.index("mat")

            print("\nRelevante gewichten voor 'kat+zit+op→mat' patroon:")
            print(f"Gewicht 'kat'→VL1-1: {simulation.weights_input_hidden1[kat_idx, vl1_1_idx]:.1f}")
            print(f"Gewicht VL1-1→VL2-1: {simulation.weights_hidden1_hidden2[vl1_1_idx, vl2_1_idx]:.1f}")
            print(f"Gewicht VL2-1→'mat': {simulation.weights_hidden2_output[vl2_1_idx, mat_idx]:.1f}")

# Voer de simulatie uit
run_simulation()
