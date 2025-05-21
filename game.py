import numpy as np


class WordsHumansYieldSimulation:
    def __init__(self):
        # Define the vocabularies
        self.input_vocab = [
            "he",
            "the",
            "a",
            "on",
            "in",
            "under",
            "next to",
            "cat",
            "dog",
            "child",
            "plant",
            "car",
            "sits",
            "stands",
            "lies",
            "plays",
            "walks",
            "big",
            "small",
            "red",
        ]

        self.output_vocab = [
            "mat",
            "table",
            "chair",
            "floor",
            "door",
            "garden",
            "book",
            "ball",
            "cabinet",
            "lamp",
        ]

        # Network dimensions
        self.input_size = len(self.input_vocab)
        self.hidden1_size = 5  # Hidden Layer 1
        self.hidden2_size = 5  # Hidden Layer 2
        self.output_size = len(self.output_vocab)

        # Initialize weights with random small values between -2 and +2
        self.weights_input_hidden1 = np.random.uniform(
            -2, 2, (self.input_size, self.hidden1_size)
        )
        self.weights_hidden1_hidden2 = np.random.uniform(
            -2, 2, (self.hidden1_size, self.hidden2_size)
        )
        self.weights_hidden2_output = np.random.uniform(
            -2, 2, (self.hidden2_size, self.output_size)
        )

        # For storing activations during network use
        self.input_activations = np.zeros(self.input_size)
        self.hidden1_activations = np.zeros(self.hidden1_size)
        self.hidden2_activations = np.zeros(self.hidden2_size)
        self.output_scores = np.zeros(self.output_size)

    def activation_function(self, x):
        # Simple threshold activation: 1 if > 0, otherwise 0
        return np.where(x > 0, 1, 0)

    def process_input(self, sentence):
        """Activate input words based on the sentence (simulate Input processors)"""
        # Reset activations
        self.input_activations = np.zeros(self.input_size)

        # Activate input neurons based on words in the sentence
        for word in sentence.lower().split():
            if word in self.input_vocab:
                idx = self.input_vocab.index(word)
                self.input_activations[idx] = 1

        return self.input_activations

    def process_hidden1(self):
        """Calculate activations for Hidden Layer 1 (simulate HL1 neurons)"""
        # Calculate inputs for hidden layer 1
        hidden1_inputs = np.dot(self.input_activations, self.weights_input_hidden1)

        # Apply activation function
        self.hidden1_activations = self.activation_function(hidden1_inputs)

        return self.hidden1_activations, hidden1_inputs

    def process_hidden2(self):
        """Calculate activations for Hidden Layer 2 (simulate HL2 neurons)"""
        # Calculate inputs for hidden layer 2
        hidden2_inputs = np.dot(self.hidden1_activations, self.weights_hidden1_hidden2)

        # Apply activation function
        self.hidden2_activations = self.activation_function(hidden2_inputs)

        return self.hidden2_activations, hidden2_inputs

    def process_output(self):
        """Calculate scores for output words (simulate Output processors)"""
        # Calculate output scores (no activation function for the output)
        self.output_scores = np.dot(
            self.hidden2_activations, self.weights_hidden2_output
        )

        # Find the word with the highest score
        predicted_idx = np.argmax(self.output_scores)
        predicted_word = self.output_vocab[predicted_idx]

        return predicted_word, self.output_scores

    def forward_pass(self, sentence):
        """Perform a complete forward pass through the network"""
        self.process_input(sentence)
        self.process_hidden1()
        self.process_hidden2()
        return self.process_output()

    def backpropagation(self, correct_word, learning_rate=1):
        """Adjust weights with backpropagation (simulate Backpropagation trainers)"""
        # If the prediction is correct, no need to adjust weights
        predicted_idx = np.argmax(self.output_scores)
        predicted_word = self.output_vocab[predicted_idx]

        if predicted_word == correct_word:
            return True, []  # Correct, no adjustments

        # If incorrect, adjust weights
        correct_idx = self.output_vocab.index(correct_word)

        # List to track which weights were adjusted
        weight_adjustments = []

        # Simple weight adjustment strategy:
        # 1. Strengthen connections from active HL2 neurons to correct output
        # 2. Weaken connections from active HL2 neurons to incorrect output
        for i in range(self.hidden2_size):
            if self.hidden2_activations[i] == 1:
                # Strengthen connection to correct output
                old_weight = self.weights_hidden2_output[i, correct_idx]
                self.weights_hidden2_output[i, correct_idx] += learning_rate
                weight_adjustments.append(
                    (
                        f"HL2-{i + 1} → {correct_word}",
                        old_weight,
                        self.weights_hidden2_output[i, correct_idx],
                    )
                )

                # Weaken connection to incorrect output
                old_weight = self.weights_hidden2_output[i, predicted_idx]
                self.weights_hidden2_output[i, predicted_idx] -= learning_rate
                weight_adjustments.append(
                    (
                        f"HL2-{i + 1} → {predicted_word}",
                        old_weight,
                        self.weights_hidden2_output[i, predicted_idx],
                    )
                )

        return False, weight_adjustments  # Incorrect, weights adjusted

    def print_network_state(self):
        """Print the current state of the network (simulate what participants see)"""
        print("\n--- NETWORK STATUS ---")

        # Input layer
        print("\nINPUT LAYER:")
        active_inputs = [
            f"'{self.input_vocab[i]}'"
            for i in range(self.input_size)
            if self.input_activations[i] == 1
        ]
        if active_inputs:
            print(f"Active words: {', '.join(active_inputs)}")
        else:
            print("No active words")

        # Hidden layer 1
        print("\nHIDDEN LAYER 1:")
        for i in range(self.hidden1_size):
            state = "ACTIVE (1)" if self.hidden1_activations[i] == 1 else "INACTIVE (0)"
            print(f"Neuron HL1-{i + 1}: {state}")

        # Hidden layer 2
        print("\nHIDDEN LAYER 2:")
        for i in range(self.hidden2_size):
            state = "ACTIVE (1)" if self.hidden2_activations[i] == 1 else "INACTIVE (0)"
            print(f"Neuron HL2-{i + 1}: {state}")

        # Output layer
        print("\nOUTPUT LAYER:")
        for i in range(self.output_size):
            print(f"'{self.output_vocab[i]}': Score {self.output_scores[i]:.1f}")

        predicted_idx = np.argmax(self.output_scores)
        print(
            f"\nPREDICTION: '{self.output_vocab[predicted_idx]}' (Score: {self.output_scores[predicted_idx]:.1f})"
        )

    def set_strategic_weights(self):
        """Initialize some strategic weights to accelerate learning"""
        # Reset weights to small values
        self.weights_input_hidden1 = np.random.uniform(
            -0.5, 0.5, self.weights_input_hidden1.shape
        )
        self.weights_hidden1_hidden2 = np.random.uniform(
            -0.5, 0.5, self.weights_hidden1_hidden2.shape
        )
        self.weights_hidden2_output = np.random.uniform(
            -0.5, 0.5, self.weights_hidden2_output.shape
        )

        # Pattern: "cat" + "sits" + "on" should activate "mat"
        cat_idx = self.input_vocab.index("cat")
        sits_idx = self.input_vocab.index("sits")
        on_idx = self.input_vocab.index("on")
        mat_idx = self.output_vocab.index("mat")

        # Make "cat", "sits" and "on" strong activators of HL1-1
        self.weights_input_hidden1[cat_idx, 0] = 3
        self.weights_input_hidden1[sits_idx, 0] = 2
        self.weights_input_hidden1[on_idx, 0] = 2

        # Make HL1-1 a strong activator of HL2-1
        self.weights_hidden1_hidden2[0, 0] = 3

        # Make HL2-1 a strong activator of "mat"
        self.weights_hidden2_output[0, mat_idx] = 3

        # Pattern: "child" + "plays" should activate "ball"
        child_idx = self.input_vocab.index("child")
        plays_idx = self.input_vocab.index("plays")
        ball_idx = self.output_vocab.index("ball")

        # Make "child" and "plays" strong activators of HL1-2
        self.weights_input_hidden1[child_idx, 1] = 3
        self.weights_input_hidden1[plays_idx, 1] = 2

        # Make HL1-2 a strong activator of HL2-2
        self.weights_hidden1_hidden2[1, 1] = 3

        # Make HL2-2 a strong activator of "ball"
        self.weights_hidden2_output[1, ball_idx] = 3

        print(
            "Strategic weights initialized for 'cat sits on' → 'mat' and 'child plays' → 'ball'"
        )

    def simulate_game_rounds(self, training_data, learning_rate=1):
        """Simulate multiple rounds of the game, as in the workshop"""
        print("\n===== WORDS HUMANS YIELD GAME SIMULATION =====")
        print("\nTraining rounds:")

        for round_num, (sentence, correct_word) in enumerate(training_data, 1):
            print(f"\n----- ROUND {round_num} -----")
            print(f"Training sentence: '{sentence} ___' (Correct: '{correct_word}')")

            # Step 1: Input processors activate words
            print("\nStep 1: Input processors activate words")
            self.process_input(sentence)
            active_words = [
                self.input_vocab[i]
                for i in range(self.input_size)
                if self.input_activations[i] == 1
            ]
            print(f"Active words: {', '.join(active_words)}")

            # Step 2: Hidden Layer 1 calculates activations
            print("\nStep 2: Hidden Layer 1 calculates activations")
            activations, raw_inputs = self.process_hidden1()
            for i in range(self.hidden1_size):
                status = "ACTIVE (1)" if activations[i] == 1 else "INACTIVE (0)"
                print(f"HL1-{i + 1}: Sum = {raw_inputs[i]:.1f} → {status}")

            # Step 3: Hidden Layer 2 calculates activations
            print("\nStep 3: Hidden Layer 2 calculates activations")
            activations, raw_inputs = self.process_hidden2()
            for i in range(self.hidden2_size):
                status = "ACTIVE (1)" if activations[i] == 1 else "INACTIVE (0)"
                print(f"HL2-{i + 1}: Sum = {raw_inputs[i]:.1f} → {status}")

            # Step 4: Output processors calculate scores
            print("\nStep 4: Output processors calculate scores")
            predicted_word, scores = self.process_output()
            for i in range(self.output_size):
                print(f"'{self.output_vocab[i]}': Score {scores[i]:.1f}")

            print(f"\nPrediction: '{predicted_word}'")
            print(f"Correct answer: '{correct_word}'")

            # Step 5: Backpropagation (if needed)
            is_correct, adjustments = self.backpropagation(correct_word, learning_rate)

            if is_correct:
                print("\nPrediction is CORRECT! No weight adjustments needed.")
            else:
                print(
                    "\nPrediction is INCORRECT. Backpropagation trainers adjust weights:"
                )
                for connection, old_weight, new_weight in adjustments:
                    print(f"  {connection}: {old_weight:.1f} → {new_weight:.1f}")

                # Show new prediction after adjustments
                print("\nAfter weight adjustments:")
                new_pred, _ = self.forward_pass(sentence)
                print(f"New prediction would be: '{new_pred}'")

            print("\n" + "=" * 50)


# Define training data
training_data = [
    ("the cat sits on the", "mat"),
    ("the child plays with the", "ball"),
    ("the dog lies on the", "floor"),
    ("the plant stands on the", "table"),
    ("the child stands next to the", "door"),
    ("the cat plays in the", "garden"),
    ("the big dog sits under the", "table"),
    ("the child reads a", "book"),
]

# Define test data
test_data = [
    ("the small cat lies on the", "mat"),
    ("the child stands on the", "chair"),
    ("the red car stands next to the", "door"),
    ("the dog plays with the", "ball"),
]


# Start the simulation
def run_simulation():
    # Create a new instance of the simulation
    simulation = WordsHumansYieldSimulation()

    # Optional: set strategic weights for faster learning
    simulation.set_strategic_weights()

    # 1. Show example of one prediction
    print("\n===== EXAMPLE PREDICTION =====")
    sentence = "the cat sits on the"
    prediction, _ = simulation.forward_pass(sentence)
    print(f"Input: '{sentence} ___'")
    print(f"Prediction: '{prediction}'")
    simulation.print_network_state()

    # 2. Simulate several game rounds
    print("\n\n===== SIMULATION OF GAME ROUNDS =====")
    simulation.simulate_game_rounds(training_data[:4])  # Use first 4 training examples

    # 3. Test the network on new sentences
    print("\n===== TESTING THE NETWORK =====")
    print("Now we test the network on new sentences it hasn't seen before:")

    for sentence, correct_word in test_data:
        predicted_word, _ = simulation.forward_pass(sentence)
        is_correct = "✓ CORRECT" if predicted_word == correct_word else "✗ INCORRECT"

        print(f"\nInput: '{sentence} ___'")
        print(
            f"Prediction: '{predicted_word}' (Correct: '{correct_word}') {is_correct}"
        )

        # Show weights and activations for cat+sits+on → mat example if present
        if "cat" in sentence and ("sits" in sentence or "lies" in sentence):
            cat_idx = simulation.input_vocab.index("cat")
            hl1_1_idx = 0  # HL1-1
            hl2_1_idx = 0  # HL2-1
            mat_idx = simulation.output_vocab.index("mat")

            print("\nRelevant weights for 'cat+sits+on→mat' pattern:")
            print(
                f"Weight 'cat'→HL1-1: {simulation.weights_input_hidden1[cat_idx, hl1_1_idx]:.1f}"
            )
            print(
                f"Weight HL1-1→HL2-1: {simulation.weights_hidden1_hidden2[hl1_1_idx, hl2_1_idx]:.1f}"
            )
            print(
                f"Weight HL2-1→'mat': {simulation.weights_hidden2_output[hl2_1_idx, mat_idx]:.1f}"
            )


# Run the simulation
run_simulation()
