import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# USER INPUT VARIABLES
# ==========================================
N_NEURONS = 128          # Number of neurons (Size of the network)
MAX_PATTERNS = 30        # Maximum number of patterns to test up to
NUM_TRIALS = 15          # How many times to repeat (smoothing)
THEORETICAL_RATIO = 0.15 # Theoretical capacity constant (~0.15 * N)

# ==========================================
# HOPFIELD NETWORK CLASS
# ==========================================
class HopfieldNetwork:
    def __init__(self, n_neurons):     # FIXED
        self.n_neurons = n_neurons
        self.weights = np.zeros((n_neurons, n_neurons))

    def train(self, patterns):
        n_patterns, n_features = patterns.shape
        
        # Hebbian Rule: W = P.T * P / N
        self.weights = np.dot(patterns.T, patterns) / self.n_neurons
        
        # Remove self-connections
        np.fill_diagonal(self.weights, 0)

    def recall(self, pattern, steps=5):
        state = pattern.copy()
        for _ in range(steps):
            activation = np.dot(self.weights, state)
            state = np.where(activation >= 0, 1, -1)
        return state

# ==========================================
# EXPERIMENT LOGIC
# ==========================================
def run_capacity_experiment():
    print(f"Running Capacity Test for N={N_NEURONS} neurons...")

    x_values = range(1, MAX_PATTERNS + 1)
    y_values = []

    for num_patterns in x_values:
        trial_accuracies = []

        for _ in range(NUM_TRIALS):
            patterns = np.random.choice([-1, 1], size=(num_patterns, N_NEURONS))

            net = HopfieldNetwork(N_NEURONS)
            net.train(patterns)

            success = 0
            for p in patterns:
                recalled = net.recall(p)
                if np.array_equal(p, recalled):
                    success += 1

            trial_accuracies.append(success / num_patterns)

        y_values.append(np.mean(trial_accuracies))

    return x_values, y_values

# ==========================================
# PLOTTING
# ==========================================
def plot_results(x, y):
    theoretical_capacity = int(THEORETICAL_RATIO * N_NEURONS)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linewidth=2, label='Recall Success Rate')

    plt.axvline(x=theoretical_capacity, linestyle='--', linewidth=2, color='r',
                label=f'Theoretical Capacity ({theoretical_capacity})')

    plt.xlabel("Number of Stored Patterns")
    plt.ylabel("Recall Success Rate")
    plt.title("Hopfield Network Capacity Test")
    plt.grid(True)
    plt.legend()
    plt.ylim(-0.05, 1.05)
    plt.show()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":   # FIXED
    x_data, y_data = run_capacity_experiment()
    plot_results(x_data, y_data)
