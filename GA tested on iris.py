from sklearn.datasets import load_iris
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def forward_propagation(X, weights, biases):
    """
    Performs forward propagation through an MLP with predefined weights and biases.
    """
    with torch.no_grad():
        model.fc1.weight = nn.Parameter(torch.tensor(weights[0], dtype=torch.float32))
        model.fc1.bias = nn.Parameter(torch.tensor(biases[0], dtype=torch.float32))
        model.fc2.weight = nn.Parameter(torch.tensor(weights[1], dtype=torch.float32))
        model.fc2.bias = nn.Parameter(torch.tensor(biases[1], dtype=torch.float32))
        model.fc3.weight = nn.Parameter(torch.tensor(weights[2], dtype=torch.float32))
        model.fc3.bias = nn.Parameter(torch.tensor(biases[2], dtype=torch.float32))

    outputs = model(X)
    return outputs


def initialize_population(pop_size, layers):
    """Initialize a population of weight matrices and biases."""
    population = []
    for _ in range(pop_size):
        individual = {
            "weights": [np.random.randn(layers[i+1], layers[i]) for i in range(len(layers)-1)],
            "biases": [np.random.randn(layers[i+1]) for i in range(len(layers)-1)]
        }
        population.append(individual)
    return population

def fitness_function(individual, X, y):
    """Calculate fitness as inverse error."""
    output = forward_propagation(X, individual["weights"], individual["biases"])
    # print(np.shape(output))
    true_labels = y  # assumed to be LongTensor of class indices

#-----------------------Log Loss--------------------------------------
    # # Gather the log-probs for the correct classes
    # log_probs = output[range(len(true_labels)), true_labels]

    # # Negative log likelihood
    # error = -log_probs.mean()
    # print("Manual NLL Loss:", error.item())
#---------------------------------------------------

    error = criterion(output, y)
    # print("Manual NLL Loss:",error)
    out = 1 / (1 + (error.item()))# sigmoid like function
    # print("Manual NLL Loss:",out)
    return out

def select_parents(population, fitnesses, num_parents):
    """Select best individuals based on fitness."""
    sorted_indices = np.argsort(fitnesses)[::-1]
    # print(fitnesses[::-1])
    return [population[i] for i in sorted_indices[:num_parents]]

def crossover(parent1, parent2):
    """Perform exchange crossover between two parents."""
    child = {"weights": [], "biases": []}
    for w1, w2, b1, b2 in zip(parent1["weights"], parent2["weights"], parent1["biases"], parent2["biases"]):
        mask = np.random.rand(*w1.shape) > 0.5  # Random binary mask
        w_child = np.where(mask, w1, w2)  # Exchange crossover
        b_mask = np.random.rand(*b1.shape) > 0.5
        b_child = np.where(b_mask, b1, b2)
        child["weights"].append(w_child)
        child["biases"].append(b_child)
    return child

def mutate(individual, mutation_rate=0.1):
    """Mutate by completely replacing a randomly chosen weight or bias with a new random value."""
    for i in range(len(individual["weights"])):
        if np.random.rand() < mutation_rate:
            # Choose a random position in the weight matrix
            w_shape = individual["weights"][i].shape
            rand_w_index = tuple(np.random.randint(dim) for dim in w_shape)
            # Replace with a new random value (e.g., from normal distribution)
            individual["weights"][i][rand_w_index] = np.random.randn()
            
            # Same for bias
            b_shape = individual["biases"][i].shape
            rand_b_index = tuple(np.random.randint(dim) for dim in b_shape)
            individual["biases"][i][rand_b_index] = np.random.randn()
    
    return individual


def genetic_algorithm(X, y, layers, pop_size=25, generations=1, mutation_rate=0.1):
    """Optimize MLP weights using Genetic Algorithm."""
    population = initialize_population(pop_size, layers)
    for gen in range(generations):
        fitnesses = [fitness_function(ind, X, y) for ind in population]
        parents = select_parents(population, fitnesses, num_parents=pop_size//2)
        children = [crossover(parents[i], parents[i+1]) for i in range(0, len(parents)-1, 2)]
        children = [mutate(child, mutation_rate) for child in children]
        population = parents + children  # New generation
        # print(fitnesses)
    best_individual = max(population, key=lambda ind: fitness_function(ind, X, y))
    print(fitness_function(best_individual, X, y))
    return best_individual["weights"], best_individual["biases"]


# Define categories
CATEGORIES = ['agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings', 'chaparral', 'denseresidential',
              'forest', 'freeway', 'golfcourse', 'harbor', 'intersection', 'mediumresidential', 'mobilehomepark',
              'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential', 'storagetanks', 'tenniscourt']
CATE2ID = {v: k for k, v in enumerate(CATEGORIES)}

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

iris = load_iris()

# Features and labels
X = iris.data            # shape: (150, 4)
y = iris.target          # shape: (150,)

print("Data Loaded")
# Convert to PyTorch tensors
train_X = torch.tensor(train_image_feats.astype(np.float32))
test_X = torch.tensor(test_image_feats.astype(np.float32))

# Convert labels to integer values based on CATE2ID mapping
train_y = np.array([y[label] for label in train_y])  # Convert string labels to integers
test_y = np.array([y[label] for label in test_y])  # Convert string labels to integers

train_y = torch.tensor(train_y, dtype=torch.long)
test_y = torch.tensor(test_y, dtype=torch.long)

# Define the MLP Model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes=21):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

layers = [150, 128, 64, 21]

# Instantiate and move the model to the device
model = MLP(input_size=layers[0], hidden_size1=layers[1], hidden_size2=layers[2], num_classes=len(CATEGORIES)).to(device)
criterion = nn.CrossEntropyLoss()


# Example MLP with 2 input neurons, 3 hidden neurons, and 1 output neuron
X = train_X #  input
y = train_y  #  expected output


for i in range(0,5):
  weights, biases = genetic_algorithm(X, y, layers)

  # Activation functions for each layer
  # activation_functions = [relu, sigmoid]

  # Forward pass
  output = forward_propagation(X, weights, biases)
  # print(weights)

  _, predicted = torch.max(output, 1)
  # print(np.shape(predicted))

  correct, total = 0, 0

  total = y.size(0)
  correct = (predicted == y).sum().item()

  test_accuracy = correct / total*100
  print(f"Test Accuracy: {test_accuracy:.4f}")
