import numpy as np
import random
import string
import json
import os
import tkinter as tk
from tkinter import messagebox
import threading

# Constants and Parameters
BRAIN_SIZE = 100000
BRAIN_SYNAPSE_COUNT = 1000000
LEARNING_RATE = 0.1
MAX_WORD_LENGTH = 25
TRAINING_CYCLES = 100000
stop_training = False

# Neuron class
class Neuron:
    def __init__(self, id):
        self.id = id
        self.threshold = random.uniform(0.1, 0.9)
        self.input_weights = []
        self.output_weights = []
        self.input_signal = 0
        self.output_signal = 0
        
    def receive_input_signal(self, signal):
        self.input_signal += signal
        
    def calculate_output_signal(self):
        self.output_signal = 1 / (1 + np.exp(-self.input_signal))  # Sigmoid activation function
        
    def fire(self):
        for neuron, weight in self.output_weights:
            neuron.receive_input_signal(weight * self.output_signal)
        self.input_signal = 0  # Reset input signal after firing

# Synapse class
class Synapse:
    def __init__(self, input_neuron, output_neuron):
        self.input_neuron = input_neuron
        self.output_neuron = output_neuron
        self.weight = random.uniform(0.1, 0.9)
        
    def update_weight(self):
        if self.input_neuron.output_signal > 0:
            self.weight += LEARNING_RATE * self.input_neuron.output_signal * (1 - self.output_neuron.output_signal)

# Helper functions
def generate_word(length):
    return ''.join(random.choice(string.ascii_lowercase) for i in range(length))

def get_neuron_index(char):
    char_to_index = ord(char) - ord('a')
    neuron_index = char_to_index % BRAIN_SIZE
    return neuron_index

def feed_data_to_brain(data, brain_neurons, brain_synapses):
    for word in data:
        for char in word.lower():
            if char.isalpha():
                neuron_index = get_neuron_index(char)
                brain_neurons[neuron_index].receive_input_signal(1)
    
        for neuron in brain_neurons:
            neuron.calculate_output_signal()
            if neuron.output_signal > 0:
                neuron.fire()
        for neuron in brain_neurons:
            neuron.input_signal = 0  # Reset input signal for all neurons

        # Update synapse weights
        for synapse in brain_synapses:
            synapse.update_weight()

# Function to save the brain state to a JSON file
def save_brain_state(brain_neurons, brain_synapses, file_path='seed.json'):
    try:
        neuron_state = [{'id': neuron.id, 'threshold': neuron.threshold, 'output_signal': neuron.output_signal} for neuron in brain_neurons]
        synapse_state = [{'input_neuron': synapse.input_neuron.id, 'output_neuron': synapse.output_neuron.id, 'weight': synapse.weight} for synapse in brain_synapses]
        
        with open(file_path, 'w') as f:
            json.dump({'neurons': neuron_state, 'synapses': synapse_state}, f, indent=4)
        print(f"Brain state saved to {file_path}")
    except IOError as e:
        print(f"An error occurred while saving the brain state: {e}")
def load_brain_state(file_path='seed.json'):
    with open(file_path, 'r') as f:
        state = json.load(f)
    
    # Reconstruct neurons
    brain_neurons = [Neuron(neuron['id']) for neuron in state['neurons']]
    for neuron, saved_state in zip(brain_neurons, state['neurons']):
        neuron.threshold = saved_state['threshold']
        neuron.output_signal = saved_state['output_signal']
    
    # Reconstruct synapses
    brain_synapses = []
    for saved_state in state['synapses']:
        input_neuron = brain_neurons[saved_state['input_neuron']]
        output_neuron = brain_neurons[saved_state['output_neuron']]
        synapse = Synapse(input_neuron, output_neuron)
        synapse.weight = saved_state['weight']
        brain_synapses.append(synapse)
        input_neuron.output_weights.append((output_neuron, synapse.weight))
        output_neuron.input_weights.append((input_neuron, synapse.weight))
    
    return brain_neurons, brain_synapses

def generate_response(brain_neurons, max_length=MAX_WORD_LENGTH):
    response_words = [generate_word(random.randint(1, max_length))]
    for _ in range(random.randint(1, max_length) - 1):
        last_word = response_words[-1]
        last_char = last_word[-1]
        neuron_index = get_neuron_index(last_char)
        last_neuron = brain_neurons[neuron_index]
        if last_neuron.output_weights:
            next_neuron = max(last_neuron.output_weights, key=lambda x: x[1])[0]
            next_char_index = (next_neuron.id % 26) + ord('a')
            next_char = chr(next_char_index)
            next_word = generate_word(random.randint(1, max_length))
            next_word = next_char + next_word[1:]
            response_words.append(next_word)
    response = ' '.join(response_words)
    total_activation = calculate_total_activation(brain_neurons)
    return response, total_activation


def initialize_brain():
    choice = input("Do you want to load an existing brain state from 'seed.json'? (yes/no): ").strip().lower()
    if choice == 'yes':
        try:
            return load_brain_state()
        except FileNotFoundError:
            print("No existing brain state found. Creating a new one.")
    
    # Initialize a new brain with neurons and synapses
    brain_neurons = [Neuron(i) for i in range(BRAIN_SIZE)]
    brain_synapses = []

    for _ in range(BRAIN_SYNAPSE_COUNT):
        input_neuron = random.choice(brain_neurons)
        output_neuron = random.choice(brain_neurons)
        synapse = Synapse(input_neuron, output_neuron)
        brain_synapses.append(synapse)
        input_neuron.output_weights.append((output_neuron, synapse.weight))
        output_neuron.input_weights.append((input_neuron, synapse.weight))
    
    return brain_neurons, brain_synapses

def calculate_total_activation(brain_neurons):
    return sum(neuron.output_signal for neuron in brain_neurons)

# Function to automate the training process
def train_brain(brain_neurons, brain_synapses, cycles=TRAINING_CYCLES):
    global stop_training
    for _ in range(cycles):
        if stop_training:
            print("Training stopped by user.")
            break

        random_word = generate_word(random.randint(1, MAX_WORD_LENGTH))
        feed_data_to_brain([random_word], brain_neurons, brain_synapses)

# Function to stop the training
def stop_training_process():
    global stop_training
    stop_training = True

def update_stats(brain_neurons, brain_synapses, stats_text, best_response):
    active_neurons = len([neuron for neuron in brain_neurons if neuron.output_signal > neuron.threshold])
    average_weight = sum([synapse.weight for synapse in brain_synapses]) / len(brain_synapses)
    max_weight = max(synapse.weight for synapse in brain_synapses)
    min_weight = min(synapse.weight for synapse in brain_synapses)
    max_neuron_output = max(neuron.output_signal for neuron in brain_neurons)
    min_neuron_output = min(neuron.output_signal for neuron in brain_neurons)
    
    stats = (
        f"Total Synapses: {len(brain_synapses)}",
        f"Active Neurons: {active_neurons}",
        f"Average Synapse Weight: {average_weight:.3f}",
        f"Max Synapse Weight: {max_weight:.3f}",
        f"Min Synapse Weight: {min_weight:.3f}",
        f"Max Neuron Output: {max_neuron_output:.3f}",
        f"Min Neuron Output: {min_neuron_output:.3f}",
        f"Best Response: {best_response['response']}",
        f"Activation: {best_response['activation']:.3f}"
    )
    stats_text.set("\n".join(stats))


def start_gui_in_thread(brain_neurons, brain_synapses, best_response):
    # Start the GUI in a separate thread and pass the best_response
    gui_thread = threading.Thread(target=initialize_gui, args=(brain_neurons, brain_synapses, best_response))
    gui_thread.daemon = True  # Make sure the thread will not prevent the program from exiting
    gui_thread.start()

def initialize_gui(brain_neurons, brain_synapses, best_response):
    root = tk.Tk()
    root.title("Brain Stats Monitor")
    stats_text = tk.StringVar()
    update_stats(brain_neurons, brain_synapses, stats_text, best_response)
    tk.Label(root, textvariable=stats_text, justify=tk.LEFT).pack()

    # Button to stop the training
    tk.Button(root, text="Stop Training", command=stop_training_process).pack()

    def update_gui_stats():
        update_stats(brain_neurons, brain_synapses, stats_text, best_response)
        root.after(1000, update_gui_stats)
    
    root.after(1000, update_gui_stats)
    tk.Button(root, text="Exit", command=root.destroy).pack()
    root.mainloop()

# Store the best response at the global level
best_response = {
    'response': '',
    'activation': -np.inf
}

def interactive_brain_simulation():
    brain_neurons, brain_synapses = initialize_brain()
    start_gui_in_thread(brain_neurons, brain_synapses, best_response)
    print("Interactive brain simulation started.")
    while True:
        user_input = input("Enter command (train/chat/exit): ").strip().lower()
        if user_input == 'exit':
            save_brain_state(brain_neurons, brain_synapses)
            os._exit(0)
        elif user_input == 'train':
            train_brain(brain_neurons, brain_synapses)
            print("Training completed.")
        elif user_input == 'chat':
            message = input("You: ").strip()
            words = message.split()
            feed_data_to_brain(words, brain_neurons, brain_synapses)
            response, activation = generate_response(brain_neurons)
            print(f"Brain: {response}")
            if activation > best_response['activation']:
                best_response['response'] = response
                best_response['activation'] = activation
                print(f"New best response with activation {activation:.3f}: {response}")
        else:
            print("Invalid command.")

if __name__ == '__main__':
    interactive_brain_simulation()
