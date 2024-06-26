import numpy as np

class MultiClassPerceptron:
    def __init__(self, learning_rate, epochs, vocab):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.vocab = vocab
        self.weights = []
        self.accuracy_dict = {'Epoch': [], 'Training Accuracy %': [], 'Test Accuracy %': []}

