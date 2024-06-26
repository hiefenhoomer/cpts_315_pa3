import copy
from perceptrons import Perceptron
import numpy as np


class AveragedPerceptron(Perceptron.Perceptron):
    def __init__(self, learning_rate, vocab):
        super().__init__(learning_rate, vocab)
        # We need cumulative weights to average out here.
        self.cumulative_weights = copy.deepcopy(self.weights)
        # Need to keep count of the iteration to average weights.
        self.counter = 0

    # Adjust weights should increment the counter and add to the cumulative weights.
    def adjust_weights(self, feature_vector, correction_factor):
        super().adjust_weights(feature_vector, correction_factor)
        # Add the cumulative weights.
        self.cumulative_weights = np.add(self.cumulative_weights, copy.deepcopy(self.weights))
        # Increment count.
        self.counter += 1

    def get_final_weights(self):
        # Get the averaged weights.
        return self.cumulative_weights / self.counter

    def train(self, processed_training_path, training_labels_path, processed_test_path, test_labels_path, epochs):
        # Call the super class.
        accuracy_dict = super().train(processed_training_path, training_labels_path, processed_test_path,
                                      test_labels_path, epochs)
        self.weights = self.get_final_weights()
        averaged_training = self.check_accuracy(processed_training_path, training_labels_path)
        averaged_test = self.check_accuracy(processed_test_path, test_labels_path)
        accuracy_dict['Epoch'].append('Averaged Perceptron')
        accuracy_dict['Training Accuracy %'].append(averaged_training)
        accuracy_dict['Test Accuracy %'].append(averaged_test)

        return accuracy_dict
