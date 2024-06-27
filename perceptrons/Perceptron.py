import numpy as np


class Perceptron:
    def __init__(self, learning_rate, vocab):
        # Initialize the weights to extend the length of the vocabulary.
        self.weights = np.zeros(len(vocab))
        # The vocabulary is a dictionary of words corresponding to their indices alphabetically.
        self.vocab = vocab
        # Learning rate.
        self.learning_rate = learning_rate
        # Useful for formatting
        self.accuracy_dict = {'Epoch': [], 'Training Accuracy %': [], 'Test Accuracy %': [], 'Training Failed': [], 'Test Failed': []}

    def line_to_feature_vector(self, line):
        # Feature vector should be initialized to be the length of the vocabulary.
        feature_vector = np.zeros(len(self.vocab))
        line = line.strip()
        words = line.split(' ')
        for word in words:
            # Get the index each word belongs in from the dictionary.
            index = self.vocab.get(word)
            # Add 1 to the frequency of this word to the feature_vector.
            feature_vector[index] += 1
        return feature_vector

    def predict(self, feature_vector):
        # np.sign returns -1, 0, or 1.
        result = np.sign(np.dot(feature_vector, self.weights))
        # Need to ensure a positive or zero result otherwise the prediction will be incorrect.
        return 1 if result > 0 else 0

    def adjust_weights(self, feature_vector, correction_factor):
        # w + η · yt · xt - in this case n is the learning rate, yt is our actual value, and xt is our feature vector.
        self.weights = np.add(self.weights, self.learning_rate * correction_factor * feature_vector)

    def train(self, processed_training_path, training_labels_path, processed_test_path, test_labels_path, epochs):
        # For each epoch
        for epoch in range(epochs):
            # Keep count of correct results and incorrect results.
            correct_count = 0
            total_count = 0
            incorrect_count = 0
            # Open label and training files.
            with open(processed_training_path, 'r') as processed_file, open(training_labels_path, 'r') as labels_file:
                # Skip if line == '\n' - this means we're at the end of the file.
                for line in processed_file:
                    if line == '\n':
                        continue

                    # Create the feature vector.
                    feature_vector = self.line_to_feature_vector(line)
                    label = labels_file.readline()
                    label = label.strip()
                    if not label:
                        raise ValueError('Looks like you did something wrong.')

                    # Get our prediction and the actual values.
                    prediction = self.predict(feature_vector)
                    actual = int(label)
                    total_count += 1

                    # Adjust the weights if the prediction is incorrect.
                    if prediction != actual:
                        incorrect_count += 1
                        # Must have a positive or negative correction factor, otherwise we lose a degree of adjustment.
                        if prediction < actual:
                            correction_factor = 1
                        else:
                            correction_factor = -1
                        self.adjust_weights(feature_vector, correction_factor)
                    else:
                        correct_count += 1

                processed_file.close()
                labels_file.close()

                # Create a table to get the current accuracy of the perceptron.
                self.accuracy_dict['Epoch'].append(epoch)
                training_percentage_correct = int(correct_count / total_count * 100)
                self.accuracy_dict['Training Accuracy %'].append(training_percentage_correct)
                self.accuracy_dict['Training Failed'].append(incorrect_count)
                test_percent_correct = self.check_accuracy(processed_test_path, test_labels_path, 'Test Failed')
                self.accuracy_dict['Test Accuracy %'].append(test_percent_correct)

        return self.accuracy_dict

    def check_accuracy(self, processed_path, labels_path, table_column):
        correct_count = 0
        total_count = 0
        failed = 0
        with open(processed_path, 'r') as processed_file, open(labels_path, 'r') as labels_file:
            for line in processed_file:
                if line == '\n':
                    continue
                label = labels_file.readline()

                total_count += 1
                feature_vector = self.line_to_feature_vector(line)

                prediction = self.predict(feature_vector)
                actual = int(label)

                if prediction == actual:
                    correct_count += 1
                else:
                    failed += 1

            labels_file.close()
            self.accuracy_dict[table_column].append(failed)
        return int(correct_count / total_count * 100)
