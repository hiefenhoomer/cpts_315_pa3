import numpy as np


class MultiClassPerceptron:
    # Let's just fill the constructor with everything. Less to think about later on.
    def __init__(self, learning_rate, epochs, ocr_labels, codes, vocab, images, test_ocr_labels, test_images):
        self.learning_rate = learning_rate
        self.epochs = epochs
        # Hash codes to their respective index.
        self.codes_to_idx = codes
        self.codes_length = len(self.codes_to_idx)
        # Images are 8 x 16 grids. These looked the most like letters.
        self.images = images
        # Hash vocabulary to index.
        self.vocab_to_idx = vocab
        # Length of the vocabulary. The size of the weight array at each code index.
        self.vocab_length = len(self.vocab_to_idx)
        # Initializing weights: [codes_idx][vocab_idx]
        self.weights = np.zeros((len(codes), len(vocab)))
        # Labels for each test image.
        self.ocr_labels = ocr_labels

        # Testing data.
        self.test_ocr_labels = test_ocr_labels
        self.test_images = test_images
        self.accuracy_dict = {'Epoch': [], 'Training Accuracy %': [], 'Test Accuracy %': [], 'Training Failed': [], 'Test Failed': []}

    def create_feature_vector(self, features):
        # Feature vector must be the size of this: len(code_idx)
        feature_vector = np.zeros(self.vocab_length)
        # Count the frequency of each feature.
        for feature in features:
            feature_vector[self.vocab_to_idx[feature]] = 1
        return feature_vector

    def predict(self, image, weights):
        feature_vector = self.create_feature_vector(image)
        # Get the dot product of all items in indices corresponding
        dot_products = [np.dot(feature_vector, weight) for weight in weights]
        # Get the max dot product in the list.
        return np.argmax(dot_products), feature_vector

    def modify_weights(self, prediction_idx, actual_idx, feature_vector):
        # nx = η · xt
        nx = feature_vector * self.learning_rate
        # wyt = wyt + η · xt // update the weights
        self.weights[actual_idx] += nx
        # wyˆt = wyˆt − η · xt // update the weights
        self.weights[prediction_idx] -= nx

    def test(self):
        # Initialize the count.
        counter = 0
        success = 0
        failure = 0
        # Zip the codes and features together to iterate over both lists simultaneously.
        image_code_pair = zip(self.test_images, self.test_ocr_labels)
        for image, actual in image_code_pair:
            # Get the prediction index and feature vector. These are important for updating the weights.
            prediction_idx, feature_vector = self.predict(image, self.weights)
            counter += 1
            # Determine the accuracy of this iteration.
            if self.codes_to_idx[actual] == prediction_idx:
                success += 1
            else:
                failure += 1

        # Get the total percent correct.
        percent_correct = str(int((success / counter) * 100))
        self.accuracy_dict['Test Accuracy %'].append(percent_correct)
        self.accuracy_dict['Test Failed'].append(failure)
        return

    def train(self):
        # For epoch...
        for epoch in range(self.epochs):
            self.accuracy_dict['Epoch'].append(epoch)
            # Initialize counters.
            counter = 0
            failure = 0
            success = 0
            # Zip the labels and codes for ease of use.
            image_code_pair = zip(self.images, self.ocr_labels)
            for image, actual in image_code_pair:
                prediction_idx, feature_vector = self.predict(image, self.weights)
                counter += 1
                # If the actual code doesn't match the prediction, adjust the weights.
                if self.codes_to_idx[actual] != prediction_idx:
                    self.modify_weights(prediction_idx, self.codes_to_idx[actual], feature_vector)
                    failure += 1
                else:
                    success += 1

            percent_correct = str(int((success / counter) * 100))
            self.accuracy_dict['Training Accuracy %'].append(percent_correct)
            self.accuracy_dict['Training Failed'].append(failure)
            self.test()
