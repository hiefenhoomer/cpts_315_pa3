import numpy as np


class MultiClassPerceptron:
    # Let's just fill the constructor with everything. Less to think about later on.
    def __init__(self, learning_rate, epochs, ocr_labels, codes, vocab, images):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.codes_to_idx = codes
        self.codes_length = len(self.codes_to_idx)
        # Images are 8 x 16 grids. These looked the most like letters.
        self.images = images
        self.vocab = vocab
        self.vocab_length = len(self.vocab)
        self.weights = np.zeros((len(codes), len(vocab)))
        self.ocr_labels = ocr_labels

    def create_feature_vector(self, features):
        feature_vector = np.zeros(self.vocab_length)
        for feature in features:
            feature_vector[self.vocab[feature]] = 1
        return feature_vector

    def predict(self, image):
        feature_vector = self.create_feature_vector(image)
        predictions = [np.dot(feature_vector, self.weights[i]) for i in range(self.codes_length)]
        return np.argmax(predictions), feature_vector

    def rebalance(self, prediction_idx, actual_idx, feature_vector):
        nx = feature_vector * self.learning_rate
        self.weights[actual_idx] += nx
        self.weights[prediction_idx] -= nx

    def train(self):
        for epoch in range(self.epochs):
            counter = 0
            success = 0
            image_code_pair = zip(self.images, self.ocr_labels)
            for image, actual in image_code_pair:
                prediction_idx, feature_vector = self.predict(image)
                counter += 1
                if self.codes_to_idx[actual] != prediction_idx:
                    self.rebalance(prediction_idx, self.codes_to_idx[actual], feature_vector)
                else:
                    success += 1

            percent_correct = str(int((success / counter) * 100))
            print(f'Epoch {epoch + 1}, Percent Correct: {percent_correct}%')
