import numpy as np


class MultiClassPerceptron:
    def __init__(self, learning_rate, epochs, vocab, classes):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.vocab = vocab
        # Classes have been ordered in the main class for class_to_index, e.g., { a:0, b:1, c:2,...}
        self.class_to_index = {cls: idx for idx, cls in enumerate(classes)}
        # Reverse of the above operation.
        self.index_to_classes = {idx: cls for idx, cls in enumerate(classes)}
        # Features are being sorted arbitrarily.
        self.feature_to_index = {feature: idx for idx, feature in enumerate(vocab.keys())}
        # Reverse of the above operation.
        self.index_to_feature = {idx: feature for idx, feature in enumerate(vocab.keys())}
        # Here we have our weights: [[features],[classes]]. Because the features have been
        # ordered consistently, it doesn't matter the order the indexes take.
        self.weights = np.zeros((len(vocab), len(classes)))
        # Total classes.
        self.class_count = len(classes)
        # Using this for formatting.
        self.accuracy_dict = {'Epoch': [], 'Training Accuracy %': [], 'Test Accuracy %': []}
        self.feature_to_class_matrix = self.init_vocab_vectors()

    def init_vocab_vectors(self):
        feature_to_class_matrix = {}
        for feature, classes in self.vocab.items():
            feature_to_class_matrix[feature] = np.zeros(self.class_count)
            for cls in classes:
                feature_to_class_matrix[feature][self.class_to_index[cls]] = 1.0
        return feature_to_class_matrix

    def predict_picture(self, image):
        feature_indices = [self.feature_to_index[portion] for portion in image]
        result = []
        for feature_index in feature_indices:
            feature_classes = self.feature_to_class_matrix[self.index_to_feature[feature_index]]
            feature_weight = self.weights[feature_index]
            result.append(feature_classes * feature_weight)
        summation = np.zeros(self.class_count)
        for matrix in result:
            summation + matrix
        return feature_indices, np.argmax(summation)

    def update_weights(self, feature_indices, predicted_index, actual_index):
        for feature_index in feature_indices:
            feature = self.index_to_feature[feature_index]
            self.weights[feature_index][actual_index] += self.learning_rate * self.feature_to_class_matrix[feature][actual_index]
            self.weights[feature_index][predicted_index] -= self.learning_rate * self.feature_to_class_matrix[feature][predicted_index]

    def train(self, processed_ocr_train, ocr_train_label_path):
        for epoch in range(self.epochs):
            count = 0
            correct = 0
            with open(processed_ocr_train, 'r') as train, open(ocr_train_label_path, 'r') as ocr_train_label:
                picture = []

                for line in train:
                    line = line.strip()
                    if line == '00000000':
                        continue
                    if line == '':
                        feature_indices, predicted_class_index = self.predict_picture(picture)
                        actual_class_index = self.class_to_index[ocr_train_label.readline().strip()]

                        if predicted_class_index != actual_class_index:
                            self.update_weights(feature_indices, predicted_class_index, actual_class_index)
                        else:
                            correct += 1

                        picture = []
                    if line == '':
                        continue
                    picture.append(line)
                    count += 1
            print('Percentage correct: ' + str(int(correct / count * 100)) + '%\n')


