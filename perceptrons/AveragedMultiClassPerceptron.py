from perceptrons.MultiClassPerceptron import MultiClassPerceptron
import copy


class AveragedMultiClassPerceptron(MultiClassPerceptron):
    def __init__(self, learning_rate, epochs, ocr_labels, codes, vocab, images, test_ocr_labels, test_images):
        super().__init__(learning_rate, epochs, ocr_labels, codes, vocab, images, test_ocr_labels, test_images)
        self.weights_summation = copy.deepcopy(self.weights)
        self.summation_count = 0

    def modify_weights(self, prediction_idx, actual_idx, feature_vector):
        super().modify_weights(prediction_idx, actual_idx, feature_vector)
        self.weights_summation += self.weights
        self.summation_count += 1

    def train(self):
        super().train()

        train_data = zip(self.images, self.ocr_labels)
        test_data = zip(self.test_images, self.test_ocr_labels)
        weights_avg = self.weights_summation / self.summation_count

        train_percentage, train_failed = self.average_perceptron_data(train_data, weights_avg)
        test_percentage, test_failed = self.average_perceptron_data(test_data, weights_avg)

        self.accuracy_dict['Epoch'].append('Averaged Perceptron %:')
        self.accuracy_dict['Training Accuracy %'].append(train_percentage)
        self.accuracy_dict['Test Accuracy %'].append(test_percentage)
        self.accuracy_dict['Training Failed'].append(train_failed)
        self.accuracy_dict['Test Failed'].append(test_failed)

        return self.accuracy_dict

    def average_perceptron_data(self, data, weights_avg):
        count = 0
        success = 0
        failed = 0

        for img, actual in data:
            predicted_idx, feature = self.predict(img, weights_avg)
            if self.codes_to_idx[actual] == predicted_idx:
                success += 1
            else:
                failed += 1
            count += 1

        return str(int((success/count) * 100)), failed

