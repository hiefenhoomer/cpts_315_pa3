import Preprocessing
from perceptrons.AveragedMultiClassPerceptron import AveragedMultiClassPerceptron

from perceptrons.AveragedPerceptron import AveragedPerceptron
from perceptrons.MultiClassPerceptron import MultiClassPerceptron

raw_train_path = 'raw/traindata.txt'
raw_test_path = 'raw/testdata.txt'
raw_ocr_train_path = 'raw/ocr_train.txt'
raw_ocr_test_path = 'raw/ocr_test.txt'

training_labels_path = 'labels/trainlabels.txt'
test_labels_path = 'labels/testlabels.txt'
ocr_training_labels_path = 'labels/ocrtrainlabels.txt'
ocr_test_labels_path = 'labels/ocrtestlabels.txt'

stop_list_path = 'stoplist.txt'
ocr_stop_list_path = 'ocrstoplist.txt'

processed_train_path = 'processed/processed_training.txt'
processed_test_path = 'processed/processed_test.txt'
processed_ocr_train_path = 'processed/processed_ocr_train.txt'
processed_ocr_test_path = 'processed/processed_ocr_test.txt'

success_perceptron_path = 'results/success_perceptron.txt'
success_multi_class_perceptron_path = 'results/success_multi_class_perceptron.txt'

learning_rate = 1
epochs = 20

if __name__ == '__main__':
    # Look at this mess! We put all the garbage over here to make coding easier.

    Preprocessing.clean_file(raw_train_path, processed_train_path, stop_list_path)
    Preprocessing.clean_file(raw_test_path, processed_test_path, stop_list_path)
    averaged_perceptron = AveragedPerceptron(learning_rate, Preprocessing.create_vocabulary(processed_train_path))
    success_perceptron_dict = averaged_perceptron.train(processed_train_path, training_labels_path,
                                                       processed_test_path, test_labels_path, epochs)
    Preprocessing.write_table_to_file(success_perceptron_dict, success_perceptron_path)

    Preprocessing.create_processed_ocr(raw_ocr_train_path, processed_ocr_train_path, ocr_training_labels_path)
    Preprocessing.create_processed_ocr(raw_ocr_test_path, processed_ocr_test_path, ocr_test_labels_path)

    code_dict, ocr_labels, multi_class_vocab, images = Preprocessing.create_ocr_vocabulary(processed_ocr_train_path, ocr_training_labels_path)
    a, test_ocr_labels, b, test_images = Preprocessing.create_ocr_vocabulary(processed_ocr_test_path, ocr_test_labels_path)

    multi_class_perceptron = MultiClassPerceptron(learning_rate, epochs, ocr_labels, code_dict, multi_class_vocab, images, test_ocr_labels, test_images)
    multi_class_perceptron.train()

    averaged_multi_class_perceptron = AveragedMultiClassPerceptron(learning_rate, epochs, ocr_labels, code_dict, multi_class_vocab, images, test_ocr_labels, test_images)
    success_dict = averaged_multi_class_perceptron.train()

    Preprocessing.write_table_to_file(success_dict, success_multi_class_perceptron_path)
