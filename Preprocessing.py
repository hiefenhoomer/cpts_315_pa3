import numpy as np
import os
import tabulate
import pandas as pd
from tabulate import tabulate


def load_stop_list(stop_list_path):
    words = []
    with open(stop_list_path, 'r') as file:
        for line in file:
            line = line.strip()
            words.append(line)
        file.close()
    return words


def clean_file(raw_path, processed_path, stop_list):
    stop_list_words = load_stop_list(stop_list)
    if os.path.exists(processed_path):
        os.remove(processed_path)
    with open(raw_path, 'r') as raw:
        with open(processed_path, 'w') as processed:
            for line in raw:
                words = line.strip().split(' ')
                cleaned_words = [word for word in words if word not in stop_list_words]
                cleaned_string = ' '.join(cleaned_words)
                processed.write(f'{cleaned_string}\n')
            processed.close()
        raw.close()


def create_vocabulary(processed_path):
    vocab = []
    with open(processed_path, 'r') as proc:
        for line in proc:
            if line == '\n':
                continue
            words = line.strip().split(' ')
            words.sort()
            for word in words:
                vocab.append(word)
        vocab.sort()
        vocab_dict = {}
        count = 0
        for i in range(len(vocab)):
            if vocab[i] in vocab_dict.keys():
                continue
            vocab_dict[vocab[i]] = count
            count += 1
        proc.close()
    return vocab_dict


def create_processed_ocr(ocr_input_path, ocr_output_path, ocr_label_path):
    if os.path.exists(ocr_output_path):
        os.remove(ocr_output_path)
    if os.path.exists(ocr_label_path):
        os.remove(ocr_label_path)

    with open(ocr_input_path, 'r') as ocr_input, open(ocr_output_path, 'w') as ocr_output, open(ocr_label_path,
                                                                                                'w') as ocr_label:
        for line in ocr_input:
            line = line.strip()
            if line == '':
                continue
            items = line.split('\t')
            items = items[1:3]

            img_str = items[0][2:]
            img_arr = [b for b in img_str]

            image = np.array(img_arr).reshape((16, 8))
            for image_line in image:
                ocr_output.write(''.join(image_line) + '\n')
            ocr_output.write('\n')

            char = items[1]
            ocr_label.write(char + '\n')

        ocr_input.close()
        ocr_output.close()
        ocr_label.close()


def create_multi_class_vocab(processed_ocr_train_path, ocr_label_path):
    vocab = {}
    with open(processed_ocr_train_path, 'r') as features, open(ocr_label_path, 'r') as classes:
        feature_list = []
        for feature in features:
            feature = feature.strip()
            if feature == '00000000':
                continue

            if feature == '':
                label = classes.readline().strip()
                for feat in feature_list:
                    if feat not in vocab:
                        vocab[feat] = []
                    if label not in vocab[feat]:
                        vocab[feat].append(label)
                        vocab[feat].sort()
                continue

            feature_list.append(feature)
        return vocab


def all_classes(ocr_train_labels_path):
    classes = []
    with open(ocr_train_labels_path, 'r') as labels:
        for label in labels:
            if label not in classes:
                classes.append(label)
        labels.close()
    return sorted(classes)


def write_table_to_file(df_dict, file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    df = pd.DataFrame(df_dict)
    table = tabulate(df, headers='keys', tablefmt='fancy_grid', showindex='false')
    with open(file_path, 'w') as file:
        file.write('Averaged and Standard Perceptron Success:\n\n')
        file.write(table)
