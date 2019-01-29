import re
import hazm
import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report


class Helpers:

    @staticmethod
    def remove_extra_alphabet(s):
        return re.sub(r'(.)\1{2,}', r'\1\1', s)

    @staticmethod
    def max_sent_words(sents):
        l = []
        for s in sents:
            words = hazm.word_tokenize(s)
            l.append(len(words))
        return max(l)

    @staticmethod
    def word2idx(word_model, word):
        return word_model.wv.vocab[word].index

    @staticmethod
    def idx2word(word_model, idx):
        return word_model.wv.index2word[idx]

    @staticmethod
    def print_binary_old(model, inputs, y_test):
        # binary test result
        y_pred = model.predict(inputs)
        print(confusion_matrix(y_test, y_pred.round()))
        print(classification_report(y_test, y_pred.round(), digits=4))

    @staticmethod
    def print_categorical_old(model, inputs, y_test):
        # categorical test result
        y_prob = model.predict(inputs)
        y_pred_class = y_prob.argmax(axis=-1)
        y_test_final = y_test.argmax(axis=-1)
        print(confusion_matrix(y_test_final, y_pred_class))
        print(classification_report(y_test_final, y_pred_class, digits=4))

    @staticmethod
    def classification_report_csv(report, filename):
        report_data = []
        lines = report.split('\n')
        for line in lines[2:-3]:
            row = {}
            row_data = line.split(' ')
            row_data = list(filter(None, row_data))
            row['class'] = row_data[0]
            row['precision'] = float(row_data[1])
            row['recall'] = float(row_data[2])
            row['f1_score'] = float(row_data[3])
            row['support'] = float(row_data[4])
            report_data.append(row)
        dataframe = pd.DataFrame.from_dict(report_data)
        filename = filename + '.csv'
        dataframe.to_csv(filename, index=False)
