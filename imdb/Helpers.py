import nltk

from sklearn.metrics import confusion_matrix, classification_report


class Helpers:
    @staticmethod
    def max_sent_words(sents):
        l = []
        for s in sents:
            words = nltk.word_tokenize(s)
            l.append(len(words))
        return max(l)

    @staticmethod
    def word2idx(word_model, word):
        return word_model.wv.vocab[word].index

    @staticmethod
    def idx2word(word_model, idx):
        return word_model.wv.index2word[idx]

    @staticmethod
    def print_binary(model, inputs, y_test):
        # binary test result
        y_pred = model.predict(inputs)
        print(confusion_matrix(y_test, y_pred.round()))
        print(classification_report(y_test, y_pred.round(), digits=4))

    @staticmethod
    def print_categorical(model, inputs, y_test):
        # categorical test result
        y_prob = model.predict(inputs)
        y_pred_class = y_prob.argmax(axis=-1)
        y_test_final = y_test.argmax(axis=-1)
        print(confusion_matrix(y_test_final, y_pred_class))
        print(classification_report(y_test_final, y_pred_class, digits=4))
