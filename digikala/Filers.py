import pandas as pd
import numpy as np
import os
import hazm
import gensim
import pickle

from digikala.Helpers import Helpers
from hazm import POSTagger, WordTokenizer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from enum import Enum
from itertools import chain
from gensim.models import Doc2Vec
from keras.utils import np_utils


class Filers:
    class EnumTypes(Enum):
        ALL_INDEXES = 1
        ALL_WORDS = 2
        NOTNOUN_INDEXES = 3
        NOTNOUN_WORDS = 4
        ADJ_INDEXES = 5
        ADJ_WORDS = 6
        ADV_INDEXES = 7
        ADV_WORDS = 8
        VERB_INDEXES = 9
        VERB_WORDS = 10
        ADJ_ADV_VERB_INDEXES = 11
        ADJ_ADV_VERB_WORDS = 12

        def check_pos_tag(self, pos):
            if self in (Filers.EnumTypes.ALL_INDEXES, Filers.EnumTypes.ALL_WORDS):
                return True
            if self in (Filers.EnumTypes.NOTNOUN_INDEXES, Filers.EnumTypes.NOTNOUN_WORDS):
                return pos != 'N'
            if self in (Filers.EnumTypes.ADJ_INDEXES, Filers.EnumTypes.ADJ_WORDS):
                return pos == 'AJ'
            if self in (Filers.EnumTypes.ADV_INDEXES, Filers.EnumTypes.ADV_WORDS):
                return pos == 'ADV'
            if self in (Filers.EnumTypes.VERB_INDEXES, Filers.EnumTypes.VERB_WORDS):
                return pos == 'V'
            if self in (Filers.EnumTypes.ADJ_ADV_VERB_INDEXES, Filers.EnumTypes.ADJ_ADV_VERB_WORDS):
                return pos == 'AJ' or pos == 'ADV' or pos == 'V'

    @staticmethod
    def make_tagged_sentences():
        tagger = POSTagger(model='resources/postagger.model')
        sentences_df = pd.read_pickle('./base_outputs/digikala_dictionary.pkl')
        words_tagged = []
        for idx_s, sent in enumerate(sentences_df['text']):
            words = hazm.word_tokenize(sent)
            words_tagged.append(tagger.tag(words))

        with open("./base_outputs/tagged_sentences.pkl", "wb") as fp:
            pickle.dump(words_tagged, fp)

    @staticmethod
    def make_data_for_network(enum_type):
        word_model = gensim.models.Word2Vec.load('./trained_word2vec/word2vec_model_150.bin')
        sentences_df = pd.read_pickle('./base_outputs/digikala_dictionary.pkl')
        with open("./base_outputs/tagged_sentences.pkl", "rb") as fp:
            tagged_sentences = pickle.load(fp)
        sent_n = len(sentences_df['text'])
        x = []
        for idx_t, words_tagged in enumerate(tagged_sentences):
            if enum_type in (Filers.EnumTypes.ALL_INDEXES,
                             Filers.EnumTypes.NOTNOUN_INDEXES,
                             Filers.EnumTypes.ADJ_INDEXES,
                             Filers.EnumTypes.ADV_INDEXES,
                             Filers.EnumTypes.VERB_INDEXES,
                             Filers.EnumTypes.ADJ_ADV_VERB_INDEXES):
                sentence = []
            else:
                sentence = ''
            for idx_w, w in enumerate(words_tagged):
                if (w[0] in word_model.wv.vocab) and enum_type.check_pos_tag(w[1]):
                    if enum_type in (Filers.EnumTypes.ALL_INDEXES,
                                     Filers.EnumTypes.NOTNOUN_INDEXES,
                                     Filers.EnumTypes.ADJ_INDEXES,
                                     Filers.EnumTypes.ADV_INDEXES,
                                     Filers.EnumTypes.VERB_INDEXES,
                                     Filers.EnumTypes.ADJ_ADV_VERB_INDEXES):
                        word_index = Helpers.word2idx(word_model, w[0])
                        sentence.append(word_index)
                    else:
                        sentence = sentence + w[0] + ' '
                # else:
                #     continue
            print(str(idx_t + 1) + '/' + str(sent_n))
            x.append(sentence)

        filename = './prepared_data/sentences_' + enum_type.name.lower() + '.li'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as fp:
            pickle.dump(x, fp)

    @staticmethod
    def load_data(enum_type):
        filename_x = './prepared_data/sentences_' + enum_type.name.lower() + '.li'
        with open(filename_x, 'rb') as fp:
            x = pickle.load(fp)

        digikala_df = pd.read_pickle('./base_outputs/digikala_dictionary.pkl')
        y = digikala_df['label'].values

        nonzero_indices = (y != 0).nonzero()[0]
        negative_indices = (y == -1).nonzero()[0]
        y[negative_indices] = 0

        x_tmp = []
        for i in nonzero_indices:
            x_tmp.append(x[i])
        x = x_tmp

        return x, y[nonzero_indices]

    @staticmethod
    def load_data_percentage(enum_type, categorical=False):
        x, y = Filers.load_data(enum_type)
        if categorical:
            y = np_utils.to_categorical(y, num_classes=2)
        return train_test_split(x, y, test_size=.1, random_state=42, stratify=y)

    @staticmethod
    def cluster_data():
        # filename_x = './prepared_data/sentences_' + enum_type.name.lower() + '.li'
        # with open(filename_x, 'rb') as fp:
        # x = pickle.load(fp)

        digikala_df = pd.read_pickle('./base_outputs/digikala_dictionary.pkl')
        y = digikala_df['label'].values
        nonzero_indices = (y != 0).nonzero()[0]
        # negative_indices = (y == -1).nonzero()[0]
        # y[negative_indices] = 0

        data_df_nonzero = digikala_df.loc[nonzero_indices]
        data_df_nonzero = data_df_nonzero.reset_index(drop=True)
        y = data_df_nonzero['label'].values
        # tokenizer = WordTokenizer()
        # count_vect = CountVectorizer(tokenizer=tokenizer.tokenize, ngram_range=(1, 3), max_features=4000)
        # tfidf_vect = TfidfVectorizer(tokenizer=tokenizer.tokenize, ngram_range=(1, 3), max_features=4000)

        # x_vectorized = count_vect.fit_transform(x)
        # sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        # for train_index, test_index in sss.split(x_vectorized[nonzero_indices], y[nonzero_indices]):
        #     x_train, x_test = x_vectorized[train_index], x_vectorized[test_index]
        #     y_train, y_test = y[train_index], y[test_index]
        #
        # indices_except_test = np.setdiff1d(np.nonzero(y == y)[0], test_index)
        #
        # x_new = x_vectorized[indices_except_test]
        # y_new = y[indices_except_test]
        #
        d2v_model = Doc2Vec.load("./trained_doc2vec/d2v.bin")
        kmeans_model = KMeans(n_clusters=2, init='k-means++', max_iter=100)
        X = kmeans_model.fit(d2v_model.docvecs.doctag_syn0)
        labels = kmeans_model.labels_.tolist()
        y_pred_kmeans = kmeans_model.fit_predict(d2v_model.docvecs.doctag_syn0)

        # kmeans = KMeans(n_clusters=2, n_init=20, n_jobs=-1)
        # y_pred_kmeans = kmeans.fit_predict(x_new)

        label_plus_one_0 = 0
        label_plus_one_0_index_list = []
        label_zero_0 = 0
        label_zero_0_index_list = []
        label_minus_one_0 = 0
        label_minus_one_0_index_list = []
        label_plus_one_1 = 0
        label_plus_one_1_index_list = []
        label_zero_1 = 0
        label_zero_1_index_list = []
        label_minus_one_1 = 0
        label_minus_one_1_index_list = []

        for item_index, item in enumerate(y_pred_kmeans):
            if item == 0:
                if y[item_index] == 1:
                    label_plus_one_0 += 1
                    label_plus_one_0_index_list.append(item_index)
                elif y[item_index] == 0:
                    label_zero_0 += 1
                    label_zero_0_index_list.append(item_index)
                elif y[item_index] == -1:
                    label_minus_one_0 += 1
                    label_minus_one_0_index_list.append(item_index)

            if item == 1:
                if y[item_index] == 1:
                    label_plus_one_1 += 1
                    label_plus_one_1_index_list.append(item_index)
                elif y[item_index] == 0:
                    label_zero_1 += 1
                    label_zero_1_index_list.append(item_index)
                elif y[item_index] == -1:
                    label_minus_one_1 += 1
                    label_minus_one_1_index_list.append(item_index)

        # with open('./base_outputs/cluster_indices.pkl', 'wb') as f:
        #     pickle.dump([label_plus_one_0_index_list, label_zero_0_index_list, label_minus_one_0_index_list,
        #                  label_plus_one_1_index_list, label_zero_1_index_list, label_minus_one_1_index_list], f)

        label_plus_one_0_percent = label_plus_one_0 / len(y) * 100
        label_zero_0_percent = label_zero_0 / len(y) * 100
        label_minus_one_0_percent = label_minus_one_0 / len(y) * 100
        label_plus_one_1_percent = label_plus_one_1 / len(y) * 100
        label_zero_1_percent = label_zero_1 / len(y) * 100
        label_minus_one_1_percent = label_minus_one_1 / len(y) * 100

        print(label_plus_one_0_percent)
        print(label_zero_0_percent)
        print(label_minus_one_0_percent)
        print(label_plus_one_1_percent)
        print(label_zero_1_percent)
        print(label_minus_one_1_percent)

    @staticmethod
    def load_clustered_data(enum_type):
        with open('./base_outputs/cluster_indices.pkl', 'rb') as f:
            label_plus_one_0_index_list, label_zero_0_index_list, label_minus_one_0_index_list, \
            label_plus_one_1_index_list, label_zero_1_index_list, label_minus_one_1_index_list = pickle.load(f)
        positive_indices = list(
            chain(label_plus_one_1_index_list, label_plus_one_0_index_list, label_zero_0_index_list))
        negative_indices = list(
            chain(label_minus_one_1_index_list, label_minus_one_0_index_list, label_zero_1_index_list))

        filename_x = './prepared_data/sentences_' + enum_type.name.lower() + '.li'
        with open(filename_x, 'rb') as fp:
            x = pickle.load(fp)

        digikala_df = pd.read_pickle('./base_outputs/digikala_dictionary.pkl')
        y = digikala_df['label'].values
        nonzero_indices = (y != 0).nonzero()[0]
        # negative_indices = (y == -1).nonzero()[0]
        # y[negative_indices] = 0

        x_nonzero = []
        for i in nonzero_indices:
            x_nonzero.append(x[i])

        x_train = []
        x_test = []
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        for train_index, test_index in sss.split(x_nonzero, y[nonzero_indices]):
            for i in train_index:
                x_train.append(x[i])
            for i in test_index:
                x_test.append(x[i])
            y_train, y_test = y[train_index], y[test_index]

        y_test_negative_indices = (y_test == -1).nonzero()[0]
        y_test[y_test_negative_indices] = 0

        indices_except_test = np.setdiff1d(np.nonzero(y == y)[0], test_index)
        x_train_tmp = []
        for i in indices_except_test:
            x_train_tmp.append(x[i])
        x_train = x_train_tmp
        y[negative_indices] = 0
        y[positive_indices] = 1
        y_train = y[indices_except_test]

        return x_train, x_test, y_train, y_test
