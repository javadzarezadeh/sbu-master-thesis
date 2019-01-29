import os
import pandas as pd
import nltk
import pickle
import gensim

from imdb.Helpers import Helpers
from sklearn.model_selection import train_test_split
from enum import Enum


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
                return pos != 'NOUN'
            if self in (Filers.EnumTypes.ADJ_INDEXES, Filers.EnumTypes.ADJ_WORDS):
                return pos == 'ADJ'
            if self in (Filers.EnumTypes.ADV_INDEXES, Filers.EnumTypes.ADV_WORDS):
                return pos == 'ADV'
            if self in (Filers.EnumTypes.VERB_INDEXES, Filers.EnumTypes.VERB_WORDS):
                return pos == 'VERB'
            if self in (Filers.EnumTypes.ADJ_ADV_VERB_INDEXES, Filers.EnumTypes.ADJ_ADV_VERB_WORDS):
                return pos == 'ADJ' or pos == 'ADV' or pos == 'VERB'

    @staticmethod
    def make_tagged_sentences():
        df_pathes = ['./base_outputs/imdb_train_dictionary.pkl', './base_outputs/imdb_test_dictionary.pkl']
        for path in df_pathes:
            sentences_df = pd.read_pickle(path)
            sent_n = len(sentences_df['text'])
            words_tagged = []
            for idx_s, sent in enumerate(sentences_df['text']):
                words = nltk.word_tokenize(sent)
                print(str(idx_s + 1) + '/' + str(sent_n))
                words_tagged.append(nltk.pos_tag(words, tagset='universal'))

            if path is df_pathes[0]:
                filename = './base_outputs/tagged_sentences' + '_train' + '.pkl'
            else:
                filename = './base_outputs/tagged_sentences' + '_test' + '.pkl'
            with open(filename, 'wb') as fp:
                pickle.dump(words_tagged, fp)

    @staticmethod
    def make_data_for_network(enum_type):
        word_model = gensim.models.Word2Vec.load('./trained_word2vec/word2vec_model_150.bin')
        df_pathes = ['./base_outputs/imdb_train_dictionary.pkl', './base_outputs/imdb_test_dictionary.pkl']
        for path in df_pathes:
            sentences_df = pd.read_pickle(path)
            if path == df_pathes[0]:
                tagged_path = "./base_outputs/tagged_sentences_train.pkl"
            else:
                tagged_path = "./base_outputs/tagged_sentences_test.pkl"
            with open(tagged_path, "rb") as fp:
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

            if path == df_pathes[0]:
                filename = './prepared_data/sentences_train_' + enum_type.name.lower() + '.li'
            else:
                filename = './prepared_data/sentences_test_' + enum_type.name.lower() + '.li'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'wb') as fp:
                pickle.dump(x, fp)

    @staticmethod
    def load_data(enum_type):

        filename_train_x = './prepared_data/sentences_train_' + enum_type.name.lower() + '.li'
        with open(filename_train_x, 'rb') as fp:
            x_train = pickle.load(fp)

        filename_test_x = './prepared_data/sentences_test_' + enum_type.name.lower() + '.li'
        with open(filename_test_x, 'rb') as fp:
            x_test = pickle.load(fp)

        train_df = pd.read_pickle('./base_outputs/imdb_train_dictionary.pkl')
        y_train = train_df['label'].values

        test_df = pd.read_pickle('./base_outputs/imdb_test_dictionary.pkl')
        y_test = test_df['label'].values

        return x_train, x_test, y_train, y_test

    @staticmethod
    def load_data_percentage(enum_type):
        x, y = Filers.load_data(enum_type)
        return train_test_split(x, y, test_size=.2, random_state=42)
