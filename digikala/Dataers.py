import pandas as pd
import os
import multiprocessing

from digikala.Helpers import Helpers
from hazm import Normalizer
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


class Dataers:

    @staticmethod
    def fetch_sentences():
        dgk_file = '../../Sentiment_Datasets/digikala_dataset.txt'
        normalizer = Normalizer()

        with open(dgk_file, 'r') as f:
            lines = f.readlines()

        data = {'text': [],
                'label': []}
        for l in lines:
            text_temp = Helpers.remove_extra_alphabet(l[2:])
            normalized_text = normalizer.normalize(text_temp)
            data['text'].append(normalized_text)
            data['label'].append(int(l[0:2]))

        df = pd.DataFrame(data, columns=['text', 'label'])
        filename = './base_outputs/digikala_dictionary.pkl'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_pickle(filename)

    @staticmethod
    def train_word2vec_model():
        # data_df = pd.read_pickle('./base_outputs/digikala_dictionary.pkl')
        # y = data_df['label'].values
        # zero_indices = (y == 0).nonzero()[0]
        # texts = data_df['text'][zero_indices]
        #
        # filename = './trained_word2vec/comments.txt'
        # os.makedirs(os.path.dirname(filename), exist_ok=True)
        # with open(filename, 'w') as f:
        #     f.writelines(texts)

        # train model
        model = Word2Vec(LineSentence('./trained_word2vec/comments.txt'),
                         size=512,
                         window=5,
                         min_count=0,
                         workers=multiprocessing.cpu_count()
                         )
        # save model
        model.save('./trained_word2vec/word2vec_model_512.bin')
        model.wv.save_word2vec_format('./trained_word2vec/word2vec_word_vector_512.txt', binary=False)
