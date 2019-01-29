import os
import pandas as pd
import multiprocessing

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


class Dataers:
    @staticmethod
    def fetch_sentences():
        imdb_dir = '/home/jayzee/aclImdb'
        train_dir = os.path.join(imdb_dir, 'train')
        test_dir = os.path.join(imdb_dir, 'test')
        directory_paths = [train_dir, test_dir]

        dir_name_unsup = os.path.join(imdb_dir, 'train/unsup')
        data = {'text': [],
                'label': []}
        for fname in os.listdir(dir_name_unsup):
            if fname[-4:] == '.txt':
                f = open(os.path.join(dir_name_unsup, fname))
                data['text'].append(f.read())
                f.close()
        df = pd.DataFrame(data, columns=['text'])
        filename = './base_outputs/imdb' + '_unsup' + '_dictionary.pkl'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_pickle(filename)

        for dir_path in directory_paths:
            data = {'text': [],
                    'label': []}
            for label_type in ['neg', 'pos']:
                dir_name = os.path.join(dir_path, label_type)
                for fname in os.listdir(dir_name):
                    if fname[-4:] == '.txt':
                        f = open(os.path.join(dir_name, fname))
                        data['text'].append(f.read())
                        f.close()
                        if label_type == 'neg':
                            data['label'].append(0)
                        else:
                            data['label'].append(1)
            df_tmp = pd.DataFrame(data, columns=['text', 'label'])
            df = df_tmp.sample(frac=1, random_state=42).reset_index(drop=True)
            if dir_path is train_dir:
                filename = './base_outputs/imdb' + '_train' + '_dictionary.pkl'
            elif dir_path is test_dir:
                filename = './base_outputs/imdb' + '_test' + '_dictionary.pkl'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            df.to_pickle(filename)

    @staticmethod
    def train_word2vec_model():
        # data_train_df = pd.read_pickle('./base_outputs/imdb_train_dictionary.pkl')
        # data_unsup_df = pd.read_pickle('./base_outputs/imdb_unsup_dictionary.pkl')
        # data_df = pd.concat([data_train_df, data_unsup_df], sort=False)
        # texts = data_df['text']
        #
        # filename = './trained_word2vec/comments.txt'
        # os.makedirs(os.path.dirname(filename), exist_ok=True)
        # with open(filename, 'w') as f:
        #     f.writelines(texts)

        # train model
        model = Word2Vec(LineSentence('./trained_word2vec/comments.txt'),
                         size=64,
                         window=5,
                         min_count=0,
                         workers=multiprocessing.cpu_count()
                         )
        # save model
        model.save('./trained_word2vec/word2vec_model_64.bin')
        model.wv.save_word2vec_format('./trained_word2vec/word2vec_word_vector_64.txt', binary=False)
