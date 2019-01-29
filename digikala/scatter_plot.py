from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from bidi.algorithm import get_display
import arabic_reshaper
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# from nltk.tokenize import word_tokenize
from digikala.Filers import Filers
from hazm import WordTokenizer
import pandas as pd
from gensim.models.doc2vec import TaggedLineDocument
import multiprocessing
import gensim
from gensim.models.doc2vec import TaggedDocument
from collections import namedtuple
import os

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import os, sys, email
import gensim
from gensim.models import Doc2Vec
from string import punctuation
import timeit
from sklearn.cluster import KMeans
from sklearn import metrics
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from collections import OrderedDict


def train_doc2vec():
    SentimentDocument = namedtuple('SentimentDocument', 'words tags sentiment')
    data_df = pd.read_pickle('./base_outputs/digikala_dictionary.pkl')
    y = data_df['label'].values
    nonzero_indices = (y != 0).nonzero()[0]
    negative_indices = (y == -1).nonzero()[0]
    # data_df_nonzero = data_df.loc[nonzero_indices]
    # data_df_nonzero = data_df_nonzero.reset_index(drop=True)
    tokenizer = WordTokenizer()
    alldocs = []
    for comment_idx, comment in enumerate(data_df['text']):
        words = tokenizer.tokenize(comment)
        tags = [comment_idx]
        if data_df['label'][comment_idx] == 1:
            sentiment = [1.0]
        elif data_df['label'][comment_idx] == -1:
            sentiment = [0.0]
        elif data_df['label'][comment_idx] == 0:
            sentiment = [None]

        alldocs.append(SentimentDocument(words, tags, sentiment))

    cores = multiprocessing.cpu_count()
    # assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"
    # model = Doc2Vec(dm=0, min_count=2, sample=0, workers=cores)
    model = Doc2Vec(min_count=0, sample=0, workers=cores)
    model.build_vocab(alldocs)
    model.train(alldocs, total_examples=model.corpus_count, epochs=10)
    filename = './trained_doc2vec/d2v.bin'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    model.save(filename)


def clustering():
    d2v_model = Doc2Vec.load("./trained_doc2vec/d2v.bin")
    kmeans_model = KMeans(n_clusters=2, init='k-means++', max_iter=100)
    X = kmeans_model.fit(d2v_model.docvecs.doctag_syn0)
    labels = kmeans_model.labels_.tolist()
    l = kmeans_model.fit_predict(d2v_model.docvecs.doctag_syn0)

    # tsne = TSNE(n_components=2, random_state=0)
    pca = PCA(n_components=2).fit(d2v_model.docvecs.doctag_syn0)
    datapoint = pca.transform(d2v_model.docvecs.doctag_syn0)
    # datapoint = tsne.fit_transform(d2v_model.docvecs.doctag_syn0)
    plt.figure
    label1 = ["#FFFF00", "#008000"]
    color = [label1[i] for i in labels]
    plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color)

    centroids = kmeans_model.cluster_centers_
    centroidpoint = pca.transform(centroids)
    # centroidpoint = tsne.fit_transform(centroids)
    plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1], marker='^', s=150, c='#000000')
    plt.show()


def display_closestwords_tsnescatterplot(model, word):
    arr = np.empty((0, 100), dtype='f')
    word_labels = [word]

    # get close words
    close_words = model.wv.similar_by_word(word, topn=99)

    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)

    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(get_display(arabic_reshaper.reshape(label)), xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
    plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)
    plt.show()


# train_doc2vec()
clustering()
# from gensim.models.doc2vec import Doc2Vec
#
# model= Doc2Vec.load("./trained_doc2vec/d2v.bin")
# tokenizer = WordTokenizer()
# #to find the vector of a document which is not in training data
# test_data = tokenizer.tokenize("من دیجی دوست نیستم")
# v1 = model.infer_vector(test_data)
# print("V1_infer", v1)
#
# # to find most similar doc using tags
# similar_doc = model.docvecs.most_similar('من')
# print(similar_doc)
#
#
# # to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
# print(model.docvecs['من'])
