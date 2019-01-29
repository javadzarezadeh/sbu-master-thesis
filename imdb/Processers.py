from imdb.Filers import Filers
from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Flatten, Dense, Embedding, Activation, BatchNormalization
import numpy as np
import gensim
import multiprocessing
# from hazm import Normalizer
from sklearn.model_selection import train_test_split, StratifiedKFold, ShuffleSplit, cross_val_score, cross_val_predict, \
    cross_validate
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from keras import preprocessing, optimizers
from keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout, Conv1D, GlobalMaxPooling1D, \
    GlobalAveragePooling1D, MaxPooling1D, Concatenate, Input, Reshape
from keras.models import Sequential, Model
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

from keras.engine import Layer, InputSpec
import tensorflow as tf
import nltk
import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.layers import Convolution1D
from keras import initializers, regularizers, constraints, optimizers, layers


class Processers:

    @staticmethod
    def train_cnn_original(enum_type):
        # constants
        maxlen = 100
        # vector_size = 150
        drop_rate = 0.5
        batch_size = 100
        epochs = 500
        optimizer = optimizers.adam(lr=0.001)
        filter_size_a = 2
        filter_size_b = 3
        filter_size_c = 4
        nb_filters = 150

        word_model = gensim.models.Word2Vec.load('./trained_word2vec/word2vec_model_64.bin')
        pretrained_weights = word_model.wv.syn0
        vocab_size, emdedding_size = pretrained_weights.shape

        # prepare data
        x_train, x_test, y_train, y_test = Filers.load_data(enum_type)
        x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
        x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

        earlystopping = EarlyStopping()
        callbacks = [earlystopping]

        # model initialization
        my_input = Input(shape=x_train[1].shape)
        embedding = Embedding(input_dim=vocab_size,
                              output_dim=emdedding_size,
                              weights=[pretrained_weights],
                              trainable=False)(my_input)

        # A branch
        conv_a = Conv1D(filters=nb_filters,
                        kernel_size=filter_size_a,
                        # activation='relu',
                        )(embedding)
        pooled_conv_a = GlobalMaxPooling1D()(conv_a)

        # B branch
        conv_b = Conv1D(filters=nb_filters,
                        kernel_size=filter_size_b,
                        # activation='relu',
                        )(embedding)
        pooled_conv_b = GlobalMaxPooling1D()(conv_b)

        # C branch
        conv_c = Conv1D(filters=nb_filters,
                        kernel_size=filter_size_c,
                        # activation='relu',
                        )(embedding)
        pooled_conv_c = GlobalMaxPooling1D()(conv_c)

        concat = Concatenate()([pooled_conv_a, pooled_conv_b, pooled_conv_c])
        concat_dropped = Dropout(drop_rate)(concat)

        # prob = Dense(units=2,
        #              activation='softmax')(concat_dropped)

        prob = Dense(units=1,
                     activation='sigmoid')(concat_dropped)

        model = Model(my_input, prob)
        # model.compile(loss='categorical_crossentropy',
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        model.summary()
        history = model.fit(x_train, y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_split=0.1,
                            callbacks=callbacks)

        # model evaluating
        score = model.evaluate(x_test, y_test)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])

        # binary test result
        # y_pred = model.predict(x_test)
        # print(y_pred[1])
        # print(y_test[1])
        # print(confusion_matrix(y_test, y_pred))
        # print(classification_report(y_test, y_pred))

        # # binary test result
        prediction = model.predict(x_test)
        y_pred = (prediction > 0.5)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred, digits=4))

        # categorical test result
        # y_prob = model.predict(x_test)
        # y_pred_class = y_prob.argmax(axis=-1)
        # y_test_final = y_test.argmax(axis=-1)
        # print(confusion_matrix(y_test_final, y_pred_class))
        # print(classification_report(y_test_final, y_pred_class, digits=4))

    @staticmethod
    def train_vdcnn_pos(*enum_types):
        # constants
        maxlen = 128
        batch_size = 100
        epochs = 20
        optimizer = optimizers.adam()
        nb_filter_a = 16
        nb_filter_b = 32
        nb_filter_c = 64
        nb_filter_d = 128

        word_model = gensim.models.Word2Vec.load('./trained_word2vec/word2vec_model_150.bin')
        pretrained_weights = word_model.wv.syn0
        vocab_size, emdedding_size = pretrained_weights.shape

        earlystopping = EarlyStopping()
        callbacks = [earlystopping]

        count_vect = CountVectorizer(tokenizer=nltk.tokenize.word_tokenize, ngram_range=(1, 3), max_features=4000)
        # tfidf = TfidfTransformer()
        tfidf_vect = TfidfVectorizer(tokenizer=nltk.tokenize.word_tokenize, ngram_range=(1, 3), max_features=4000)

        # prepare data
        list_of_x_train = []
        list_of_x_test = []
        for enum_type in enum_types:
            x_train, x_test, y_train, y_test = Filers.load_data(enum_type)
            if enum_type in (Filers.EnumTypes.ALL_INDEXES,
                             Filers.EnumTypes.NOTNOUN_INDEXES,
                             Filers.EnumTypes.ADJ_INDEXES,
                             Filers.EnumTypes.ADV_INDEXES,
                             Filers.EnumTypes.VERB_INDEXES,
                             Filers.EnumTypes.ADJ_ADV_VERB_INDEXES):
                x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
                x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
            else:
                x_train = tfidf_vect.fit_transform(x_train)
                x_test = tfidf_vect.fit_transform(x_test)
            list_of_x_train.append(x_train)
            list_of_x_test.append(x_test)

        # model initialization
        input_a = Input(shape=(maxlen,))
        # input_b = Input(shape=(4000,))

        # dense_tfidf = Dense(10)(input_b)
        # reshaped_idf = Reshape((4000, 1))(input_b)
        # conv_tfidf = Conv1D(nb_filter_a, kernel_size=3, padding="valid", activation='relu')(reshaped_idf)
        # pool_idf = GlobalMaxPooling1D()(conv_tfidf)
        embedding = Embedding(input_dim=vocab_size,
                              output_dim=emdedding_size,
                              weights=[pretrained_weights],
                              trainable=False)
        input_encoded_a = embedding(input_a)
        conv1 = Conv1D(nb_filter_a, kernel_size=3, padding="valid")
        conv1_a = conv1(input_encoded_a)

        conv2 = Conv1D(nb_filter_a, kernel_size=3, padding="same", activation='relu')
        conv2_a = conv2(conv1_a)
        batch2_a = BatchNormalization()(conv2_a)
        conv3 = Conv1D(nb_filter_a, kernel_size=3, padding="same", activation='relu')
        conv3_a = conv3(batch2_a)
        batch3_a = BatchNormalization()(conv3_a)
        pool1 = MaxPooling1D(pool_size=2, strides=3)
        pool1_a = pool1(batch3_a)

        conv4 = Conv1D(nb_filter_b, kernel_size=3, padding="same", activation='relu')
        conv4_a = conv4(pool1_a)
        batch4_a = BatchNormalization()(conv4_a)
        conv5 = Conv1D(nb_filter_b, kernel_size=3, padding="same", activation='relu')
        conv5_a = conv5(batch4_a)
        batch5_a = BatchNormalization()(conv5_a)
        pool2 = MaxPooling1D(pool_size=2, strides=3)
        pool2_a = pool2(batch5_a)

        conv6 = Conv1D(nb_filter_c, kernel_size=3, padding="same", activation='relu')
        conv6_a = conv6(pool2_a)
        batch6_a = BatchNormalization()(conv6_a)
        conv7 = Conv1D(nb_filter_c, kernel_size=3, padding="same", activation='relu')
        conv7_a = conv7(batch6_a)
        batch7_a = BatchNormalization()(conv7_a)
        pool3 = MaxPooling1D(pool_size=2, strides=3)
        pool3_a = pool3(batch7_a)

        conv8 = Conv1D(nb_filter_d, kernel_size=3, padding="same", activation='relu')
        conv8_a = conv8(pool3_a)
        batch8_a = BatchNormalization()(conv8_a)
        conv9 = Conv1D(nb_filter_d, kernel_size=3, padding="same", activation='relu')
        conv9_a = conv9(batch8_a)
        batch9_a = BatchNormalization()(conv9_a)
        pool4 = MaxPooling1D(pool_size=2, strides=3)
        pool4_a = pool4(batch9_a)
        flat = Flatten()(pool4_a)
        # concat = Concatenate()([flat, pool_idf])
        dense1 = Dense(nb_filter_d, activation='relu')(flat)
        dense2 = Dense(nb_filter_c, activation='relu')(dense1)
        dense3 = Dense(nb_filter_c, activation='relu')(dense2)
        prob = Dense(1, activation='sigmoid')(dense3)
        model = Model(inputs=[input_a,
                              # input_b,
                              ], outputs=prob)
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        model.summary()
        history = model.fit(x=list_of_x_train, y=y_train,
                            epochs=epochs,
                            validation_split=0.1,
                            batch_size=batch_size,
                            # callbacks=callbacks
                            )

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

        # model evaluating
        score = model.evaluate(list_of_x_test, y_test)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])

        # binary test result
        # y_pred = model.predict(x_test)
        # print(y_pred[1])
        # print(y_test[1])
        # print(confusion_matrix(y_test, y_pred))
        # print(classification_report(y_test, y_pred))

        # binary test result
        prediction = model.predict(list_of_x_test)
        y_pred = (prediction > 0.5)
        # y_pred = model.predict(list_of_x_test)X_te
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred, digits=4))

        # categorical test result
        # y_prob = model.predict(x_test)
        # y_pred_class = y_prob.argmax(axis=-1)
        # y_test_final = y_test.argmax(axis=-1)
        # print(confusion_matrix(y_test_final, y_pred_class))
        # print(classification_report(y_test_final, y_pred_class, digits=4))

    @staticmethod
    def train_lstm_kaggle(*enum_types):

        tokenizer = Tokenizer(num_words=6000)
        # prepare data
        list_of_x_train = []
        list_of_x_test = []
        for enum_type in enum_types:
            x_train, x_test, y_train, y_test = Filers.load_data(enum_type)
            if enum_type in (Filers.EnumTypes.ALL_INDEXES,
                             Filers.EnumTypes.NOTNOUN_INDEXES,
                             Filers.EnumTypes.ADJ_INDEXES,
                             Filers.EnumTypes.ADV_INDEXES,
                             Filers.EnumTypes.VERB_INDEXES,
                             Filers.EnumTypes.ADJ_ADV_VERB_INDEXES):
                x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=128)
                x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=128)
            else:
                tokenizer.fit_on_texts(x_train)
                tokenizer.fit_on_texts(x_test)
                x_train = tokenizer.texts_to_sequences(x_train)
                x_test = tokenizer.texts_to_sequences(x_test)
                X_t = pad_sequences(x_train, maxlen=128)
                X_te = pad_sequences(x_test, maxlen=128)
            list_of_x_train.append(x_train)
            list_of_x_test.append(x_test)

        model = Sequential()
        model.add(Embedding(6000, 128))
        model.add(Bidirectional(LSTM(32, return_sequences=True)))
        model.add(GlobalMaxPool1D())
        model.add(Dense(20, activation="relu"))
        model.add(Dropout(0.05))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(X_t, y_train, batch_size=100, epochs=3, validation_split=0.2)

        prediction = model.predict(X_te)
        y_pred = (prediction > 0.5)

        # binary test result
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred, digits=4))

    @staticmethod
    def train_cnn_modified(*enum_types):
        # constants
        maxlen = 100
        # vector_size = 150
        drop_rate = 0.5
        batch_size = 100
        epochs = 20
        optimizer = optimizers.adam(lr=0.001)
        filter_size_a = 2
        filter_size_b = 3
        filter_size_c = 4
        nb_filters = 150

        word_model = gensim.models.Word2Vec.load('./trained_word2vec/word2vec_model_150.bin')
        pretrained_weights = word_model.wv.syn0
        vocab_size, emdedding_size = pretrained_weights.shape

        count_vect = CountVectorizer(tokenizer=nltk.tokenize.word_tokenize, ngram_range=(1, 3), max_features=4000)
        # tfidf = TfidfTransformer()
        tfidf_vect = TfidfVectorizer(tokenizer=nltk.tokenize.word_tokenize, ngram_range=(1, 3), max_features=4000)

        # prepare data
        list_of_x_train = []
        list_of_x_test = []
        for enum_type in enum_types:
            x_train, x_test, y_train, y_test = Filers.load_data(enum_type)
            if enum_type in (Filers.EnumTypes.ALL_INDEXES,
                             Filers.EnumTypes.NOTNOUN_INDEXES,
                             Filers.EnumTypes.ADJ_INDEXES,
                             Filers.EnumTypes.ADV_INDEXES,
                             Filers.EnumTypes.VERB_INDEXES,
                             Filers.EnumTypes.ADJ_ADV_VERB_INDEXES):
                x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
                x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
            else:
                x_train = tfidf_vect.fit_transform(x_train)
                x_test = tfidf_vect.fit_transform(x_test)
            list_of_x_train.append(x_train)
            list_of_x_test.append(x_test)

        # model initialization
        input_a = Input(shape=(maxlen,))
        input_b = Input(shape=(maxlen,))
        embedding = Embedding(input_dim=vocab_size,
                              output_dim=emdedding_size,
                              weights=[pretrained_weights],
                              trainable=False)(input_a)

        embedding2 = Embedding(input_dim=vocab_size,
                               output_dim=emdedding_size,
                               weights=[pretrained_weights],
                               trainable=False)(input_b)

        embedding_dropped = Dropout(drop_rate)(embedding)

        embedding_dropped2 = Dropout(drop_rate)(embedding2)

        # A branch
        conv_a = Conv1D(filters=nb_filters,
                        kernel_size=filter_size_a,
                        activation='relu',
                        )(embedding_dropped)
        pooled_conv_a = GlobalMaxPooling1D()(conv_a)
        pooled_conv_dropped_a = Dropout(drop_rate)(pooled_conv_a)

        # B branch
        conv_b = Conv1D(filters=nb_filters,
                        kernel_size=filter_size_b,
                        activation='relu',
                        )(embedding_dropped)
        pooled_conv_b = GlobalMaxPooling1D()(conv_b)
        pooled_conv_dropped_b = Dropout(drop_rate)(pooled_conv_b)

        # C branch
        conv_c = Conv1D(filters=nb_filters,
                        kernel_size=filter_size_c,
                        activation='relu',
                        )(embedding_dropped)
        pooled_conv_c = GlobalMaxPooling1D()(conv_c)
        pooled_conv_dropped_c = Dropout(drop_rate)(pooled_conv_c)

        attention_dense = Dense(units=16)(embedding_dropped2)
        pooled_attention = GlobalMaxPooling1D()(attention_dense)

        concat = Concatenate()([pooled_conv_dropped_a, pooled_conv_dropped_b, pooled_conv_dropped_c, pooled_attention])
        concat_dropped = Dropout(drop_rate)(concat)

        # prob = Dense(units=2,
        #              activation='softmax')(concat_dropped)

        prob = Dense(units=1,
                     activation='sigmoid')(concat_dropped)

        model = Model(inputs=[input_a,
                              input_b,
                              ], outputs=prob)
        # model.compile(loss='categorical_crossentropy',
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        model.summary()
        history = model.fit(x=list_of_x_train, y=y_train,
                            epochs=epochs,
                            batch_size=batch_size)

        # model evaluating
        score = model.evaluate(list_of_x_test, y_test)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])

        # binary test result
        # y_pred = model.predict(x_test)
        # print(y_pred[1])
        # print(y_test[1])
        # print(confusion_matrix(y_test, y_pred))
        # print(classification_report(y_test, y_pred))

        # binary test result
        prediction = model.predict(list_of_x_test)
        y_pred = (prediction > 0.5)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred, digits=4))

        # categorical test result
        # y_prob = model.predict(x_test)
        # y_pred_class = y_prob.argmax(axis=-1)
        # y_test_final = y_test.argmax(axis=-1)
        # print(confusion_matrix(y_test_final, y_pred_class))
        # print(classification_report(y_test_final, y_pred_class, digits=4))

    @staticmethod
    def train_cnn_lstm(*enum_types):
        # constants
        maxlen = 100
        # vector_size = 150
        drop_rate = 0.5
        batch_size = 100
        epochs = 20
        optimizer = optimizers.adam(lr=0.001)
        filter_size_a = 2
        filter_size_b = 3
        filter_size_c = 4
        nb_filters = 150
        lstm_units = 100

        word_model = gensim.models.Word2Vec.load('./trained_word2vec/word2vec_model_150.bin')
        pretrained_weights = word_model.wv.syn0
        vocab_size, emdedding_size = pretrained_weights.shape

        count_vect = CountVectorizer(tokenizer=nltk.tokenize.word_tokenize, ngram_range=(1, 3), max_features=4000)
        tfidf_vect = TfidfVectorizer(tokenizer=nltk.tokenize.word_tokenize, ngram_range=(1, 3), max_features=4000)

        earlystopping = EarlyStopping()
        callbacks = [earlystopping]

        # prepare data
        list_of_x_train = []
        list_of_x_test = []
        for enum_type in enum_types:
            x_train, x_test, y_train, y_test = Filers.load_data(enum_type)
            if enum_type in (Filers.EnumTypes.ALL_INDEXES,
                             Filers.EnumTypes.NOTNOUN_INDEXES,
                             Filers.EnumTypes.ADJ_INDEXES,
                             Filers.EnumTypes.ADV_INDEXES,
                             Filers.EnumTypes.VERB_INDEXES,
                             Filers.EnumTypes.ADJ_ADV_VERB_INDEXES):
                x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
                x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
            else:
                x_train = tfidf_vect.fit_transform(x_train)
                x_test = tfidf_vect.fit_transform(x_test)
            list_of_x_train.append(x_train)
            list_of_x_test.append(x_test)

        # model initialization
        my_input = Input(shape=x_train[1].shape)
        embedding = Embedding(input_dim=vocab_size,
                              output_dim=emdedding_size,
                              weights=[pretrained_weights],
                              trainable=False)

        input_encoded_a = embedding(my_input)
        # conv1 = Conv1D(nb_filters, kernel_size=3, padding="valid")
        # conv1_a = conv1(input_encoded_a)
        #
        # conv2 = Conv1D(nb_filters, kernel_size=3, padding="same", activation='relu')
        # conv2_a = conv2(conv1_a)
        # batch2_a = BatchNormalization()(conv2_a)
        # conv3 = Conv1D(nb_filters, kernel_size=3, padding="same", activation='relu')
        # conv3_a = conv3(batch2_a)
        # batch3_a = BatchNormalization()(conv3_a)
        # pool1 = MaxPooling1D(pool_size=2, strides=3)
        # pool1_a = pool1(batch3_a)

        conv_a = Conv1D(filters=nb_filters,
                        kernel_size=filter_size_b,
                        activation='relu',
                        )(input_encoded_a)

        pooled_conv_b = MaxPooling1D(pool_size=2)(conv_a)
        # pooled_conv_b = GlobalMaxPooling1D()(conv_a)

        lstm_a = GRU(lstm_units,
                      # return_sequences=True,
                      dropout=drop_rate)(pooled_conv_b)

        # pooled_conv_h = MaxPooling1D(pool_size=3)(conv_h)

        # pooled_conv = GlobalMaxPooling1D()(conv_h)
        # concat = Concatenate()([pooled_conv_a, pooled_conv_b, pooled_conv_c])
        # concat_dropped = Dropout(drop_rate)(pooled_conv)

        prob = Dense(units=1,
                     activation='sigmoid')(lstm_a)

        model = Model(my_input, prob)
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        model.summary()
        history = model.fit(list_of_x_train, y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_split=0.1,
                            # callbacks=callbacks
                            )

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

        # model evaluating
        scores = model.evaluate(list_of_x_test, y_test)
        y_pred = model.predict(list_of_x_test)
        print('Test score:', scores[0])
        print('Test accuracy:', scores[1])

        # binary test result
        prediction = model.predict(list_of_x_test)
        y_pred = (prediction > 0.5)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred, digits=4))

    @staticmethod
    def train_cnn_lstm_paper(*enum_types):
        # constants
        maxlen = 100
        # vector_size = 150
        drop_rate = 0.5
        batch_size = 100
        epochs = 100
        optimizer = optimizers.adam(lr=0.001)
        filter_size_a = 2
        filter_size_b = 3
        filter_size_c = 4
        nb_filters = 150
        lstm_units = 100

        word_model = gensim.models.Word2Vec.load('./trained_word2vec/word2vec_model_150.bin')
        pretrained_weights = word_model.wv.syn0
        vocab_size, emdedding_size = pretrained_weights.shape

        count_vect = CountVectorizer(tokenizer=nltk.tokenize.word_tokenize, ngram_range=(1, 3), max_features=4000)
        tfidf_vect = TfidfVectorizer(tokenizer=nltk.tokenize.word_tokenize, ngram_range=(1, 3), max_features=4000)

        earlystopping = EarlyStopping()
        callbacks = [earlystopping]

        # prepare data
        list_of_x_train = []
        list_of_x_test = []
        for enum_type in enum_types:
            x_train, x_test, y_train, y_test = Filers.load_data(enum_type)
            if enum_type in (Filers.EnumTypes.ALL_INDEXES,
                             Filers.EnumTypes.NOTNOUN_INDEXES,
                             Filers.EnumTypes.ADJ_INDEXES,
                             Filers.EnumTypes.ADV_INDEXES,
                             Filers.EnumTypes.VERB_INDEXES,
                             Filers.EnumTypes.ADJ_ADV_VERB_INDEXES):
                x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
                x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
            else:
                x_train = tfidf_vect.fit_transform(x_train)
                x_test = tfidf_vect.fit_transform(x_test)
            list_of_x_train.append(x_train)
            list_of_x_test.append(x_test)
            # model initialization
            my_input = Input(shape=x_train[1].shape)
            embedding = Embedding(input_dim=vocab_size,
                                  output_dim=emdedding_size,
                                  weights=[pretrained_weights],
                                  trainable=True)

            # embedding = Dropout(0.50)(embedding)
            input_encoded_a = embedding(my_input)

            conv1 = Conv1D(nb_filters, kernel_size=4, padding="valid", activation='relu')
            conv1_a = conv1(input_encoded_a)
            batch1_a = BatchNormalization()(conv1_a)
            pool1 = MaxPooling1D(pool_size=2)
            pool1_a = pool1(batch1_a)

            conv2 = Conv1D(nb_filters, kernel_size=5, padding="valid", activation='relu')
            conv2_a = conv2(input_encoded_a)
            batch2_a = BatchNormalization()(conv2_a)
            pool2 = MaxPooling1D(pool_size=2)
            pool2_a = pool2(batch2_a)

            concat = Concatenate()([pool1_a, pool2_a])

            concat = Dropout(0.15)(concat)

            concat = GRU(100)(concat)

            concat = Dense(400, activation='relu', init='he_normal')(concat)
            # W_constraint=maxnorm(3), b_constraint=maxnorm(3),
            # name='mlp')(x)

            concat = Dropout(0.10, name='drop')(concat)

            output = Dense(1, init='he_normal',
                           activation='sigmoid', name='output')(concat)

            model = Model(my_input, output)
            model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        model.summary()
        history = model.fit(list_of_x_train, y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_split=0.1,
                            callbacks=callbacks)

        # model evaluating
        scores = model.evaluate(list_of_x_test, y_test)
        y_pred = model.predict(list_of_x_test)
        print('Test score:', scores[0])
        print('Test accuracy:', scores[1])

        # binary test result
        prediction = model.predict(list_of_x_test)
        y_pred = (prediction > 0.5)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred, digits=4))
