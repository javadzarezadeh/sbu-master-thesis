import numpy as np
import gensim
from digikala.Filers import Filers
from digikala.Helpers import Helpers
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedKFold
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.utils import indexable
from keras import preprocessing, optimizers
from keras.utils import np_utils
from keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout, Conv1D, Conv2D, GlobalMaxPooling1D, \
    GlobalAveragePooling1D, MaxPooling1D, Concatenate, Input, BatchNormalization, Activation, Flatten, Reshape, GRU
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping
from itertools import chain
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

from keras.engine import Layer, InputSpec
import tensorflow as tf
from hazm import WordTokenizer
import matplotlib.pyplot as plt


class Processers:

    @staticmethod
    def train_cnn_original(*enum_types):
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

        word_model = gensim.models.Word2Vec.load('./trained_word2vec/word2vec_model_512.bin')
        pretrained_weights = word_model.wv.syn0
        vocab_size, emdedding_size = pretrained_weights.shape

        tokenizer = WordTokenizer()
        count_vect = CountVectorizer(tokenizer=tokenizer.tokenize, ngram_range=(1, 3), max_features=4000)
        tfidf_vect = TfidfVectorizer(tokenizer=tokenizer.tokenize, ngram_range=(1, 3), max_features=4000)

        earlystopping = EarlyStopping()
        callbacks = [earlystopping]

        # prepare data
        list_of_x_train = []
        list_of_x_test = []
        for enum_type in enum_types:
            x_train, x_test, y_train, y_test = Filers.load_data_percentage(enum_type)
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
        my_input = Input(shape=(maxlen,))
        embedding = Embedding(input_dim=vocab_size,
                              output_dim=emdedding_size,
                              weights=[pretrained_weights],
                              trainable=True)(my_input)

        embedding_dropped = Dropout(drop_rate)(embedding)

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

        concat = Concatenate()([pooled_conv_dropped_a, pooled_conv_dropped_b, pooled_conv_dropped_c])
        concat_dropped = Dropout(drop_rate)(concat)

        # prob = Dense(units=2,
        #              activation='softmax')(concat_dropped)

        prob = Dense(units=1,
                     activation='sigmoid')(concat_dropped)

        model = Model(my_input, prob)
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

        # categorical test result
        # y_prob = model.predict(list_of_x_test)
        # y_pred_class = y_prob.argmax(axis=-1)
        # y_test_final = y_test.argmax(axis=-1)
        # print(confusion_matrix(y_test_final, y_pred_class))
        # print(classification_report(y_test_final, y_pred_class, digits=4))

        # binary test result
        prediction = model.predict(list_of_x_test)
        y_pred = (prediction > 0.5)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred, digits=4))

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
        tokenizer = WordTokenizer()
        count_vect = CountVectorizer(tokenizer=tokenizer.tokenize, ngram_range=(1, 3), max_features=4000)
        # tfidf = TfidfTransformer()
        tfidf_vect = TfidfVectorizer(tokenizer=tokenizer.tokenize, ngram_range=(1, 3), max_features=4000)

        earlystopping = EarlyStopping()
        callbacks = [earlystopping]

        # prepare data
        list_of_x_train = []
        list_of_x_test = []
        for enum_type in enum_types:
            x_train, x_test, y_train, y_test = Filers.load_data_percentage(enum_type)
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
        # print(classification_report(y_test, y_pred, digits=4))

        filename = 'vdcnn_' + str(enum_types[0])
        report = classification_report(y_test, y_pred, digits=4)
        Helpers.classification_report_csv(report, filename)

    @staticmethod
    def train_cnn_original_kfold(*enum_types):
        # constants
        maxlen = 100
        # vector_size = 150
        drop_rate = 0.5
        batch_size = 100
        epochs = 6
        optimizer = optimizers.adam(lr=0.001)
        filter_size_a = 2
        filter_size_b = 3
        filter_size_c = 4
        nb_filters = 150

        word_model = gensim.models.Word2Vec.load('./trained_word2vec/word2vec_model_150.bin')
        pretrained_weights = word_model.wv.syn0
        vocab_size, emdedding_size = pretrained_weights.shape

        count_vect = CountVectorizer(tokenizer=WordTokenizer.tokenize, ngram_range=(1, 3), max_features=4000)
        # tfidf = TfidfTransformer()
        tfidf_vect = TfidfVectorizer(tokenizer=WordTokenizer.tokenize, ngram_range=(1, 3), max_features=4000)

        # prepare data
        list_of_x = []
        for enum_type in enum_types:
            x, y = Filers.load_data(enum_type)
            if enum_type in (Filers.EnumTypes.ALL_INDEXES,
                             Filers.EnumTypes.NOTNOUN_INDEXES,
                             Filers.EnumTypes.ADJ_INDEXES,
                             Filers.EnumTypes.ADV_INDEXES,
                             Filers.EnumTypes.VERB_INDEXES,
                             Filers.EnumTypes.ADJ_ADV_VERB_INDEXES):
                x = preprocessing.sequence.pad_sequences(x, maxlen=maxlen)
            else:
                x = tfidf_vect.fit_transform(x)
            list_of_x.append(x)

        kfold = StratifiedKFold(n_splits=10, shuffle=False, random_state=42)
        cvscores = []
        confusion_matrices = []
        precisions_positive = []
        precisions_negative = []
        recalls_positive = []
        recalls_negative = []
        f1s_positive = []
        f1s_negative = []
        supports_positive = []
        supports_negative = []
        for train, test in kfold.split(x, y):
            # model initialization
            my_input = Input(shape=(maxlen,))
            embedding = Embedding(input_dim=vocab_size,
                                  output_dim=emdedding_size,
                                  weights=[pretrained_weights],
                                  trainable=True)(my_input)

            embedding_dropped = Dropout(drop_rate)(embedding)

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

            concat = Concatenate()([pooled_conv_dropped_a, pooled_conv_dropped_b, pooled_conv_dropped_c])
            concat_dropped = Dropout(drop_rate)(concat)

            # prob = Dense(units=2,
            #              activation='softmax')(concat_dropped)

            prob = Dense(units=1,
                         activation='sigmoid')(concat_dropped)

            model = Model(my_input, prob)
            model.compile(loss='binary_crossentropy',
                          optimizer=optimizer,
                          metrics=['accuracy'])
            model.summary()
            history = model.fit(x[train], y[train],
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_split=0.1)

            # # model evaluating
            scores = model.evaluate(x[test], y[test])
            y_pred = model.predict(x[test])

            confusion_matrices.append(confusion_matrix(y[test], y_pred.round()))
            p_p, r_p, f1_p, s_p = precision_recall_fscore_support(y[test], y_pred.round(), labels=[1])
            p_n, r_n, f1_n, s_n = precision_recall_fscore_support(y[test], y_pred.round(), labels=[0])
            precisions_positive.append(p_p)
            precisions_negative.append(p_n)
            recalls_positive.append(r_p)
            recalls_negative.append(r_n)
            f1s_positive.append(f1_p)
            f1s_negative.append(f1_n)
            supports_positive.append(s_p)
            supports_negative.append(s_n)
            # print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
            cvscores.append(scores[1] * 100)

        precisions_average_positive = np.mean(np.array(precisions_positive)) * 100
        precisions_average_negative = np.mean(np.array(precisions_negative)) * 100
        recalls_average_positive = np.mean(np.array(recalls_positive)) * 100
        recalls_average_negative = np.mean(np.array(recalls_negative)) * 100
        f1s_average_positive = np.mean(np.array(f1s_positive)) * 100
        f1s_average_negative = np.mean(np.array(f1s_negative)) * 100
        supports_average_positive = np.mean(np.array(supports_positive))
        supports_average_negative = np.mean(np.array(supports_negative))

        for cm in confusion_matrices:
            print(cm)
        print('Positive Precision mean: %.2f%%' % precisions_average_positive)
        print('Negative Precision mean: %.2f%%' % precisions_average_negative)
        print('Positive Recall mean: %.2f%%' % recalls_average_positive)
        print('Negative Recall mean: %.2f%%' % recalls_average_negative)
        print('Positive F1-score mean: %.2f%%' % f1s_average_positive)
        print('Negative F1-score mean: %.2f%%' % f1s_average_negative)

        print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
        # print('Test score:', score[0])
        # print('Test accuracy:', score[1])

    @staticmethod
    def train_vdcnn_pos_kfold(enum_type):
        # constants
        maxlen = 128
        batch_size = 100
        epochs = 8
        optimizer = optimizers.adam()
        nb_filter_a = 16
        nb_filter_b = 32
        nb_filter_c = 64
        nb_filter_d = 128

        word_model = gensim.models.Word2Vec.load('./trained_word2vec/word2vec_model_150.bin')
        pretrained_weights = word_model.wv.syn0
        vocab_size, emdedding_size = pretrained_weights.shape

        # prepare data
        x, y = Filers.load_data(enum_type)
        x = preprocessing.sequence.pad_sequences(x, maxlen=maxlen)

        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cvscores = []
        confusion_matrices = []
        precisions_positive = []
        precisions_negative = []
        recalls_positive = []
        recalls_negative = []
        f1s_positive = []
        f1s_negative = []
        supports_positive = []
        supports_negative = []
        for train, test in kfold.split(x, y):
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
            history = model.fit(x[train], y[train],
                                epochs=epochs,
                                batch_size=batch_size)

            # # model evaluating
            scores = model.evaluate(x[test], y[test])
            y_pred = model.predict(x[test])

            confusion_matrices.append(confusion_matrix(y[test], y_pred.round()))
            p_p, r_p, f1_p, s_p = precision_recall_fscore_support(y[test], y_pred.round(), labels=[1])
            p_n, r_n, f1_n, s_n = precision_recall_fscore_support(y[test], y_pred.round(), labels=[0])
            precisions_positive.append(p_p)
            precisions_negative.append(p_n)
            recalls_positive.append(r_p)
            recalls_negative.append(r_n)
            f1s_positive.append(f1_p)
            f1s_negative.append(f1_n)
            supports_positive.append(s_p)
            supports_negative.append(s_n)
            # print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
            cvscores.append(scores[1] * 100)

        precisions_average_positive = np.mean(np.array(precisions_positive)) * 100
        precisions_average_negative = np.mean(np.array(precisions_negative)) * 100
        recalls_average_positive = np.mean(np.array(recalls_positive)) * 100
        recalls_average_negative = np.mean(np.array(recalls_negative)) * 100
        f1s_average_positive = np.mean(np.array(f1s_positive)) * 100
        f1s_average_negative = np.mean(np.array(f1s_negative)) * 100
        supports_average_positive = np.mean(np.array(supports_positive))
        supports_average_negative = np.mean(np.array(supports_negative))

        for cm in confusion_matrices:
            print(cm)
        print('Positive Precision mean: %.2f%%' % precisions_average_positive)
        print('Negative Precision mean: %.2f%%' % precisions_average_negative)
        print('Positive Recall mean: %.2f%%' % recalls_average_positive)
        print('Negative Recall mean: %.2f%%' % recalls_average_negative)
        print('Positive F1-score mean: %.2f%%' % f1s_average_positive)
        print('Negative F1-score mean: %.2f%%' % f1s_average_negative)

        print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
        # print('Test score:', score[0])
        # print('Test accuracy:', score[1])

    @staticmethod
    def train_cnn_modified(*enum_types):
        # constants
        maxlen = 128
        # vector_size = 150
        drop_rate = 0.5
        batch_size = 100
        epochs = 1
        optimizer = optimizers.adam(lr=0.001)
        filter_size_a = 2
        filter_size_b = 3
        filter_size_c = 4
        nb_filters = 150

        word_model = gensim.models.Word2Vec.load('./trained_word2vec/word2vec_model_150.bin')
        pretrained_weights = word_model.wv.syn0
        vocab_size, emdedding_size = pretrained_weights.shape

        tokenizer = WordTokenizer()
        count_vect = CountVectorizer(tokenizer=tokenizer.tokenize, ngram_range=(1, 3), max_features=4000)
        # tfidf = TfidfTransformer()
        tfidf_vect = TfidfVectorizer(tokenizer=tokenizer.tokenize, ngram_range=(1, 3), max_features=4000)

        # prepare data
        list_of_x_train = []
        list_of_x_test = []
        for enum_type in enum_types:
            # x_train, x_test, y_train, y_test = Filers.load_data_percentage(enum_type)
            x_train, x_test, y_train, y_test = Filers.load_data_percentage(enum_type)
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
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        model.summary()
        history = model.fit(list_of_x_train, y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_split=0.1)

        # model evaluating
        scores = model.evaluate(list_of_x_test, y_test)
        y_pred = model.predict(list_of_x_test)
        print('Test score:', scores[0])
        print('Test accuracy:', scores[1])

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

        tokenizer = WordTokenizer()
        count_vect = CountVectorizer(tokenizer=tokenizer.tokenize, ngram_range=(1, 3), max_features=4000)
        tfidf_vect = TfidfVectorizer(tokenizer=tokenizer.tokenize, ngram_range=(1, 3), max_features=4000)

        earlystopping = EarlyStopping()
        callbacks = [earlystopping]

        # prepare data
        list_of_x_train = []
        list_of_x_test = []
        for enum_type in enum_types:
            x_train, x_test, y_train, y_test = Filers.load_data_percentage(enum_type)
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
        # print(classification_report(y_test, y_pred, digits=4))

        filename = 'gru_' + str(enum_types[0])
        report = classification_report(y_test, y_pred, digits=4)
        Helpers.classification_report_csv(report, filename)

    @staticmethod
    def train_cnn_lstm_paper(*enum_types):
        # constants
        maxlen = 100
        # vector_size = 150
        drop_rate = 0.5
        batch_size = 100
        epochs = 100
        optimizer = optimizers.adam(lr=0.001)
        filter_size_a = 5
        # filter_size_b = 3
        # filter_size_c = 4
        nb_filters = 200
        lstm_units = 100

        word_model = gensim.models.Word2Vec.load('./trained_word2vec/word2vec_model_150.bin')
        pretrained_weights = word_model.wv.syn0
        vocab_size, emdedding_size = pretrained_weights.shape

        tokenizer = WordTokenizer()
        count_vect = CountVectorizer(tokenizer=tokenizer.tokenize, ngram_range=(1, 3), max_features=4000)
        tfidf_vect = TfidfVectorizer(tokenizer=tokenizer.tokenize, ngram_range=(1, 3), max_features=4000)

        earlystopping = EarlyStopping()
        callbacks = [earlystopping]

        # prepare data
        list_of_x_train = []
        list_of_x_test = []
        for enum_type in enum_types:
            x_train, x_test, y_train, y_test = Filers.load_data_percentage(enum_type)
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
