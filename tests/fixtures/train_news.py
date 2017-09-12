import h5py
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Dense, Conv1D, Dropout, GlobalMaxPooling1D, MaxPooling1D
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.datasets import fetch_20newsgroups


MAX_NUM_WORDS = 10000
MAX_SEQUENCE_LENGTH = 150


def get_data():
    newsgroups = fetch_20newsgroups()
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(newsgroups.data)
    sequences = tokenizer.texts_to_sequences(newsgroups.data)
    X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    y = np_utils.to_categorical(newsgroups.target)
    return X, y


def build_model(input_dim, num_classes):
    model = Sequential()
    model.add(Embedding(MAX_NUM_WORDS, 50, input_length=MAX_SEQUENCE_LENGTH))
    model.add(Dropout(0.2))

    model.add(Conv1D(filters=128, kernel_size=3, activation='elu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation='softmax'))
    model.compile(
            loss='categorical_crossentropy',
            optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def train_model():
    seed = 7
    np.random.seed(seed)
    X, y = get_data()
    print('X shape:', X.shape)
    print('Y shape:', y.shape)
    clf = build_model(X.shape[-1], y.shape[-1])
    clf.fit(X, y,
            shuffle=True, validation_split=.2,
            batch_size=64, epochs=16)
    return clf


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train a convnet on 20news.')
    parser.add_argument('--outfile', type=str, default='news_model.keras',
                        help='Where to save the model')
    args = parser.parse_args()

    clf = train_model()
    clf.save(args.outfile)

