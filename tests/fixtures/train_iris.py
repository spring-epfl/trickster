import h5py
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.datasets import load_iris


def get_data():
    iris = load_iris()
    X = iris['data']
    y = np_utils.to_categorical(iris['target'])
    return X, y


def build_model():
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_model():
    seed = 7
    np.random.seed(seed)
    X, y = get_data()
    print('X shape:', X.shape)
    print('Y shape:', y.shape)
    clf = build_model()
    clf.fit(X, y,
            shuffle=True, validation_split=.2,
            batch_size=10, epochs=300)
    return clf


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train MLP on Iris dataset.')
    parser.add_argument('--outfile', type=str, default='iris_model.keras',
                        help='Where to save the model')
    args = parser.parse_args()

    clf = train_model()
    clf.save(args.outfile)

