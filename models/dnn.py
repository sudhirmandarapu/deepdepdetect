from models.model import Model

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt


class DNN(Model):
    def __init__(self, train_x, train_y, test_x, test_y, learning_rate):
        Model.__init__(self)
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.learning_rate = learning_rate

    def train(self):
        train_x = self.min_max_normalized(self.train_x)
        test_x = self.min_max_normalized(self.test_x)
        train_x = np.nan_to_num(train_x)
        test_x = np.nan_to_num(test_x)

        model = keras.Sequential()
        model.add(keras.layers.Dense(32, activation='relu'))
        model.add(keras.layers.Dense(16, activation='relu'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))

        model.compile(
            optimizer=tf.train.AdamOptimizer(),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        history = model.fit(
            train_x,
            self.train_y,
            batch_size=5,
            epochs=30,
            validation_data=(test_x, self.test_y),
            verbose=1
        )

        history_dict = history.history
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1, len(acc) + 1)

        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()
        return model
