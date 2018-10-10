from models.model import Model
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold


class DNN(Model):
    def __init__(self, train_x, train_y, test_x, test_y, learning_rate):
        Model.__init__(self)
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.learning_rate = learning_rate

    def train(self):
        #  train_x = self.min_max_normalized(self.train_x)
        #  test_x = self.min_max_normalized(self.test_x)
        #  train_x = np.nan_to_num(train_x)
        #  test_x = np.nan_to_num(test_x)
        train_x = self.train_x
        test_x = self.test_x
        train_y = self.train_y
        test_y = self.test_y

        model = keras.Sequential()
        model.add(keras.layers.Dense(300, activation='relu'))
        #  model.add(keras.layers.Dense(16, activation='relu'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))

        model.compile(
            optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        history = model.fit(
            train_x,
            train_y,
            batch_size=10,
            epochs=200,
            validation_data=(test_x, test_y),
            verbose=1
        )

        history_dict = history.history
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1, len(acc) + 1)

        plt.plot(epochs, acc, 'b', label='train accuracy')
        plt.plot(epochs, val_acc, '#000000', label='test accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()

        plt.show()
        return model


class CrossValidationDNN(Model):
    def __init__(self, train_x, train_y, test_x, test_y, learning_rate, folds):
        Model.__init__(self)
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.learning_rate = learning_rate
        self.folds = folds
        self.model = None

    def create(self):
        self.model = None  # Clear model.
        model = keras.Sequential()
        model.add(keras.layers.Dense(32, activation='relu'))
        model.add(keras.layers.Dense(32, activation='relu'))
        model.add(keras.layers.Dense(32, activation='relu'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))

        model.compile(
            optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        self.model = model

    def train(self, x_train, y_train, x_val, y_val):
        history = self.model.fit(
            x_train,
            y_train,
            batch_size=10,
            epochs=200,
            validation_data=(x_val, y_val),
            verbose=0
        )
        return history

    @staticmethod
    def _plot_training(history):
        history_dict = history.history
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1, len(acc) + 1)

        plt.plot(epochs, acc, 'b', label='train accuracy')
        plt.plot(epochs, val_acc, '#000000', label='test accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()

        plt.show()

    def train_with_cross_validation(self):
        skf = StratifiedKFold(n_splits=self.folds, shuffle=True)
        for index, (train_indices, val_indices) in enumerate(skf.split(self.train_x, self.train_y)):
            print("Training on fold " + str(index+1) + "/"+str(self.folds)+"...")
            # Generate batches from indices.
            x_train, x_val = self.train_x[train_indices], self.train_x[val_indices]
            y_train, y_val = self.train_y[train_indices], self.train_y[val_indices]
            self.create()

            print("Training new iteration on " + str(x_train.shape[0]) + " training samples, " + str(x_val.shape[0]) +
                  " validation samples, this may be a while...")

            history = self.train(x_train, y_train, x_val, y_val)
            self._plot_training(history)
            accuracy_history = history.history['acc']
            val_accuracy_history = history.history['val_acc']
            print(
                "Last training accuracy: "
                + str(accuracy_history[-1])
                + ", last validation accuracy: "
                + str(val_accuracy_history[-1])
            )
