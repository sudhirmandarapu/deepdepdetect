from models.model import Model
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from keras import backend


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
        model.add(keras.layers.Dense(300, activation='relu'))
        model.add(keras.layers.Dense(300, activation='relu'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))

        model.compile(
            optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 'mae']
        )

        history = model.fit(
            train_x,
            train_y,
            batch_size=10,
            epochs=160,
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
        model.add(keras.layers.Dense(300, activation='relu'))
        model.add(keras.layers.Dense(300, activation='relu'))
        model.add(keras.layers.Dense(300, activation='relu'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))

        model.compile(
            optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', rmse]
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

        #plt.plot(epochs, acc, 'b', label='train accuracy')
        #plt.plot(epochs, val_acc, '#000000', label='test accuracy')
        #plt.title('Training and validation accuracy')
        #plt.xlabel('epoch')
        #plt.ylabel('accuracy')
        #plt.legend()

        '''
        plt.title('Root mean square error of model over time')
        plt.xlabel('epoch')
        plt.ylabel('RMSE')
        plt.plot(epochs, history.history['rmse'], label="Training RMSE")
        plt.plot(epochs, history.history['val_rmse'], label="Testing RMSE")
        print(history.history['val_rmse'])
        plt.show()
        '''

        #plt.show()

    def train_with_cross_validation(self):
        skf = StratifiedKFold(n_splits=self.folds, shuffle=True)
        root_mean_squared_error = []
        acc = []
        val_root_mean_squared_errors = []
        val_acc = []
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
            root_mean_squared_error.append(history.history['rmse'][-1])
            val_root_mean_squared_errors.append(history.history['val_rmse'][-1])
            acc.append(accuracy_history[-1])
            val_acc.append(val_accuracy_history[-1])
            print(
                "Last training accuracy: "
                + str(accuracy_history[-1])
                + ", last validation accuracy: "
                + str(val_accuracy_history[-1])
            )
        print("Avg rmse: "+str(sum(root_mean_squared_error)/len(root_mean_squared_error))
              + ", Avg val_rmse: "+str(+sum(val_root_mean_squared_errors)/len(val_root_mean_squared_errors)))

        print("Avg acc: "+str(sum(acc)/len(acc))
              + ", Avg val_acc: "+str(+sum(val_acc)/len(val_acc)))


def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))