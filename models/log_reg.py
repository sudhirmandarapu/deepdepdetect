from models.model import Model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class LogRegModel(Model):
    def __init__(self, train_x, train_y, test_x, test_y):
        Model.__init__(self)
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

    def train(self):
        # Normalize the train and test sets.
        # train_x = self.min_max_normalized(self.train_x)
        # test_x = self.min_max_normalized(self.test_x)

        # There are some nan due to the fact that the max and min for those features are both 0. In this case, just
        # set the value to 0
        # train_x = np.nan_to_num(train_x)
        # test_x = np.nan_to_num(test_x)

        train_x = self.train_x
        test_x = self.test_x

        # Initialize the weights of the logistic regression.
        W = tf.Variable(tf.random_normal(shape=[train_x.shape[1], 1]))
        b = tf.Variable(tf.random_normal(shape=[1, 1]))
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        # Define the placeholders.
        data = tf.placeholder(dtype=tf.float32, shape=[None, train_x.shape[1]])
        target = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        assert data.shape[1] == W.shape[0]

        # Declare the model that has to be learned.
        mod = tf.matmul(data, W) + b

        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=mod, labels=target))

        # Other hyper-parameters.
        learning_rate = 0.001
        batch_size = 10
        iter_num = 10000

        # Define the optimizer, and the goal which is the minimize the loss.
        opt = tf.train.GradientDescentOptimizer(learning_rate)
        goal = opt.minimize(loss)

        # Sigmoid activation function.
        prediction = tf.round(tf.sigmoid(mod))
        correct = tf.cast(tf.equal(prediction, target), dtype=tf.float32)
        accuracy = tf.reduce_mean(correct)

        loss_trace = []
        train_acc = []
        test_acc = []

        # Training part.
        for epoch in range(iter_num):
            # Generate random batch index.
            batch_index = np.random.choice(train_x.shape[0], size=batch_size)
            batch_train_x = train_x[batch_index]
            batch_train_y = np.matrix(self.train_y[batch_index]).T
            sess.run(goal, feed_dict={data: batch_train_x, target: batch_train_y})
            temp_loss = sess.run(loss, feed_dict={data: batch_train_x, target: batch_train_y})
            # Convert into a matrix, and the shape of the placeholder to correspond.
            temp_train_acc = sess.run(accuracy, feed_dict={data: train_x, target: np.matrix(self.train_y).T})
            temp_test_acc = sess.run(accuracy, feed_dict={data: test_x, target: np.matrix(self.test_y).T})

            # Recode the result. TODO: Understand this part.
            loss_trace.append(temp_loss)
            train_acc.append(temp_train_acc)
            test_acc.append(temp_test_acc)

            # Output:
            if (epoch + 1) % 300 == 0:
                print('epoch: {:4d} loss: {:5f} train_acc: {:5f} test_acc: {:5f}'.format(epoch + 1, temp_loss,
                                                                                         temp_train_acc, temp_test_acc))
        plt.plot(train_acc, 'b-', label='train accuracy')
        plt.plot(test_acc, 'k-', label='test accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('Train and Test Accuracy')
        plt.legend(loc='best')
        plt.show()
