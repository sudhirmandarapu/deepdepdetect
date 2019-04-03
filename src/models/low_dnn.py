import tensorflow as tf

# Parameters
learning_rate = 0.01
num_steps = 500
batch_size = 10
display_step = 10

# Network Parameters
n_hidden_1 = 16
n_hidden_2 = 16
n_hidden_3 = 8
num_input = 81  # Number of features.
num_classes = 1  # Depressed or not.


class LowDNN:
    def __init__(self, train_x, train_y, test_x, test_y):
        self.train_x = train_x
        self.train_y = train_y.reshape((train_y.shape[0], 1))
        self.test_x = test_x
        self.test_y = test_y.reshape((test_y.shape[0], 1))

    def neural_network(self, x_dict):
        # TF Estimator input is a dict, in case of multiple inputs
        x = x_dict['examples']
        layer_1 = tf.layers.dense(x, n_hidden_1)
        layer_2 = tf.layers.dense(layer_1, n_hidden_2)
        layer_3 = tf.layers.dense(layer_2, n_hidden_3)
        out_layer = tf.layers.dense(layer_3, num_classes)
        return out_layer

    def model_fn(self, features, labels, mode):
        logits = self.neural_network(features)

        predictions = tf.argmax(logits, axis=1)
        confidences = tf.nn.softmax(logits)

        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits,
                labels=tf.cast(labels, dtype=tf.float32)
            )
        )
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        acc_op = tf.metrics.accuracy(labels=labels, predictions=predictions)

        logging_hook = tf.train.LoggingTensorHook({"loss": loss, "accuracy": acc_op[1]}, every_n_iter=display_step)

        # TF Estimators requires to return a EstimatorSpec, that specify the different ops for training, evaluating...
        estim_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            eval_metric_ops={'accuracy': acc_op},
            training_hooks=[logging_hook]
        )

        return estim_specs

    def train(self):
        tf.logging.set_verbosity(tf.logging.INFO)
        model = tf.estimator.Estimator(self.model_fn)
        # Define the input function for training
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'examples': self.train_x}, y=self.train_y,
            batch_size=batch_size, num_epochs=None, shuffle=True)

        # Train the Model
        model.train(input_fn, steps=num_steps)

        # Evaluate the Model
        # Define the input function for evaluating
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'examples': self.test_x}, y=self.test_y,
            batch_size=batch_size, shuffle=False)
        # Use the Estimator 'evaluate' method
        e = model.evaluate(input_fn)

        print("Testing Accuracy:", e['accuracy'])
