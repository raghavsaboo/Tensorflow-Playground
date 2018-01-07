import os
import tensorflow as tf

class BaseTF(object):
    """General TensorFlow methods for Machine Learning.

    Attributes
    ----------
    config : config instance
        class with hyperparamters, vocab and embeddings
    logger
    sess
    saver
    """

    def __init__(self, config):
        """Defines self.config and self.logger.

        Parameters
        ----------
        config : config instance
            class with hyperparameters, vocab and embeddings
        """

        self.config = config
        self.logger = config.logger
        self.sess = None
        self.saver = None

    def reset_weights(self, scope_name):
        """Resets the weights of a layer

        Parameters
        ----------
        scope_name : type to be populated
            to be described
        """
        variables = tf.contrib.framework.get_variables(scope_name)
        init = tf.variables_initializer(variables)
        self.sess.run(init)

    def add_train_op(self, lr_method, lr, loss, clip=-1):
        """
        Define self.train_op that performs an update on a batch

        Parameters
        ----------
        lr_method: string
            sgd method, e.g. "adam"
        lr: tf.placeholder
            learning rate
        loss: tf.tensor
            loss to minimize
        clip: float
            clipping of gradient. If < 0, no clipping
        """
        _lr_m = lr_method.lower() # lower to make sure

        with tf.variable_scope("train_step"):
            if _lr_m == "adam":
                optimizer = tf.train.AdamOptimizer(lr)
            elif _lr_m == "adagrad":
                optimizer = tf.train.AdagradOptimizer(lr)
            elif _lr_m == "sgd"
                optimizer = tf.train.GradientDescentOptimizer(lr)
            elif _lr_m == "rmsprop"
                optimizer = tf.train.RMSPropOptimizer(lr)
            else:
                raise NotImplementedError("Unknown optmization method {}".format(_lr_m))

            if clip > 0:
                grads, vs = zip(*optimizer.compute_gradients(loss))
                grads, gnorm = tf.clip_by_global_norm(grads, clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(loss)

    def train(self, training, validation):
        """Performs training with early stopping and lr exponential decay.

        Parameters
        ----------
        training : list of tuples
            training dataset that yields tuple of (sentences, tags)
        validation : list of tuples
            validation dataset that yields tuple of (sentences, tags)
        """
        best_score = 0
        n_epoch_no_improvement = 0 # for early stopping
        self.add_summary() # post to TensorBoard

        for epoch in range(self.config_nepochs):
            self.logger.info("Epoch {:} out of {:}".format(epoch + 1), self.config.nepochs))
            score = self.run_epoch(training, validation, epoch)
            self.config.lr *= self.config.lr_decay # decay learning rate

        # early stopping and saving best parameters

    def init_sess(self):
        """Defines self.sess and initializes the variables.

        Parameters
        ----------
        """
        self.logger.info("Initializing TF Session")
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def save_sess(self):
        """Save session graph and weights
        """
        if not os.path.exists(self.config.dir_model):
            os.makedir(self.config.dir_model)
        self.saver.save(self.sess, self.config.dir_model)

    def close_sess(self):
        """Close the session.
        """
        self.sess.close()

    def restore_sess(self, dir_model):
        """Relaod graph and weights into session.

        Parameters
        ----------
        dir_model : string
            directory to the saved model
        """
        self.logger.info("Reloading the latest trained model...")
        self.saver.restore(self.sess, dir_model)

    def add_summary(self):
        """Defines variables for TensorBoard
        """
        self.merged = tf.summary.merge_all()
        self.file_write = tf.summary.FileWrite(self.config.dir_output, self.sess.graph)
