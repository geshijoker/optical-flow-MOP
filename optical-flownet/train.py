from unsupervised import unsupervised_loss
import tensorflow as tf

class Trainer():
    def __init__(self, train_batch_fn, val_batch_fn, batch_size = 10, loss = None, epochs = 5, learning_rate = 5e3):
        self.train_batch_fn = train_batch_fn
        self.val_batch_fn = val_batch_fn
        if loss = None:
            self.loss_fn = unsupervised_loss
        self.lr = learning_rate
        self.batch_size = batch_size

    def run(self):
        num_train = tf.shape(self.train_batch_fn)[0]
        iterations_per_epoch = max(num_train / self.batch_size, 1)
        num_iterations = self.epochs * iterations_per_epoch
        for t in xrange(num_iterations):
            train()

    def get_train_and_loss_ops(self, batch, learning_rate):
        if self.params['flownet'] == 'resnet':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
        else:
            opt = tf.train.AdamOptimizer(beta1=0.9, beta2=0.999,
                                         learning_rate=learning_rate)
        # def _add_summaries():
        #     _add_loss_summaries()
        #     _add_param_summaries()
        #     if self.debug:
        #         _add_image_summaries()

        if len(self.devices) == 1:
            loss_ = self.loss_fn(batch, self.params, self.normalization)
            train_op = opt.minimize(loss_)
            # _add_summaries()
        else:
            tower_grads = []
            with tf.variable_scope(tf.get_variable_scope()):
                for i, devid in enumerate(self.devices):
                    with tf.device(devid):
                        with tf.name_scope('tower_{}'.format(i)) as scope:
                            loss_ = self.loss_fn(batch, self.params, self.normalization)
                            # _add_summaries()

                            # Reuse variables for the next tower.
                            tf.get_variable_scope().reuse_variables()

                            # Retain the summaries from the final tower.
                            tower_summaries = tf.get_collection(tf.GraphKeys.SUMMARIES,
                                                                scope)
                            grads = opt.compute_gradients(loss_)
                            tower_grads.append(grads)

            grads = average_gradients(tower_grads)
            apply_gradient_op = opt.apply_gradients(grads)
            train_op = apply_gradient_op

        return train_op, loss_



    def train(self, batch_size = 20):
        # N = tf.shape(self.train_batch_fn)[0]

        # num_batches = N / batch_size
        # if N % batch_size != 0:
        #     num_batches += 1
        # y_pred = []
        # for i in xrange(num_batches):
        #     start = i * batch_size
        #     end = (i + 1) * batch_size
        #     train_op, loss_ = self.get_train_and_loss_ops(batch, self.lr)
        

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            if g is not None:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)
        if grads != []:
            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
    return average_grads