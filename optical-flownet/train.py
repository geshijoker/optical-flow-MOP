from unsupervised import unsupervised_loss
import tensorflow as tf

class Trainer():
    def __init__(self, train_batch_fn, val_batch_fn = None, batch_size = 10, loss = None, epochs = 5, learning_rate = 2e-3):
        self.train_batch_fn = train_batch_fn
        if not val_batch_fn == None:
            self.val_batch_fn = val_batch_fn
        if not loss == None:
            self.loss_fn = loss
        else:
            self.loss_fn = unsupervised_loss
        self.lr = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.devices = ["/gpu:0"]

    def run(self):
        num_train = tf.shape(self.train_batch_fn)[1]
        iterations_per_epoch = tf.cast(tf.maximum(num_train / self.batch_size, 1), tf.int32)
        num_iterations = self.epochs * iterations_per_epoch
        #for t in xrange(num_iterations):
        #for i in range(2):
        self.train()
        # with tf.Session() as sess:
        #     tf.global_variables_initializer().run()

        #     coord = tf.train.Coordinator()
        #     threads = tf.train.start_queue_runners(coord=coord)

        #     image_tensor = sess.run([num_iterations])
        #     print(image_tensor)
        #     print(tf.shape(image_tensor))

        #     coord.request_stop()
        #     coord.join(threads)

    def get_train_and_loss_ops(self, batch, learning_rate):
        opt = tf.train.AdamOptimizer(beta1=0.9, beta2=0.999,
                                        learning_rate=learning_rate)
        # def _add_summaries():
        #     _add_loss_summaries()
        #     _add_param_summaries()
        #     if self.debug:
        #         _add_image_summaries()

        
        loss_ = self.loss_fn(batch, params=None, normalization=None, return_flow=False)
        #print("tf.trainable_variables(): ", tf.trainable_variables())
        train_op = opt.minimize(loss_)
        #grads = opt.compute_gradients(loss_)


        return train_op, loss_
        #return grads



    def train(self):
        learning_rate_ = tf.placeholder(tf.int32, name="learning_rate")
        global_step_ = tf.placeholder(tf.int32, name="global_step")
        train_op, loss_ = self.get_train_and_loss_ops(self.train_batch_fn, learning_rate_)
        #grads_ = self.get_train_and_loss_ops(self.train_batch_fn, learning_rate_)


        # num_batches = N / batch_size
        # if N % batch_size != 0:
        #     num_batches += 1
        # y_pred = []
        # for i in xrange(num_batches):
        #     start = i * batch_size
        #     end = (i + 1) * batch_size
        #     train_op, loss_ = self.get_train_and_loss_ops(batch, self.lr)
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=sess_config) as sess:
            tf.global_variables_initializer().run()

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            learning_rate = self.lr
            for i in range(2):
                print("Loop:", i)
                feed_dict = {learning_rate_: learning_rate}
                print("train_op:", type(train_op))
                print("loss_:",type(loss_))
                print("feed_dict:",type(feed_dict))
                _, loss  = sess.run([train_op, loss_], feed_dict=feed_dict)
                print(loss)
                print(tf.shape(loss))

            # feed_dict = {learning_rate_: learning_rate}
            # grads  = sess.run([grads_], feed_dict=feed_dict)
            # print("grads", grads)
            coord.request_stop()
            coord.join(threads)
