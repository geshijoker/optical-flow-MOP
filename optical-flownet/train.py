from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import numpy as np
from multiprocessing import Process
#from matplotlib.pyplot import plot, show

import tensorflow as tf
from tensorflow.python.client import timeline
import tensorflow.contrib.slim as slim

class Trainer():
    def __init__(self, train_batch_fn, eval_batch_fn, params,
                 train_summaries_dir, eval_summaries_dir, ckpt_dir,
                 normalization, debug=False, experiment="", interactive_plot=False,
                 supervised=False, devices=None):

        self.train_summaries_dir = train_summaries_dir
        self.eval_summaries_dir = eval_summaries_dir
        self.ckpt_dir = ckpt_dir
        self.params = params
        self.debug = debug
        self.train_batch_fn = train_batch_fn
        self.eval_batch_fn = eval_batch_fn
        self.normalization = normalization
        self.experiment = experiment
        self.interactive_plot = interactive_plot
        self.plot_proc = None
        self.supervised = supervised
        self.loss_fn = supervised_loss if supervised else unsupervised_loss
        self.devices = devices or '/gpu:0'
        self.shared_device = devices[0] if len(devices) == 1 else '/cpu:0'

    def run(self, min_iter, max_iter):
        """Train (at most) from min_iter + 1 to max_iter.
        If checkpoints are found in ckpt_dir,
        they must be have a global_step within [min_iter, max_iter]. In this case,
        training is continued from global_step + 1 until max_iter is reached.
        """
        save_interval = self.params['save_interval']

        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        if ckpt is not None:
            ckpt_path = ckpt.model_checkpoint_path
            global_step = int(ckpt_path.split('/')[-1].split('-')[-1])
            assert global_step >= min_iter, 'training stage not reached'

            start_iter = global_step + 1
            if start_iter > max_iter:
                print('-- train: max_iter reached')
                return
        else:
            start_iter = min_iter + 1

        print('-- training from i = {} to {}'.format(start_iter, max_iter))

        assert (max_iter - start_iter + 1) % save_interval == 0
        for i in range(start_iter, max_iter + 1, save_interval):
            self.train(i, i + save_interval - 1, i - (min_iter + 1))
            self.eval(1)

        if self.plot_proc:
            self.plot_proc.join()

    def eval(self, num):
        assert num == 1

         with tf.Graph().as_default():
            inputs = self.eval_batch_fn()
            im1, im2, input_shape = inputs[:3]
            truths = inputs[3:]

            height, width, _ = tf.unstack(tf.squeeze(input_shape), num=3, axis=0)
            im1 = resize_input(im1, height, width, 384, 1280)
            im2 = resize_input(im2, height, width, 384, 1280)

            _, flow, flow_bw = unsupervised_loss(
                (im1, im2),
                params=self.params,
                normalization=self.normalization,
                augment=False, return_flow=True)

            im1 = resize_output(im1, height, width, 3)
            im2 = resize_output(im2, height, width, 3)
            flow = resize_output_flow(flow, height, width, 2)
            flow_bw = resize_output_flow(flow_bw, height, width, 2)

            variables_to_restore = tf.all_variables()

            images_ = [image_warp(im1, flow) / 255,
                       flow_to_color(flow),
                       1 - (1 - occlusion(flow, flow_bw)[0]) * create_outgoing_mask(flow) ,
                       forward_warp(flow_bw) < DISOCC_THRESH]
            image_names = ['warped image', 'flow', 'occ', 'reverse disocc']
    