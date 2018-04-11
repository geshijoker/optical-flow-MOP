#!/usr/bin/env python

# ----------------------------------------------------------------------------
# The 2017 DAVIS Challenge on Video Object Segmentation
#-----------------------------------------------------------------------------
# Copyright (c) 2017 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi
# ----------------------------------------------------------------------------

"""
Read and write segmentation in indexed format.

EXAMPLE:
    python experiments/read_write_segmentation.py

"""

#import davis_loader
import numpy as np
import t_loader as loader
import tensorflow as tf


TRAIN    = 'train'
VAL      = 'val'
TESTDEV  = 'test-dev'
TRAINVAL = 'train-val'

label_directory_base = "../data/trainval_DAVIS/Annotations/480p/classic-car/"
label_file_list = ["00000.png", "00001.png", "00002.png"]
for i in range(len(label_file_list)):
	label_file_list[i] = label_directory_base + label_file_list[i]

data_directory_base = "../data/trainval_DAVIS/JPEGImages/480p/classic-car/"
data_file_list = ["00000.jpg", "00001.jpg", "00002.jpg"]
for i in range(len(data_file_list)):
	data_file_list[i] = data_directory_base + data_file_list[i]

data_queue = tf.train.string_input_producer(data_file_list,shuffle=False)
label_queue = tf.train.string_input_producer(label_file_list,shuffle=False)

data_reader = tf.WholeFileReader()
file_name, data_file = data_reader.read(data_queue)
data = tf.image.decode_jpeg(data_file,channels=3)
with tf.Session() as sess:
    # Required to get the filename matching to run.
    tf.initialize_all_variables().run()

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Get an image tensor and print its value.
    image_tensor = sess.run([data])
    print(image_tensor)

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)


