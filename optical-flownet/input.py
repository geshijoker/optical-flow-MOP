import numpy as np
import tensorflow as tf
from input_config import *

dir_base = "../data/DAVIS/"

def read_data_label_dir(txt_dir):
    """
    Data structure of data_dic/labels_dic: {'bear': ["diretory/of/the/image",...], 'car': []}
    """
    data_dic = {}
    labels_dic = {}
    with open(dir_base + txt_dir) as f:
        lines = f.read().splitlines()
        for line in lines:
            pair = line.strip().split()
            class_name = pair[0].split("/")[-2]
            if not class_name in data_dic:
                data_dic[class_name] = [dir_base + pair[0]]
            else:
                data_dic[class_name].append(dir_base + pair[0])
            if not class_name in labels_dic:
                labels_dic[class_name] = [dir_base + pair[1]]
            else:
                labels_dic[class_name].append(dir_base + pair[1])
    return data_dic, labels_dic

def read_jpeg_image(filenames, num_epochs=None):
    """Given a list of filenames, constructs a reader op for images."""
    filename_queue = tf.train.string_input_producer(filenames,
        shuffle=False, capacity=len(filenames))
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)
    image_uint8 = tf.image.decode_jpeg(value, channels=3)
    image = tf.cast(image_uint8, tf.float32)
    return image

def read_png_image(filenames, num_epochs=None):
    """Given a list of filenames, constructs a reader op for images."""
    filename_queue = tf.train.string_input_producer(filenames,
        shuffle=False, capacity=len(filenames))
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)
    image_uint8 = tf.image.decode_png(value, channels=3)
    image = tf.cast(image_uint8, tf.float32)
    return image

def _resize_crop_or_pad(tensor):
    # return tf.image.resize_bilinear(tf.expand_dims(tensor, 0), [height, width])
    return tf.image.resize_image_with_crop_or_pad(tensor, reshape_height, reshape_width)

def input_raw(filenames, swap_images=False, labels=False, batch_size=None, num_threads=1):
    """Constructs input of raw data.
    Args:
        swap_images: for each pair (im1, im2), also include (im2, im1)
    Returns:
        image_1: batch of first images
        image_2: batch of second images
    """
    file_len = len(filenames)
    if not file_len % 2 == 0:
        file_len -= 1

    filenames_1 = []
    filenames_2 = []
    for i in range(file_len):
        if i % 2 == 0:
            filenames_1.append(filenames[i])
        else:
            filenames_2.append(filenames[i])

    with tf.variable_scope('train_inputs'):
        if labels == True:
            image_1 = read_png_image(filenames_1)
            image_2 = read_png_image(filenames_2)
        else:
            image_1 = read_jpeg_image(filenames_1)
            image_2 = read_jpeg_image(filenames_2)

        print("reshape: ", reshape_height)
        image_1 = tf.reshape(_resize_crop_or_pad(image_1), [reshape_height, reshape_width, channel])
        image_2 = tf.reshape(_resize_crop_or_pad(image_2), [reshape_height, reshape_width, channel])

        #print(image_1.get_shape())
        # if self.normalize:
        #     image_1 = self._normalize_image(image_1)
        #     image_2 = self._normalize_image(image_2)
        if batch_size == None:
            batch_size = len(filenames_1)
        return tf.train.batch(
            [image_1, image_2],
            batch_size=batch_size,
            num_threads=num_threads)
def main():
    # Follow the demo below to read dataset.
    data_dic, labels_dic = read_data_label_dir("/ImageSets/480p/train.txt")
    images = input_raw(data_dic['bear'])

    with tf.Session() as sess:
        # Required to get the filename matching to run.
        tf.global_variables_initializer().run()

        # Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Get an image tensor and print its value.
        image_tensor = sess.run([images])
        print(images)

        # Finish off the filename queue coordinator.
        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    main()