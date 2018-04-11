import tensorflow as tf
import os
import settings

def read_filenames_from_txt(filename_file):
  with open(filename_file) as f:
    lines = f.read().splitlines()
  return lines
def create_queue(filenames,data_dir_path,label_dir_path):
  '''
  return two queue
  data_queue, label_queue

  '''
  #TODO: add label queue
  data_filenames = [os.path.join(data_dir_path,f+'.jpg') for f in filenames]
  label_filenames = [os.path.join(label_dir_path,f+'.png') for f in filenames]
  #print(data_filenames)
  data_queue = tf.train.string_input_producer(data_filenames,shuffle=False)
  label_queue = tf.train.string_input_producer(label_filenames,shuffle=False)
  #print(label_queue)
  return data_queue,label_queue

def read_PAS(data_queue, label_queue):
  """Reads and parses examples from CIFAR10 data files.
  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.
  Args:
    filename_queue: A queue of strings with the filenames to read from.
  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

  class PASRecord(object):
    pass
  result = PASRecord()

  data_reader = tf.WholeFileReader()
  file_name, data_file = data_reader.read(data_queue)
  data = tf.image.decode_jpeg(data_file,channels=3)
  result.data_file_name = file_name
  result.data = data

  #label_reader = tf.WholeFileReader()
  label_name, label_file = data_reader.read(label_queue)
  label = tf.image.decode_png(label_file,channels=1,dtype = tf.uint8)
  result.label_file_name = label_name
  result.label = label
  result.data = tf.image.resize_images(result.data,[50,50])
  #result.label = tf.image.resize_area(result.label,[50,50])
  result.label = tf.image.resize_images(result.label, [50,50], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=False)
  #result.label = tf.decode_raw(result.label, tf.float32, little_endian=None, name=None)
  #result.label = tf.image.resize_image_with_crop_or_pad(result.label,50,50)
  result.label = tf.reduce_sum(result.label,2)
  
  tf.add_to_collection('label',result.label)
  #assert we are reading same data file and corresponding label file?
  #data_basename = os.path.basename(file_name)
  #label_basename = os.path.basename(label_name)
  #assert(data_basename==label_basename)

  #TODO: prepocess of image should go here
  #TODO: remove resize here
  
  
  #print(result.data)
  #print(result.label)
  result.data = tf.cast(result.data, tf.float32)
  '''
  label_bytes = 1  # 2 for CIFAR-100
  result.height = 32
  result.width = 32
  result.depth = 3
  image_bytes = result.height * result.width * result.depth
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  record_bytes = label_bytes + image_bytes

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the CIFAR-10 format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8 that is record_bytes long.
  record_bytes = tf.decode_raw(value, tf.uint8)

  # The first bytes represent the label, which we convert from uint8->int32.
  result.label = tf.cast(
      tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(
      tf.strided_slice(record_bytes, [label_bytes],
                       [label_bytes + image_bytes]),
      [result.depth, result.height, result.width])
  # Convert from [depth, height, width] to [height, width, depth].
  result.uint8image = tf.transpose(depth_major, [1, 2, 0])
  '''
  return result


def generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.
  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.
  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  #TODO: min after queuesize properly from outside
  num_preprocess_threads = 16
  '''
  if batch_size == 1:
    images,label_batch = tf.expand_dims(image, 0), tf.expand_dims(label, 0)
    print("Note: using batch_size == 1, only expend dimension of images and labels")
  el
  '''
  if settings.shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.summary.image('images', images)

  return images, label_batch