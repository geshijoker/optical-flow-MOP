"""This module provides the loss function.
"""

import tensorflow as tf

def flow_loss(img1, img2, optical):
    """Calculate the loss from the logits and the labels.
    Args:
      img1: tensor, float - the first frame.
      img2: tensor, float - the second frame.
      optical: tensor, float - the optical estimation from flownet.
    Returns:
      loss: Loss tensor of type float.
    """
    with tf.name_scope('optical_flow_loss'):
        losses = {}

        im2_warped = image_warp(im2, flow_fw)
        im1_warped = image_warp(im1, flow_bw)

        im_diff_fw = im1 - im2_warped
        im_diff_bw = im2 - im1_warped

        losses['photo'] =  (photometric_loss(im_diff_fw, mask_fw) +
                        photometric_loss(im_diff_bw, mask_bw))

        losses['grad'] = (gradient_loss(im1, im2_warped, mask_fw) +
                      gradient_loss(im2, im1_warped, mask_bw))

        losses['smooth_1st'] = (smoothness_loss(flow_fw) +
                            smoothness_loss(flow_bw))

    return loss

def photometric_loss(im_diff, mask):
    return charbonnier_loss(im_diff, mask, beta=255)

def conv2d(x, weights):
    return tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME')

def gradient_loss(im1, im2_warped, mask):
    with tf.variable_scope('gradient_loss'):
        mask_x = create_mask(im1, [[0, 0], [1, 1]])
        mask_y = create_mask(im1, [[1, 1], [0, 0]])
        gradient_mask = tf.tile(tf.concat(axis=3, values=[mask_x, mask_y]), [1, 1, 1, 3])
        diff = _gradient_delta(im1, im2_warped)
        return charbonnier_loss(diff, mask * gradient_mask)

def create_mask(tensor, paddings):
    with tf.variable_scope('create_mask'):
        shape = tf.shape(tensor)
        inner_width = shape[1] - (paddings[0][0] + paddings[0][1])
        inner_height = shape[2] - (paddings[1][0] + paddings[1][1])
        inner = tf.ones([inner_width, inner_height])

        mask2d = tf.pad(inner, paddings)
        mask3d = tf.tile(tf.expand_dims(mask2d, 0), [shape[0], 1, 1])
        mask4d = tf.expand_dims(mask3d, 3)
        return tf.stop_gradient(mask4d)

def _gradient_delta(im1, im2_warped):
    with tf.variable_scope('gradient_delta'):
        filter_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] # sobel filter
        filter_y = np.transpose(filter_x)
        weight_array = np.zeros([3, 3, 3, 6])
        for c in range(3):
            weight_array[:, :, c, 2 * c] = filter_x
            weight_array[:, :, c, 2 * c + 1] = filter_y
        weights = tf.constant(weight_array, dtype=tf.float32)

        im1_grad = conv2d(im1, weights)
        im2_warped_grad = conv2d(im2_warped, weights)
        diff = im1_grad - im2_warped_grad
        return diff

def smoothness_loss(flow):
    with tf.variable_scope('smoothness_loss'):
        delta_u, delta_v, mask = _smoothness_deltas(flow)
        loss_u = charbonnier_loss(delta_u, mask)
        loss_v = charbonnier_loss(delta_v, mask)
        return loss_u + loss_v

def _smoothness_deltas(flow):
    with tf.variable_scope('smoothness_delta'):
        mask_x = create_mask(flow, [[0, 0], [0, 1]])
        mask_y = create_mask(flow, [[0, 1], [0, 0]])
        mask = tf.concat(axis=3, values=[mask_x, mask_y])

        filter_x = [[0, 0, 0], [0, 1, -1], [0, 0, 0]]
        filter_y = [[0, 0, 0], [0, 1, 0], [0, -1, 0]]
        weight_array = np.ones([3, 3, 1, 2])
        weight_array[:, :, 0, 0] = filter_x
        weight_array[:, :, 0, 1] = filter_y
        weights = tf.constant(weight_array, dtype=tf.float32)

        flow_u, flow_v = tf.split(axis=3, num_or_size_splits=2, value=flow)
        delta_u = conv2d(flow_u, weights)
        delta_v = conv2d(flow_v, weights)
        return delta_u, delta_v, mask

def charbonnier_loss(x, mask=None, truncate=None, alpha=0.45, beta=1.0, epsilon=0.001):
    """Compute the generalized charbonnier loss of the difference tensor x.
    All positions where mask == 0 are not taken into account.
    Args:
        x: a tensor of shape [num_batch, height, width, channels].
        mask: a mask of shape [num_batch, height, width, mask_channels],
            where mask channels must be either 1 or the same number as
            the number of channels of x. Entries should be 0 or 1.
    Returns:
        loss as tf.float32
    """
    with tf.variable_scope('charbonnier_loss'):
        batch, height, width, channels = tf.unstack(tf.shape(x))
        normalization = tf.cast(batch * height * width * channels, tf.float32)

        error = tf.pow(tf.square(x * beta) + tf.square(epsilon), alpha)

        if mask is not None:
            error = tf.multiply(mask, error)

        if truncate is not None:
            error = tf.minimum(error, truncate)

        return tf.reduce_sum(error) / normalization