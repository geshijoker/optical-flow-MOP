import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers

from input import *
from input_config import *

FLOW_SCALE = 5.0
dir_base = "../data/DAVIS/"

def flownet(im1, im2, flownet_spec='S', full_resolution=False, train_all=False,
            backward_flow=False):
    num_batch, height, width, _ = tf.unstack(tf.shape(im1))
    flownet_num = len(flownet_spec)
    assert flownet_num > 0
    flows_fw = []
    flows_bw = []
    for i, name in enumerate(flownet_spec):
        assert name in ('S', 's')
        channel_mult = 1 if name in ('S') else 3 / 8
        full_res = full_resolution and i == flownet_num - 1

        def scoped_block():
            if name.lower() == 's':
                def _flownet_s(im1, im2, flow=None):
                    if flow is not None:
                        flow = tf.image.resize_bilinear(flow, [height, width]) * 4 * FLOW_SCALE
                        warp = image_warp(im2, flow)
                        diff = tf.abs(warp - im1)
                        if not train_all:
                            flow = tf.stop_gradient(flow)
                            warp = tf.stop_gradient(warp)
                            diff = tf.stop_gradient(diff)

                        inputs = tf.concat([im1, im2, flow, warp, diff], axis=3)
                        inputs = tf.reshape(inputs, [num_batch, height, width, 14])
                    else:
                        inputs = tf.concat([im1, im2], 3)

                    print("channel_mult:", channel_mult)
                    return flownet_s(inputs,
                                     full_res=full_res,
                                     channel_mult=channel_mult)
                
                stacked = len(flows_fw) > 0
                with tf.variable_scope('flownet_s') as scope:
                    flow_fw = _flownet_s(im1, im2, flows_fw[-1][0] if stacked else None)
                    flows_fw.append(flow_fw)
                    if backward_flow:
                        scope.reuse_variables()
                        flow_bw = _flownet_s(im2, im1, flows_bw[-1][0]  if stacked else None)
                        flows_bw.append(flow_bw)

        scope_name = "stack_{}_flownet".format(i)
        with tf.variable_scope(scope_name):
            scoped_block()

    if backward_flow:
        return flows_fw, flows_bw
    return flows_fw

def _leaky_relu(x):
    with tf.variable_scope('leaky_relu'):
        return tf.maximum(0.1 * x, x)

def nchw_to_nhwc(tensors):
    return [tf.transpose(t, [0, 2, 3, 1]) for t in tensors]

def nhwc_to_nchw(tensors):
    return [tf.transpose(t, [0, 3, 1, 2]) for t in tensors]

def flownet_s(inputs, channel_mult=1, full_res=False):
    """Given stacked inputs, returns flow predictions in decreasing resolution.
    Uses FlowNetSimple.
    """
    m = channel_mult

    with slim.arg_scope([slim.conv2d, slim.batch_norm, slim.conv2d_transpose],
                        data_format='NHWC',
                        weights_regularizer=slim.l2_regularizer(0.0004),
                        weights_initializer=layers.variance_scaling_initializer(),
                        activation_fn=_leaky_relu):
        conv1 = slim.conv2d(inputs, int(64 * m), 7, stride=2, scope='conv1')
        batch1 = slim.batch_norm(conv1, scope="batch1")
        conv2 = slim.conv2d(batch1, int(128 * m), 5, stride=2, scope='conv2')
        batch2 = slim.batch_norm(conv2, scope="batch2")
        conv3 = slim.conv2d(batch2, int(256 * m), 5, stride=2, scope='conv3')
        batch3 = slim.batch_norm(conv3, scope="batch3")
        conv3_1 = slim.conv2d(batch3, int(256 * m), 3, stride=1, scope='conv3_1')
        batch3_1 = slim.batch_norm(conv3_1, scope="batch3_1")
        conv4 = slim.conv2d(batch3_1, int(512 * m), 3, stride=2, scope='conv4')
        batch4 = slim.batch_norm(conv4, scope="batch4")
        conv4_1 = slim.conv2d(batch4, int(512 * m), 3, stride=1, scope='conv4_1')
        batch4_1 = slim.batch_norm(conv4_1, scope="batch4_1")
        conv5 = slim.conv2d(batch4_1, int(512 * m), 3, stride=2, scope='conv5')
        batch5 = slim.batch_norm(conv5, scope="batch5")
        conv5_1 = slim.conv2d(batch5, int(512 * m), 3, stride=1, scope='conv5_1')
        batch5_1 = slim.batch_norm(conv5_1, scope="batch5_1")
        conv6 = slim.conv2d(batch5_1, int(1024 * m), 3, stride=2, scope='conv6')
        batch6 = slim.batch_norm(conv6, scope="batch6")
        conv6_1 = slim.conv2d(batch6, int(1024 * m), 3, stride=1, scope='conv6_1')


        res = _flownet_upconv(conv6_1, conv5_1, conv4_1, conv3_1, conv2, conv1, inputs,
                              channel_mult=channel_mult, full_res=full_res)
        return res

def _flownet_upconv(conv6_1, conv5_1, conv4_1, conv3_1, conv2, conv1=None, inputs=None,
                    channel_mult=1, full_res=False, channels=2):
    m = channel_mult

    flow6 = slim.conv2d(conv6_1, channels, 3, scope='flow6',
                        activation_fn=None)
    deconv5 = slim.conv2d_transpose(conv6_1, int(512 * m), 4, stride=2,
                                   scope='deconv5')
    flow6_up5 = slim.conv2d_transpose(flow6, channels, 4, stride=2,
                                     scope='flow6_up5',
                                     activation_fn=None)
    concat5 = tf.concat([conv5_1, deconv5, flow6_up5], 3)
    flow5 = slim.conv2d(concat5, channels, 3, scope='flow5',
                       activation_fn=None)

    deconv4 = slim.conv2d_transpose(concat5, int(256 * m), 4, stride=2,
                                   scope='deconv4')
    flow5_up4 = slim.conv2d_transpose(flow5, channels, 4, stride=2,
                                     scope='flow5_up4',
                                     activation_fn=None)
    concat4 = tf.concat([conv4_1, deconv4, flow5_up4], 3)
    flow4 = slim.conv2d(concat4, channels, 3, scope='flow4',
                       activation_fn=None)

    deconv3 = slim.conv2d_transpose(concat4, int(128 * m), 4, stride=2,
                                   scope='deconv3')
    flow4_up3 = slim.conv2d_transpose(flow4, channels, 4, stride=2,
                                     scope='flow4_up3',
                                     activation_fn=None)
    concat3 = tf.concat([conv3_1, deconv3, flow4_up3], 3)
    flow3 = slim.conv2d(concat3, channels, 3, scope='flow3',
                       activation_fn=None)

    deconv2 = slim.conv2d_transpose(concat3, int(64 * m), 4, stride=2,
                                   scope='deconv2')
    flow3_up2 = slim.conv2d_transpose(flow3, channels, 4, stride=2,
                                     scope='flow3_up2',
                                     activation_fn=None)
    concat2 = tf.concat([conv2, deconv2, flow3_up2], 3)
    flow2 = slim.conv2d(concat2, channels, 3, scope='flow2',
                       activation_fn=None)

    flows = [flow2, flow3, flow4, flow5, flow6]

    if full_res:
        with tf.variable_scope('full_res'):
            deconv1 = slim.conv2d_transpose(concat2, int(32 * m), 4, stride=2,
                                           scope='deconv1')
            flow2_up1 = slim.conv2d_transpose(flow2, channels, 4, stride=2,
                                             scope='flow2_up1',
                                             activation_fn=None)
            concat1 = tf.concat([conv1, deconv1, flow2_up1], 3)
            flow1 = slim.conv2d(concat1, channels, 3, scope='flow1',
                                activation_fn=None)

            deconv0 = slim.conv2d_transpose(concat1, int(16 * m), 4, stride=2,
                                           scope='deconv0')
            flow1_up0 = slim.conv2d_transpose(flow1, channels, 4, stride=2,
                                             scope='flow1_up0',
                                             activation_fn=None)
            concat0 = tf.concat([inputs, deconv0, flow1_up0], 3)
            flow0 = slim.conv2d(concat0, channels, 3, scope='flow0',
                                activation_fn=None)

            flows = [flow0, flow1] + flows

    return flows

def main():
    data_dic, labels_dic = read_data_label_dir("/ImageSets/480p/train.txt")
    im1, im2 = input_raw(data_dic['bear'])

    flows_fw = flownet(im1, im2, flownet_spec='S', full_resolution=True, train_all=False, backward_flow=False)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        image_tensor = sess.run(flows_fw)
        print(flows_fw)

        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    main()