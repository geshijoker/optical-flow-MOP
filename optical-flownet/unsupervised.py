import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from input import *
from input_config import *
from flownet import *
#from ops import downsample
from loss import *

FLOW_SCALE = 5.0
dir_base = "../data/DAVIS/"

LOSSES = ['grad', 'photo', 'smooth_1st']

def _track_loss(op, name):
    tf.add_to_collection('losses', tf.identity(op, name=name))

def unsupervised_loss(batch, params=None, normalization=None, return_flow=False):

    #channel_mean = tf.constant(normalization[0]) / 255.0
    channel_mean = tf.constant(1.0) / 255.0
    im1, im2 = batch
    im1 = im1 / 255.0
    im2 = im2 / 255.0
    print("im1 shape",im1.get_shape())
    im_shape = tf.shape(im1)[1:3]

    im1_geo, im2_geo = im1, im2
    im1_photo, im2_photo = im1, im2

    # Images for loss comparisons with values in [0, 1] (scale to original using * 255)
    im1_norm = im1_geo
    im2_norm = im2_geo
    # Images for neural network input with mean-zero values in [-1, 1]
    im1_photo = im1_photo - channel_mean
    im2_photo = im2_photo - channel_mean

    #flownet_spec = params.get('flownet', 'S')
    #full_resolution = params.get('full_res')
    #train_all = params.get('train_all')
    flownet_spec='S'
    full_resolution=True
    train_all=False
    backward_flow=False

    flows_fw = flownet(im1_photo, im2_photo,
                        flownet_spec=flownet_spec,
                        full_resolution=full_resolution,
                        backward_flow=backward_flow,
                        train_all=train_all)

    flows_fw = flows_fw[-1]
    print("flows_fw",flows_fw)

    # -------------------------------------------------------------------------
    # Losses
    layer_weights = [12.7, 4.35, 3.9, 3.4, 1.1]
    layer_patch_distances = [3, 2, 2, 1, 1]
    if full_resolution:
        layer_weights = [12.7, 5.5, 5.0, 4.35, 3.9, 3.4, 1.1]
        layer_patch_distances = [3, 3] + layer_patch_distances
        im1_s = im1_norm
        im2_s = im2_norm
        #mask_s = border_mask
        final_flow_scale = FLOW_SCALE * 4
        final_flow_fw = flows_fw[0] * final_flow_scale
        #final_flow_bw = flows_bw[0] * final_flow_scale
    else:
        print("im1_norm shape",im1_norm.get_shape())
        print("im2_norm",im2_norm.get_shape())
        print("im_shape",im_shape.get_shape())
        tf.Print(im_shape,[im_shape])
        print("flows_fw[0]",flows_fw[0].get_shape())
        im1_s = im1_norm
        im2_s = im2_norm
        #im1_s = downsample(im1_norm, 4)
        #im2_s = downsample(im2_norm, 4)
        #mask_s = downsample(border_mask, 4)
        final_flow_scale = FLOW_SCALE
        im_shape = tf.constant([448, 832],dtype=tf.int32)
        final_flow_fw = tf.image.resize_bilinear(flows_fw[0], im_shape) * final_flow_scale * 4
        #final_flow_bw = tf.image.resize_bilinear(flows_bw[0], im_shape) * final_flow_scale * 4
        print("final_flow_fw", final_flow_fw.get_shape())

    combined_losses = dict()
    combined_loss = 0.0
    for loss in LOSSES:
        combined_losses[loss] = 0.0

    #if params.get('pyramid_loss'):
        #flow_enum = enumerate(zip(flows_fw, flows_bw))
    #else:
        #flow_enum = [(0, (flows_fw[0], flows_bw[0]))]
    flows_bw = flows_fw
    flow_enum = [(0, (flows_fw[0], flows_bw[0]))]

    for i, flow_pair in flow_enum:
        layer_name = "loss" + str(i + 2)

        flow_scale = final_flow_scale / (2 ** i)

        with tf.variable_scope(layer_name):
            layer_weight = layer_weights[i]
            flow_fw_s, flow_bw_s = flow_pair

            losses = flow_loss(im1_s, im2_s, 
        	   flow_fw_s * flow_scale)

        layer_loss = 0.0

        layer_loss = layer_loss + 1.0 * losses['grad'] + 1.0 * losses['photo'] + 3.0 * losses['smooth_1st']
        combined_losses['grad'] = combined_losses['grad'] + layer_weight * losses['grad']
        combined_losses['photo'] = combined_losses['photo'] + layer_weight * losses['photo']
        combined_losses['smooth_1st'] = combined_losses['smooth_1st'] + layer_weight * losses['smooth_1st']
        combined_loss += layer_weight * layer_loss

        #im1_s = downsample(im1_s, 2)
        #im2_s = downsample(im2_s, 2)
        #mask_s = downsample(mask_s, 2)

    regularization_loss = tf.losses.get_regularization_loss()
    #print("regularization_loss: ", regularization_loss.get_shape())
    final_loss = combined_loss + regularization_loss

    #_track_loss(final_loss, 'loss/combined')

    #for loss in LOSSES:
        #_track_loss(combined_losses[loss], 'loss/' + loss)
        #weight_name = loss + '_weight'
        #if params.get(weight_name):
            #weight = tf.identity(params[weight_name], name='weight/' + loss)
            #tf.add_to_collection('params', weight)

    #print(final_loss.get_shape())
    #final_loss = combined_loss
    print("final_loss",final_loss.get_shape())

    if not return_flow:
        return final_loss

    return final_loss, final_flow_fw, final_flow_bw

def main():
    data_dic, labels_dic = read_data_label_dir("/ImageSets/480p/train.txt")
    im1, im2 = input_raw(data_dic['bear'])

    batch = [im1, im2]
    final_loss = unsupervised_loss(batch, params=None, normalization=None, return_flow=False)
    #flows_fw = flownet(im1, im2, flownet_spec='S', full_resolution=False, train_all=False, backward_flow=False)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        image_tensor = sess.run([final_loss])
        print(image_tensor)
        print(tf.shape(image_tensor))
        #print(final_loss)
        #sess.run(tf.Print(final_loss,[final_loss]))

        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    main()
