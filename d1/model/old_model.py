import tensorflow as tf
import numpy as np
import math
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util


def get_transform_K(inputs, is_training, bn_decay=None, K = 3):
    """ Transform Net, input is BxNx1xK gray image
        Return:
            Transformation matrix of size KxK """
    # batch_size = inputs.get_shape()[0].value
    # num_point = inputs.get_shape()[1].value
    num_point = inputs.shape[1]

    net = tf_util.conv2d(inputs, 256, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='tconv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='tconv2', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point, 1], padding='VALID', scope='tmaxpool')

    net = tf.squeeze(net, axis=[1, 2])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='tfc2', bn_decay=bn_decay)

    with tf.variable_scope('transform_feat') as sc:
        weights = tf.get_variable('weights', [256, K*K], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        biases = tf.get_variable('biases', [K*K], initializer=tf.constant_initializer(0.0), dtype=tf.float32) + tf.constant(np.eye(K).flatten(), dtype=tf.float32)
        transform = tf.nn.bias_add(tf.matmul(net, weights), biases)

    #transform = tf_util.fully_connected(net, 3*K, activation_fn=None, scope='tfc3')
    transform = tf.reshape(transform, [-1, K, K])
    return transform

def get_transform(pc_xyz, is_training, bn_decay=None):
    """ Transform Net, input is BxNx3
        Return:
            Transformation matrix of size 3xK """
    # batch_size = point_cloud.get_shape()[0].value
    # num_point = point_cloud.get_shape()[1].value
    num_point = pc_xyz.shape[1]
    pc_xyz = tf.expand_dims(pc_xyz, -1)

    net = tf_util.conv2d(pc_xyz, 64, [1,3], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='tconv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='tconv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='tconv4', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point, 1], padding='VALID', scope='tmaxpool')

    net = tf.squeeze(net, axis=[1, 2])
    net = tf_util.fully_connected(net, 128, bn=True, is_training=is_training, scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 128, bn=True, is_training=is_training, scope='tfc2', bn_decay=bn_decay)

    with tf.variable_scope('transform_XYZ') as sc:
        weights = tf.get_variable('weights', [128, 3* 3], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        biases = tf.get_variable('biases', [3 * 3], initializer=tf.constant_initializer(0.0), dtype=tf.float32) + tf.constant([1,0,0,0,1,0,0,0,1], dtype=tf.float32)
        transform = tf.nn.bias_add(tf.matmul(net, weights), biases)

    transform = tf.reshape(transform, [-1, 3, 3])
    return transform

def get_model(pc_xyz, pc_features, is_training, num_class, 
              bn_decay=None, use_t_net=True, pc_cls=None, cls_num=1, **kwargs):
    """ ConvNet baseline, input is BxNx3 gray image """
    end_points = {}
    num_point = pc_xyz.shape[1]

    if use_t_net:
        with tf.variable_scope('transform_net1') as sc:
            transform = get_transform(pc_xyz, is_training, bn_decay)
        pc_xyz = tf.matmul(pc_xyz, transform)
    # B x N x 3 x 1
    pc_xyz = tf.expand_dims(pc_xyz, -1) 

    ## to B * N * 1 * 1
    pc_features = tf.expand_dims(pc_features, -1)
    # to B x N x 4 x 1
    pc_xyz = tf.concat(axis=2, values=[pc_xyz, pc_features])

    # block one [64, 128, 128]
    K = 4
    out1 = tf_util.conv2d(pc_xyz, 64, [1, K], 
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv1', bn_decay=bn_decay)
    out2 = tf_util.conv2d(out1, 128, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv2', bn_decay=bn_decay)
    out3 = tf_util.conv2d(out2, 128, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv3', bn_decay=bn_decay)

    if use_t_net:
        with tf.variable_scope('transform_net2') as sc:
            K = 128
            transform = get_transform_K(out3, is_training, bn_decay, K)
        end_points['transform'] = transform
        net_transformed = tf.matmul(tf.reshape(out3, [-1, num_point, 128]), transform)
        net_transformed = tf.expand_dims(net_transformed, [2])
    else:
        net_transformed = out3

    # block two [512, 2048]
    out4 = tf_util.conv2d(net_transformed, 512, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv4', bn_decay=bn_decay)
    out5 = tf_util.conv2d(out4, 2048, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv5', bn_decay=bn_decay)
    out_max = tf_util.max_pool2d(out5, [num_point, 1], padding='VALID', scope='maxpool')

    # segmentation network
    if pc_cls is not None:
        # pc_labels.shape (batch_size, cls_num)
        one_hot_label_expand = tf.reshape(pc_cls, [-1, 1, 1, cls_num])
        out_max = tf.concat(axis=3, values=[out_max, one_hot_label_expand])

    out_max = tf.tile(out_max, [1, num_point, 1, 1])
    concat = tf.concat(axis=3, values=[out_max, out1, out2, out3, out4, out5])

    # block [256, 256, 128] -> 2
    net2 = tf_util.conv2d(concat, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
                        bn=True, is_training=is_training, scope='seg/conv1')
    net2 = tf_util.dropout(net2, keep_prob=0.8, is_training=is_training, scope='seg/dp1')
    net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
                        bn=True, is_training=is_training, scope='seg/conv2')
    net2 = tf_util.dropout(net2, keep_prob=0.8, is_training=is_training, scope='seg/dp2')
    net2 = tf_util.conv2d(net2, 128, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
                        bn=True, is_training=is_training, scope='seg/conv3')
    net2 = tf_util.conv2d(net2, num_class, [1,1], padding='VALID', stride=[1,1], activation_fn=None, 
                        bn=False, scope='seg/conv4')

    net2 = tf.reshape(net2, [-1, num_point, num_class])

    return net2, end_points