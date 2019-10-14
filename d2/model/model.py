import os
import sys
# BASE_DIR = os.path.dirname(os.path.dirname(__file__))
# sys.path.append(BASE_DIR)
sys.path.append(os.path.join(os.getcwd(), 'utils'))
import tensorflow as tf
from tensorflow.python.ops import array_ops
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module, input_transformation_net
from selection_util import focal_loss_sigmoid_on_2_classification, lovasz_hinge

def lower_keys(dict):
    return { k.lower(): v for k, v in dict.items()}

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    From tensorflow tutorial: cifar10/cifar10_multi_gpu_train.py
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
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def placeholder_inputs(num_points, feature_channels):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(None, num_points, 3), name="xyz_pl")
    features_pl = tf.placeholder(tf.float32, shape=(None, num_points, feature_channels), name="features_pl")
    labels_pl = tf.placeholder(tf.uint8, shape=(None, num_points, 2), name="labels_pl")
    return pointclouds_pl, features_pl, labels_pl

def get_model(pc_xyz, pc_features, is_training, num_class, 
              bn_decay=None, max_num_groups=2048, group_size = 32, **kwargs):
    """ Semantic segmentation PointNet, input is BxNx3, output B x num_class """
    l0_xyz = pc_xyz
    l0_points = pc_features # tf.expand_dims(pc_features, -1)
    end_points = { 'l0_xyz': l0_xyz }

    # rotation_mat = input_transformation_net(tf.concat([l0_xyz, l0_points], axis=-1), is_training, bn_decay)
    # l0_xyz = tf.matmul(l0_xyz, rotation_mat)

    # Layer 1
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=max_num_groups, 
                                                       radius=0.1, nsample=group_size, mlp=[32,32,64],
                                                       mlp2=None, is_training=is_training, 
                                                       bn_decay=bn_decay, scope='layer1')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=max_num_groups>>2, 
                                                       radius=0.2, nsample=group_size, mlp=[64,64,128],
                                                       mlp2=None, is_training=is_training, 
                                                       bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=max_num_groups>>4, 
                                                       radius=0.4, nsample=group_size, mlp=[128,128,256],
                                                       mlp2=None, is_training=is_training, 
                                                       bn_decay=bn_decay, scope='layer3')
    l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=max_num_groups>>6, 
                                                       radius=0.8, nsample=group_size, mlp=[256,256,512],
                                                       mlp2=None, is_training=is_training, 
                                                       bn_decay=bn_decay, scope='layer4')

    #l0_points = None
    # Feature Propagation layers
    l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256,256], is_training, bn_decay, scope='fa_layer1')
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], is_training, bn_decay, scope='fa_layer2')
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay, scope='fa_layer3')
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, tf.concat([l0_xyz,l0_points],axis=-1), l1_points, [128,128,128], is_training, bn_decay, scope='fa_layer4')

    # FC layers
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    end_points['feats'] = net 
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net, num_class, 1, padding='VALID', activation_fn=None, scope='fc2')

    return net, end_points

def get_loss(pred, label, fn, cls_weights, gamma=2, alpha=0.8, **kwargs):
    """ pred: BxNxC,
        label: BxNxC,
	"""
    # your class weights
    class_weights = tf.constant([cls_weights])
    if fn == 'focal':
        # Focal loss
        classify_loss = focal_loss_sigmoid_on_2_classification(label, pred, alpha=alpha, gamma=gamma)
        classify_loss = tf.reduce_sum(classify_loss) / tf.to_float(tf.count_nonzero(classify_loss))
    elif fn == 'weight_sigmoid_ce':
        weights = class_weights * tf.to_float(label)
        classify_loss = tf.losses.sigmoid_cross_entropy(label, pred, weights)
    elif fn == 'lovasz':
        # lovasz_hinge
        pred = pred[:, :, 1] # turn to BxN
        label = tf.argmax(label, axis=2) # turn to BxN
        classify_loss = lovasz_hinge(pred, label)
    else:
        #weight_softmax_ce
        weights = tf.reduce_sum(class_weights * tf.to_float(label), axis=2)
        classify_loss = tf.losses.softmax_cross_entropy(label, pred, weights)

    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss

def get_metrix(pred, label):
    pred = tf.argmax(pred, 2, output_type=tf.int32)
    label = tf.argmax(label, 2, output_type=tf.int32)

    T1 = tf.equal(pred, 1)
    T2 = tf.equal(label, 1)
    F1 = tf.equal(pred, 0)
    F2 = tf.equal(label, 0)

    return dict({
        'T1&T2': tf.reduce_sum(tf.cast(tf.logical_and(T1, T2), tf.float32), axis=1),
        'F1&F2': tf.reduce_sum(tf.cast(tf.logical_and(F1, F2), tf.float32), axis=1),
        'T1|T2': tf.reduce_sum(tf.cast(tf.logical_or(T1, T2), tf.float32), axis=1),
        'T1': tf.reduce_sum(tf.cast(T1, tf.float32), axis=1),
        'F1': tf.reduce_sum(tf.cast(F1, tf.float32), axis=1),
        'T2': tf.reduce_sum(tf.cast(T2, tf.float32), axis=1),
        'F2': tf.reduce_sum(tf.cast(F2, tf.float32), axis=1),
    })

def inference_model(input_fn, is_training_pl, cfg):
    print("--- Get model and loss")
    pointclouds, features, labels = input_fn()
    pointclouds = tf.split(pointclouds, cfg.NUM_GPUS, name='input_xyz')
    tower_features = tf.split(features, cfg.NUM_GPUS, name='input_feat')
    tower_labels = tf.split(labels, cfg.NUM_GPUS, name='input_label')
    tower_pred = []
    tower_loss = []
    tower_mdict = []
    # rank_pls = tf.split(rank_pl, cfg.NUM_GPUS) if USE_RANK else None
    with tf.variable_scope(tf.get_variable_scope()) as outter_scope:
        for i in range(cfg.NUM_GPUS):
            with tf.device(tf_util.assign_to_device('/gpu:%d'%(i), "/cpu:0")), tf.name_scope('gpu_%d' % (i)):
                # Evenly split input data to each GPU
                pc_batch = pointclouds[i]
                features_batch = tower_features[i]
                label_batch = tower_labels[i]

                # Get model and loss
                pred, _ = get_model(pc_batch, features_batch, is_training_pl, cfg.NUM_CLASSES,  **lower_keys(cfg.MODEL))

                loss = get_loss(pred, label_batch, **lower_keys(cfg.LOSS))
                tf.summary.scalar('loss', loss)

                with tf.name_scope("compute_mdict"):
                     tower_mdict.append(get_metrix(pred, label_batch))

                tower_pred.append(pred)
                tower_loss.append(loss)
            outter_scope.reuse_variables()

    # Merge pred and losses from multiple GPUs
    total_pred = tf.concat(tower_pred, 0) # B x N x 2
    total_loss = tf.reduce_mean(tower_loss)
    # features -> (batch, num_points, channels)
    m_dict = tower_mdict[0]
    for d in tower_mdict[1:]:
        for k, v in d.items():
            m_dict[k] = tf.concat((m_dict[k], v), 0)

    return total_loss, m_dict, tf.argmax(tf.nn.softmax(tf.reshape(total_pred, (-1, 2))), -1)

def model_fn(input_fn, optimizer, bn_decay, is_training_pl, batch, cfg):
    print("--- Get model and loss")
    pointclouds, features, labels = input_fn()
    pointclouds = tf.split(pointclouds, cfg.NUM_GPUS, name='input_xyz')
    tower_features = tf.split(features, cfg.NUM_GPUS, name='input_feat')
    tower_labels = tf.split(labels, cfg.NUM_GPUS, name='input_label')
    tower_grads = []
    tower_pred = []
    tower_loss = []
    tower_mdict = []
    # rank_pls = tf.split(rank_pl, cfg.NUM_GPUS) if USE_RANK else None
    with tf.variable_scope(tf.get_variable_scope()) as outter_scope:
        for i in range(cfg.NUM_GPUS):
            with tf.device(tf_util.assign_to_device('/gpu:%d'%(i), "/cpu:0")), tf.name_scope('gpu_%d' % (i)):
                # Evenly split input data to each GPU
                pc_batch = pointclouds[i]
                features_batch = tower_features[i]
                label_batch = tower_labels[i]

                # Get model and loss
                pred, _ = get_model(
                    pc_batch, features_batch, is_training_pl, cfg.NUM_CLASSES, 
                    bn_decay=bn_decay, **lower_keys(cfg.MODEL))

                loss = get_loss(pred, label_batch, **lower_keys(cfg.LOSS))
                tf.summary.scalar('loss', loss)

                with tf.name_scope("compute_gradients"):
                    tower_grads.append(optimizer.compute_gradients(loss))

                with tf.name_scope("compute_mdict"):
                     tower_mdict.append(get_metrix(pred, label_batch))

                tower_pred.append(pred)
                tower_loss.append(loss)
            outter_scope.reuse_variables()

    with tf.name_scope("apply_gradients"), tf.device("/cpu:0"):
        # Get training operator
        grads = average_gradients(tower_grads)
        # NOTE, get this
        train_op = optimizer.apply_gradients(grads, global_step=batch)
        # NOTE: get this
        total_loss = tf.reduce_mean(tower_loss)

    # Merge pred and losses from multiple GPUs
    # total_pred = tf.concat(tower_pred, 0)

    # features -> (batch, num_points, channels)
    m_dict = tower_mdict[0]
    for d in tower_mdict[1:]:
        for k, v in d.items():
            m_dict[k] = tf.concat((m_dict[k], v), 0)

    return train_op, total_loss, m_dict