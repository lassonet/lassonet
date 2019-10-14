import argparse
import os
import sys
import random
import math
from datetime import datetime
import re

import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
from tqdm import tqdm, trange

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

import d2dataset
from config import Config
import selection_util as sl_util

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help='Path to config file')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
FLAGS = parser.parse_args()

cfg = Config(FLAGS.config)
cfg.BATCH_SIZE = int(cfg.BATCH_SIZE / cfg.NUM_GPUS)
cfg.NUM_GPUS = 1

NUM_POINTS = cfg.DATASET['NUM_POINTS']
LOG_DIR = os.path.join(FLAGS.log_dir,  cfg.OUTPUT_PATH)
sys.path.insert(0, LOG_DIR)
MODEL = importlib.import_module('model')

ckpt_epoch_p = re.compile(r'best_model_epoch_(\d+).ckpt')
ckpt_file = max([f for f in os.listdir(LOG_DIR) if 'best_model_epoch_' in f], key=lambda f: int(ckpt_epoch_p.findall(f)[0]))
CKPT_FILE = os.path.join(LOG_DIR, 'best_model_epoch_' + ckpt_epoch_p.findall(ckpt_file)[0] + '.ckpt')

test_file_list = cfg.DATASET['TRAIN_FILES']

DATA_PATH = os.path.join(ROOT_DIR, 'data')
DATASET = d2dataset.D2Dataset([os.path.join(DATA_PATH, f) for f in test_file_list], **sl_util.lower_keys(cfg.DATASET))

def log_string(out_str):
    tqdm.write(out_str)

def inference():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        _, _, _, _, val_pc_data, val_features_data, val_label_data, _ = DATASET.split(cfg.NUM_GPUS)
        print('val_pc_data.shape:', val_pc_data.shape, 
              'val_features_data.shape', val_features_data.shape, 
              'val_label_data.shape', val_label_data.shape)

        pointclouds_pl, features_pl, labels_pl = MODEL.placeholder_inputs(NUM_POINTS, val_features_data.shape[-1])
        is_training_pl = tf.constant(False, tf.bool)
        # Note the global_step=batch parameter to minimize.
        # That tells the optimizer to helpfully increment the 'batch' parameter
        # for you every time it trains.

        test_dataset = tf.data.Dataset \
                .from_tensor_slices((pointclouds_pl, features_pl, labels_pl)) \
                .batch(cfg.BATCH_SIZE) \
                .prefetch(cfg.BATCH_SIZE)

        # create a iterator of the correct shape and type
        iter = tf.data.Iterator.from_structure(test_dataset.output_types,
                                                test_dataset.output_shapes)

        # create the initialisation operations
        test_init_op = iter.make_initializer(test_dataset)

        # -------------------------------------------
        # Get model and loss on multiple GPU devices
        # -------------------------------------------
        # MODEL.get_model(pointclouds, features, is_training_pl, cfg.NUM_CLASSES, bn_decay=bn_decay)
        def input_fn():
            return iter.get_next()
        total_loss, m_dict, _ = MODEL.inference_model(input_fn, is_training_pl, cfg)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)
        # restore variables
        saver.restore(sess, CKPT_FILE)

        ops = {'metrix': m_dict, 'loss': total_loss}

        log_string('**** HYPER  PARMS ****')
        log_string(str(cfg))

        val_num_baches = math.ceil(val_pc_data.shape[0] / cfg.BATCH_SIZE)

        sess.run(test_init_op, feed_dict = {
                    pointclouds_pl:val_pc_data, 
                    features_pl:val_features_data, 
                    labels_pl:val_label_data
                })
        eval_one_epoch(sess, ops, val_num_baches)

# evaluate on randomly chopped scenes
def eval_one_epoch(sess, ops, num_batches):
    """ ops: dict mapping from string to tf ops """
    loss_sum = 0
    metrix = {
        'T1': [],
        'F1': [],
        'T2': [],
        'F2': [],
        'T1&T2': [],
        'F1&F2': [],
        'T1|T2': [],
    }

    t1 = datetime.now()
    log_string('---- EVALUATION at %s----' % (str(t1)))
    for _ in trange(num_batches, leave=False, desc='val'):
        loss_val, m_dict = sess.run([ops['loss'], ops['metrix']])
        loss_sum += loss_val
        for k, v in m_dict.items():
            metrix[k] += v.tolist()

    delta = datetime.now() - t1
    print('average_group_number:', DATASET.average_group_number())
    print('=======>#batches:', num_batches, 'delta:', delta.seconds + delta.microseconds/1E6)

    mean_loss = loss_sum / float(num_batches)
    log_string('---- mean loss: {:.4f}'.format(mean_loss))
    results, split_statics = DATASET.cal_real_metrix(metrix)
    for line in sl_util.printAsTabel(results, split_statics):
        log_string(line)


    return results[-1]['iou']

if __name__ == "__main__":
    log_string('pid: %s' % (str(os.getpid())))
    inference()
