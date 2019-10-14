import argparse
import os
import sys
import random
import math
from datetime import datetime

import h5py
import numpy as np
import tensorflow as tf
import importlib
from tqdm import tqdm, trange

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

import d2dataset
from config import Config
import selection_util as sl_util
from train_test_op import train_one_epoch, eval_one_epoch

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help='Path to config file')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
FLAGS = parser.parse_args()

cfg = Config(FLAGS.config)

NUM_POINTS = cfg.DATASET['NUM_POINTS']

MODEL = importlib.import_module('model')  # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'model')
LOG_DIR = os.path.join(FLAGS.log_dir,  os.path.splitext(os.path.basename(FLAGS.config))[0].replace('.', '') + '_'+ datetime.now().strftime("%Y%m%d%H%M"))
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR, exist_ok=True)
os.system('cp -r %s %s' % (MODEL_FILE, LOG_DIR))  # bkp of model def
os.system('cp %s %s' % (os.path.abspath(__file__), LOG_DIR))  # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

train_file_list = cfg.DATASET['TRAIN_FILES']
DATA_PATH = os.path.join(ROOT_DIR, 'data')
DATASET = d2dataset.D2Dataset([os.path.join(DATA_PATH, f) for f in train_file_list], **sl_util.lower_keys(cfg.DATASET))

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    tqdm.write(out_str)

def train():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        train_pc_data, train_features_data, train_label_data, _, \
        val_pc_data, val_features_data, val_label_data, _ = DATASET.strong_split(cfg.NUM_GPUS)
        print('train_pc_data.shape:', train_pc_data.shape, 
              'train_features_data.shape', train_features_data.shape, 
              'train_label_data.shape', train_label_data.shape,
              'val_pc_data.shape:', val_pc_data.shape, 
              'val_features_data.shape', val_features_data.shape, 
              'val_label_data.shape', val_label_data.shape)

        pointclouds_pl, features_pl, labels_pl = MODEL.placeholder_inputs(NUM_POINTS, train_features_data.shape[-1])
        is_training_pl = tf.placeholder(tf.bool, shape=())
        # Note the global_step=batch parameter to minimize.
        # That tells the optimizer to helpfully increment the 'batch' parameter
        # for you every time it trains.
        batch = tf.get_variable('batch', [],
                                initializer=tf.constant_initializer(0), trainable=False)
        
        train_dataset = tf.data.Dataset \
                .from_tensor_slices((pointclouds_pl, features_pl, labels_pl)) \
                .batch(cfg.BATCH_SIZE) \
                .prefetch(cfg.BATCH_SIZE) \
                .shuffle(buffer_size=3 * cfg.BATCH_SIZE) \
                .repeat()
        test_dataset = tf.data.Dataset \
                .from_tensor_slices((pointclouds_pl, features_pl, labels_pl)) \
                .batch(cfg.BATCH_SIZE) \
                .prefetch(cfg.BATCH_SIZE)

        # create a iterator of the correct shape and type
        iter = tf.data.Iterator.from_structure(train_dataset.output_types,
                                                train_dataset.output_shapes)

        # create the initialisation operations
        train_init_op = iter.make_initializer(train_dataset)
        test_init_op = iter.make_initializer(test_dataset)

        print("--- Get training operator")
        # Get training operator
        bn_decay = MODEL.get_bn_decay(batch, cfg)
        tf.summary.scalar('bn_decay', bn_decay)
        learning_rate = MODEL.get_learning_rate(batch, cfg)
        tf.summary.scalar('learning_rate', learning_rate)

        if cfg.OPTIMIZER['TYPE'] == 'momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=cfg.OPTIMIZER['MOMENTUM'])
        elif cfg.OPTIMIZER['TYPE'] == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate)

        # -------------------------------------------
        # Get model and loss on multiple GPU devices
        # -------------------------------------------
        def input_fn():
            return iter.get_next()
        train_op, total_loss, m_dict = MODEL.model_fn(input_fn, optimizer, bn_decay, is_training_pl, batch, cfg)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        sess.run(tf.global_variables_initializer())

        ops = {
            'is_training_pl': is_training_pl,
            'metrix': m_dict,
            'loss': total_loss,
            'train_op': train_op,
            'merged': merged,
            'step': batch,
        }

        log_string('**** HYPER  PARMS ****')
        log_string(str(cfg))

        best_iou = -1
        it = 0
        train_num_batches = math.ceil(train_pc_data.shape[0] / cfg.BATCH_SIZE)
        val_num_baches = math.ceil(val_pc_data.shape[0] / cfg.BATCH_SIZE)

        sess.run(train_init_op, feed_dict = {
            pointclouds_pl:train_pc_data, 
            features_pl:train_features_data, 
            labels_pl:train_label_data
        })

        for epoch in trange(0, cfg.MAX_EPOCH, desc='epochs'):
            log_string('---- Begin EPOCH %03d at %s----' % (epoch, str(datetime.now())))
            sys.stdout.flush()
            loss, it = train_one_epoch(sess, ops, train_num_batches, train_writer, it, epoch, DATASET)
            log_string('---- END EPOCH %03d loss: %f ---- \n' % (epoch, loss))

            if epoch % 5 == 0:
                sess.run(test_init_op, feed_dict = {
                    pointclouds_pl:val_pc_data, 
                    features_pl:val_features_data, 
                    labels_pl:val_label_data
                })
                iou = eval_one_epoch(sess, ops, val_num_baches, test_writer, best_iou, epoch, DATASET, log_string)
                sess.run(train_init_op, feed_dict = {
                    pointclouds_pl:train_pc_data, 
                    features_pl:train_features_data, 
                    labels_pl:train_label_data
                })
            if iou > best_iou:
                best_iou = iou
                save_path = saver.save(sess, os.path.join(LOG_DIR, "best_model_epoch_%03d.ckpt" % (epoch)))
                log_string("Model saved in file: %s\n" % save_path)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s\n" % save_path)

if __name__ == "__main__":
    log_string('pid: %s' % (str(os.getpid())))
    train()
    LOG_FOUT.close()
