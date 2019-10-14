from datetime import datetime

import tensorflow as tf
from tqdm import tqdm, trange
import selection_util as sl_util

def train_one_epoch(sess, ops, num_batches, train_writer, it, epoch, dataset):
    """ ops: dict mapping from string to tf ops """
    is_training = True
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

    with trange(num_batches, leave=False, desc='train') as pbar:
        for _ in pbar:
            summary, step, _, loss_val, m_dict = sess.run(
                [ops['merged'], ops['step'],
                ops['train_op'],
                ops['loss'], ops['metrix']], 
                feed_dict={ops['is_training_pl']: is_training},
            )

            loss_sum += loss_val
            for k, v in m_dict.items():
                metrix[k] += v.tolist()

            train_writer.add_summary(summary, step)
            it += 1
            pbar.set_postfix(dict(total_it=it))

    mean_loss = loss_sum / float(num_batches)

    if epoch % 5 == 0:
        results, _ = dataset.cal_real_metrix(metrix)
        total = results[-1]
        py_summary = tf.Summary(value=[
            tf.Summary.Value(tag="iou", simple_value=total['iou']),
            # tf.Summary.Value(tag="acc", simple_value=total['acc']),
            tf.Summary.Value(tag="pos_pre", simple_value=total['pos_pre']), 
            tf.Summary.Value(tag="pos_rec", simple_value=total['pos_rec']), 
            # tf.Summary.Value(tag="neg_pre", simple_value=total['neg_pre']), 
            # tf.Summary.Value(tag="neg_rec", simple_value=total['neg_rec'])
        ])
        train_writer.add_summary(py_summary, epoch)

    return mean_loss, it

def eval_one_epoch(sess, ops, num_batches, test_writer, best_iou, epoch, 
    dataset, log_string):
    """ ops: dict mapping from string to tf ops """
    is_training = False
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

    log_string('---- EPOCH %03d EVALUATION at %s----' % (epoch, str(datetime.now())))
    for _ in trange(num_batches, leave=False, desc='val'):
        summary, step, loss_val, m_dict = sess.run([
            ops['merged'], ops['step'],
            ops['loss'], ops['metrix']], 
            feed_dict={ops['is_training_pl']: is_training
        })
        test_writer.add_summary(summary, step)
        loss_sum += loss_val
        for k, v in m_dict.items():
            metrix[k] += v.tolist()

    mean_loss = loss_sum / float(num_batches)
    results, split_statics = dataset.cal_real_metrix(metrix)
    total = results[-1]
    py_summary = tf.Summary(value=[
        tf.Summary.Value(tag="iou", simple_value=total['iou']),
        # tf.Summary.Value(tag="acc", simple_value=total['acc']),
        tf.Summary.Value(tag="pos_pre", simple_value=total['pos_pre']), 
        tf.Summary.Value(tag="pos_rec", simple_value=total['pos_rec']), 
        # tf.Summary.Value(tag="neg_pre", simple_value=total['neg_pre']), 
        # tf.Summary.Value(tag="neg_rec", simple_value=total['neg_rec'])
    ])

    test_writer.add_summary(py_summary, epoch)

    log_string('---- mean loss: {:.4f} ---- best iou: {:.4f} -----'.format(mean_loss, best_iou))
    for line in sl_util.printAsTabel(results, split_statics):
        log_string(line)

    return total['iou']
