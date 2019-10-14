import tensorflow as tf

from .model import placeholder_inputs, get_model, get_loss, average_gradients, model_fn, inference_model

def get_learning_rate(batch, cfg):
    learning_rate = tf.train.exponential_decay(
        cfg.OPTIMIZER['BASE_LR'],  # Base learning rate.
        batch * cfg.BATCH_SIZE,  # Current index into the dataset.
        cfg.OPTIMIZER['DECAY_STEP'],          # Decay step.
        cfg.OPTIMIZER['DECAY_RATE'],          # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch, cfg):
    bn_momentum = tf.train.exponential_decay(
        cfg.BN['INIT_DECAY'],
        batch*cfg.BATCH_SIZE,
        cfg.BN['DECAY_STEP'],
        cfg.BN['DECAY_RATE'],
        staircase=True)
    bn_decay = tf.minimum(cfg.BN['DECAY_CLIP'], 1 - bn_momentum)
    return bn_decay