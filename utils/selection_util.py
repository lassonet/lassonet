import tensorflow as tf
import numpy as np
import math

def lower_keys(dict):
    return { k.lower(): v for k, v in dict.items()}

def printAsTabel(results, split_statics):
    total = results[-1]
    total_statics = split_statics[-1]
    lines = []
    for result, statics in zip(results[:-1], split_statics[:-1]):
        line = None

        for m_type, item in result.items():
            # header
            if line is None:
                line =  ' ' * 9 + ' |' + ' |'.join(['{:>6}'] * (1 + len(item.keys()))).format(*item.keys(), 'total') + '\n'
                sep = '-' * len(line) + '\n'
                line += ' ' * 9 + ' |' + ' |'.join(['T:{:>4}'] * (1 + len(item.keys()))).format(*[statics[k][0] for k in item], total_statics[0]) + '\n'
                line += ' ' * 9 + ' |' + ' |'.join(['V:{:>4}'] * (1 + len(item.keys()))).format(*[statics[k][1] for k in item], total_statics[1]) + '\n'
                line += sep
            # row
            if m_type == 'w_scene':
                line += '{:>9} |'.format(m_type) + ' |'.join(['{:6d}'] * (1 + len(item.keys()))).format(*item.values(), total[m_type]) + '\n'
            else:
                line += '{:>9} |'.format(m_type) + ' |'.join(['{:.4f}'] * (1 + len(item.keys()))).format(*item.values(), total[m_type]) + '\n'
        lines.append(line)
    
    return lines

def printAsTabelTest(results, split_statics):
    total = results[-1]
    total_statics = split_statics[-1]
    lines = []
    for result, statics in zip(results[:-1], split_statics[:-1]):
        line = None

        for m_type, item in result.items():
            # header
            if line is None:
                line =  ' ' * 9 + ' |' + ' |'.join(['{:>6}'] * (1 + len(item.keys()))).format(*item.keys(), 'total') + '\n'
                sep = '-' * len(line) + '\n'
                line += ' ' * 9 + ' |' + ' |'.join(['V:{:>4}'] * (1 + len(item.keys()))).format(*[statics[k] for k in item], total_statics) + '\n'
                line += sep
            # row
            line += '{:>9} |'.format(m_type) + ' |'.join(['{:.4f}'] * (1 + len(item.keys()))).format(*item.values(), total[m_type]) + '\n'
        lines.append(line)
    
    return lines

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '#'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

def cal_rank(data):
    sub_xyz, xyz, r = data
    ranks = []
    for x, y, z in sub_xyz:
        bound_x = (x-r <= xyz[:, 0]) & (xyz[:, 0] <= x+r)
        bound_y = (y-r <= xyz[:, 1]) & (xyz[:, 1] <= y+r)
        bound_z = ( -1 <= xyz[:, 2]) & (xyz[:, 2] <= z)
        inbox_idx = bound_x & bound_y & bound_z
        inbox_points = np.sum(inbox_idx)
        ranks.append(inbox_points)
    return ranks

def convert_to_cam_coordinate(xyz, cam_param):
        '''
            Arguments:
            xyz np.array [N x 3]
            cam_param (tx, ty, tz, rx, ry, rz, worldInverseMat, projectionMat)
        '''
        tx, ty, tz = cam_param[:3]
        rx, ry, rz = cam_param[3:6]
        # need to convert from column major
        # world_inverse_mat = cam_param[6:22].reshape((4, 4)).T

        cosYaw = math.cos(-rz)
        sinYaw = math.sin(-rz)
        Rz = np.array(([
            [cosYaw, sinYaw, 0],
            [-sinYaw, cosYaw, 0],
            [0, 0, 1]
        ]), dtype=np.float32)
        
        cosRoll = math.cos(-rx)
        sinRoll = math.sin(-rx)
        Rx = np.array([
            [1, 0, 0],
            [0, cosRoll, sinRoll],
            [0, -sinRoll, cosRoll]
        ], dtype=np.float32)

        # 1. translate
        cXYZ = xyz - [tx, ty, tz]
        # 2. rotate inverse of ZXY
        cXYZ = cXYZ @ (Rz @ Rx)

        return cXYZ.astype(np.float32)

def convert_to_projection_coordinate(cxyz, cam_param):
    # need to convert from column major
    # map to x:[-1, 1], y:[-1, 1], z:[-1, 1]
    projection_mat = cam_param[22:38].reshape((4, 4)).T

    # add one more homogenose
    pXYZ = np.ones((cxyz.shape[0], 4), dtype=np.float32)
    pXYZ[:, :3] = cxyz
    pXYZ = (projection_mat @ pXYZ.T).T
    # divide w
    # from https://stackoverflow.com/questions/16202348/numpy-divide-row-by-row-sum
    pXYZ = pXYZ / pXYZ[:, -1, None]

    return pXYZ[:, :3]

def old_closer_to_the_inside_point(xyz, inside, direction = 1, space = 1):
    inside_xyz = xyz[inside == 1]
    offset_z = np.max(inside_xyz[:, 2]) if direction == 1 else np.min(inside_xyz[:, 2])
    if space == 0:
        offset_z = np.mean(inside_xyz[:, 2])
    mean_x = np.mean(inside_xyz[:, 0])
    mean_y = np.mean(inside_xyz[:, 1])

    xyz = xyz - [mean_x, mean_y, offset_z]
    return xyz

def closer_to_the_inside_point(xyz, inside, direction = 1, space = 1):
    inside_xyz = xyz[inside == 1]
    
    mean_x = np.mean(inside_xyz[:, 0])
    mean_y = np.mean(inside_xyz[:, 1])
    mean_z = np.mean(inside_xyz[:, 2])
    # mean_x = -1.0
    # mean_y = -1.0
    # mean_z = 1.0
    d = math.sqrt(mean_x **2 + mean_y**2 + mean_z**2)
    d2 = math.sqrt(mean_x **2 + mean_z **2)
    
    sinBeta = mean_x / d2
    cosBeta = -mean_z / d2
    Ry = np.array([
        [cosBeta, 0, -sinBeta],
        [0, 1, 0],
        [sinBeta, 0, cosBeta]
    ], dtype=np.float64)

    sinGama = -mean_y / d
    cosGama = abs(d2 / d)
    Rx = np.array([
        [1, 0, 0],
        [0, cosGama, sinGama],
        [0, -sinGama, cosGama]
    ], dtype=np.float64)

    Rmt = Ry @ Rx
    xyz = xyz @ Rmt
    # print('check===>', np.array([[mean_x, mean_y, mean_z]], dtype=np.float64) @ Ry @ Rx)

    inside_xyz = xyz[inside == 1]
    offset_z = np.max(inside_xyz[:, 2])
    xyz[:, 2] = xyz[:, 2] - offset_z

    # xyz = xyz - [mean_x, mean_y, offset_z]
    return xyz


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard


# --------------------------- Lovasz BINARY LOSSES ---------------------------


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)
        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """

    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        return loss

    # deal with the void prediction case (only void pixels)
    loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                   lambda: tf.reduce_sum(logits) * 0.,
                   compute_loss,
                   strict=True,
                   name="loss"
                   )
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels


# --------------------------- Focal BINARY LOSSES ---------------------------

def focal_loss_sigmoid_on_2_classification(labels, logtis, alpha=0.5, gamma=2):
    y_pred = tf.to_float(tf.sigmoid(logtis[:, :, 1])) # 转换成概率值
    labels = tf.to_float(tf.argmax(labels, axis=2)) # int -> float

    loss = -labels * alpha * ((1 - y_pred) ** gamma) * tf.log(tf.clip_by_value(y_pred, 1e-9, 1.0)) \
     -(1 - labels) * (1 - alpha) * (y_pred ** gamma) * tf.log(tf.clip_by_value(1- y_pred, 1e-9, 1.0))
    return loss

