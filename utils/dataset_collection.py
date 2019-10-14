import os
import sys
from os import path
from datetime import datetime
import time
import json
import math
import multiprocessing
from multiprocessing import Pool

import numpy as np
import h5py

import selection_util as sl_util

WORLD_SPACE = 0
CAMERA_SPACE = 1
NDC_SPACE = 2

def concat(main, sub):
    if type(main) is list:
        if type(sub) is not list:
            sub = [t for t in sub]
        main += sub
        return main
    else:
        return sub if main is None else np.concatenate((main, sub), axis=0)

# full attrs: x,y,z, isIn, dx, dy, dz
class DatasetCollection:

    # should add more features, such as split (train/test), and samples
    def __init__(self, h5_filepaths, space = 0, num_points= 40 * 1024, force_generate_random_index=False, closer=1, max_npoints=2048, reshape=False, **kwargs):
        '''
            Argument:
                h5_filepath (string), path to the h5 file
        '''
        self.space = space
        self.npoints = num_points
        self.closer = closer
        self.max_npoints = max_npoints

        self.px_w = 2.0 / 1920.0

        # load data
        self.h5_filepaths = h5_filepaths
        self.records = None
        self.targets = None
        self.scenes = None
        self.record_2_cls = None
        self.record_2_scene = None
        self.record_2_cam_params = None
        self.record_2_hl = None
        self.train_scenes = []
        self.test_scenes = []

        self.cache_files = []
        self.group_interval = []
        self.record_interval = []
        self.scene_interval = []
        self.test_group_idx = []
        self.eval_return = [0, 1, 2] # 0-cls, 1-highlight, 2-total
        self.max_cls = len(h5_filepaths)
        self.filter_cache_prefix = '_filter_bbox_'

        if reshape:
            self.records = []
            self.targets = []
            self.scenes = []

        for cls_i, h5_filepath in enumerate(h5_filepaths):
            # store interval begin
            beg = 0 if self.records is None else len(self.records)
            scene_beg = 0 if self.scenes is None else len(self.scenes)
            # get data from file
            f = h5py.File(h5_filepath)
            tmp_records = f.get('record')[()]
            tmp_targets = f.get('target')[()]
            tmp_record_2_scene = f.get('record_2_scene')[()].astype(np.uint16)
            tmp_scenes = f.get('scene')[()]
            tmp_record_2_cam_params = f.get('record_2_cam_pos')[()]
            tmp_record_2_hl = f.get('recrod_2_highlight')[()]
            f.close()
            tmp_record_2_cls = np.full((len(tmp_records)), cls_i)

            # concat
            self.records = concat(self.records, [t for t in tmp_records])
            self.targets = concat(self.targets, [t for t in tmp_targets])
            tmp_record_2_scene += scene_beg
            self.record_2_scene = concat(self.record_2_scene, tmp_record_2_scene) 
            self.scenes = concat(self.scenes, [s for s in tmp_scenes])
            self.record_2_cam_params = concat(self.record_2_cam_params, tmp_record_2_cam_params)
            self.record_2_hl = concat(self.record_2_hl, tmp_record_2_hl)
            self.record_2_cls = concat(self.record_2_cls, tmp_record_2_cls)

            # update interval with end
            self.record_interval.append((beg, len(self.records)))
            self.scene_interval.append((scene_beg, len(self.scenes)))

        if reshape:
            self.records = np.array(self.records, dtype=object)
            self.targets = np.array(self.targets, dtype=object)
            self.scenes = np.array(self.scenes, dtype=object) 
        elif type(self.records) is list:
            self.records = np.array(self.records, dtype=np.bool)
            self.targets = np.array(self.targets, dtype=np.bool)
            self.scenes = np.array(self.scenes, dtype=np.float32) 

        print('load records.shape:', self.records.shape,
            'targets.shape', self.targets.shape,
            'record_2_scene.shape:', self.record_2_scene.shape,
            'scenes.shape:', self.scenes.shape,
            'record_2_cam_params.shape:', self.record_2_cam_params.shape)

        self.record_2_keep_idx, self.group_2_record_idx, \
        self.record_2_outside_T, self.record_2_outside_F, \
        self.train_scenes, self.test_scenes = self._filter_and_group_outside(force_generate_random_index)
        
        print('Finish load the dataset ' + ','.join(h5_filepaths), 'dataset len: %d' % (len(self)))

    def _filter_and_group_outside(self, force_regenerate):
        record_2_keep_idx = []
        train_scenes = []
        test_scenes = []
        group_2_record_idx = None
        record_2_outside_T = None
        record_2_outside_F = None
        for i, h5_filepath in enumerate(self.h5_filepaths):
            group_beg = 0 if group_2_record_idx is None else len(group_2_record_idx)
            h5_dirname = path.dirname(h5_filepath)
            h5_filename = path.basename(h5_filepath)[:-3]
            record_beg, _ = self.record_interval[i]
            random_index_files = [f for f in os.listdir(h5_dirname) if (f.endswith('.h5') 
                and h5_filename + self.filter_cache_prefix + str(self.npoints) in f)
            ]

            if not force_regenerate and len(random_index_files) != 0:
                # if not force_regenerate and cache file exist
                # then load the cache file
                latest_date = 0
                latest_file = ''
                for f in random_index_files:
                    datetime_str = f[-15:-3]
                    ts = datetime.strptime(datetime_str, "%Y%m%d%H%M").timestamp()
                    if ts > latest_date:
                        latest_date = ts
                        latest_file = f
                cache_file = path.join(h5_dirname, latest_file)
                self.cache_files.append(cache_file)

                h5 = h5py.File(cache_file, 'a')
                record_2_keep_idx = concat(record_2_keep_idx,[t for t in h5.get('record_2_keep_idx')[()]])
                group_2_record_idx = h5.get('group_2_record_idx')[()] if group_2_record_idx is None else\
                                    np.concatenate((group_2_record_idx, [(record_idx + record_beg, offset) for record_idx, offset in  h5.get('group_2_record_idx')[()]]), axis=0)
                record_2_outside_T = concat(record_2_outside_T, h5.get('record_2_outside_T')[()])
                record_2_outside_F = concat(record_2_outside_F, h5.get('record_2_outside_F')[()])
                scene_beg, scene_end = self.scene_interval[i]
                train_scenes = concat(train_scenes, [sId + scene_beg for sId in h5.get('train_scene')[()]])
                test_scenes = concat(test_scenes, [sId + scene_beg for sId in h5.get('test_scene')[()]])
            else:
                # otherwise, generate
                tmp_record_2_keep_idx, tmp_group_2_record_idx, \
                tmp_record_2_outside_T, tmp_record_2_outside_F = self._filter_outside_new_(i, h5_filename)
                
                record_2_keep_idx = concat(record_2_keep_idx, tmp_record_2_keep_idx)
                group_2_record_idx = tmp_group_2_record_idx if group_2_record_idx is None else\
                                    np.concatenate((group_2_record_idx, [(record_idx + record_beg, offset) for record_idx, offset in tmp_group_2_record_idx]), axis=0)
                record_2_outside_T = concat(record_2_outside_T, tmp_record_2_outside_T)
                record_2_outside_F = concat(record_2_outside_F, tmp_record_2_outside_F)

                # strong split
                scene_2_group = {}
                for tmp_record_idx, offset in tmp_group_2_record_idx:
                    record_idx = tmp_record_idx + record_beg
                    scene_idx = self.record_2_scene[record_idx]
                    scene_2_group.setdefault(scene_idx, []).append(str(tmp_record_idx) + '_' + str(offset))
                scene_groups_count = { scene_idx:len(group_idx_arr) for scene_idx, group_idx_arr in scene_2_group.items() }
 
                attempt = 0
                tmp_scenes = scene_groups_count.keys()
                tmp_test_idx = []
                tmp_test_scene = []
                tmp_test_count = round(sum(scene_groups_count.values()) * 0.1)
                while len(tmp_test_idx) < tmp_test_count:
                    # pick the first
                    candidates = [scene_idx for scene_idx, count in scene_groups_count.items() if (count+len(tmp_test_idx)) < (10 + int(tmp_test_count)) ]
                    if len(candidates) == 0:
                        break
                    try_scene_idx = np.random.choice(candidates, 1)[0] # max to 1.5x test_count
                    tmp_test_scene.append(try_scene_idx)
                    tmp_test_idx += scene_2_group[try_scene_idx]

                    del scene_groups_count[try_scene_idx]
                    del scene_2_group[try_scene_idx]
                    attempt += 1
                    if attempt > 1000:
                        raise Exception('Try too many times in strong split')

                tmp_train_scene = [scene_id for scene_id in tmp_scenes if scene_id not in tmp_test_scene]

                train_scenes = concat(train_scenes, tmp_train_scene)
                test_scenes = concat(test_scenes, tmp_test_scene)

                ## remove scene_id offset
                scene_beg, scene_end = self.scene_interval[i]
                tmp_train_scene = [scene_id - scene_beg for scene_id in tmp_train_scene]
                tmp_test_scene = [scene_id - scene_beg for scene_id in tmp_test_scene]

                # save the random index to cache
                cache_file = path.join(h5_dirname,
                                        h5_filename + self.filter_cache_prefix + str(self.npoints) + '_' +
                                        datetime.now().strftime("%Y%m%d%H%M") + '.h5')
                self.cache_files.append(cache_file)
                h5_fout = h5py.File(cache_file)
                first_len = len(tmp_record_2_keep_idx[0])
                if all( len(t) == first_len for t in tmp_record_2_keep_idx ): # self.npoints == self.max_npoints:
                    h5_fout.create_dataset('record_2_keep_idx', data=tmp_record_2_keep_idx,
                            compression='gzip', compression_opts=4, dtype='uint32')
                else:
                    dt = h5py.special_dtype(vlen=np.dtype('uint32'))
                    h5_fout.create_dataset('record_2_keep_idx', shape=(len(tmp_record_2_keep_idx), ),  
                                        data=tmp_record_2_keep_idx,
                                        compression='gzip', compression_opts=4, dtype=dt)
                h5_fout.create_dataset('group_2_record_idx', data=tmp_group_2_record_idx,
                                    compression='gzip', compression_opts=4, dtype='uint32')
                h5_fout.create_dataset('record_2_outside_T', data=tmp_record_2_outside_T,
                            compression='gzip', compression_opts=4, dtype='uint32')
                h5_fout.create_dataset('record_2_outside_F', data=tmp_record_2_outside_F,
                            compression='gzip', compression_opts=4, dtype='uint32')
                h5_fout.create_dataset('train_scene', data=tmp_train_scene,
                            compression='gzip', compression_opts=4, dtype='uint32')
                h5_fout.create_dataset('test_scene', data=tmp_test_scene,
                            compression='gzip', compression_opts=4, dtype='uint32')
                h5_fout.close()
            self.group_interval.append((group_beg, len(group_2_record_idx)))
        return record_2_keep_idx, group_2_record_idx, record_2_outside_T, record_2_outside_F, train_scenes, test_scenes

    def _filter_outside_new_(self, dataset_idx, filename= ''):
        ''' Use this function to filter the points outside the frustum
        '''
        record_beg, record_end = self.record_interval[dataset_idx]
        record_2_keep_idx = []
        record_2_outside_T = []
        record_2_outside_F = []
        group_2_record_idx = []
        # set up the number
        for i, inside in enumerate(self.records[record_beg:record_end]):
            if self.npoints == self.max_npoints:
                # keep all
                keep_idx = np.arange(self.npoints)
                record_2_keep_idx.append(keep_idx)

                group_2_record_idx.append((i, 0))
                record_2_outside_T.append(0)
                record_2_outside_F.append(0)
                sl_util.printProgressBar(i, record_end - record_beg, prefix='Filter ' + filename + ' outsides:', suffix='Complete', length=50)
                continue

            offset_i = i + record_beg
            scene_idx = self.record_2_scene[offset_i]
            label = self.targets[offset_i]
            xyz = self.scenes[scene_idx]
            cam_params = self.record_2_cam_params[offset_i]

            # convert to s2
            xyz = sl_util.convert_to_cam_coordinate(xyz, cam_params)
            xyz = sl_util.convert_to_projection_coordinate(xyz, cam_params)
            # get inside bbox in s2
            inside_xyz = xyz[inside == 1]
            inside_min_xy = np.min(inside_xyz, axis=0)
            inside_max_xy = np.max(inside_xyz, axis=0)

            # inside_points = np.sum(inside)
            # in_out_factor = inside_points / inside.shape[0]
            # extend as 10 px
            inside_min_xy -= self.px_w * 1.0
            inside_max_xy += self.px_w * 1.0
            # filter use new bbox
            bound_x = (inside_min_xy[0] <= xyz[:, 0]) & (xyz[:, 0] <= inside_max_xy[0])
            bound_y = (inside_min_xy[1] <= xyz[:, 1]) & (xyz[:, 1] <= inside_max_xy[1])
            extend_inbox_idx = bound_x & bound_y

            inside_points = np.sum(extend_inbox_idx)
            round_up_offset = inside_points % self.npoints
            round_up_offset = self.npoints - round_up_offset if round_up_offset > 0 else 0
            round_up_to_npoints = round_up_offset + inside_points
            assert(round_up_to_npoints % self.npoints == 0), 'Round up npoints should be divided by self.npoints'

            # handle the index problem
            filter_idx = extend_inbox_idx == False
            # get idx of outside candidates closer
            tmp_r = 1.0
            outside_idx = np.zeros((0))
            while outside_idx.shape[0] < round_up_offset:
                tmp_r += 1.0
                tmp_min_xy = inside_min_xy - self.px_w * tmp_r
                tmp_max_xy = inside_max_xy + self.px_w * tmp_r
                # filter use iterative bbox
                bound_x = (tmp_min_xy[0] <= xyz[:, 0]) & (xyz[:, 0] <= tmp_max_xy[0])
                bound_y = (tmp_min_xy[1] <= xyz[:, 1]) & (xyz[:, 1] <= tmp_max_xy[1])
                outside_idx = bound_x & bound_y
                assert(len(outside_idx) >= round_up_offset), 'The length of outside_idx should be greater than round_up_offset before filtering'
                outside_idx[extend_inbox_idx == True] = False
                outside_idx = np.argwhere(outside_idx)[:, 0]

            # random pick round_up_offset to false 
            if len(outside_idx) > 0:
                random_pick_offset_idx = np.random.choice(outside_idx, round_up_offset, replace=False)
                filter_idx[random_pick_offset_idx] = False

            # assert(self.max_npoints == round_up_to_npoints + np.sum(filter_idx))

            # filter the points
            keep_idx = np.argwhere(filter_idx == False)[:, 0]
            np.random.shuffle(keep_idx)
            record_2_keep_idx.append(keep_idx)

            num_groups = int(keep_idx.shape[0] / self.npoints)
            group_2_record_idx += [(i, offset) for offset in range(num_groups)]

            # 
            outside_label = label[filter_idx]
            record_2_outside_T.append(np.sum(outside_label == 1))
            record_2_outside_F.append(np.sum(outside_label == 0))
            sl_util.printProgressBar(i, record_end - record_beg, prefix='Filter ' + filename + ' outsides:', suffix='Complete', length=50)
        return record_2_keep_idx, group_2_record_idx, record_2_outside_T, record_2_outside_F

    def __getitem__(self, group_idx):
        # find the record
        record_index, group_idx_offset = self.group_2_record_idx[group_idx]
        record = self.records[record_index]
        target = self.targets[record_index]
        cls_i = self.record_2_cls[record_index]
        scene_idx = self.record_2_scene[record_index]
        scene = self.scenes[scene_idx]
        cam_params = self.record_2_cam_params[record_index]

        # find the subscene index
        subscene_point_index = self.record_2_keep_idx[record_index][
                self.npoints * group_idx_offset : self.npoints * (group_idx_offset +1)]

        # load the data
        inside = record[subscene_point_index].astype(np.uint8)
        label = target[subscene_point_index].astype(np.uint8)
        xyz = scene[subscene_point_index].astype(np.float32)
        space_convertor = [
            sl_util.convert_to_cam_coordinate,
            sl_util.convert_to_projection_coordinate
        ]
        for i in range(self.space):
            xyz = space_convertor[i](xyz, cam_params)
        
        if self.closer in [0, 1]:
            closer_fn = sl_util.old_closer_to_the_inside_point if self.closer == 0 else sl_util.closer_to_the_inside_point
            xyz = closer_fn(xyz, inside, -1 if self.space == NDC_SPACE else 1, space=self.space)

        inside = np.expand_dims(inside, -1)
        return xyz, inside, label, cls_i
    
    def __len__(self):
        return len(self.group_2_record_idx)

    def sample_for_strong_split(self, idx, num_gpus=1):
        num_unit = len(idx) / num_gpus
        train_count = math.floor(0.9 * num_unit)
        train_count *= num_gpus
        test_count = len(idx) - train_count
        
        # turn record_2_scene -> scene_2_record
        scene_2_group = {}
        for group_idx in idx:
            record_index, group_idx_offset = self.group_2_record_idx[group_idx]
            scene_idx = self.record_2_scene[record_index]
            scene_2_group.setdefault(scene_idx, []).append(group_idx)

        scene_groups_count = { scene_idx:len(group_idx_arr) for scene_idx, group_idx_arr in scene_2_group.items() }
        attempt = 0
        test_idx = []
        while len(test_idx) < test_count:
            # pick the first
            try_scene_idx = np.random.choice([scene_idx for scene_idx, count in scene_groups_count.items() if count < int(test_count * 1.0) ], 1) # max to 1.5x test_count
            try_scene_idx = try_scene_idx[0]
            test_idx += scene_2_group[try_scene_idx]
            del scene_groups_count[try_scene_idx]
            del scene_2_group[try_scene_idx]
            attempt += 1
            if attempt > 1000:
                raise Exception('Try too many times in strong split')

        train_idx = []
        for group_idx_arr in scene_2_group.values():
            train_idx += group_idx_arr

        return train_idx, test_idx

    def strong_split(self, num_gpus = 1):
        idx = list(range(len(self)))
        train_scene = self.train_scenes
        val_scene = self.test_scenes
        scene_2_group = {}
        for group_idx in idx:
            record_index, group_idx_offset = self.group_2_record_idx[group_idx]
            scene_idx = self.record_2_scene[record_index]
            scene_2_group.setdefault(scene_idx, []).append(group_idx)
        # sync
        train_idx = []
        for scene in train_scene:
            train_idx += scene_2_group[scene]
        np.random.shuffle(train_idx)

        test_idx = []
        for scene in val_scene:
            test_idx += scene_2_group[scene]
        np.random.shuffle(test_idx)

        # add offset
        offset = len(train_idx) % num_gpus
        offset = 0 if offset == 0 else num_gpus - offset
        offset_idx = np.random.choice(train_idx, offset, replace=False)
        train_idx += offset_idx.tolist()

        offset = len(test_idx) % num_gpus
        offset = 0 if offset == 0 else num_gpus - offset
        offset_idx = np.random.choice(test_idx, offset, replace=False)
        self.test_group_idx = test_idx + offset_idx.tolist()
        idx = train_idx + self.test_group_idx
        tick = len(train_idx)

        xyz = []
        features = []
        labels = []
        clss = []
        for i in idx:
            x, feat, y, cls_i = self[i]
            xyz.append(x)
            features.append(feat)
            one_hot_y = np.zeros((self.npoints, 2), dtype=np.uint8)
            one_hot_y[:, 0] = np.where(y == 0, 1, 0)
            one_hot_y[:, 1] = y
            labels.append(one_hot_y)
            one_hot_cls = [0] * self.max_cls
            one_hot_cls[cls_i] = 1
            clss.append(one_hot_cls)
        xyz = np.array(xyz, dtype=np.float32)
        features = np.array(features, dtype=np.float32)
        labels = np.array(labels, dtype=np.uint8)
        clss = np.array(clss, dtype=np.uint8)

        # count train
        train_split_statics = [{}, {}]
        val_split_statics = [{}, {}]
        # train
        for group_idx in idx[:tick]:
            record_index, _ = self.group_2_record_idx[group_idx]
            cls = self.record_2_cls[record_index]
            highlight = self.record_2_hl[record_index]
            train_split_statics[0][cls] = train_split_statics[0].get(cls, 0) + 1
            train_split_statics[1][highlight] = train_split_statics[1].get(highlight, 0) + 1
        # val
        for group_idx in idx[tick:]:
            record_index, _ = self.group_2_record_idx[group_idx]
            cls = self.record_2_cls[record_index]
            highlight = self.record_2_hl[record_index]
            # count
            val_split_statics[0][cls] = val_split_statics[0].get(cls, 0) + 1
            val_split_statics[1][highlight] = val_split_statics[1].get(highlight, 0) + 1
        # merge
        self.split_statics = [{}, {}, (tick, len(self.test_group_idx))]
        for i, item in enumerate(train_split_statics):
            for k, v in item.items():
                self.split_statics[i][k] = (train_split_statics[i].get(k, 0), val_split_statics[i].get(k, 0))

        self.naive_iou = [{}, {}, []]
        self.naive_pos_pre = [{}, {}, []]
        self.naive_pos_rec = [{}, {}, []]
        T2 = {}
        T3 = {}
        T3_and_T2 = {}
        T3_or_T2 = {}
        for i, group_idx in enumerate(self.test_group_idx):
            record_index, _ = self.group_2_record_idx[group_idx]
            _, feat, y, _ = self[group_idx]
            T2[record_index] = T2.get(record_index, 0) + np.sum((y == 1))
            T3[record_index] = T3.get(record_index, 0) + np.sum((feat[:, -1] == 1))
            T3_and_T2[record_index] = T3_and_T2.get(record_index, 0) + np.sum((feat[:, -1] == 1) & (y == 1))
            T3_or_T2[record_index] = T3_or_T2.get(record_index, 0) + np.sum((feat[:, -1] == 1) | (y == 1))

        naive_iou = [{}, {}, []]
        naive_pos_pre = [{}, {}, []]
        naive_pos_rec = [{}, {}, []]
        for record_index in T3_and_T2:
            cls = self.record_2_cls[record_index]
            highlight = self.record_2_hl[record_index]
            if T3.get(record_index) != 0:
                v = T3_and_T2.get(record_index) / T3.get(record_index)
                naive_pos_pre[0].setdefault(cls, []).append(v)
                naive_pos_pre[1].setdefault(highlight, []).append(v)
                naive_pos_pre[2].append(v)
            #
            if T2.get(record_index) != 0:
                v = T3_and_T2.get(record_index) / T2.get(record_index)
                naive_pos_rec[0].setdefault(cls, []).append(v)
                naive_pos_rec[1].setdefault(highlight, []).append(v)
                naive_pos_rec[2].append(v)

            if T3_and_T2.get(record_index):
                v = T3_and_T2.get(record_index) / T3_or_T2.get(record_index)
                naive_iou[0].setdefault(cls, []).append(v) 
                naive_iou[1].setdefault(highlight, []).append(v) 
                naive_iou[2].append(v)
        self.naive_iou = [ {k:np.mean(v) for k, v in item.items() } for item in naive_iou[:-1]  ]
        self.naive_iou.append(np.mean(naive_iou[-1]))
        self.naive_pos_pre = [ {k:np.mean(v) for k, v in item.items() } for item in naive_pos_pre[:-1]  ]
        self.naive_pos_pre.append(np.mean(naive_pos_pre[-1]))
        self.naive_pos_rec = [ {k:np.mean(v) for k, v in item.items() } for item in naive_pos_rec[:-1]  ]
        self.naive_pos_rec.append(np.mean(naive_pos_rec[-1]))

        return xyz[:tick], features[:tick], labels[:tick], clss[:tick], \
               xyz[tick:], features[tick:], labels[tick:], clss[tick:]

    def average_group_number(self):
        record_id_to_group_number = {}
        for group_idx in enumerate(self.test_group_idx):
            record_idx, _ = self.group_2_record_idx[group_idx]
            record_id_to_group_number[record_idx] = record_id_to_group_number.get(record_idx, 0) + 1

        sum = 0
        for k, v in record_id_to_group_number.items():
            sum += v
        
        return sum / len(record_id_to_group_number)

    def cal_real_metrix(self, metrix):
        T1 = {}
        F1 = {}
        T2 = {}
        F2 = {}
        T1_and_T2 = {}
        T1_or_T2 = {}
        F1_and_F2 = {}
        
        for i, group_idx in enumerate(self.test_group_idx): # record is big, group_idx is small
            record_index, _ = self.group_2_record_idx[group_idx]
            T1[record_index] = T1.get(record_index, 0) + metrix.get('T1')[i]
            T2[record_index] = T2.get(record_index, 0) + metrix.get('T2')[i]
            F1[record_index] = F1.get(record_index, 0) + metrix.get('F1')[i]
            F2[record_index] = F2.get(record_index, 0) + metrix.get('F2')[i]
            T1_and_T2[record_index] = T1_and_T2.get(record_index, 0) + metrix.get('T1&T2')[i]
            F1_and_F2[record_index] = F1_and_F2.get(record_index, 0) + metrix.get('F1&F2')[i]
            T1_or_T2[record_index] = T1_or_T2.get(record_index, 0) + metrix.get('T1|T2')[i]

        cal_result = {
            'iou': [{}, {}, []], 
            'acc': [{}, {}, []],
            'pos_pre': [{}, {}, []], 
            'pos_rec': [{}, {}, []],
        }

        for record_index in T1:
            T2[record_index] += self.record_2_outside_T[record_index] # T2
            T1_or_T2[record_index] += self.record_2_outside_T[record_index] # T2
            F1[record_index] += self.record_2_outside_F[record_index]
            F2[record_index] += self.record_2_outside_F[record_index]
            F1_and_F2[record_index] += self.record_2_outside_F[record_index]
            cls = self.record_2_cls[record_index]
            highlight = self.record_2_hl[record_index]
            # 
            if T1_and_T2.get(record_index) != 0:
                v = T1_and_T2.get(record_index) / T1_or_T2.get(record_index)
                cal_result['iou'][0].setdefault(cls, []).append(v)
                cal_result['iou'][1].setdefault(highlight, []).append(v)
                cal_result['iou'][2].append(v)

            #
            v = (T1_and_T2[record_index] + F1_and_F2[record_index]) / (T2.get(record_index) + F2.get(record_index))
            cal_result['acc'][0].setdefault(cls, []).append(v)
            cal_result['acc'][1].setdefault(highlight, []).append(v)
            cal_result['acc'][2].append(v)
            #pos_pre
            if T1.get(record_index) != 0:
                v = T1_and_T2.get(record_index) / T1.get(record_index)
                cal_result['pos_pre'][0].setdefault(cls, []).append(v)
                cal_result['pos_pre'][1].setdefault(highlight, []).append(v)
                cal_result['pos_pre'][2].append(v)
            #pos_rec
            if T2.get(record_index) != 0:
                v = T1_and_T2.get(record_index) / T2.get(record_index)
                cal_result['pos_rec'][0].setdefault(cls, []).append(v)
                cal_result['pos_rec'][1].setdefault(highlight, []).append(v)
                cal_result['pos_rec'][2].append(v)

        m_dict = [['iou', 'acc', 'pos_pre', 'pos_rec']] * 2
        results = [ { m_type: { k: np.mean(arr) for k, arr in cal_result[m_type][i].items()  } for m_type in m_types } for i, m_types in enumerate(m_dict) ]

        for i, result in enumerate(results):
            result['naive_iou'] = self.naive_iou[i]
            result['naive_pos_pre'] = self.naive_pos_pre[i]
            result['naive_pos_rec'] = self.naive_pos_rec[i]

        total = { m_type: np.mean(cal_result[m_type][2]) for m_type in ['iou', 'acc', 'pos_pre', 'pos_rec']}
        total['naive_iou'] = self.naive_iou[-1]
        total['naive_pos_pre'] = self.naive_pos_pre[-1]
        total['naive_pos_rec'] = self.naive_pos_rec[-1]
        results.append(total)

        return [results[idx] for idx in self.eval_return], [self.split_statics[idx] for idx in self.eval_return]
