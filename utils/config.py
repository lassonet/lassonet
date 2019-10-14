import yaml

def merge(default, cfg):
    for key, value in cfg.items():
        if isinstance(value, dict):
            # get node or create one
            if isinstance(default, dict):
                node = default.setdefault(key, {})
                default[key] = merge(node, value)
            else:
                node = getattr(default, key, {}) # default.setdefault(key, {})
                setattr(default, key, merge(node, value))
        else:
            if isinstance(default, dict):
                default[key] = value
            else:
                setattr(default, key, value)

    return default

class Config:

    def __init__(self, filename):
        self.NUM_GPUS = 1 # How many gpus to use [default: 1]
        self.BATCH_SIZE = 8 # Batch Size during training [default: 8]
        self.MAX_EPOCH = 201 # Epoch to run [default: 201]
        self.NUM_CLASSES = 2
        self.DATASET = {
            'TRAIN_FILES': ['full_cam_d2_a1.h5'],
            'SPACE': 1, # Which space to convert the data [default: 1]
            'NUM_POINTS': 40960, # How many points per data sample [default: 40960]
            'CLOSER': 1
        }
        self.LOSS = {
            'FN': 'weight_softmax_ce', # The loss function [default: weight_softmax_ce]
            'CLS_WEIGHTS': [1.0, 5.0],
            "ALPHA": 0.8,
            "GAMMA": 2
        }
        self.OPTIMIZER = {
            'TYPE': 'adam', # adam or momentum [default: adam]
            'DECAY_STEP': 200000, # Decay step for lr decay [default: 200000]
            'DECAY_RATE': 0.7, # Decay rate for lr decay [default: 0.7]
            'MOMENTUM': 0.9, # Initial learning rate [default: 0.9]
            'BASE_LR': 0.001, # Initial learning rate [default: 0.001]
        }
        self.BN = {
            'INIT_DECAY': 0.5,
            'DECAY_RATE': 0.5,
            'DECAY_STEP': float(200000),
            'DECAY_CLIP': 0.99
        }
        self.MODEL = {
            'MAX_NUM_GROUPS': 2048,
            'GROUP_SIZE': 32,
            'MODEL_TYPE': 'part', # or scene or 'pn'
            'USE_T_NET': False,
            'USE_CLS': True
        }
        self.OUTPUT_PATH = ''
        self.PORT = 11221

        with open(filename, 'r') as stream:
            cfg = yaml.safe_load(stream)
        merge(self, cfg)
        num_cls = 0
        already_has_d2 = False
        for tf in self.DATASET['TRAIN_FILES']:
            if '_d1_a' in tf:
                num_cls += 1
            elif '_d2_a' in tf and not already_has_d2:
                num_cls += 1

        self.DATASET['NUM_CLS'] = num_cls

        # assert
        assert(self.BATCH_SIZE % self.NUM_GPUS == 0)

    def __str__(self):
        return "Config : " + str(self.__dict__)
