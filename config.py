"""
config.py
"""

import os
import sys
import socket
import time
from easydict import EasyDict as edict

__C = edict()
cfg = __C

BASE_PATH = os.path.dirname(__file__)

__C.BATCH_SIZE = 1
__C.SAMPLE_POINTS = 8192  # None for no sampling
__C.MAX_EPOCH = 200

__C.LEARNING_RATE_BASE = 1e-3
__C.LEARNING_RATE_DECAY_STEP = 200000
__C.LEARNING_RATE_DECAY_RATE = 0.7

__C.BN_INIT_DECAY = 0.5
__C.BN_DECAY_DECAY_RATE = 0.5
__C.BN_DECAY_DECAY_STEP = float(__C.LEARNING_RATE_DECAY_STEP)
__C.BN_DECAY_CLIP = 0.99

__C.CALC_GRAPH = "ae_fixsph_coord_cd"
__C.CALC_GRAPH_FILE = os.path.join(BASE_PATH, 'model', __C.CALC_GRAPH + '.py')

__C.LOG_DIR = 'log'
__C.RUN_NAME = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()) + '-' + socket.gethostname() + '-' + 'TestRun'
__C.WRITE_DIR = os.path.join(__C.LOG_DIR, __C.RUN_NAME)

__C.PARTNET = "/disk1/data/partnet/data_v0"
__C.CAT = 'Chair'

# GPU RELATED CONFIG
__C.GPU = 0
