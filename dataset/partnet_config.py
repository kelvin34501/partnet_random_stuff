"""
partnet_config.py
"""

import os
import sys
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Fallback choice of partnet dataset
__C.partnet = "/disk1/data/partnet/data_v0"

# meta
__C.columns = ['anno_id', 'model_id', 'cat', 'model_path']

