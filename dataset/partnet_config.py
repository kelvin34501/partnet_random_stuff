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

# parts
__C.parts_columns = ['global_id', 'anno_id', 'part_relative_id', 'name', 'text', 'objs', 'depth', 'height']
__C.part_parent_child_columns = ['parent_global_id', 'child_global_id']
__C.part_sibling_columns = ['sibling_a_id', 'sibling_b_id']
__C.part_leaf_columns = ['leaf_global_id']
