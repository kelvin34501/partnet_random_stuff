import os
import sys

BASE_PATH = os.path.dirname(__file__)
sys.path.append(BASE_PATH)
from dataset_util import Dataset
from mesh_util import load_pc, get_pc, get_bbox, get_brect, get_bbox_extent, draw_boxes3d
from partnet_config import cfg
from partnet_meta_constructor import PartnetMetaConstructor
from preprocess import *

import trimesh
import pymesh
import random
import numpy as np
import pandas as pd

from itertools import combinations
from tqdm import tqdm


class PartnetBBoxConstructor():
    def __init__(self, meta_constructor, bbox_dir=None):
        self.meta_constructor = meta_constructor
        self.meta = self.meta_constructor.df
        self.parts = self.meta_constructor.parts
        self.part_parent_child = self.meta_constructor.part_parent_child
        self.part_sibling = self.meta_constructor.part_sibling
        self.part_leaf = self.meta_constructor.part_leaf

        if bbox_dir is None:
            self.bbox_dir = cfg.bbox_dir
        else:
            self.bbox_dir = bbox_dir

    @staticmethod
    def _load_mesh(desc):
        path = desc['objs_dir']
        objs = desc['objs']
        mesh_list = []
        for obj in eval(objs):
            obj_path = os.path.join(path, obj + '.obj')
            mesh_tmp = load_pc(obj_path)
            mesh_tmp, info = pymesh.remove_isolated_vertices(mesh_tmp)
            mesh_list.append(mesh_tmp)
        mesh = pymesh.merge_meshes(mesh_list)
        return mesh

    def construct_bbox(self, use_cache=True):
        if use_cache:
            return None

        for i in tqdm(range(len(self.parts))):
            part_desc = self.parts.iloc[i]
            part_global_id = part_desc['global_id']

            part_mesh = self._load_mesh(part_desc)
            try:
                bbox, extents, transform = get_bbox(part_mesh)
            except Exception as e:
                bbox, extents, transform = get_brect(part_mesh)

            np.savetxt(os.path.join(cfg.bbox_dir, part_global_id + '.txt'), bbox)


class PartnetBBoxDataset(Dataset):
    def __init__(self, meta_constructor, get_mode='global_id', bbox_dir=None):
        self.meta_constructor = meta_constructor
        self.meta = self.meta_constructor.df
        self.parts = self.meta_constructor.parts

        if bbox_dir is None:
            self.bbox_dir = cfg.bbox_dir
        else:
            self.bbox_dir = bbox_dir

        self.get_mode = get_mode

    def _get_by_global_id(self, global_id):
        assert isinstance(global_id, str)
        res = np.loadtxt(os.path.join(self.bbox_dir, global_id + '.txt'))
        return res

    def _get_by_index(self, index):
        assert isinstance(index, int)
        global_id = self.parts.iloc[index]['global_id']
        res = np.loadtxt(os.path.join(self.bbox_dir, global_id + '.txt'))
        return res

    def __getitem__(self, item):
        if self.get_mode == 'global_id':
            return self._get_by_global_id(item)
        else:
            return self._get_by_index(item)

    def __len__(self):
        return len(self.parts)


if __name__ == '__main__':
    m = PartnetMetaConstructor(cfg.partnet)
    m.construct_meta()
    a = PartnetBBoxConstructor(m)
    a.construct_bbox()
    bd = PartnetBBoxDataset(m)
    print(bd['0_1'])
    print(get_bbox_extent(bd['0_1']))
