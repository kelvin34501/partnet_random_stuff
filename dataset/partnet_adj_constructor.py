import os
import sys

BASE_PATH = os.path.dirname(__file__)
sys.path.append(BASE_PATH)
from dataset_util import Dataset
from mesh_util import load_pc, get_pc
from partnet_config import cfg
from partnet_meta_constructor import PartnetMetaConstructor
from preprocess import *

import trimesh
import pymesh
import random
import numpy as np
import pandas as pd

from itertools import combinations


class PartnetAdjacencyConstructor():
    def __init__(self, meta_constructor, graph_dir=None, use_cache=True):
        self.meta_constructor = meta_constructor
        self.meta = self.meta_constructor.df
        self.parts = self.meta_constructor.parts
        self.part_parent_child = self.meta_constructor.part_parent_child
        self.part_sibling = self.meta_constructor.part_sibling
        self.part_leaf = self.meta_constructor.part_leaf

        if graph_dir is None:
            self.graph_dir = cfg.graph_dir
        else:
            self.graph_dir = graph_dir
        self.use_cache = use_cache

    def _get_part_id_of_instance(self, item_id):
        all_parts = self.parts[self.parts['item_id'] == item_id]['global_id']
        joined_parts_id = self.part_leaf[self.part_leaf['leaf_global_id'].isin(all_parts)]['leaf_global_id']
        joined_parts = self.parts[self.parts['global_id'].isin(joined_parts_id)]
        return joined_parts

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

    def construct_adj_graph(self):
        index_list = list(self.meta.index)
        for item_id in index_list:
            leaf_desc = self._get_part_id_of_instance(item_id)
            res_list = []
            for local_id in range(len(leaf_desc)):
                mesh = self._load_mesh(leaf_desc.iloc[local_id])
                res_list.append((local_id, mesh))
            adj_mat = np.zeros((len(leaf_desc), len(leaf_desc)))
            for part_a, part_b in combinations(res_list, 2):
                part_a_id, part_a_mesh = part_a
                part_b_id, part_b_mesh = part_b



if __name__ == '__main__':
    m = PartnetMetaConstructor(cfg.partnet)
    m.construct_meta()
    a = PartnetAdjacencyConstructor(m)
    a.construct_adj_graph()
