import os
import sys

BASE_PATH = os.path.dirname(__file__)
sys.path.append(BASE_PATH)
from dataset_util import Dataset
from mesh_util import load_pc, get_pc, draw_boxes3d, get_bbox_volume, get_bbox_extent
from partnet_config import cfg
from partnet_meta_constructor import PartnetMetaConstructor
from partnet_bbox_constructor import PartnetBBoxDataset
from preprocess import *
from gjk import gjk_calc
from multiprocessing import Process

import trimesh
import pymesh
import random
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

from itertools import combinations


class PartnetAdjacencyConstructor():
    def __init__(self, meta_constructor, graph_dir=None):
        self.meta_constructor = meta_constructor
        self.meta = self.meta_constructor.df
        self.parts = self.meta_constructor.parts
        self.part_parent_child = self.meta_constructor.part_parent_child
        self.part_sibling = self.meta_constructor.part_sibling
        self.part_leaf = self.meta_constructor.part_leaf

        self.bbox_dataset = PartnetBBoxDataset(self.meta_constructor)

        if graph_dir is None:
            self.graph_dir = cfg.graph_dir
        else:
            self.graph_dir = graph_dir

    def _get_part_of_instance(self, item_id):
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

    def construct_adj_graph(self, verbose=False, use_cache=True):
        if use_cache:
            return None

        if verbose:
            from mayavi import mlab

        index_list = list(self.meta.index)
        progress = tqdm(index_list)
        for item_id in progress:
            if verbose:
                print("============")
            leaf_desc = self._get_part_of_instance(item_id)
            leaf_id = list(leaf_desc['global_id'])
            leaf_bbox = [self.bbox_dataset[id] for id in leaf_id]
            mesh_list = []
            for i in range(len(leaf_desc)):
                mesh_list.append(self._load_mesh(leaf_desc.iloc[i]))
            leaf_id_map = {leaf_id[i]: i for i in range(0, len(leaf_id))}

            adj_mat = np.eye(len(leaf_id))
            adj_dir_mat = np.eye(len(leaf_id))
            if verbose:
                print(leaf_id_map)
            for id_a, id_b in combinations(leaf_id, 2):
                bbox_a = leaf_bbox[leaf_id_map[id_a]]
                bbox_b = leaf_bbox[leaf_id_map[id_b]]
                try:
                    bbox_dist = gjk_calc.calc(bbox_a, bbox_b)
                except Exception as e:
                    progress.write(e)
                    progress.write('=======')
                    progress.write('GJK Error Detected for {}'.format(item_id))
                    progress.write('More information:')
                    progress.write(self.meta.iloc[item_id])
                    bbox_dist = 10.0
                adj_mat[leaf_id_map[id_a], leaf_id_map[id_b]] = bbox_dist
                adj_mat[leaf_id_map[id_b], leaf_id_map[id_a]] = bbox_dist
                if get_bbox_volume(bbox_a) >= get_bbox_volume(bbox_b):
                    adj_dir_mat[leaf_id_map[id_a], leaf_id_map[id_b]] = bbox_dist
                    adj_dir_mat[leaf_id_map[id_b], leaf_id_map[id_a]] = 1.0
                else:
                    adj_dir_mat[leaf_id_map[id_a], leaf_id_map[id_b]] = 1.0
                    adj_dir_mat[leaf_id_map[id_b], leaf_id_map[id_a]] = bbox_dist
            if verbose:
                print(adj_mat)
                for mesh in mesh_list:
                    mlab.triangular_mesh(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], mesh.faces)
                draw_boxes3d(np.stack(leaf_bbox))
                mlab.show()
            adj_res = adj_mat.copy()
            adj_res = np.logical_not(adj_res).astype(np.int)
            adj_dir_res = adj_dir_mat.copy()
            adj_dir_res = np.logical_not(adj_dir_res).astype(np.int)

            # dump things
            with open(os.path.join(self.graph_dir, str(item_id) + '_mapping.pkl'), "wb") as stream:
                pickle.dump(leaf_id_map, stream)
            np.savetxt(os.path.join(self.graph_dir, str(item_id) + '_dist.txt'), adj_mat)
            np.savetxt(os.path.join(self.graph_dir, str(item_id) + '.txt'), adj_res)
            np.savetxt(os.path.join(self.graph_dir, str(item_id) + '_dist_directional.txt'), adj_dir_mat)
            np.savetxt(os.path.join(self.graph_dir, str(item_id) + '_directional.txt'), adj_dir_res)


class PartnetAdjacencyDataset(Dataset):
    def __init__(self, meta_constructor, graph_dir=None):
        self.meta_constructor = meta_constructor
        self.meta = self.meta_constructor.df
        self.bbox_dataset = PartnetBBoxDataset(self.meta_constructor)

        if graph_dir is None:
            self.graph_dir = cfg.graph_dir
        else:
            self.graph_dir = graph_dir

    @staticmethod
    def toposort(adjmat):
        closed = []
        adjmat = adjmat.copy()
        try:
            vertmap = np.arange(adjmat.shape[0])
        except Exception as exc:
            return [0]

        while vertmap.size > 0:
            indegree = np.sum(adjmat, axis=0)
            picked = np.random.choice(np.where(indegree == 0)[0])
            closed.append(vertmap[picked])
            vertmap = np.delete(vertmap, picked, axis=0)
            adjmat = np.delete(np.delete(adjmat, picked, axis=0), picked, axis=1)
        return closed

    def __getitem__(self, index):
        adjmat = np.loadtxt(os.path.join(self.graph_dir, '{}_directional.txt'.format(index)))
        with open(os.path.join(self.graph_dir, '{}_mapping.pkl'.format(index)), "rb") as stream:
            idmap = pickle.load(stream)
        rev_idmap = {v: k for k, v in idmap.items()}
        sortee = self.toposort(adjmat)
        res = [self.bbox_dataset[rev_idmap[idx]] for idx in sortee]
        return res

    def __len__(self):
        return len(self.meta)


if __name__ == '__main__':
    m = PartnetMetaConstructor(cfg.partnet)
    m.construct_meta()
    # a = PartnetAdjacencyConstructor(m)
    # a.construct_adj_graph(use_cache=False)
    d = PartnetAdjacencyDataset(m)
    for idx in range(len(d)):
        print([get_bbox_extent(bbox) for bbox in d[idx]])
