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
import threading
from queue import Queue
import multiprocessing as mp
from tqdm import tqdm


class PartnetDataset(Dataset):
    def __init__(self, path=None, cat=None, cache_maxsize=None,
                 traverse="instance", return_mode="vertex"):
        self.meta_constructor = PartnetMetaConstructor(path)
        self.meta_constructor.construct_meta()
        self.meta = self.meta_constructor.df
        self.parts = self.meta_constructor.parts
        self.part_parent_child = self.meta_constructor.part_parent_child
        self.part_sibling = self.meta_constructor.part_sibling
        self.part_leaf = self.meta_constructor.part_leaf
        self.length = len(self.meta)
        self.length_part = len(self.parts)
        self.cat = cat
        self.traverse = traverse
        self.return_mode = return_mode
        self.cache = {}
        self.cache_maxsize = cache_maxsize

        self.reload_category(self.cat)

    def reload_category(self, cat=None):
        # self.meta_in_action is a view of self.meta
        if cat is not None:
            if isinstance(cat, str):
                self.meta_in_action = self.meta[self.meta['cat'] == cat]
            else:
                self.meta_in_action = self.meta[self.meta['cat'].isin(cat)]
            self.parts_in_action = self._get_parts_from_cat(cat)
            self.leafs_in_action = self._get_leafs_from_cat(cat)
        else:
            self.meta_in_action = self.meta
            self.parts_in_action = self.parts
            self.leafs_in_action = self._get_leafs_from_cat(None)
        self.reload_traverse(self.traverse)

    def reload_traverse(self, traverse):
        self.traverse = traverse
        if self.traverse == 'instance':
            self.length_in_action = len(self.meta_in_action)
        elif self.traverse == 'part':
            self.length_in_action = len(self.parts_in_action)
        else:
            self.length_in_action = len(self.leafs_in_action)
        self.cache = {}

    def _get_part_id_from_item_id(self, item_id):
        return list(self.parts[self.parts['item_id'] == item_id]['global_id'])

    def _get_parts_from_cat(self, cat):
        if isinstance(cat, str):
            item_ids = list(self.meta[self.meta['cat'] == cat].index)
        else:
            item_ids = list(self.meta[self.meta['cat'].isin(cat)].index)
        return self.parts[self.parts['item_id'].isin(item_ids)]

    def _get_leafs_from_cat(self, cat):
        if cat is None:
            return self.parts[self.parts['global_id'].isin(self.part_leaf['leaf_global_id'])]
        else:
            if isinstance(cat, str):
                item_ids = list(self.meta[self.meta['cat'] == cat].index)
            else:
                item_ids = list(self.meta[self.meta['cat'].isin(cat)].index)
            global_ids = self.parts[self.parts['item_id'].isin(item_ids)]['global_id']
            leaf_global_ids = self.part_leaf[self.part_leaf['leaf_global_id'].isin(global_ids)]['leaf_global_id']
            return self.parts[self.parts['global_id'].isin(leaf_global_ids)]

    def _getitem_instance(self, index):
        if index in self.cache:
            # print("!!!CACHED")
            mesh = self.cache[index]
        else:
            path = self.meta_in_action.iloc[index]['model_path']
            mesh = load_pc(path)
            if len(self.cache) == self.cache_maxsize:
                pop_key = random.choice(list(self.cache.keys()))
                self.cache.pop(pop_key)
                # print("pop", pop_key)
            self.cache[index] = mesh
            # print("ins", index)

        if self.return_mode == "vertex":
            res = get_pc(mesh)
        else:
            res = mesh
        return res

    def _getitem_part(self, index):
        if index in self.cache:
            mesh = self.cache[index]
        else:
            path = self.parts_in_action.iloc[index]['objs_dir']
            objs = self.parts_in_action.iloc[index]['objs']
            mesh_list = []
            for obj in eval(objs):
                obj_path = os.path.join(path, obj + '.obj')
                mesh_tmp = load_pc(obj_path)
                mesh_tmp, info = pymesh.remove_isolated_vertices(mesh_tmp)
                mesh_list.append(mesh_tmp)

            mesh = pymesh.merge_meshes(mesh_list)
            if len(self.cache) == self.cache_maxsize:
                pop_key = random.choice(list(self.cache.keys()))
                self.cache.pop(pop_key)
            self.cache[index] = mesh

        if self.return_mode == "vertex":
            res = get_pc(mesh)
        else:
            res = mesh
        return res

    def _getitem_leaf(self, index):
        if index in self.cache:
            mesh = self.cache[index]
        else:
            path = self.leafs_in_action.iloc[index]['objs_dir']
            objs = self.leafs_in_action.iloc[index]['objs']
            mesh_list = []
            for obj in eval(objs):
                obj_path = os.path.join(path, obj + '.obj')
                mesh_tmp = load_pc(obj_path)
                mesh_tmp, info = pymesh.remove_isolated_vertices(mesh_tmp)
                mesh_list.append(mesh_tmp)
            mesh = pymesh.merge_meshes(mesh_list)
            if len(self.cache) == self.cache_maxsize:
                pop_key = random.choice(list(self.cache.keys()))
                self.cache.pop(pop_key)
            self.cache[index] = mesh

        if self.return_mode == "vertex":
            res = get_pc(mesh)
        else:
            res = mesh
        return res

    def __getitem__(self, index):
        if self.traverse == 'instance':
            return self._getitem_instance(index)
        elif self.traverse == 'part':
            return self._getitem_part(index)
        else:
            return self._getitem_leaf(index)

    def __len__(self):
        return self.length_in_action


class PartnetDataLoader(threading.Thread):
    def __init__(self, dataset, batch_size=32, max_epoch=200, queue_maxsize=500, aligned=True,
                 preprocess_callback_list=(identity,)):
        super(PartnetDataLoader, self).__init__()
        self.dataset = dataset
        self.dataset_len = len(self.dataset)
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.num_batches = self.dataset_len // self.batch_size
        self.aligned = aligned
        self.preprocess_callback_list = list(preprocess_callback_list)

        self.queue = Queue(maxsize=queue_maxsize)
        self.stopped = False

    def run(self):
        while not self.stopped:
            self.dataset_len = len(self.dataset)
            index_mapping = np.arange(self.dataset_len)
            np.random.shuffle(index_mapping)
            self.num_batches = self.dataset_len // self.batch_size

            for batch_idx in range(self.num_batches):
                if self.stopped:
                    return None

                start_idx = batch_idx * self.batch_size
                end_idx = (batch_idx + 1) * self.batch_size

                pc_list = []
                for i in range(start_idx, end_idx):
                    pc_list.append(self.dataset[i])
                # vanilla = self.concat_pc(pc_list)
                # self.queue.put(vanilla)

                for ppf in self.preprocess_callback_list:
                    mapped = list(map(ppf, pc_list))
                    if self.aligned:
                        res_mapped = self.concat_pc(mapped)
                    elif self.batch_size == 1:
                        res_mapped = mapped[0]
                    self.queue.put(res_mapped)

    def fetch(self):
        if self.stopped:
            return None
        return self.queue.get()

    def shutdown(self):
        self.stopped = True
        print(">>> Shutting Down Dataloader")
        while not self.queue.empty():
            self.queue.get()
        print("=== Shut Down Dataloader: All Queued Data Cleared")

    @staticmethod
    def concat_pc(pc_list):
        res = np.stack(pc_list)
        return res

    def _benchmark_compare(self, batch_idx):
        start_idx = batch_idx * self.batch_size
        end_idx = (batch_idx + 1) * self.batch_size

        pc_list = []
        for i in range(start_idx, end_idx):
            pc_list.append(self.dataset[i])

        ppf = self.preprocess_callback_list[0]
        mapped = list(map(ppf, pc_list))
        res_mapped = self.concat_pc(mapped)

        return res_mapped


if __name__ == '__main__':
    partnet_dataset = PartnetDataset()
    # for i in tqdm(range(len(partnet_dataset))):
    #     a = partnet_dataset[i]

    lder = PartnetDataLoader(partnet_dataset, preprocess_callback_list=[sample_choice_points_factory(2048, )])
    print(len(partnet_dataset))
    partnet_dataset.reload_category('Knife')
    print(len(partnet_dataset))
    partnet_dataset.reload_traverse('part')
    print(len(partnet_dataset))
    partnet_dataset.reload_traverse('leaf')
    print(len(partnet_dataset))
    partnet_dataset.return_mode = 'else'
