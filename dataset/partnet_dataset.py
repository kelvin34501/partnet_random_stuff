import os
import sys

BASE_PATH = os.path.dirname(__file__)
sys.path.append(BASE_PATH)
from dataset_util import Dataset, load_pc, get_pc
from partnet_config import cfg
from partnet_meta_constructor import PartnetMetaConstructor
from preprocess import *

import trimesh
import random
import numpy as np
import pandas as pd
import threading
from queue import Queue
import multiprocessing as mp
from tqdm import tqdm


class PartnetDataset(Dataset):
    def __init__(self, path=None, meta_cache=None, cat=None, cache_maxsize=None,
                 return_mode="vertex"):
        self.meta_constructor = PartnetMetaConstructor(path, meta_cache)
        self.meta_constructor.construct_meta()
        self.meta = self.meta_constructor.df
        self.length = len(self.meta)
        self.return_mode = return_mode
        self.cache = {}
        self.cache_maxsize = cache_maxsize

        # self.meta_in_action is a view of self.meta
        if cat is not None:
            if isinstance(cat, str):
                self.meta_in_action = self.meta[self.meta['cat'] == cat]
            else:
                self.meta_in_action = self.meta[self.meta['cat'].isin(cat)]
        else:
            self.meta_in_action = self.meta
        self.length_in_action = len(self.meta_in_action)

    def reload_category(self, cat=None):
        if cat is not None:
            if isinstance(cat, str):
                self.meta_in_action = self.meta[self.meta['cat'] == cat]
            else:
                self.meta_in_action = self.meta[self.meta['cat'].isin(cat)]
        else:
            self.meta_in_action = self.meta
        self.length_in_action = len(self.meta_in_action)

    def __getitem__(self, index):
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

    def __len__(self):
        return self.length_in_action


class PartnetDataLoader(threading.Thread):
    def __init__(self, dataset, batch_size=32, max_epoch=200, queue_maxsize=500,
                 preprocess_callback_list=(identity,)):
        super(PartnetDataLoader, self).__init__()
        self.dataset = dataset
        self.dataset_len = len(self.dataset)
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.num_batches = self.dataset_len // self.batch_size
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
                    res_mapped = self.concat_pc(mapped)
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

    lder = PartnetDataLoader(partnet_dataset, preprocess_callback_list=[sample_points_factory(2048, )])
    partnet_dataset.reload_category('Knife')

    lder.start()
    import time

    for i in range(10):
        start = time.time()
        ret = lder.fetch()
        # ret = lder.compare(i)
        end = time.time()
        print(ret.shape, end - start)

    lder.shutdown()
