import os
import sys
import json
import multiprocessing as mp
from itertools import zip_longest, count, repeat

import pandas as pd
from tqdm import tqdm

BASE_PATH = os.path.dirname(__file__)
sys.path.append(BASE_PATH)
from partnet_config import cfg


def grouper(iterable, n, padvalue=None):
    return zip_longest(*[iter(iterable)] * n, fillvalue=padvalue)


class PartnetMetaConstructor():
    def __init__(self, path=None, cache=None):
        if path is None:
            self.path = cfg.partnet
        else:
            self.path = path
        self.cache_file = cache
        self.df = pd.DataFrame(columns=cfg.columns)

    @staticmethod
    def bfs(tree):
        queue = [*tree]
        while len(queue) > 0:
            node = queue.pop(0)
            print(node['name'])
            if 'children' in node:
                for child in node['children']:
                    queue.append(child)

    @staticmethod
    def dfs(tree):
        stack = [*zip(repeat(0), tree[::-1])]
        while len(stack) > 0:
            depth, node = stack.pop(-1)
            print('  ' * depth, node['name'])
            if 'children' in node:
                for child in node['children']:
                    stack.append((depth + 1, child))

    def _parse_part_tree(self, item_id, dirname):
        item_path = os.path.join(self.path, dirname)
        result_path = os.path.join(item_path, 'result.json')
        result_merged_path = os.path.join(item_path, 'result_after_merging.json')
        with open(result_path) as stream:
            tree = json.load(stream)
        self.dfs(tree)
        print("====")
        with open(result_merged_path) as stream:
            tree = json.load(stream)
        self.dfs(tree)
        sys.exit(2)

    def _parse_meta(self, item_id, dirname):
        item_path = os.path.join(self.path, dirname)
        meta_path = os.path.join(item_path, "meta.json")
        pointcloud_path = os.path.join(item_path, 'point_sample', 'sample-points-all-pts-nor-rgba-10000.ply')
        with open(meta_path, "r") as stream:
            desc = json.load(stream)
            self.df.loc[item_id] = {'anno_id': desc['anno_id'],
                                    'model_id': desc['model_id'],
                                    'cat': desc['model_cat'],
                                    'model_path': pointcloud_path}
        self._parse_part_tree(item_id, dirname)

    def _construct_meta(self):
        for i, dirname in enumerate(tqdm(os.listdir(self.path))):
            self._parse_meta(i, dirname)

    def construct_meta(self, use_cache=True):
        print(">>> Start Constructing Meta")
        if self.cache_file is None:
            self.cache_file = os.path.join(BASE_PATH, 'cache.csv')

        if use_cache and os.path.exists(self.cache_file):
            print(">>> Using Cached Meta")
            self.df = pd.read_csv(self.cache_file, usecols=cfg.columns)
        else:
            self._construct_meta()
            self.df.to_csv(self.cache_file)
        print("=== Completed Constructed Meta")


if __name__ == '__main__':
    print(pd.__version__)
    from partnet_config import cfg

    m = PartnetMetaConstructor(cfg.partnet)
    m.construct_meta(use_cache=False)
    print(m.df.loc[:, 'cat'].unique())
