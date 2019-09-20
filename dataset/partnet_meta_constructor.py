import os
import sys
import json
import multiprocessing as mp
from itertools import zip_longest, count, repeat

import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from itertools import permutations

BASE_PATH = os.path.dirname(__file__)
sys.path.append(BASE_PATH)
from partnet_config import cfg


def grouper(iterable, n, padvalue=None):
    return zip_longest(*[iter(iterable)] * n, fillvalue=padvalue)


class PartnetMetaConstructor():
    def __init__(self, path=None):
        self.cache_file = os.path.join(BASE_PATH, 'cache.csv')
        self.cache_parts = os.path.join(BASE_PATH, 'cache_parts.csv')
        self.cache_part_parent_child = os.path.join(BASE_PATH, 'cache_part_parent_child.csv')
        self.cache_part_sibling = os.path.join(BASE_PATH, 'cache_part_sibling.csv')
        self.cache_part_leaf = os.path.join(BASE_PATH, 'cache_part_leaf.csv')
        if path is None:
            self.path = cfg.partnet
        else:
            self.path = path
        self.df = None
        self.parts = None
        self.part_parent_child = None
        self.part_sibling = None
        self.part_leaf = None

    @staticmethod
    def _postfix(tree):
        if tree is None:
            return None

        res = []
        h_list = []
        if 'children' in tree:
            for child in tree['children']:
                obj_list, h = PartnetMetaConstructor._postfix(child)
                res += obj_list
                h_list.append(h)
            tree['objs'] = res
            tree['height'] = max(h_list) + 1
        else:
            res += tree['objs']
            tree['height'] = 1
        return res, tree['height']

    def _hier(self, tree, item_id, dirname):
        queue = [(0, tree)]
        visited = set()
        while len(queue) > 0:
            depth, node = queue.pop(0)
            if repr(node) not in visited:
                visited.add(repr(node))
                # print('  ' * depth, depth, node['height'], node['objs'])
                global_id = str(item_id) + '_' + str(node['id'])
                objs_path = os.path.join(self.path, dirname, 'objs')
                self.parts.append({
                    'global_id': global_id,
                    'item_id': item_id,
                    'part_relative_id': node['id'],
                    'name': node['name'],
                    'text': node['text'],
                    'objs_dir': objs_path,
                    'objs': node['objs'],
                    'depth': depth,
                    'height': node['height']
                })

                if 'children' in node:
                    for child in node['children']:
                        queue.append((depth + 1, child))
                        self.part_parent_child.append({
                            'parent_global_id': global_id,
                            'child_global_id': str(item_id) + '_' + str(child['id'])
                        })
                    if len(node['children']) > 1:
                        for fst, snd in permutations(node['children'], 2):
                            self.part_sibling.append({
                                'sibling_a_id': str(item_id) + '_' + str(fst['id']),
                                'sibling_b_id': str(item_id) + '_' + str(snd['id'])
                            })
                else:
                    self.part_leaf.append({
                        'leaf_global_id': global_id
                    })

    def _parse_part_tree(self, item_id, dirname):
        # print(item_id, dirname)
        item_path = os.path.join(self.path, dirname)
        result_path = os.path.join(item_path, 'result.json')
        with open(result_path) as stream:
            tree_raw = json.load(stream)
            assert len(tree_raw) == 1
            tree = deepcopy(tree_raw[0])
        self._postfix(tree)
        self._hier(tree, item_id, dirname)

    def _parse_meta(self, item_id, dirname):
        item_path = os.path.join(self.path, dirname)
        meta_path = os.path.join(item_path, "meta.json")
        pointcloud_path = os.path.join(item_path, 'point_sample', 'sample-points-all-pts-nor-rgba-10000.ply')
        with open(meta_path, "r") as stream:
            desc = json.load(stream)
            self.df.append({'anno_id': desc['anno_id'],
                            'model_id': desc['model_id'],
                            'cat': desc['model_cat'],
                            'model_path': pointcloud_path})
        self._parse_part_tree(item_id, dirname)
        return True

    def _construct_meta(self):
        counter = 0
        for i, dirname in enumerate(tqdm(os.listdir(self.path))):
            ret = self._parse_meta(i, dirname)
            if ret:
                counter += 1

    def construct_meta(self, use_cache=True):
        print(">>> Start Constructing Meta")
        if use_cache and os.path.exists(self.cache_file):
            print(">>> Using Cached Meta")
            self.df = pd.read_csv(self.cache_file, usecols=cfg.columns)
            self.parts = pd.read_csv(self.cache_parts, usecols=cfg.parts_columns)
            self.part_parent_child = pd.read_csv(self.cache_part_parent_child, usecols=cfg.part_parent_child_columns)
            self.part_sibling = pd.read_csv(self.cache_part_sibling, usecols=cfg.part_sibling_columns)
            self.part_leaf = pd.read_csv(self.cache_part_leaf, usecols=cfg.part_leaf_columns)
        else:
            self.df = []
            self.parts = []
            self.part_parent_child = []
            self.part_sibling = []
            self.part_leaf = []

            self._construct_meta()
            print(">>> Creating Pandas DataFrames")

            self.df = pd.DataFrame(self.df, columns=cfg.columns)
            self.parts = pd.DataFrame(self.parts, columns=cfg.parts_columns)
            self.part_parent_child = pd.DataFrame(self.part_parent_child, columns=cfg.part_parent_child_columns)
            self.part_sibling = pd.DataFrame(self.part_sibling, columns=cfg.part_sibling_columns)
            self.part_leaf = pd.DataFrame(self.part_leaf, columns=cfg.part_leaf_columns)

            print(">>> Caching Pandas DataFrames")
            self.df.to_csv(self.cache_file)
            self.parts.to_csv(self.cache_parts)
            self.part_parent_child.to_csv(self.cache_part_parent_child)
            self.part_sibling.to_csv(self.cache_part_sibling)
            self.part_leaf.to_csv(self.cache_part_leaf)
        print("=== Completed Constructed Meta")


if __name__ == '__main__':
    print(pd.__version__)
    from partnet_config import cfg

    m = PartnetMetaConstructor(cfg.partnet)
    m.construct_meta(use_cache=False)
    print(m.df.loc[:, 'cat'].unique())
