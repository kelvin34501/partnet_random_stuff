import numpy as np
# import open3d as o3d
from util import stdout_redirected
import pymesh


def load_pc(path):
    # with stdout_redirected():
    #     res = o3d.io.read_point_cloud(path)
    res = pymesh.load_mesh(path)
    return res


def dump_pc(path, pc):
    # with stdout_redirected():
    #     o3d.io.write_point_cloud(path, pc)
    pymesh.save_mesh(path, pc)


def get_pc(pc):
    return pc.vertices


class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


def gen_mask(length, percent=0.6):
    all = np.arange(length)
    thresh = int(np.ceil(length * percent))
    np.random.shuffle(all)
    return all[:thresh], all[thresh:]


class DatasetDispatcher(Dataset):
    def __init__(self, totalset, mask):
        self.mapping = dict(enumerate(mask))
        self.totalset = totalset
        self.length = len(self.mapping)

    def __getitem__(self, index):
        return self.totalset[self.mapping[index]]

    def __len__(self):
        return self.length
