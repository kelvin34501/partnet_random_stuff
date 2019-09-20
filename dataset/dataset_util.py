import numpy as np


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
