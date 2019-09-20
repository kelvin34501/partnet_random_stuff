from dataset.partnet_dataset import PartnetDataset, PartnetDataLoader
from dataset.preprocess import identity

from config import cfg
from dataset.mesh_util import get_bbox, draw_boxes3d

DATASET = PartnetDataset(path=cfg.PARTNET, cat=cfg.CAT)  # cache_maxsize=None -> no pop
DATASET.reload_traverse('leaf')
DATASET.return_mode = 'mesh'
DATALOADER = PartnetDataLoader(DATASET,
                               batch_size=cfg.BATCH_SIZE, max_epoch=cfg.MAX_EPOCH, aligned=False,
                               preprocess_callback_list=(identity,))

import numpy as np

if __name__ == '__main__':
    import time

    DATALOADER.start()
    for i in range(100):
        start = time.time()
        a = DATALOADER.fetch()
        end = time.time()
        print(type(a), end - start)
        bbox, _, _ = get_bbox(a)
        flag = True
        if flag:
            from mayavi import mlab

            mlab.triangular_mesh(a.vertices[:, 0], a.vertices[:, 1], a.vertices[:, 2], a.faces)
            draw_boxes3d(np.expand_dims(bbox, 0))
            mlab.show()
    DATALOADER.shutdown()
