from dataset.partnet_dataset import PartnetDataset, PartnetDataLoader
from dataset.preprocess import sample_points_factory

from config import cfg

DATASET = PartnetDataset(path=cfg.PARTNET, cat=cfg.CAT)  # cache_maxsize=None -> no pop
DATALOADER = PartnetDataLoader(DATASET,
                               batch_size=cfg.BATCH_SIZE, max_epoch=cfg.MAX_EPOCH,
                               preprocess_callback_list=(sample_points_factory(cfg.SAMPLE_POINTS),))

if __name__ == '__main__':
    import time

    DATALOADER.start()
    for i in range(100):
        start = time.time()
        a = DATALOADER.fetch()
        end = time.time()
        print(a.shape, end - start)
    DATALOADER.shutdown()
