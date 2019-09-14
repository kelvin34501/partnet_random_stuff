import numpy as np


def identity(x):
    return x


def sample_points_factory(num_points):
    def _sample_points(x):
        total_points = x.shape[0]
        mask = np.arange(total_points)
        np.random.shuffle(mask)
        mask = mask[:num_points]
        return x[mask, :]

    return _sample_points
