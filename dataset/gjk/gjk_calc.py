import ctypes
import numpy as np
import os

BASE_PATH = os.path.dirname(__file__)
lib = ctypes.cdll.LoadLibrary(os.path.join(BASE_PATH, 'gjk_wrapper.so'))
dist_squared_func = lib.polyhedron_intersect_polyhedron
dist_squared_func.restype = ctypes.c_double


def calc(bbox_a, bbox_b):
    res = dist_squared_func(
        ctypes.c_void_p(bbox_a.ctypes.data),
        ctypes.c_int(len(bbox_a)),
        ctypes.c_void_p(bbox_b.ctypes.data),
        ctypes.c_int(len(bbox_b))
    )
    return np.sqrt(res)


if __name__ == '__main__':
    b1 = np.array([[0, 0, 0],
                   [0, 0, 1],
                   [0, 1, 0],
                   [0, 1, 1],
                   [1, 0, 0],
                   [1, 0, 1],
                   [1, 1, 0],
                   [1, 1, 1]], dtype=np.double)
    b2 = np.array([[-1, -1, -1],
                   [-1, -1, 2],
                   [-1, 2, -1],
                   [-1, 2, 2],
                   [2, -1, -1],
                   [2, -1, 2],
                   [2, 2, -1],
                   [2, 2, 2]], dtype=np.double)
    calc(b1, b2)
