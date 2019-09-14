import sys
import os
from contextlib import contextmanager
import numpy as np


def print_args(args):
    opts = vars(args)
    print("======= Arguments ========")
    for k, v in sorted(opts.items()):
        print("{:<20}- {}".format(k, v))
    print("==========================")


def mute(func):
    def mute_wrapper(*args, **kwargs):
        sys.stdout = open(os.devnull, 'w')
        res = func(*args, **kwargs)
        sys.stdout = sys.__stdout__
        return res

    return mute_wrapper


@contextmanager
def stdout_redirected(to=os.devnull):
    """
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    """
    fd = sys.stdout.fileno()

    # assert that Python and C stdio write using the same file descriptor
    # assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w')  # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different L


def mute_c(func):
    def mute_wrapper(*args, **kwargs):
        with stdout_redirected():
            res = func(*args, **kwargs)
        return res

    return mute_wrapper


def sample_sphere(num_points, radius=1):
    theta_vec = np.random.rand(num_points) * 2 * np.pi
    phi_vec = np.random.rand(num_points) * np.pi

    z = radius * np.cos(theta_vec)
    sin_theta_vec = np.sin(theta_vec)
    x = radius * np.multiply(sin_theta_vec, np.cos(phi_vec))
    y = radius * np.multiply(sin_theta_vec, np.sin(phi_vec))
    res = np.column_stack((x, y, z)).astype(np.float32)
    return res


if __name__ == '__main__':
    sample_sphere(2048, 1)
