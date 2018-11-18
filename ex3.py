import numpy as np
from scipy.ndimage.filters import convolve

BASE_FILTER = np.array([1, 1])


def build_gaussian(size):
    gaus = np.array(BASE_FILTER).astype(np.uint64)
    for i in range(size - 2):
        gaus = np.convolve(gaus, BASE_FILTER)
    return gaus * (2 ** -(size - 1))


def build_gaussian_pyramid(im, max_levels, filter_size):
    pass


if __name__ == '__main__':
    # b = np.arange(0, 100)
    # print(b)
    # c = b.reshape([10, 10])
    # print(c[::2, ::2])
