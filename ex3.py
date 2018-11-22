import imageio
import numpy as np
from imageio import imread
from scipy.ndimage.filters import convolve1d
from skimage.color import rgb2grey
import matplotlib.pyplot as plt
import scipy.misc

BASE_FILTER = np.array([1, 1])
RGB = 2
GRAYSCALE = 1
FACTOR = 256


def read_image(filename, representation):
    """
    this function read an image.
    :param filename: image name or path to read.
    :param representation: 1 for RGB, 2 for GRAYSCALE
    :return: an image in float64 format.
    """
    if representation == GRAYSCALE:
        image = imread(filename)
        return rgb2grey(image)
    elif representation == RGB:
        image = imread(filename)
        return image / FACTOR
    else:
        exit()


def imdisplay(image, representation):
    """
    this function displays an image.
    :param filename: the name of the file or path.
    :param representation: 1 for rgb image, 2 for GRAYSCALE.
    :return:
    """
    if representation == GRAYSCALE:
        plt.imshow(image, cmap='gray')
    elif representation == RGB:
        plt.imshow(image)
    plt.show()


def build_filter(size):
    gaus = np.array(BASE_FILTER).astype(np.uint64)
    for i in range(size - 2):
        gaus = np.convolve(gaus, BASE_FILTER)
    return gaus * (2 ** -(size - 1))


def reduce(im, filter):
    """
    reduces size by 2
    :param im:
    :return:
    """
    im = convolve1d(im, filter, mode='constant')
    im = convolve1d(im.T, filter, mode='constant')
    return im.T[::2, ::2]


def expand(im, filter):
    x, y = im.shape
    im = np.insert(im, np.arange(1, y + 1, 1), 0, axis=1)
    im = np.insert(im, np.arange(1, x + 1, 1), 0, axis=0)
    im = convolve1d(im, 2 * filter, mode='constant')
    im = convolve1d(im.T, 2 * filter, mode='constant')
    return im.T


def build_gaussian_pyramid(im, max_levels, filter_size):
    gaussian_pyramid = list()
    gaussian_pyramid.append(im)
    filter = build_filter(filter_size)
    for i in range(max_levels):
        x, y = im.shape
        if x < 16 or y < 16:
            break
        im = reduce(im, filter)
        gaussian_pyramid.append(im)
    return gaussian_pyramid, filter.reshape((1, filter_size))


def build_laplacian_pyramid(im, max_levels, filter_size):
    gaussian_pyramid, filter = build_gaussian_pyramid(im, max_levels, filter_size)
    laplacian_pyramid = list()
    for i in range(len(gaussian_pyramid) - 1):
        laplacian = gaussian_pyramid[i] - expand(gaussian_pyramid[i + 1], filter.reshape((filter_size,)))
        laplacian_pyramid.append(laplacian)
    laplacian_pyramid.append(gaussian_pyramid[-1])
    return laplacian_pyramid, filter


def laplacian_to_image(lpyr, filter_vec, coeff):
    x, y = filter_vec.shape
    while len(lpyr) > 1:
        lpyr[-2] = (lpyr[-2] + coeff[-1] * expand(lpyr[-1], filter_vec.reshape((y,))))
        lpyr = lpyr[:-1]
        coeff = coeff[:-1]
    return lpyr[0]


def render_pyramid(pyr, levels):
    width = sum([pyr[i].shape[1] for i in range(levels)])
    new_image = np.zeros((pyr[0].shape[0], width))
    old_y = 0
    for i in range(levels):
        new_image[0:pyr[i].shape[0], old_y:old_y + pyr[i].shape[1]] = np.interp(pyr[i],
                                                                                (np.min(pyr[i]), np.max(pyr[i])),
                                                                                (0, 1))
        old_y += pyr[i].shape[1]
    return new_image


def display_pyramid(pyr, levels):
    plt.imshow(render_pyramid(pyr, levels), cmap="grey")
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    lap1, filter1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    lap2, filter2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    gaus, gaus_filter = build_gaussian_pyramid(mask.astype(np.float64), max_levels, filter_size_mask)
    lap_out = []
    for i in range(len(lap1)):
        lap_out.append(gaus[i] * lap1[i] + (1 - gaus[i]) * lap2[i])
    coef = [1 for i in range(len(lap_out))]
    return laplacian_to_image(lap_out, filter1, coef)


if __name__ == '__main__':
    apple = read_image("apple.jpg", 1)
    pear = read_image("pear.jpg", 1)
    plt.imshow(apple, cmap="gray")
    plt.show()
    plt.imshow(pear, cmap="gray")
    plt.show()
    mask = np.zeros_like(pear)
    mask[:, 150:] = 1
    print(mask)
    print(apple.shape, pear.shape, mask.shape)
    res = pyramid_blending(apple, pear, mask, 10, 3, 3)
    plt.imshow(res, cmap="gray")
    plt.show()
