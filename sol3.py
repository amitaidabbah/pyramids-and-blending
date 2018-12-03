import os
import numpy as np
from imageio import imread
from scipy.ndimage.filters import convolve1d
from skimage.color import rgb2grey
import matplotlib.pyplot as plt

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
        if image.ndim == 3:
            return rgb2grey(image)
        return image / 255
    elif representation == RGB:
        image = imread(filename)
        return image / 255
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
    """
    this function buils the gaussian filter vector.
    :param size: the size of the gaussian filter
    :return: the normalized gaussian vector of size size
    """
    gaus = np.array(BASE_FILTER).astype(np.uint64)
    for i in range(size - 2):
        gaus = np.convolve(gaus, BASE_FILTER)
    return gaus * (2 ** -(size - 1))


def reduce(im, filter):
    """
    reduce the image size by 2.
    :param im: image to reduce
    :param filter: size of blur filter to use
    :return: reduced image
    """
    im = convolve1d(im, filter, mode='constant')
    im = convolve1d(im.T, filter, mode='constant')
    return im.T[::2, ::2]


def expand(im, filter):
    """
    expand the image size by 2.
    :param im: image to expand
    :param filter: size of blur filter to use
    :return: expanded image
    """
    x, y = im.shape
    im = np.insert(im, np.arange(1, y + 1, 1), 0, axis=1)
    im = np.insert(im, np.arange(1, x + 1, 1), 0, axis=0)
    im = convolve1d(im, 2 * filter, mode='constant')
    im = convolve1d(im.T, 2 * filter, mode='constant')
    return im.T


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    this method builds a gaussian pyramid of an input grayscale image.
    :param im: input grayscale image.
    :param max_levels: maximum number of levels in the pyramid.
    :param filter_size: size of gaussian blur to use.
    :return: (the gaussian pyramid as a list, filter vector (1,filter_size))
    """
    gaussian_pyramid = list()
    gaussian_pyramid.append(im)
    filter = build_filter(filter_size)
    for i in range(max_levels-1):
        x, y = im.shape
        if x < 16 or y < 16:
            break
        im = reduce(im, filter)
        gaussian_pyramid.append(im)
    return gaussian_pyramid, filter.reshape((1, filter_size))


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    this method builds a laplacia pyramid of an input grayscale image.
    :param im: input grayscale image.
    :param max_levels: maximum number of levels in the pyramid.
    :param filter_size: size of gaussian blur to use.
    :return: (the laplacian pyramid as a list, filter vector (1,filter_size))
    """
    gaussian_pyramid, filter = build_gaussian_pyramid(im, max_levels, filter_size)
    laplacian_pyramid = list()
    for i in range(len(gaussian_pyramid) - 1):
        laplacian = gaussian_pyramid[i] - expand(gaussian_pyramid[i + 1], filter.reshape((filter_size,)))
        laplacian_pyramid.append(laplacian)
    laplacian_pyramid.append(gaussian_pyramid[-1])
    return laplacian_pyramid, filter


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    this function constructs laplacian pyramid back to an image.
    :param lpyr: the laplacian pyramid
    :param filter_vec: the vector returned by the function to create the pyramid.
    :param coeff: a list of coefficients specifying weight of each level of the pyramid
    :return: reconstructed image.
    """
    x, y = filter_vec.shape
    while len(lpyr) > 1:
        lpyr[-2] = (lpyr[-2] + coeff[-1] * expand(lpyr[-1], filter_vec.reshape((y,))))
        lpyr = lpyr[:-1]
        coeff = coeff[:-1]
    return lpyr[0]


def render_pyramid(pyr, levels):
    """
    creates an image with all the levels of the pyramid size by side.
    :param pyr:the pyramid to render
    :param levels:number of levels to render
    :return:
    """
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
    """
    displays the pyramid on screen.
    :param pyr: pyramid to display
    :param levels: num of levels to display
    :return:
    """
    plt.imshow(render_pyramid(pyr, levels), cmap="grey")
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    this method blends 2 different GRAYSACLE images using a given mask image.
    :param im1: first image to blend
    :param im2: second image to blend
    :param mask: mask to use
    :param max_levels: maximum number of levels when building the pyramid
    :param filter_size_im: filter size to use on images when building the pyrmaid
    :param filter_size_mask: filter size to use on mask when building the pyrmaid
    :return: the blended rendered image
    """
    lap1, filter1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    lap2, filter2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    gaus, gaus_filter = build_gaussian_pyramid(mask.astype(np.float64), max_levels, filter_size_mask)
    lap_out = []
    for i in range(len(lap1)):
        lap_out.append(gaus[i] * lap1[i] + (1 - gaus[i]) * lap2[i])
    coef = [1 for i in range(len(lap_out))]
    return laplacian_to_image(lap_out, filter1, coef)


def pyramid_blend_RGB(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    this method blends 2 different RGB images using a given mask image.
    :param im1: first image to blend
    :param im2: second image to blend
    :param mask: mask to use
    :param max_levels: maximum number of levels when building the pyramid
    :param filter_size_im: filter size to use on images when building the pyrmaid
    :param filter_size_mask: filter size to use on mask when building the pyrmaid
    :return: the blended rendered image
    """
    res_R = pyramid_blending(im1[:, :, 0], im2[:, :, 0], mask, max_levels, filter_size_im, filter_size_mask)
    res_G = pyramid_blending(im1[:, :, 1], im2[:, :, 1], mask, max_levels, filter_size_im, filter_size_mask)
    res_B = pyramid_blending(im1[:, :, 2], im2[:, :, 2], mask, max_levels, filter_size_im, filter_size_mask)
    res = np.stack((res_R, res_G, res_B), axis=-1)
    return res


def combine_plot(im1, im2, mask, im_blend):
    """
    polt all images in the same figure
    :param im1: 1st RGB image
    :param im2: 2nd RGB image
    :param mask: grayscale image.
    :param im_blend: result RGB image.
    """
    plt.figure(facecolor='black')
    plt.subplot(2, 2, 1)
    plt.imshow(im1)
    plt.subplot(2, 2, 2)
    plt.imshow(im2)
    plt.subplot(2, 2, 3)
    plt.imshow(mask, cmap='gray')
    plt.subplot(2, 2, 4)
    plt.imshow(im_blend)
    plt.show()


def path(filename):
    """
    changes filename to relative path
    :param filename: file name to change
    :return: relative path
    """
    return os.path.join(os.path.dirname(__file__), filename)


def blending_example1():
    """
    this performs an example image blending
    :return: image 1, image 2, mask as bool , result
    """
    eye = read_image(path('eye.jpg'), 2)
    moon = read_image(path('moon.jpg'), 2)
    mask = read_image(path('moonmaskinv.jpg'), 1)
    res = pyramid_blend_RGB(eye, moon, mask, 10, 3, 3)
    combine_plot(eye, moon, mask, np.clip(res, 0, 1))
    return eye, moon, mask.astype(np.bool), res


def blending_example2():
    """
    this performs an example image blending
    :return: image 1, image 2, mask as bool , result
    """
    model = read_image(path("model.jpg"), 2)
    dolphin = read_image(path("dolphins.jpg"), 2)
    mask = read_image(path("dolphinsmaskinv.jpg"), 1)
    res = pyramid_blend_RGB(model, dolphin, mask, 10, 15, 3)
    combine_plot(model, dolphin, mask, np.clip(res, 0, 1))
    return model, dolphin, mask.astype(np.bool), res

if __name__ == '__main__':
    monkey = read_image("monkey.jpg",1)
    gpyr, filter_vec = build_gaussian_pyramid(monkey, 3, 3)
    print(len(gpyr))