import cv2 as cv
import numpy as np


class SimilarImageException(Exception):
    pass


class UnmatchedKeyException(Exception):
    pass


def pad(image):
    height, width = image.shape[0: 2]
    return np.pad(image, ((0, 8 - height & 7), (0, 8 - width & 7), (0, 0)), 'constant')


def cut(image, shape):
    height, width = shape
    return image[0: height, 0: width]


def pseudo_random_perm_mat(shape):
    perm_mat = np.arange(shape[0] * shape[1]).reshape(shape)
    perm_mat = np.random.permutation(perm_mat)
    return perm_mat


def inv_perm_mat(perm_mat):
    flat_perm_mat = perm_mat.flatten()
    inv = np.zeros(flat_perm_mat.size)
    inv = inv.astype(np.int32)
    for i in range(flat_perm_mat.size):
        inv[flat_perm_mat[i]] = i
    inv = inv.reshape(perm_mat.shape)
    return inv


def permute_pxl(image, perm_mat):
    perm_watermark = np.zeros_like(image)
    height, width = image.shape
    for i in range(height):
        for j in range(width):
            perm = perm_mat[i][j]
            perm_watermark[divmod(perm, width)] = image[i][j]
    return perm_watermark


def pseudo_random_permute(image):
    pxl_perm_mat = pseudo_random_perm_mat(image.shape)
    image = permute_pxl(image, pxl_perm_mat)
    return image, pxl_perm_mat


def dct(channel):
    height, width = channel.shape[0: 2]
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            channel[i: i + 8, j: j + 8] = cv.dct(channel[i: i + 8, j: j + 8])
    return channel


def idct(channel):
    height, width = channel.shape[0: 2]
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            channel[i: i + 8, j: j + 8] = cv.idct(channel[i: i + 8, j: j + 8])
    return channel
