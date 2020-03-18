import cv2 as cv
import numpy as np

import utils


def map_robustness(robustness):
    return 0.16 * robustness + 8


def embed_freq_domain(luma, watermark, delta):
    height, width = luma.shape
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            watermark_blk = watermark[i >> 2: (i >> 2) + 2, j >> 1: (j >> 1) + 4].flatten()
            for k in range(8):
                loc_i, loc_j = (k, 7 - k) if k < 4 else (k - 4, 10 - k)
                depth = np.abs(luma[i + loc_i][j + loc_j] - luma[i + loc_j][j + loc_i]) + delta
                if watermark_blk[k]:
                    if luma[i + loc_i][j + loc_j] <= luma[i + loc_j][j + loc_i]:
                        luma[i + loc_i][j + loc_j] = luma[i + loc_j][j + loc_i] + depth
                else:
                    if luma[i + loc_i][j + loc_j] >= luma[i + loc_j][j + loc_i]:
                        luma[i + loc_j][j + loc_i] = luma[i + loc_j][j + loc_i] + depth
    return luma


def embed_watermark(host, watermark, robustness):
    old_shape = host.shape[0: 2]
    host = utils.pad(host)
    height, width = host.shape[0: 2]
    host = cv.cvtColor(host, cv.COLOR_BGR2YCrCb)
    luma = host[:, :, 0]
    luma = luma.astype(np.float)
    watermark = cv.resize(watermark, (width >> 1, height >> 2))
    watermark = cv.adaptiveThreshold(watermark, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    watermark, pxl_perm_mat = utils.pseudo_random_permute(watermark)
    luma = utils.dct(luma)
    luma = embed_freq_domain(luma, watermark, robustness)
    luma = utils.idct(luma)
    host[:, :, 0] = np.clip(luma, 0, 255)
    host = cv.cvtColor(host, cv.COLOR_YCrCb2BGR)
    host = utils.cut(host, old_shape)
    return host, pxl_perm_mat


def embed(host, watermark, robustness):
    robustness = map_robustness(robustness)
    host, pxl_perm_mat = embed_watermark(host, watermark, robustness)
    key = {
        'shape': watermark.shape,
        'pxl_perm_mat': pxl_perm_mat
    }
    return host, key


def extract_freq_domain(luma):
    height, width = luma.shape
    watermark = np.zeros((height >> 2, width >> 1))
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            watermark_blk = np.zeros(8)
            for k in range(8):
                loc_i, loc_j = (k, 7 - k) if k < 4 else (k - 4, 10 - k)
                watermark_blk[k] = 255 if luma[i + loc_i][j + loc_j] > luma[i + loc_j][j + loc_i] else 0
            watermark[i >> 2: (i >> 2) + 2, j >> 1: (j >> 1) + 4] = watermark_blk.reshape((2, 4))
    return watermark


def extract_watermark(host, pxl_perm_mat):
    host = utils.pad(host)
    host = cv.cvtColor(host, cv.COLOR_BGR2YCrCb)
    luma = host[:, :, 0]
    luma = luma.astype(np.float)
    luma = utils.dct(luma)
    watermark = extract_freq_domain(luma)
    pxl_perm_mat = utils.inv_perm_mat(pxl_perm_mat)
    watermark = utils.permute_pxl(watermark, pxl_perm_mat)
    return watermark


def extract(host, key):
    height, width = key['shape']
    pxl_perm_mat = key['pxl_perm_mat']
    watermark = extract_watermark(host, pxl_perm_mat)
    watermark = watermark.astype(np.uint8)
    watermark = cv.medianBlur(watermark, 5)
    watermark = cv.resize(watermark, (width, height))
    return watermark
