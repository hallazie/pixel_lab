# coding:utf-8

from PIL import Image

import numpy as np
import math


def chop_images():
    image = Image.open('../data/local/src_img.png')
    chunk_size = 2048
    width, height = image.size
    horz_step = math.ceil(width / chunk_size)
    vert_step = math.ceil(height / chunk_size)
    for i in range(horz_step):
        for j in range(vert_step):
            left = i * chunk_size
            right = min((i + 1) * chunk_size, width)
            up = j * chunk_size
            down = min((j + 1) * chunk_size, height)
            chunk = image.crop((left, up, right, down))
            chunk.save(f'../data/chunks/img_{i}_{j}.png')


def concat_images():
    width, height = 0, 0
    for i in range(3):
        width += Image.open(f'../data/chunks_gen/img_{i}_0.png').width
    for j in range(2):
        height += Image.open(f'../data/chunks_gen/img_0_{j}.png').height
    print(f'target image size: {width} x {height}')
    canvas = Image.fromarray(np.zeros((height, width, 3)).astype(np.uint8))
    chunk_size = 2048 * 3
    for i in range(3):
        for j in range(2):
            chunk = Image.open(f'../data/chunks_gen/img_{i}_{j}.png')
            Image.Image.paste(canvas, chunk, (i*chunk_size, j*chunk_size))
    canvas.save('../data/tar_img.png')


def crop_image():
    image = Image.open('../data/local/tar_img.png')
    image = image.crop((1000, 0, 15000, 9000))
    image.save('../data/crp_img.png')


if __name__ == '__main__':
    trim_video()
