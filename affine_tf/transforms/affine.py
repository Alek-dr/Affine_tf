from typing import Tuple
from math import radians

import tensorflow as tf
from tensorflow import cos, sin
from affine_tf.resample import bilinear_interpolation

from .utils import get_identical_coords_matrix, integer_2d_coordinates, unique_coords, full_mask, \
    one_row_coordinates, three_channels, one_channel


def rotate(image: tf.Variable, angle: float) -> Tuple[tf.Variable, tf.Variable]:
    """Поворачивает изображение на угол angle (в градуах)"""
    if len(image.shape) != 3:
        raise ValueError(f"Expected image dims: height x width x channels. Image shape: {image.shape}")
    height, width, channels = image.shape
    original_coordinates = get_identical_coords_matrix(image)
    dy = tf.divide(height, 2)
    dx = tf.divide(width, 2)
    # Сдвинуть начало координат в центр изображения
    T = tf.Variable([[1, 0, -dx],
                     [0, 1, -dy],
                     [0, 0, 1]], dtype=tf.float32)
    new_coords = tf.matmul(T, original_coordinates)
    angle = radians(angle)
    # Повернуть на угол
    R = tf.Variable([[cos(angle), -sin(angle), 0],
                     [sin(angle), cos(angle), 0],
                     [0, 0, 1]], dtype=tf.float32)
    new_coords = tf.matmul(R, new_coords)
    # Вернуть координаты обратно
    T = tf.Variable([[1, 0, dx],
                     [0, 1, dy],
                     [0, 0, 1]], dtype=tf.float32)
    new_coords = tf.matmul(T, new_coords)

    new_coords = integer_2d_coordinates(new_coords)
    new_coords = one_row_coordinates(new_coords)
    new_coords, index = unique_coords(new_coords)

    mask = full_mask(index, height * width)

    updates = tf.transpose(image, perm=[2, 0, 1])
    updates = tf.map_fn(fn=lambda t: tf.reshape(t, shape=(height * width)), elems=updates)
    updates = tf.map_fn(fn=lambda t: tf.boolean_mask(t, mask), elems=updates)

    shape = (height, width)
    new_img = tf.cond(tf.equal(channels, 3),
                      lambda: three_channels(new_coords, updates, shape),
                      lambda: one_channel(new_coords, updates, shape))

    new_img = tf.cond(tf.equal(channels, 3), lambda: new_img, lambda: tf.expand_dims(new_img, axis=2))

    new_img = tf.transpose(new_img, perm=[2, 0, 1])
    miss_values = bilinear_interpolation(new_img, new_coords)
    new_img = tf.add(new_img, miss_values)
    new_img = tf.transpose(new_img, perm=[1, 2, 0])
    return new_img


def translation(image: tf.Variable, dx: int, dy: int) -> tf.Variable:
    """Сдвигает изображение на dx и dy"""
    if len(image.shape) != 3:
        raise ValueError(f"Expected image dims: height x width x channels. Image shape: {image.shape}")
    height, width, channels = image.shape
    original_coordinates = get_identical_coords_matrix(image)
    T = tf.Variable([[1, 0, dx],
                     [0, 1, dy],
                     [0, 0, 1]], dtype=tf.float32)
    new_coords = tf.matmul(T, original_coordinates)
    new_coords = integer_2d_coordinates(new_coords)
    new_coords = one_row_coordinates(new_coords)

    updates = tf.transpose(image, perm=[2, 0, 1])
    updates = tf.map_fn(fn=lambda t: tf.reshape(t, shape=(height * width)), elems=updates)
    shape = (height, width)
    new_img = tf.cond(tf.equal(channels, 3),
                      lambda: three_channels(new_coords, updates, shape),
                      lambda: one_channel(new_coords, updates, shape))
    return new_img


def warpAffine(image: tf.Variable, T: tf.Variable) -> tf.Variable:
    if len(image.shape) != 3:
        raise ValueError(f"Expected image dims: height x width x channels. Image shape: {image.shape}")
    height, width, channels = image.shape
    original_coordinates = get_identical_coords_matrix(image)
    new_coords = tf.matmul(T, original_coordinates)
    new_coords = integer_2d_coordinates(new_coords)
    new_coords = one_row_coordinates(new_coords)
    new_coords, index = unique_coords(new_coords)

    mask = full_mask(index, height * width)

    updates = tf.transpose(image, perm=[2, 0, 1])
    updates = tf.map_fn(fn=lambda t: tf.reshape(t, shape=(height * width)), elems=updates)
    updates = tf.map_fn(fn=lambda t: tf.boolean_mask(t, mask), elems=updates)

    shape = (height, width)
    new_img = tf.cond(tf.equal(channels, 3),
                      lambda: three_channels(new_coords, updates, shape),
                      lambda: one_channel(new_coords, updates, shape))

    new_img = tf.cond(tf.equal(channels, 3), lambda: new_img, lambda: tf.expand_dims(new_img, axis=2))

    new_img = tf.transpose(new_img, perm=[2, 0, 1])
    miss_values = bilinear_interpolation(new_img, new_coords)
    new_img = tf.add(new_img, miss_values)
    new_img = tf.transpose(new_img, perm=[1, 2, 0])
    return new_img
