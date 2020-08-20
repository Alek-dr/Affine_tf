import tensorflow as tf
import numpy as np


def get_identical_coords_matrix(image: tf.Variable) -> tf.constant:
    """Возвращает координаты каждого пикселя, плюс единица в новой размеронсти"""
    size = image.shape[0] * image.shape[1]
    xvec = tf.linspace(0, image.shape[1] - 1, image.shape[1])
    xvec = tf.cast(xvec, tf.float32)
    yvec = tf.linspace(0, image.shape[0] - 1, image.shape[0])
    yvec = tf.cast(yvec, tf.float32)
    x, y = tf.meshgrid(xvec, yvec)
    x = tf.reshape(x, shape=size)
    y = tf.reshape(y, shape=size)
    ones = tf.ones(shape=size, dtype=tf.float32)
    M = tf.stack((x, y, ones), axis=0)
    return M


def one_row_coordinates(new_coords: tf.Variable) -> tf.Variable:
    """Возвращает координаты в виде [[y,x],[y,x]...[y,x]]"""
    x = tf.slice(new_coords, [0, 0], [1, -1])
    y = tf.slice(new_coords, [1, 0], [1, -1])
    new_coords = tf.stack((y, x), axis=2)
    new_coords = tf.reshape(new_coords, shape=(-1, 2))
    return new_coords


def integer_2d_coordinates(new_coords: tf.Variable) -> tf.Variable:
    """Удаляет лишнюю размерность, приводит к int32"""
    new_coords = tf.slice(new_coords, [0, 0], [2, -1])
    new_coords = tf.cast(new_coords, tf.int32)
    return new_coords


def _unique_coords(coords):
    _, index = np.unique(coords, axis=0, return_index=True)
    index = np.sort(index)
    coords = coords[index]
    return coords, index


@tf.function(input_signature=[tf.TensorSpec(None, tf.int32)])
def unique_coords(coords):
    return tf.numpy_function(_unique_coords, [coords], Tout=(tf.int32, tf.int64))


def _mask(index, n):
    mask = np.zeros(shape=(n), dtype=np.bool)
    mask[index] = True
    return mask


@tf.function(input_signature=[tf.TensorSpec(None, tf.int64), tf.TensorSpec(None, tf.int64)])
def full_mask(index, n):
    return tf.numpy_function(_mask, [index, n], Tout=tf.bool)


def three_channels(coords: tf.Variable, updates: tf.Variable, shape: tuple) -> tf.Variable:
    """"Получить новое изображение из трехканального"""
    ch1 = tf.slice(updates, [0, 0], [1, -1])
    ch1 = tf.squeeze(ch1, axis=0)
    ch2 = tf.slice(updates, [1, 0], [1, -1])
    ch2 = tf.squeeze(ch2, axis=0)
    ch3 = tf.slice(updates, [2, 0], [-1, -1])
    ch3 = tf.squeeze(ch3, axis=0)
    c1 = tf.scatter_nd(coords, ch1, shape)
    c2 = tf.scatter_nd(coords, ch2, shape)
    c3 = tf.scatter_nd(coords, ch3, shape)
    new_img = tf.stack((c1, c2, c3), axis=2)
    return new_img


def one_channel(coords: tf.Variable, updates: tf.Variable, shape: tuple) -> tf.Variable:
    """Получить новое изображение из одноканального"""
    updates = tf.squeeze(updates, axis=0)
    new_img = tf.scatter_nd(coords, updates, shape)
    return new_img
