import tensorflow as tf


def get_mask(val, max_val):
    val0 = tf.greater_equal(val, 0)
    val1 = tf.less(val, max_val)
    mask = tf.logical_and(val0, val1)
    return mask


def get_valid_mask(x, y, height, width):
    x_mask = get_mask(x, width)
    y_mask = get_mask(y, height)
    mask = tf.logical_and(x_mask, y_mask)
    return mask


def miss_values(new_coords, height, width):
    new_y = tf.slice(new_coords, [0, 0], [-1, 1])
    new_x = tf.slice(new_coords, [0, 1], [-1, 1])
    mask0 = get_valid_mask(new_x, new_y, height, width)
    mask1 = tf.identity(mask0)
    mask2d = tf.concat((mask0, mask1), axis=1)
    new_coords = tf.boolean_mask(new_coords, mask2d)
    new_coords = tf.reshape(new_coords, shape=(-1, 2))
    values = tf.ones(shape=(new_coords.shape[0]))
    mask = tf.scatter_nd(new_coords, values, shape=(height, width))
    coords = tf.where(mask == 0)
    return coords


def interpolate_values(image,
                       empty_coordinates,
                       left_up_coords,
                       right_up_coords,
                       left_down_coords,
                       right_down_coords):
    left_up_values = tf.gather_nd(image, left_up_coords)
    right_up_values = tf.gather_nd(image, right_up_coords)
    left_down_values = tf.gather_nd(image, left_down_coords)
    right_down_values = tf.gather_nd(image, right_down_coords)

    values = tf.math.add_n((left_up_values, right_up_values, left_down_values, right_down_values))
    values = tf.divide(values, 4)

    shape = (image.shape[0], image.shape[1])
    new_img = tf.scatter_nd(empty_coordinates, values, shape)
    return new_img


def bilinear_interpolation(image, new_coords):
    height, width = image.shape[1], image.shape[2]
    empty_coordinates = miss_values(new_coords, height, width)
    y = tf.slice(empty_coordinates, [0, 0], [-1, 1])
    x = tf.slice(empty_coordinates, [0, 1], [-1, 1])
    px0 = tf.squeeze(tf.subtract(x, 1), axis=1)
    py0 = tf.squeeze(tf.subtract(y, 1), axis=1)
    px1 = tf.squeeze(tf.add(x, 1), axis=1)
    py1 = tf.squeeze(tf.add(y, 1), axis=1)

    left_up_coords = tf.stack((py0, px0), axis=1)
    right_up_coords = tf.stack((py0, px1), axis=1)
    left_down_coords = tf.stack((py1, px0), axis=1)
    right_down_coords = tf.stack((py1, px1), axis=1)

    coords = (left_up_coords, right_up_coords, left_down_coords, right_down_coords)
    interp_values = tf.map_fn(fn=lambda t: interpolate_values(t, empty_coordinates, *coords), elems=image)
    return interp_values
