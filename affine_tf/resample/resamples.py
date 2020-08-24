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
    coords = tf.cast(coords, dtype=tf.int32)
    return coords


def padding_mask(coordinates, height, width):
    c = tf.transpose(coordinates)
    y = tf.slice(c, [0, 0], [1, -1])
    x = tf.slice(c, [1, 0], [1, -1])
    xm0 = tf.equal(x, 0)
    xm1 = tf.equal(x, width + 1)
    xm = tf.logical_or(xm0, xm1)
    ym0 = tf.equal(y, 0)
    ym1 = tf.equal(y, height + 1)
    ym = tf.logical_or(ym0, ym1)
    mask = tf.logical_or(xm, ym)
    m0 = tf.reshape(mask, shape=(-1, 1))
    m1 = tf.identity(m0)
    mask = tf.concat((m0, m1), axis=1)
    coords = tf.boolean_mask(coordinates, mask)
    coords = tf.reshape(coords, shape=(-1, 2))
    values = tf.ones(shape=coords.shape[0], dtype=tf.uint8)
    pad_mask = tf.scatter_nd(coords, values, tf.constant([height + 2, width + 2]))
    pad_mask = tf.clip_by_value(pad_mask, 0, 1)
    return pad_mask


def fill_channel(channel,
                 empty_coordinates,
                 den,
                 height,
                 width,
                 p0, p1,
                 p2, p3,
                 p4, p5,
                 p6, p7):
    """
    Заполнить канал channel в значениях empty_coordinates
    """
    p0v = tf.gather_nd(channel, p0)
    p1v = tf.gather_nd(channel, p1)
    p2v = tf.gather_nd(channel, p2)
    p3v = tf.gather_nd(channel, p3)
    p4v = tf.gather_nd(channel, p4)
    p5v = tf.gather_nd(channel, p5)
    p6v = tf.gather_nd(channel, p6)
    p7v = tf.gather_nd(channel, p7)

    values = tf.math.add_n((p0v, p1v, p2v, p3v, p4v, p5v, p6v, p7v))
    den = tf.cast(den, tf.float32)
    values = tf.divide(values, den)

    empty_coordinates = tf.subtract(empty_coordinates, 1)
    shape = (height, width)
    filled_channel = tf.scatter_nd(empty_coordinates, values, shape)
    return filled_channel


def interpolation(image: tf.Variable, new_coords: tf.Variable) -> tf.Variable:
    height, width = image.shape[1], image.shape[2]
    empty_coordinates = miss_values(new_coords, height, width)
    # Строим маску пропусков с паддингом 1
    empty_coordinates = tf.add(empty_coordinates, 1)
    values = tf.ones(shape=empty_coordinates.shape[0], dtype=tf.uint8)
    miss_values_mask = tf.scatter_nd(empty_coordinates, values, tf.constant([height + 2, width + 2]))
    y = tf.slice(empty_coordinates, [0, 0], [-1, 1])
    x = tf.slice(empty_coordinates, [0, 1], [-1, 1])
    # Координаты x,y вокруг пропусков
    px0 = tf.squeeze(tf.subtract(x, 1), axis=1)
    py0 = tf.squeeze(tf.subtract(y, 1), axis=1)
    px1 = tf.squeeze(tf.add(x, 1), axis=1)
    py1 = tf.squeeze(tf.add(y, 1), axis=1)

    x = tf.squeeze(x, axis=1)
    y = tf.squeeze(y, axis=1)
    # Координаты точек вокруг пропусков
    p0 = tf.stack((py0, px0), axis=1)
    p1 = tf.stack((py0, x), axis=1)
    p2 = tf.stack((py0, px1), axis=1)
    p3 = tf.stack((y, px0), axis=1)
    p4 = tf.stack((y, px1), axis=1)
    p5 = tf.stack((py1, px0), axis=1)
    p6 = tf.stack((py1, x), axis=1)
    p7 = tf.stack((py1, px1), axis=1)

    coords_around = tf.stack((p0, p1, p2, p3, p4, p5, p6, p7), axis=0)
    coords_around = tf.reshape(coords_around, shape=(-1, 2))
    coords_around = tf.cast(coords_around, tf.int32)
    values = tf.ones(shape=coords_around.shape[0], dtype=tf.uint8)
    # Маска крайних точек, которые мы попытаемся взять
    pad_mask = padding_mask(coords_around, height, width)
    # Маска точек вокруг пропусков
    t = tf.scatter_nd(coords_around, values, tf.constant([height + 2, width + 2]))
    t = tf.clip_by_value(t, 0, 1)
    # Убрать точки в паддинге
    t = tf.subtract(t, pad_mask)
    # t - маска существующих пикселей вокруг пропущенных точек
    t = tf.bitwise.bitwise_xor(miss_values_mask, t)

    # den - количество известных точек вокруг каждой неизвестной
    pd0v = tf.gather_nd(t, p0)
    pd1v = tf.gather_nd(t, p1)
    pd2v = tf.gather_nd(t, p2)
    pd3v = tf.gather_nd(t, p3)
    pd4v = tf.gather_nd(t, p4)
    pd5v = tf.gather_nd(t, p5)
    pd6v = tf.gather_nd(t, p6)
    pd7v = tf.gather_nd(t, p7)
    h = tf.stack((pd0v, pd1v, pd2v, pd3v, pd4v, pd5v, pd6v, pd7v), axis=0)
    h = tf.cast(h, tf.float32)
    den = tf.reduce_sum(h, axis=0)
    den = tf.add(den, 1)

    # Вернем координаты на место (без паддинга)
    p0 = tf.subtract(p0, 1)
    p1 = tf.subtract(p1, 1)
    p2 = tf.subtract(p2, 1)
    p3 = tf.subtract(p3, 1)
    p4 = tf.subtract(p4, 1)
    p5 = tf.subtract(p5, 1)
    p6 = tf.subtract(p6, 1)
    p7 = tf.subtract(p7, 1)
    p = (p0, p1, p2, p3, p4, p5, p6, p7)
    # Заполним пропуски для каждого канала
    channels = tf.map_fn(fn=lambda ch: fill_channel(ch, empty_coordinates, den, height, width, *p), elems=image)
    return channels
