import tensorflow as tf
import matplotlib.pyplot as plt

from affine_tf.transforms import warpAffine

if __name__ == '__main__':
    img = plt.imread("../data/11.jpg")
    plt.imshow(img)
    plt.show()

    img = tf.convert_to_tensor(img, dtype=tf.float32)

    T = tf.constant([[1, 0.25, 3],
                     [-0.1, 2, 0],
                     [0, 0, 1]], dtype=tf.float32)
    new_img = warpAffine(img, T)

    new_img = new_img.numpy()
    new_img /= 255
    plt.imshow(new_img)
    plt.show()
