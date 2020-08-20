import tensorflow as tf
import matplotlib.pyplot as plt

from transforms import warpAffine

if __name__ == '__main__':
    img = plt.imread("../data/11.jpg")
    plt.imshow(img)
    plt.show()

    img = tf.convert_to_tensor(img, dtype=tf.float32)
    dx = img.shape[1]
    T = tf.constant([[1, 0.5, 0],
                     [0.5, 1, 0],
                     [0, 0, 1]], dtype=tf.float32)
    new_img = warpAffine(img, T)

    new_img = new_img.numpy()
    new_img /= 255
    plt.imshow(new_img)
    plt.show()
