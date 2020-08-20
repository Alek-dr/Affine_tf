import tensorflow as tf
import matplotlib.pyplot as plt

from transforms import translation

if __name__ == '__main__':
    img = plt.imread("../data/11.jpg")
    img = tf.convert_to_tensor(img, dtype=tf.float32)

    transl_img = translation(img, dx=30, dy=0)

    transl_img = transl_img.numpy()
    transl_img /= 255
    plt.imshow(transl_img)
    plt.show()
