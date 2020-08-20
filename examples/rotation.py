import tensorflow as tf
import matplotlib.pyplot as plt

from transforms import rotate

if __name__ == '__main__':
    img = plt.imread("../data/11.jpg")
    plt.imshow(img)
    plt.show()

    img = tf.convert_to_tensor(img, dtype=tf.float32)
    rot_img = rotate(img, angle=30)

    rot_img = rot_img.numpy()
    rot_img /= 255
    plt.imshow(rot_img)
    plt.show()
