import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np

from affine_tf.transforms import rotate

if __name__ == '__main__':
    # img = plt.imread("../data/11.jpg")
    img = cv2.imread("../data/11.jpg", 0)
    img = np.expand_dims(img, axis=2)
    plt.imshow(img)
    plt.show()

    img = tf.convert_to_tensor(img, dtype=tf.float32)
    rot_img = rotate(img, angle=17)

    rot_img = rot_img.numpy()
    rot_img /= 255
    plt.imshow(rot_img)
    plt.show()
