import tensorflow as tf
import matplotlib.pyplot as plt

from transforms import warpAffine, rotate, translation

if __name__ == '__main__':
    img = plt.imread("../data/11.jpg")

    fig, axs = plt.subplots(2, 2)

    img = tf.convert_to_tensor(img, dtype=tf.float32)
    dx = img.shape[1]

    # reflection
    T = tf.constant([[-1, 0, dx],
                     [0, 1, 0],
                     [0, 0, 1]], dtype=tf.float32)
    refl = warpAffine(img, T)

    refl = refl.numpy()
    refl /= 255
    axs[0, 0].set_title("Reflection")
    axs[0, 0].imshow(refl)

    # rotation
    rot_img = rotate(img, angle=35)
    rot_img /= 255
    axs[0, 1].set_title("Rotation")
    axs[0, 1].imshow(rot_img)

    # shear
    T = tf.constant([[1, 0.2, 0],
                     [0.2, 1, 0],
                     [0, 0, 1]], dtype=tf.float32)
    shear = warpAffine(img, T)
    shear /= 255
    axs[1, 0].set_title("Shear")
    axs[1, 0].imshow(shear)

    # translation
    transl_img = translation(img, dx=30, dy=0)
    transl_img /= 255
    axs[1, 1].set_title("Translation")
    axs[1, 1].imshow(transl_img)

    plt.show()
