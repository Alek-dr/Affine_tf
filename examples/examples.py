import tensorflow as tf
import matplotlib.pyplot as plt

from affine_tf.transforms import warpAffine, rotate, translation, reflection

if __name__ == '__main__':
    img = plt.imread("../data/11.jpg")

    fig, axs = plt.subplots(2, 2)

    img = tf.convert_to_tensor(img, dtype=tf.float32)
    # rotation
    rot_img = rotate(img, angle=35)
    rot_img /= 255
    axs[0, 0].set_title("Rotation")
    axs[0, 0].imshow(rot_img)

    # shear
    T = tf.constant([[1, 0.2, 0],
                     [0.2, 1, 0],
                     [0, 0, 1]], dtype=tf.float32)
    shear = warpAffine(img, T)
    shear /= 255
    axs[0, 1].set_title("Shear")
    axs[0, 1].imshow(shear)

    # translation
    transl_img = translation(img, dx=40, dy=15)
    transl_img /= 255
    axs[1, 0].set_title("Translation")
    axs[1, 0].imshow(transl_img)

    # reflection
    refl = reflection(img)
    refl = refl.numpy()
    refl /= 255
    axs[1, 1].set_title("Reflection")
    axs[1, 1].imshow(refl)

    plt.show()
