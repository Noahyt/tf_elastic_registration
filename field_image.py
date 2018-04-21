import tensorflow as tf
from elastic_image import elastic_image

class field_image(elastic_image):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_rotation(self, initial_rotation = None):
        if initial_rotation is None:
            initial_rotation = 0
        with tf.variable_scope("rotation"):
            self.rotation_degree = tf.Variable(initial_rotation, trainable = True, name = "rotation_degree")

    def get_rotation(self):
        return self.rotation_degree

    def make_translation(self, initial_translation = None):
        if initial_translation is None:
            initial_translation = [0,0]
        with tf.variable_scope("translation"):
            self.translation = tf.Variable(initial_translation, trainable = True, name = "translation")

    def get_translation(self):
        return self.translation


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    data_temp = np.load('heart_rotation.npy').astype(np.float32)
    im_0 = tf.Variable(data_temp[0, :, :])
    elastic_test = field_image( im_0 , extra_pad = 20, name = "haha")

    elastic_test.make_control_points(2,3)

    cp = elastic_test.get_control_points()
    im = elastic_test.get_image()
    center_p = elastic_test.get_center_points()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        cp_eval, im_eval, center_p_eval = sess.run([cp, im, center_p])

    print(im_eval.shape)
    print(cp_eval)
    print(center_p_eval)

    plt.imsave("im", im_eval[0,:,:,1])

