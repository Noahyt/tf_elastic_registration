import tensorflow as tf
import numpy as np
from elastic_image import elastic_image

class field_image(elastic_image):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_rotation(self, initial_rotation = None, trainable = True):
        '''
        Initializes rotation variables and generates rotation_warp_list.

        :param initial_rotation: initialization value for rotation_degree
        :param trainable: sets rotation_degree trainable
        :return: None
        '''
        if initial_rotation is None:
            initial_rotation = 0
        with tf.variable_scope("rotation"):
            self.rotation_degree = tf.Variable(initial_rotation, trainable = trainable, dtype=tf.float32, name = "rotation_degree")

            # center point for all control points
            # center_point = tf.squeeze(tf.reduce_mean(self.control_points, axis=1))

            self.centered_control_points = self.control_points - self.center_point

            self.rotation_radians = self.rotation_degree * 2 * np.pi / 360

            rotation_matrix = tf.reshape(tf.stack([
                                            tf.cos(self.rotation_radians),
                                            tf.sin(self.rotation_radians),
                                            - tf.sin(self.rotation_radians),
                                            tf.cos(self.rotation_radians),
                                            ]),
                                         shape = [2,2])

            centered_rotation_points = tf.transpose(tf.tensordot(rotation_matrix, self.centered_control_points, [[0], [2]]), [1, 2, 0])

            # rotation_warp_points is a list of warp points which can be passed to elastic_image.warp()
            self.rotation_warp_points = centered_rotation_points - self.centered_control_points

    def get_rotation_warp_points(self):
        return self.rotation_warp_points

    def get_rotation_degree(self):
        return self.rotation_degree

    def make_translation(self, initial_translation = None, trainable = True):
        '''
        Initializes translation variables and generates translation_warp_list.

        :param initial_translation: Temsor [translation_axis_0, translation_axis_1]
        :param trainable:
        :return:
        '''

        if initial_translation is None:
            initial_translation = [0,0]
        with tf.variable_scope("translation"):
            self.translation = tf.Variable(initial_translation, dtype=tf.float32, trainable = trainable, name = "translation")

            # translation_warp_points is a list of warp points which can be passed to elastic_image.warp()
            self.translation_warp_points = tf.tile(self.translation[tf.newaxis, tf.newaxis, :], [1, self.num_control_points, 1])

    def get_translation_warp_points(self):
        return self.translation_warp_points

    def get_translation_vector(self):
        return self.translation


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    data_temp = np.load('heart_rotation.npy').astype(np.float32)
    im_0 = tf.Variable(data_temp[0, :, :])
    elastic_test = field_image( im_0 , size =(800,800), name = "haha")

    elastic_test.make_control_points(2,3)
    elastic_test.make_rotation(25)
    elastic_test.make_translation([50, -50])

    cp = elastic_test.get_control_points()
    im = elastic_test.get_image()
    center_p = elastic_test.get_center_point()

    warp_points_rotation = elastic_test.get_rotation_warp_points()
    warp_points_translation = elastic_test.get_translation_warp_points()

    warp_points = elastic_test.warp([warp_points_rotation, warp_points_translation])
    elastic_warped = elastic_test.get_warped()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        cp_eval, im_eval, im_warped_eval, warp_points_eval, center_p_eval = sess.run([cp, im, elastic_warped, warp_points, center_p])

    print(center_p_eval)

    print(cp_eval)

    print(warp_points_eval)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(im_eval[0, :, :, 1])
    ax[0].plot(*cp_eval[0].T[::-1])
    ax[1].imshow(im_warped_eval[0, :, :, 1])
    plt.show()


