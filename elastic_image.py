import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from modified_sparse_image_warp import sparse_image_warp


class elastic_image():

    '''
    elastic_image objects parametrizes the small scale elastic shifts for image registration.
    '''

    def __init__(self, image, field_size = (0, 0), name = "elastic_image", save_dir = None):

        self.name = name

        if save_dir is None:
            save_dir = ''
        self.save_dir = save_dir

        if field_size is None:
            field_size = image.get_shape().as_list()

        self.image, self.corner_points, self.center_point = self.pad(image, field_size)

        self.sz = self.image.get_shape().as_list()

        self.graph = tf.get_default_graph()

        self.set_tension_kernel([[0, .25, 0], [.25, -1, .25], [0, .25, 0]])

        self.num_control_points = 4

    def get_image(self):
        return self.image

    def get_corners(self):
        return self.corner_points

    def get_center_point(self):
        '''

        :return: center point for image [x, y]
        '''
        return self.center_point

    def _make_control_points_np(self,  num_points_0, num_points_1):
        x_space = np.linspace(start=self.corner_points[0, 0, 0], stop=self.corner_points[-1, -1, 0], num=num_points_0, dtype=np.int)
        y_space = np.linspace(start=self.corner_points[0, 0, 1], stop=self.corner_points[-1, -1, -1], num=num_points_1, dtype=np.int)
        x_tile = np.tile(x_space, (num_points_1, 1))
        y_tile = np.tile(y_space, (num_points_0, 1)).T
        source_control_points = np.stack([x_tile, y_tile], axis=0).T.reshape(-1, 2)
        return source_control_points[np.newaxis, :]

    def make_control_points(self, num_0, num_1):
        '''control_points are the initial x and y coordinates of nodes on an elastic_image.
        control_points has shape [1, num_points, 2]
        '''
        with tf.variable_scope("control_points"):
            self.num_0 = num_0
            self.num_1 = num_1
            self.control_points_np = self._make_control_points_np(num_0, num_1)
            self.control_points = tf.constant( self.control_points_np, tf.float32)
            self.num_control_points = num_0*num_1

    def get_control_points(self):
        return self.control_points

    def warp(self, warp_values: list, scale):
        '''Uses warp points and list of warp values to modify images.
        Warp values which may come from different sources -- such as rotation, translation, and elastic distortion.
        '''
        with tf.variable_scope("warp"):
            with tf.variable_scope("warp_points"):
                warp_points = tf.zeros_like(self.control_points)

                for warp_list in warp_values:
                    warp_points = warp_points + warp_list

                destination_control_points = warp_points + self.control_points

                source_points = self.control_points / scale
                destination_points = destination_control_points / scale

            new_size_np = np.array(self.image.get_shape().as_list())[1:3] / scale
            new_size = tf.cast(new_size_np, tf.int32)
            new_image = tf.image.resize_images(self.image, new_size, method=tf.image.ResizeMethod.BILINEAR)

            self.warped, self.dense_warp  = sparse_image_warp(new_image,
                                                              source_points,
                                                               destination_points,
                                                               interpolation_order=2,
                                                               )

        return warp_points

    def get_warped(self):
        return self.warped


    # elastic warping variables and initialization

    def make_elastic_warp(self, scale=1., initial_offsets=None, trainable=True):
        '''
        makes image warp points, sparse_warp_matrix

        :param scale: determines scale of random variation from initial_guess
        :param initial_offsets: optional, replaces random varation.
        :param trainable: set elastic_warp_points to be trainable
        :return: none
        '''
        with tf.variable_scope("elastic_warp"):
            if initial_offsets is not None:
                self.elastic_warp_points = tf.Variable(initial_offsets,
                                                       dtype=tf.float32, trainable=trainable,
                                                       name="elastic_warp_points")
                print('manually setting inital_offsets')
            else:
                assert isinstance(scale, (int, float))
                self.elastic_warp_points = tf.Variable(
                    tf.random_uniform([self.num_control_points, 2], minval=-1, maxval=1) * scale,
                    dtype=tf.float32, trainable=trainable, name="elastic_warp_points")

            self.elastic_warp_points = tf.expand_dims(self.elastic_warp_points, 0)

            # the sparse_warp_matric is the "physical" reshaping of the list of elastic_warp points
            # we need this to find the "elastic loss"
            self.sparse_warp_matrix = tf.reshape(self.elastic_warp_points, [1, self.num_0, self.num_1, 2])

    def get_sparse_warp_matrix(self):
        return self.sparse_warp_matrix

    def get_elastic_warp_points(self):
        return self.elastic_warp_points

    # loss function for elastic warp

    def init_warp_loss(self, tension = .1):
        self.warp_loss = self._grid_correlation_loss(tension)[0]

    def get_warp_loss(self):
        return self.warp_loss

    def set_tension_kernel(self, kernel):
        correlation_kernel = tf.constant(kernel)
        correlation_kernel = tf.stack([correlation_kernel, tf.zeros_like(correlation_kernel)], axis=2)
        self.correlation_kernel = tf.stack([correlation_kernel, correlation_kernel[:, :, ::-1]], axis=3)

    def _grid_correlation_loss(self, tension=None):
        if tension is None:
            tension = .1
        # symmetric padding ensures that tensors along the edge are not penalized
        tensor = tf.pad(self.sparse_warp_matrix, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="SYMMETRIC")
        convolved = tf.nn.conv2d(tensor, self.correlation_kernel, strides=[1, 1, 1, 1], padding="VALID")
        loss = tf.pow(convolved, 2)
        loss = tf.reduce_sum(loss) * tension
        return loss, convolved

    # training and summary ops

    def make_summaries(self):
        tf.summary.scalar("elastic_loss_{}".format(self.name), self.warp_loss)
        tf.summary.image("warped_alpha_{}".format(self.name), self.warped[:,:,:,1,np.newaxis], max_outputs=1)
        tf.summary.image("warped_image_{}".format(self.name), self.warped[:,:,:,0,np.newaxis], max_outputs=1)

    def dense_quiver_plot(self, sess):
        self.evaluate_dense_warp(sess)
        self.evaluate_warp(sess)

        fig, ax = plt.subplots(1, 1)

        spacing = 50

        print(self.dense_warp_eval.shape)

        all_warp = np.reshape(self.dense_warp_eval, [-1, 2])

        local_warp = self.dense_warp_eval - np.mean(all_warp, axis=0)

        ax.quiver(local_warp[0, ::spacing, ::spacing, 0], local_warp[0, ::spacing, ::spacing, 1])

        ax.set_xticklabels([])
        ax.set_yticklabels([])

        plt.savefig("{}/dense_field_{}".format(self.save_dir, self.name))
        np.savez("{}/field_data_{}".format(self.save_dir, self.name))

    def evaluate_warp(self, sess):
        self.warp_evaluated = sess.run(self.warped)

    def evaluate_dense_warp(self, sess):
        self.dense_warp_eval = sess.run(self.dense_warp)

    def pad(self, image, total_size=(0,0)):
        '''

        :param image: 2D or 4D Tensor
        :param extra_pad:
        :return:
        '''

        sz = image.get_shape().as_list()

        if len(sz) == 2:
            tf_ = tf.expand_dims(tf.expand_dims(image, 0), 3)
        else:
            assert len(sz) == 4
            tf_ = image

        sz_tf_ = tf_.get_shape().as_list()


        pad_0 = int((total_size[0] - sz_tf_[1]) / 2)
        pad_1 = int((total_size[1] - sz_tf_[2]) / 2)

        pad_shape = [[0, 0], [pad_0, pad_0],
                     [pad_1, pad_1], [0, 0]]

        corners = np.array([[[pad_0, pad_1], [pad_0, pad_1 + sz_tf_[2] - 1]],
                               [[pad_0 + sz_tf_[1] - 1, pad_1], [pad_0 + sz_tf_[1] - 1, pad_1 + sz_tf_[2] - 1]]], dtype=np.float32)


        # pads image on all spatial dimensions by  $pad
        # locates image in location of "new" image corresponding to self.displacement
        # adds new alpha dimension to channel axis

        alpha_ = tf.pad(tf.Variable(tf.ones_like(tf_), trainable = False),
                        pad_shape, mode='constant')

        tf_ = tf.pad(tf_, pad_shape, mode='constant')

        tf_ = tf.concat([tf_, alpha_], axis=3)

        #center point used for translations
        center_point = tf.constant([total_size[0] / 2, total_size[1] / 2], dtype=tf.float32)
        return tf_, corners, center_point


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    data_temp = np.load('heart_rotation.npy').astype(np.float32)
    im_0 = tf.ones([30,30])
    elastic_test = elastic_image( im_0 , field_size = (50, 70))

    elastic_test.make_control_points(2,2)

    cp = elastic_test.get_control_points()
    im = elastic_test.get_image()
    center_p = elastic_test.get_center_point()

    elastic_test.make_elastic_warp(initial_offsets=[[-6.23,15],[15,6.213],[-15,-6.213],[+6.213, -15]]) #[[+6.213, -15],[-15,-6.213], [15,6.213], [-6.213,15]]
    elastic_test.warp([elastic_test.get_elastic_warp_points()])
    elastic_warped = elastic_test.get_warped()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        cp_eval, im_eval, im_warped_eval = sess.run([cp, im, elastic_warped])

    print(cp_eval)

    fig, ax = plt.subplots(1,2)
    ax[0].imshow(im_eval[0,:,:,1])
    ax[0].plot(*cp_eval[0].T[::-1])
    ax[1].imshow(im_warped_eval[0,:,:,1])
    plt.show()

    # plt.imsave("elastic_image_test", im_eval[0,:,:,1])

