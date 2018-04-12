import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class elastic_image():

    def __init__(self, image, corners, name = "elastic_image"):
        self.name = name
        self.image = image
        self.sz = image.get_shape().as_list()
        self.corners = corners
        self.graph = tf.get_default_graph()
        self.set_tension_kernel([[0, .25, 0], [.25, -1, .25], [0, .25, 0]])


    def get_image(self):
        return self.image

    def get_corners(self):
        return self.corners

    def _make_control_points_np(self, corners, num_points_0, num_points_1):
        x_space = np.linspace(start=corners[0, 0], stop=corners[-1, 0], num=num_points_0, dtype=np.int)
        y_space = np.linspace(start=corners[0, 1], stop=corners[-1, -1], num=num_points_1, dtype=np.int)
        x_tile = np.tile(x_space, (num_points_1, 1))
        y_tile = np.tile(y_space, (num_points_0, 1)).T
        source_control_points = np.stack([x_tile, y_tile], axis=0).T.reshape(-1, 2)
        return source_control_points[np.newaxis, :]

    def make_control_points(self, num_0, num_1):
        self.num_0 = num_0
        self.num_1 = num_1
        self.control_points_np = self._make_control_points_np(self.corners, num_0, num_1)
        self.control_points = tf.constant( self.control_points_np ,tf.float32)
        self.total_points = num_0*num_1

    def get_control_points(self):
        return self.control_points

    def make_initial_guess(self, base_image):
        self.initial_guess, self.corr_v = phase_correlate_custom(base_image, self.image)
        self.initial_guess = tf.expand_dims(self.initial_guess, 0)
        self.initial_guess = tf.tile( self.initial_guess, [self.total_points, 1])
        self.initial_guess = tf.cast( self.initial_guess, tf.float32)

    def make_warp_points_and_matrix(self, scale=1., initial_offsets = None, trainable = True):
        '''
        makes image warp points, warp_tensor

        :param scale: determines scale of random variation from initial_guess
        :param initial_offsets: optional, replaces random varation.  Tensor of same shape as initial_guess.
        :param trainable: set warp_points to be trainable
        :return: none
        '''
        if initial_offsets is not None:
            self.warp_points = tf.Variable(self.initial_guess +
                                           initial_offsets,
                                           dtype=tf.float32, trainable=trainable, name="warp_points")
            print('manually setting inital_offsets')
        else:
            assert isinstance(scale, (int, float))
            self.warp_points = tf.Variable(self.initial_guess +
                                        tf.random_uniform([self.total_points, 2], minval=-1, maxval=1) * scale,
                                        dtype=tf.float32, trainable=trainable, name= "warp_points")
        #
        self.warp_matrix = tf.expand_dims(self.warp_points, 0)
        self.destination_control_points = self.warp_matrix + self.control_points
        self.warp_tensor = tf.reshape(self.warp_points,[1,self.num_0,self.num_1,2])

    def get_warp_tensor(self):
        return self.warp_tensor

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
        tensor = tf.pad(self.warp_tensor, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="SYMMETRIC")
        convolved = tf.nn.conv2d(tensor, self.correlation_kernel, strides=[1, 1, 1, 1], padding="VALID")
        loss = tf.pow(convolved, 2)
        loss = tf.reduce_sum(loss) * tension
        return loss, convolved

    def get_warp_points(self):
        return self.warp_points

    def warp(self):
        self.warped, _ = tf.contrib.image.sparse_image_warp(self.image, self.control_points, self.destination_control_points)

    def get_warped(self):
        return self.warped

    def make_summaries(self):
        tf.summary.scalar("elastic_loss_{}".format(self.name), self.warp_loss)
        tf.summary.image("warped_alpha_{}".format(self.name), self.warped[:,:,:,1,np.newaxis], max_outputs=1)
        tf.summary.image("warped_image_{}".format(self.name), self.warped[:,:,:,0,np.newaxis], max_outputs=1)

    def plot_quiver(self, sess):
        if sess:
          self.evaluate_warp(sess)
          self.evaluate_warp_points(sess)

        points_swap = self.control_points_np[0].T[::-1, :]
        absolute_coordinates = [[0], [self.sz[1]]] + [[1], [-1]] * points_swap

        fig,ax = plt.subplots(1,2,figsize = (10,10))
        ax[0].imshow(self.warp_evaluated[0,:,:,0])
        ax[1].imshow(self.warp_evaluated[0,::-1,:,1])

        Q = ax[1].quiver(absolute_coordinates[0], absolute_coordinates[1],
                         self.warp_points_eval.T[1], -self.warp_points_eval.T[0],
                         color = 'r' , pivot='tail',
                         angles = 'xy', scale_units='xy', scale=1. )

        ax[1].set_ylim([0,self.sz[1]])
        ax[1].set_xlim([0,self.sz[2]])

        plt.savefig("field_and_correction_{}.png".format(self.name))

    def evaluate_warp(self, sess):
        self.warp_evaluated = sess.run(self.warped)

    def evaluate_warp_points(self, sess):
        self.warp_points_eval = sess.run(self.warp_points)
        
    


def phase_correlate_custom(image_1, image_2, window=None):
    sz = image_1.get_shape().as_list()

    # calculate hamming window
    ham2d = tf.sqrt(tf.einsum('i,j->ij',
                              tf.contrib.signal.hamming_window(sz[1]),
                              tf.contrib.signal.hamming_window(sz[2])))

    corr_v = (tf.ifft2d(tf.fft2d(tf.cast(image_1[0, :, :, 0] * ham2d, tf.complex64)) *
                        tf.ifft2d(tf.cast(image_2[0, :, :, 0] * ham2d, tf.complex64))))
    corr_v = tf.real(corr_v)
    max_index = tf.argmax(tf.reshape(corr_v, [-1]))
    max_index = tf.unravel_index(max_index, image_2[0, :, :, 0].shape)

    #convert index shifts from circular to linear
    max_index = tf.where(tf.cast(max_index[0],tf.float32)>(sz[1]/2), [-1 * sz[1], 0 ] + max_index,  max_index)
    max_index = tf.where(tf.cast(max_index[1],tf.float32)>(sz[2]/2), [0,  -1 * sz[2]] + max_index,  max_index)
    return max_index, corr_v
