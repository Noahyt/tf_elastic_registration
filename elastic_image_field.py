import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from field_image import field_image

class elastic_image_field():
    '''
    An elastic_image_field holds field_images and coordinates their placement and rotation.
    '''

    def __init__(self, size, name="elastic_image_field"):
        self.size = size
        self.name = name
        self.field_ims = []
        self.ims = []

    def load_image(self, image, initial_rotation, initial_translation, control_points = (2,2), use_as_base = False):
        '''
        Loads image and places into field with rotation and translation


        :param elastic_image: an elastic_image to be added to the field.
        :param initial_translation: initial displacement relative to the center of the field. 1-D Tensor (x,y)
        :return: None
        '''

        fi = field_image(image , field_size = self.size)
        fi.make_control_points(*control_points)

        with tf.variable_scope("field_image_{}".format(len(self.field_ims))):
            '''initialize variables in field images'''
            if use_as_base:
                print('setting base image')
                fi.make_rotation(initial_rotation, trainable= False)
                fi.make_translation(initial_translation, trainable= False)
                fi.make_elastic_warp(scale=0)  # setting scale = 0 => no elastic warp
            else:
                fi.make_rotation(initial_rotation)
                fi.make_translation(initial_translation)
                fi.make_elastic_warp()

            self.field_ims.append(fi)

    def warp(self):
        '''perform rotation and translation operation'''

        scale = tf.placeholder(dtype=tf.float32, shape=[1])

        '''add output to field'''
        for field_image in self.field_ims:


            # image, corners = self.pad_field_image_(image)
            # image, corners = self.reduce_field_image_size(image, corners, scale)

            _ = field_image.warp([field_image.get_rotation_warp_points(),
                                      field_image.get_translation_warp_points(),
                                      field_image.get_elastic_warp_points()], scale)

            self.ims.append(field_image.get_warped())

        return scale

    def reduce_field_image_size(self, image, corners, scale):
        with tf.variable_scope("image_size_reduction"):
            new_size_np = np.array(image.get_shape().as_list())[1:3] / scale
            new_size = tf.cast(new_size_np, tf.int32)
            new_image = tf.image.resize_images(image, new_size, method=tf.image.ResizeMethod.BILINEAR)

            new_corners = corners / scale

            new_size_np = np.concatenate([[1], new_size_np.astype(np.int32), [2]])

            new_image.set_shape(new_size_np)

            return new_image, new_corners

    '''Initialization and retrieval functions for variable lists.'''

    def get_rotation_variables(self):
        collection = []
        for image in range(len(self.field_ims)):
            collection += tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                            scope="field_image_{}/rotation".format(image))
        return collection

    def get_translation_variables(self):
        collection = []
        for image in range(len(self.field_ims)):
            collection += tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                            scope="field_image_{}/translation".format(image))
        return collection

    def get_warp_variables(self):
        collection = []
        for image in range(len(self.field_ims)):
            collection += tf.get_collection(key = tf.GraphKeys.TRAINABLE_VARIABLES,
                                                scope="field_image_{}/elastic_warp".format(image))
        return collection

    '''Definition of registration loss'''

    def make_registration_loss(self):
        '''
        :return:
        mse_loss -- summed mean squared error accross all image pairs
        alpha_overlap --- overlap between all pairs of images
        '''
        with tf.variable_scope("registration_loss"):

            ims_tf = tf.squeeze(tf.stack(self.ims))

            # size = tf.cast(tf.shape(ims_tf)[1:3]/4, tf.int32)
            #
            # ims_tf = tf.image.resize_images( ims_tf, size, method=tf.image.ResizeMethod.BILINEAR)

            sz_tf = tf.shape(ims_tf)
            # sz = ims_tf.get_shape().as_list()

            def upper_tri_indices(size):
                indices = np.triu_indices(size, 1)
                indices = np.stack([indices[0], indices[1]], axis=0)
                return indices.T

            def upper_tri_tf(innn):
                out = innn - tf.matrix_band_part(innn, -1, 0)
                return out

            ims2 = tf.tile(ims_tf[tf.newaxis, :, :, :, 0], [sz_tf[0], 1, 1, 1])
            ims2 = upper_tri_tf(ims2)

            subtract_term = tf.tile(ims_tf[tf.newaxis, :, :, :, 0], [1, 1, sz_tf[0], 1])
            subtract_term = tf.reshape(subtract_term, [sz_tf[0], sz_tf[0], sz_tf[1], sz_tf[2]])
            subtract_term = upper_tri_tf(subtract_term)

            mse_loss = tf.squared_difference(ims2, subtract_term)

            alpha_overlap = tf.tile(ims_tf[tf.newaxis, :, :, :, 1], [sz_tf[0], 1, 1, 1])
            alpha_overlap = upper_tri_tf(alpha_overlap)
            alpha_ = tf.tile(ims_tf[tf.newaxis, :, :, :, 1], [1, 1, sz_tf[0], 1])
            alpha_ = tf.reshape(alpha_, [sz_tf[0], sz_tf[0], sz_tf[1], sz_tf[2]])
            alpha_ = upper_tri_tf(alpha_)
            alpha_overlap = tf.minimum(alpha_overlap, alpha_)

            # take out components that do not overlap using alpha channel
            mse_loss = tf.multiply(mse_loss, alpha_overlap)
            self.mse_loss = tf.reduce_mean(mse_loss)
        return self.mse_loss

    def get_registration_loss(self):
        return(self.mse_loss)

    def make_coherence_loss(self):
        '''reduces difference between adjacent images'''
        with tf.variable_scope('coherence_loss'):
            translations = tf.squeeze(tf.stack(self.get_translations()))
            rotations = tf.squeeze(tf.stack(self.get_rotations()))
            def tf_diff_axis_1(a):
                return a[:, 1:] - a[:, :-1]
            def tf_diff_axis_0(a):
                return a[1:] - a[:-1]

            diff = tf_diff_axis_1(translations)
            mean_diff = tf.reduce_mean(diff)

            diff = tf_diff_axis_1(diff)
            diff = tf.reduce_sum(tf.pow(diff, 2), axis =1)

            diff_rotation= tf_diff_axis_0(rotations)


            self.coherence_loss = tf.reduce_max(tf.abs(diff))/mean_diff # + tf.reduce_max(diff_rotation)/tf.reduce_mean(diff_rotation)

        return self.coherence_loss

    def get_coherence_loss(self):
        return self.coherence_loss

    ''' functions for visualization '''

    def get_total_field(self):
        pass

    def get_translations(self):
        displacements = []
        for image in self.field_ims:
            displacements.append(image.get_translation_vector())
        return(displacements)

    def get_rotations(self):
        rotations = []
        for image in self.field_ims:
            rotations.append(image.get_rotation_degree())
        return rotations

    def get_field_ims(self):
        return self.field_ims

    def get_warped_ims(self):
        return self.ims


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    data_temp = np.load('heart_rotation.npy').astype(np.float32)

    im_0 = tf.Variable(data_temp[0, :622, :])
    im_1 = tf.Variable(data_temp[1, :622, :])

    test_elastic_field = elastic_image_field([900,1000])

    test_elastic_field.load_image(im_0, 25., [100.,-100.])
    test_elastic_field.load_image(im_1, 0., [40.,40.])

    test_elastic_field.warp()

    warp_vars = test_elastic_field.get_warp_variables()

    print(warp_vars)

    field_ims = test_elastic_field.get_warped_ims()

    im = field_ims[0]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        im_eval = sess.run(im)

    plt.imsave("im_test", im_eval[0,:,:,1])

