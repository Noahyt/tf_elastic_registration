import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class elastic_image_field():
    '''
    An elastic_image_field holds field_images and coordinates their placement and rotation.
    '''

    def __init__(self, size, name="elastic_image_field"):
        self.size = size
        self.name = name
        self.field_ims = []
        self.ims = []

    def load_image(self, field_image, initial_rotation, initial_translation):
        '''
        Loads elastic image and places into field with rotation and translation


        :param elastic_image: an elastic_image to be added to the field.
        :param initial_translation: initial displacement relative to the center of the field. 1-D Tensor (x,y)
        :return: None
        '''

        with tf.variable_scope("loading_images_in_field"):
            '''initialize variables in field images'''
            field_image.make_rotation(initial_rotation)
            field_image.make_translation(initial_translation)

            self.field_ims.append(field_image)

            '''perform rotation and translation operation'''
            image = field_image.get_warped()
            image = self.rotate(image, field_image.get_rotation())
            image = self.translate(image, field_image.get_translation())

            image.set_shape([1,self.size[0], self.size[1],2])

            '''add output to field'''
            self.ims.append(image)

    def rotate(self, image, rotation_radians):
        with tf.variable_scope("rotation"):
            return tf.contrib.image.rotate(image, rotation_radians, interpolation = 'NEAREST')

    def translate(self, image, translation):
        '''

        :param image:
        :param translation:
        :return:
        '''

        with tf.variable_scope("translation"):
            #TODO(ntoyonaga) use native tensorflow
            pad_0 = int(( self.size[0] - image.get_shape().as_list()[1] ) / 2 )
            pad_1 = int(( self.size[1] - image.get_shape().as_list()[2] ) / 2 )

            print(pad_0)
            print(pad_1)

            return tf.pad(image, [[0,0], [ pad_0 + translation[0], pad_0 - translation[0]] , [ pad_1 + translation[1] , pad_1 - translation[1] ], [0,0]])

    def make_registration_loss(self):
        '''
        :return:
        mse_loss -- summed mean squared error accross all image pairs
        alpha_overlap --- overlap between all pairs of images
        '''

        print("self.ims is {}".format(self.ims))

        ims_tf = tf.squeeze(tf.stack(self.ims))

        print(ims_tf)

        sz = ims_tf.get_shape().as_list()

        def upper_tri_indices(size):
            indices = np.triu_indices(size, 1)
            indices = np.stack([indices[0], indices[1]], axis=0)
            return indices.T

        ims2 = tf.tile(ims_tf[tf.newaxis, :, :, :, 0], [sz[0], 1, 1, 1])
        ims2 = tf.gather_nd(ims2, upper_tri_indices(sz[0]))

        subtract_term = tf.tile(ims_tf[tf.newaxis, :, :, :, 0], [1, 1, sz[0], 1])
        subtract_term = tf.reshape(subtract_term, [sz[0], sz[0], sz[1], sz[2]])
        subtract_term = tf.gather_nd(subtract_term, upper_tri_indices(sz[0]))

        mse_loss = tf.squared_difference(ims2, subtract_term)

        alpha_overlap = tf.tile(ims_tf[tf.newaxis, :, :, :, 1], [sz[0], 1, 1, 1])
        alpha_overlap = tf.gather_nd(alpha_overlap, upper_tri_indices(sz[0]))
        alpha_ = tf.tile(ims_tf[tf.newaxis, :, :, :, 1], [1, 1, sz[0], 1])
        alpha_ = tf.reshape(alpha_, [sz[0], sz[0], ims_tf.shape[1], sz[2]])
        alpha_ = tf.gather_nd(alpha_, upper_tri_indices(sz[0]))

        alpha_overlap = tf.minimum(alpha_overlap, alpha_)

        # take out components that do not overlap using alpha channel
        mse_loss = tf.multiply(mse_loss, alpha_overlap)
        self.mse_loss = tf.reduce_mean(mse_loss)

    def get_registration_loss(self):
        return(self.mse_loss)

    ''' functions for visualization '''

    def get_total_field(self):
        pass

    def get_displacements(self):
        pass

    def get_field_ims(self):
        return self.field_ims

    def get_warped_ims(self):
        return self.ims


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from field_image import field_image

    data_temp = np.load('heart_rotation.npy').astype(np.float32)
    im_0 = tf.Variable(data_temp[0, :, :])
    elastic_test = field_image( im_0 , extra_pad = 20)

    elastic_test.make_control_points(2,3)

    test_elastic_field = elastic_image_field([900,1000])

    test_elastic_field.load_image(elastic_test, 0., [100,-100])

    field_ims = test_elastic_field.get_ims()

    print(field_ims)

    im = field_ims[0]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        im_eval = sess.run(im)

    plt.imsave("im_test", im_eval[0,:,:,0])

