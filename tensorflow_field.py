
# coding: utf-8
# In[2]:

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate
import time
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
from elastic_image import elastic_image
import os

def registration_to_multi_color_log(ims,  title = "multi_color_plot", directory = None):
    ims = np.log(ims + 1) / np.amax(np.log(ims + 1))

    colors = cm.rainbow(np.linspace(0, 1, ims.shape[0]))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)  # nrows, ncols, index
    ax.set_facecolor('black')
    for im_num in range(ims.shape[0]):
        colormap = LinearSegmentedColormap.from_list(
            'my_cmap', ['black', colors[im_num]], 256)

        colormap._init()  # create the _lut array, with rgba values

        colormap._lut[:, -1] = np.linspace(0, 1, colormap.N + 3)

        plt.imshow(ims[im_num],
                   interpolation='none',
                   cmap=colormap,
                   alpha=1
                   )

    if directory is None:
        directory = ''
    else:
        directory = '{}/'.format(directory)

    plt.savefig(directory + title + '.png')

    return

def individual_plots(ims, title= "individual_plots", directory = None):
    if directory is None:
        directory = ''
    else:
        directory = '{}/'.format(directory)

    # fig, ax = plt.subplots(1, ims.shape[0])

    colors = cm.rainbow(np.linspace(0, 1, ims.shape[0]))

    for im_num in range(ims.shape[0]):
        colormap = LinearSegmentedColormap.from_list(
            'my_cmap', ['black', colors[im_num]], 256)
        title_im = "{}_{}".format(title, im_num)
        im_to_plot = np.log(ims[im_num] + 1) / (np.amax(np.log(ims[im_num] + 1)))
        plt.figure()
        plt.imshow(im_to_plot,cmap=colormap)
        plt.savefig(directory + title_im + '.png')
        # ax[im_num].imshow(im_to_plot,cmap=colormap)

    return

def registration_compound(ims, title="compounded_plot", directory = None):
    if directory is None:
        directory = ''
    else:
        directory = '{}/'.format(directory)

    ims = np.log(ims + 1) / (np.amax(np.log(ims + 1)) + .0001)
    im_sum = np.sum(ims,axis = 0)/ims.shape[0]

    plt.figure()
    plt.imshow(im_sum)
    plt.title("individual_images")
    plt.colorbar()
    plt.savefig(directory + title + ".png")

    return im_sum

def load_and_pad(np_image, total_dim=None, pad=0):
    # args
    # np_image - 2D image
    # total_dim - tuple or np.array of dimension of total image field
    # pad - pad on each side of total_dim

    orig_shape = np_image.shape

    if total_dim is None:
        total_dim = orig_shape

    diff = np.array(total_dim) - np.array(np_image.shape)

    tf_ = tf.Variable(np_image, trainable = False)

    tf_ = tf.expand_dims(tf.expand_dims(tf_, 0), 3)

    # pads image on all spatial dimensions by  $pad
    # locates image in center of "new" image
    # adds new alpha dimension to channel axis


    tf_ = tf.pad(tf_, [[0, 0], [pad + int(diff[0] / 2), pad + int(diff[0] / 2)],
                       [pad + int(diff[1] / 2), pad + int(diff[1] / 2)], [0, 0]], mode='constant')

    alpha_np = np.pad(np.ones_like(np_image)[np.newaxis, :, :, np.newaxis],
                      [[0, 0], [pad + int(diff[0] / 2), pad + int(diff[0] / 2)],
                       [pad + int(diff[1] / 2), pad + int(diff[1] / 2)], [0, 0]], mode='constant')
    alpha_channel = tf.Variable(alpha_np, trainable=False)
    tf_ = tf.concat([tf_, alpha_channel], axis=3)

    corner_points = np.array([
        [pad + int(diff[0] / 2), pad + int(diff[1] / 2)],
        [pad + int(diff[0] / 2) + orig_shape[0], pad + int(diff[1] / 2)],
        [pad + int(diff[0] / 2), pad + int(diff[1] / 2) + orig_shape[1]],
        [pad + int(diff[0] / 2) + orig_shape[0], pad + int(diff[1] / 2) + orig_shape[1]]
    ])

    return tf_, corner_points


def load_n_images(np_image_array, total_dim=None, pad=0):
    images = []
    for image_iter in range(np_image_array.shape[0]):
        im_, corner_ = load_and_pad(np_image_array[image_iter], total_dim, pad)
        images.append(elastic_image(im_, corner_))
    return images

def difference_loss_n_ims(ims):

    sz = ims.get_shape().as_list()

    def upper_tri_indices(size):
        indices = np.triu_indices(size, 1)
        indices = np.stack([indices[0], indices[1]], axis=0)
        return indices.T

    ims2 = tf.tile(ims[tf.newaxis, :, :, :, 0], [sz[0], 1, 1, 1])
    ims2 = tf.gather_nd(ims2, upper_tri_indices(sz[0]))

    subtract_term = tf.tile(ims[tf.newaxis, :, :, :, 0], [1, 1, sz[0], 1])
    subtract_term = tf.reshape(subtract_term, [sz[0], sz[0], sz[1], sz[2]])
    subtract_term = tf.gather_nd(subtract_term, upper_tri_indices(sz[0]))

    mse_loss = tf.squared_difference(ims2, subtract_term)

    alpha_overlap = tf.tile(ims[tf.newaxis, :, :, :, 1], [sz[0], 1, 1, 1])
    alpha_overlap = tf.gather_nd(alpha_overlap, upper_tri_indices(sz[0]))
    alpha_ = tf.tile(ims[tf.newaxis, :, :, :, 1], [1, 1, sz[0], 1])
    alpha_ = tf.reshape(alpha_, [sz[0], sz[0], ims.shape[1], sz[2]])
    alpha_ = tf.gather_nd(alpha_, upper_tri_indices(sz[0]))

    alpha_overlap = tf.minimum(alpha_overlap, alpha_)

    # take out components that do not overlap using alpha channel
    mse_loss = tf.multiply(mse_loss, alpha_overlap)
    mse_loss = tf.reduce_mean(mse_loss)

    return mse_loss, alpha_overlap

def make_loss_multi_im(images, elastic_alpha = 1):

    ims_tf_list = [images[0].get_image()]

    for image in images[1:]:
        ims_tf_list.append(image.get_warped())

    ims_tf = tf.squeeze(tf.stack(ims_tf_list))

    mse_loss, alphas = difference_loss_n_ims(ims_tf)

    elastic_loss = []

    for image in images[1:]:
        image.init_warp_loss(.01)
        elastic_loss.append(image.get_warp_loss())
    elastic_loss_total = tf.reduce_mean(tf.squeeze(tf.stack(elastic_loss)))

    #can be modified to weight control points more

    total_loss = mse_loss +  elastic_loss_total * elastic_alpha

    return total_loss, elastic_loss_total, alphas, ims_tf


def align_images(directory):

    if not os.path.exists(directory):
        os.makedirs(directory)

    
    data_temp = np.load('noah_wrist_robot.npy').astype(np.float32)
    
    print(data_temp.shape)
    
    im_0 = data_temp[0,:,:]

    im_1 = data_temp[1,:,:]

    im_2 = data_temp[2,:,:]
    im_3 = data_temp[3,:,:]
    im_4 = data_temp[4,:,:]
    im_5 = data_temp[5,:,:]
    im_6 = data_temp[6,:,:]
    im_7 = data_temp[7,:,:]
    im_8 = data_temp[8,:,:]



    im_0 = rotate(im_0, -40, reshape = False)
    im_0 = np.clip(im_0,0, a_max= None)
    
    im_1 = rotate(im_1, -30, reshape = False)
    im_1 = np.clip(im_1,0, a_max= None)
    
    im_2 = rotate(im_2, -20, reshape = False)
    im_2 = np.clip(im_2,0, a_max= None)
    
    im_3 = rotate(im_3, -10, reshape = False)
    im_3 = np.clip(im_3,0, a_max= None)
    
    im_4 = rotate(im_4, 0, reshape = False)
    im_4 = np.clip(im_4,0, a_max= None)
    
    im_5 = rotate(im_5, 10, reshape = False)
    im_5 = np.clip(im_5,0, a_max= None)
    
    im_6 = rotate(im_6, 20, reshape = False)
    im_6 = np.clip(im_6,0, a_max= None)
    
    im_7 = rotate(im_7, 30, reshape = False)
    im_7 = np.clip(im_7,0, a_max= None)
    
    im_8 = rotate(im_8, 40, reshape = False)
    im_8 = np.clip(im_8,0, a_max= None)

    fig, ax = plt.subplots(1, 2)

    ax[0].imshow(im_4)
    ax[1].imshow(im_7)

    # [im_4, im_0, im_1, im_2, im_3, im_5, im_6, im_7, im_8]
    
    ims_np = np.stack([im_4,im_3,im_5, im_6], axis=0)
    
    ##load ims
    graph = tf.Graph()
    
    with graph.as_default():
    
        images = load_n_images(ims_np, total_dim = (900,1200), pad = 0 )
    
        num_points = [3,4] #minimum 2 in any dimension (defaults to corners)
        total_points = num_points[0]*num_points[1]
    
        start_image = images[0].get_image()
        iter = 1
    
        print(start_image.shape)
    
        for image in images[1:]:
            image.make_control_points(num_points[0],num_points[1])
            image.make_initial_guess(start_image/iter)
            image.make_warp_points_and_matrix()
            image.warp()
    
            start_image = start_image+image.get_warped()
    
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                new_image = sess.run(start_image)
                print(new_image.shape)
                plt.imsave("{}/register_image_{}.png".format(directory,iter),new_image[0,:,:,0])
    
            iter +=1
    
            start_image = tf.constant(new_image)    
    
        loss_total, elastic_loss, alphas, ims_tf = make_loss_multi_im(images , .7)
    
        #optimizer
        global_step = tf.Variable(0, trainable=False, name='global_step')
        optimizer = tf.train.AdamOptimizer(learning_rate = 0.7)
        optimizer = optimizer.minimize(loss_total, global_step=global_step)
    
        #make summaries
        for image in images[1:]:
            image.make_summaries()
        tf.summary.scalar("loss", loss_total)
        merged = tf.summary.merge_all()
        #intitialize
    
        init_op = tf.global_variables_initializer()
        #
    
        ##run
        steps = 50
        loss_ = []
    
        time_start = time.time()
        with tf.Session() as sess:
    
            sess.run(init_op)
    
            train_writer = tf.summary.FileWriter('{}/train'.format(directory), sess.graph)
    
            ims_eval = sess.run(ims_tf)
            individual_plots(ims_eval[:,:,:,0], title="raw_images", directory=directory)


            registration_to_multi_color_log(ims_eval[:,:,:,0], title="registration_before_correction")
    
            for step in range(steps):
    
                (_, loss_eval, elastic_loss_total_eval) = sess.run([optimizer, loss_total, elastic_loss])
    
                if step%10 == 0:
                    (summary_eval) = sess.run(merged)
                    train_writer.add_summary(summary_eval, step)
    
                    print("loss at step {} is {}. Elastic {}".format(step, loss_eval, elastic_loss_total_eval))
                    loss_.append(loss_eval)
    
            (ims_eval, _) = sess.run([ims_tf, loss_total])
    
            for image in images[1:]:
                image.plot_quiver(sess)
    
        time_end = time.time()
    
    print("runtime: {}".format(time_end-time_start))
    
    plt.figure()
    loss_plot = plt.plot(loss_)
    plt.savefig("loss_plot.png")
    
    registration_to_multi_color_log( ims_eval[:,:,:,0] , title = "registration_after_correction", directory = directory)
    registration_compound(ims_eval[:,:,:,0], directory=directory)


if __name__ == "__main__":
    align_images("run_data_2")