
# coding: utf-8
# In[2]:

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate
import time
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import os

from field_image import field_image
from elastic_image_field import elastic_image_field
from ycmap import get_yilei_color_map

def registration_to_multi_color_log(ims,  title = "multi_color_plot", directory = None):
    ims = np.log10(ims * 100 / np.amax(ims) + 1)

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
        im_to_plot = np.log10(ims[im_num]*100/np.amax(ims[im_num])+1)
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
        ims = np.log10(ims * 100 / np.amax(ims) + 1)
    im_sum = np.sum(ims,axis = 0)/ims.shape[0]

    plt.figure()
    plt.imshow(im_sum)
    plt.title("individual_images")
    plt.colorbar()
    plt.savefig(directory + title + ".png")

    return im_sum



''''''


def load_n_images(np_image_array, total_dim=None, pad=0):
    images = []
    for image_iter in range(np_image_array.shape[0]):
        im_ = tf.Variable(np_image_array[image_iter], trainable=False)
        print(im_)
        images.append(field_image(im_, extra_pad = 50, name = "image_{}".format(image_iter)))
    return images

def make_loss_multi_im(elastic_image_field, field_ims, elastic_alpha = 1):

    mse_loss = elastic_image_field.get_registration_loss()

    elastic_loss = []

    for image in field_ims[1:]:
        image.init_warp_loss(.01)
        elastic_loss.append(image.get_warp_loss())

    elastic_loss_total = tf.reduce_mean(tf.squeeze(tf.stack(elastic_loss)))

    #can be modified to weight control points more

    total_loss = mse_loss +  elastic_loss_total * elastic_alpha

    return total_loss, elastic_loss_total


def align_images(directory):

    if not os.path.exists(directory):
        os.makedirs(directory)

    
    data_temp = np.load('heart_rotation.npy').astype(np.float32)
    
    print(data_temp.shape)
    #TODO(ntoyonaga) images dimensions must be even
    im_0 = data_temp[0,:622,:]
    im_1 = data_temp[1,:622,:]
    im_2 = data_temp[2,:622,:]
    im_3 = data_temp[3,:622,:]
    im_4 = data_temp[4,:622,:]
    im_5 = data_temp[5,:622,:]
    im_6 = data_temp[6,:,:]
    im_7 = data_temp[7,:,:]
    im_8 = data_temp[8,:,:]

    # im_0 = rotate(im_0, -40, reshape = False)
    # im_0 = np.clip(im_0,0, a_max= None)
    #
    # im_1 = rotate(im_1, -30, reshape = False)
    # im_1 = np.clip(im_1,0, a_max= None)
    #
    # im_2 = rotate(im_2, -20, reshape = False)
    # im_2 = np.clip(im_2,0, a_max= None)
    #
    # im_3 = rotate(im_3, -10, reshape = False)
    # im_3 = np.clip(im_3,0, a_max= None)
    #
    # im_4 = rotate(im_4, 0, reshape = False)
    # im_4 = np.clip(im_4,0, a_max= None)
    #
    # im_5 = rotate(im_5, 10, reshape = False)
    # im_5 = np.clip(im_5,0, a_max= None)
    #
    # im_6 = rotate(im_6, 20, reshape = False)
    # im_6 = np.clip(im_6,0, a_max= None)
    #
    # im_7 = rotate(im_7, 30, reshape = False)
    # im_7 = np.clip(im_7,0, a_max= None)
    #
    # im_8 = rotate(im_8, 40, reshape = False)
    # im_8 = np.clip(im_8,0, a_max= None)

    # fig, ax = plt.subplots(1, 2)
    #
    # ax[0].imshow(im_4)
    # ax[1].imshow(im_7)

    # [im_4, im_0, im_1, im_2, im_3, im_5, im_6, im_7, im_8]
    
    ims_np = np.stack([im_4,im_3, im_2], axis=0)

    initial_rotations = [0., -.174, -.2]
    ims_np = np.stack([im_4, im_3, im_2], axis=0)

    initial_translation = [
        [0, 0],
        [20, 20],
        [0, 0],
    ]
    
    ##load ims
    graph = tf.Graph()
    
    with graph.as_default():
    
        images = load_n_images(ims_np, total_dim = (900,1200), pad = 0 )

        num_points = (3, 4) # minimum 2 in any dimension (defaults to corners)

        eif = elastic_image_field([800, 1000])

        for iter in range(len(images)):
            images[iter].make_control_points(*num_points)
            images[iter].make_warp_points_and_matrix()
            images[iter].warp()

            eif.load_image(images[iter], initial_rotation=initial_rotations[iter],
                           initial_translation=initial_translation[iter])

        warped_ims = tf.squeeze(tf.stack(eif.get_warped_ims()))
        field_images = eif.get_field_ims()

        eif.make_registration_loss()

        loss_total, elastic_loss = make_loss_multi_im(eif, field_images, .7)

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
    
            ims_eval = sess.run(warped_ims)

            individual_plots(ims_eval[:,:,:,0], title="raw_images", directory=directory)

            registration_to_multi_color_log(ims_eval[:,:,:,0], title="registration_before_correction")
    
            for step in range(steps):
    
                (_, loss_eval, elastic_loss_total_eval) = sess.run([optimizer, loss_total, elastic_loss])
    
                if step%10 == 0:
                    (summary_eval) = sess.run(merged)
                    train_writer.add_summary(summary_eval, step)
    
                    print("loss at step {} is {}. Elastic {}".format(step, loss_eval, elastic_loss_total_eval))
                    loss_.append(loss_eval)
    
            (ims_eval, _) = sess.run([warped_ims, loss_total])


            # for image in images[1:]:
            #     image.plot_quiver(sess)
    
        time_end = time.time()
    
    print("runtime: {}".format(time_end-time_start))
    
    plt.figure()
    loss_plot = plt.plot(loss_)
    plt.savefig("loss_plot.png")
    
    registration_to_multi_color_log( ims_eval[:,:,:,0] , title = "registration_after_correction", directory = directory)
    registration_compound(ims_eval[:,:,:,0], directory=directory)


if __name__ == "__main__":
    align_images("add_rotation_test_1")