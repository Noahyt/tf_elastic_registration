'''tensorflow_field supervises image registration using an elastic_image_field.'''


import collections
import numpy as np
import tensorflow as tf
import time
import os

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import matplotlib.pyplot as plt

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


def load_n_images(np_image_array):
    images = []
    for image_iter in range(np_image_array.shape[0]):
        im_ = tf.Variable(np_image_array[image_iter], trainable=False)
        images.append(im_)
    return images

def make_loss_multi_im(elastic_image_field, field_ims, elastic_weight = 1, coherence_weight = 1):

    with tf.variable_scope("loss"):

        elastic_image_field.make_registration_loss()
        mse_loss = elastic_image_field.get_registration_loss()

        elastic_loss = []

        for image in field_ims[1:]:
            image.init_warp_loss(.01)
            elastic_loss.append(image.get_warp_loss())

        elastic_loss_total = tf.reduce_mean(tf.squeeze(tf.stack(elastic_loss)))

        coherence_loss = elastic_image_field.make_coherence_loss()

        # can be modified to weight control points more

        total_loss = mse_loss +  elastic_loss_total * elastic_weight + coherence_loss * coherence_weight

        loss_dict = {
            'mse_loss': mse_loss,
            'coherence_loss': coherence_loss,
            'elastic_loss': elastic_loss,
            'total_loss': total_loss
        }

    return total_loss, loss_dict

class scale_tuner():

    def __init__(self, initial_scale, alpha = 2):
        self.scale = initial_scale
        self.alpha = alpha
        self.running_loss = np.zeros([1])
        self.max_derivative = 0
        self.step_since_last_change = 0

    def update_loss(self, loss):
        self.running_loss = np.concatenate((np.array([loss]), self.running_loss), axis=0)
        print(self.running_loss)
        self.running_loss = self.running_loss[0:2]
        self.step_since_last_change += 1
        if self.step_since_last_change > 1:
            self.calculate_derivatives()

    def calculate_derivatives(self):
        derivative = self.running_loss[1] - self.running_loss[0]
        if derivative > self.max_derivative:
            print("setting new max derivative {}".format(derivative))
            self.max_derivative = derivative
        elif derivative < .1 * self.max_derivative:
            self.reduce_scale()
            self.reset_running_loss()

    def reduce_scale(self):
        self.scale = self.scale * 1 / self.alpha
        print("updating scale.  now {}".format(self.scale))

    def get_scale(self):
        return self.scale

    def reset_running_loss(self):
        self.running_loss = np.zeros([3])
        self.step_since_last_change = 0
        self.max_derivative = 0

def align_images(directory, hparams = None, save_figs = False):

    if not os.path.exists(directory):
        os.makedirs(directory)

    data_temp = np.load('heart_rotation.npy').astype(np.float32)

    initial_guess = np.load('heart_rotation_ig.npz')

    initial_rotations = initial_guess["rotations"]
    initial_translations = initial_guess["translations"]


    if data_temp.shape[1] % 2 == 1:
        data_temp = data_temp[:,:-1,:]
    if data_temp.shape[2] % 2 == 1:
        data_temp = data_temp[:,:,:-1]

    ims_np = data_temp #data_temp[::4,:,:576]
    initial_rotations = initial_rotations
    initial_translations = initial_translations

    print(ims_np.shape)
    print(initial_rotations)
    print(initial_translations.shape)

    ##load ims
    graph = tf.Graph()
    
    with graph.as_default():
    
        images = load_n_images(ims_np)

        num_points = (3, 4) # minimum 2 in any dimension (defaults to corners)

        eif = elastic_image_field([1000, 1200])

        eif.load_image(images[0],
                       initial_rotation=initial_rotations[0],
                       initial_translation=initial_translations[0],
                       control_points=num_points,
                       use_as_base=True)


        for iter, image in enumerate(images[1:]):

            eif.load_image(image,
                           initial_rotation= initial_rotations[iter+1],
                           initial_translation = initial_translations[iter+1],
                           control_points = num_points)

        scale = eif.warp()

        warped_ims = tf.squeeze(tf.stack(eif.get_warped_ims()))
        field_images = eif.get_field_ims()

        loss_total, loss_dict = make_loss_multi_im(eif, field_images, 0, coherence_weight=hparams.coherence_weight)

        #optimizer
        with tf.variable_scope("optimizers"):
            global_step = tf.Variable(0, trainable=False, name='global_step')

            cartesian_optimizer = tf.train.AdamOptimizer(learning_rate = hparams.learning_rate).minimize(loss_total, global_step=global_step, var_list=eif.get_translation_variables()+eif.get_rotation_variables())
            elastic_optimizer = tf.train.AdamOptimizer(learning_rate= hparams.learning_rate).minimize(loss_total, global_step=global_step, var_list=eif.get_warp_variables())

        with tf.variable_scope("summary_ops"):
            #make summaries
            # for image in images[1:]:
            #     image.make_summaries()

            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            tf.summary.scalar("loss", loss_total)
            merged = tf.summary.merge_all()

        #intitialize
        with tf.variable_scope("init_ops"):
            init_op = tf.global_variables_initializer()
    
        ##run
        steps = hparams.num_steps
        loss_ = []

        st = scale_tuner(hparams.initial_scale, alpha = hparams.scale_tuner_alpha)

        with tf.Session() as sess:
    
            sess.run(init_op)
    
            train_writer = tf.summary.FileWriter('{}/train'.format(directory), sess.graph)

            ims_eval = sess.run(warped_ims, feed_dict={scale: [1]})

            if save_figs:
                individual_plots(ims_eval[:,:,:,0], title="raw_images", directory=directory)
                registration_to_multi_color_log(ims_eval[:,:,:,0], title="registration_before_correction", directory= directory)
                registration_to_multi_color_log(ims_eval[:, :, :, 1], title="registration_before_correction_outlines",
                                                directory=directory)

            time_start = time.time()

            for step in range(steps):


                if step%100 ==0:
                    (_, loss_dict_eval) = sess.run([cartesian_optimizer, loss_dict],
                                                                       options=run_options,
                                                                       run_metadata=run_metadata,
                                                                       feed_dict={scale: [st.get_scale()]})
                    train_writer.add_run_metadata(run_metadata, 'step%d' % step)
                else:
                    (_, loss_dict_eval) = sess.run([cartesian_optimizer, loss_dict],
                                                                       feed_dict={scale:[st.get_scale()]})

    
                if step%10 == 0:
                    # scale_val = start_scale_val - max((step - 20) * (start_scale_val - 4) / steps, 0)
                    st.update_loss(loss_dict_eval['total_loss'])
                    print("loss at step {} is {}".format(step, loss_dict_eval))
                    loss_.append(loss_dict_eval['total_loss'])

                    # if step%50 ==0:
                    #     (summary_eval) = sess.run(merged, feed_dict={scale: [1]})
                    #     train_writer.add_summary(summary_eval, step)

            time_end = time.time()

            (ims_eval, _) = sess.run([warped_ims, loss_total], feed_dict={scale:[1]})

            (final_translations, final_rotations, final_loss) = sess.run([eif.get_translations(), eif.get_rotations(), loss_dict['total_loss']], feed_dict={scale:[1]})

            # for image in images[1:]:
            #     image.plot_quiver(sess)3
    runtime = time_end - time_start
    print("runtime: {}".format(runtime))

    if save_figs:
        plt.figure()
        plt.plot(loss_)
        plt.savefig("{}/loss_plot.png".format(directory))

        plt.figure()
        plt.plot(final_rotations)
        plt.savefig("{}/calculated_rotations.png".format(directory))

        plt.figure()
        plt.plot(*np.squeeze(np.stack(final_translations, axis = 0)).T)
        plt.savefig("{}/calculated_translations.png".format(directory))

        registration_to_multi_color_log( ims_eval[:,:,:,0] , title = "registration_after_correction", directory = directory)
        registration_to_multi_color_log( ims_eval[:,:,:,1] , title = "registration_after_correction_outlines", directory = directory)

        registration_compound(ims_eval[:,:,:,0], directory=directory)

    return final_loss

def calculate_accuracy():
    '''calculates accuracy based on difference between estimated alignment and true alignment'''

    pass


if __name__ == "__main__":

    hpms = collections.namedtuple('hparams',['name', 'num_steps', 'learning_rate', 'beta', 'initial_scale','scale_tuner_alpha', 'coherence_weight' ])

    hparams_run = hpms(name='test', num_steps = 50, learning_rate=1.1, beta=1, initial_scale=35, scale_tuner_alpha = 1.1, coherence_weight=1)

    final_loss = align_images(hparams_run.name, hparams_run, save_figs=True)

    print(final_loss)