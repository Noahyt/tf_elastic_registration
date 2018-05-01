import numpy as np
from scipy.ndimage.interpolation import rotate
from skimage.transform import rescale
from scipy.ndimage import gaussian_filter
import scipy.signal as signal
import time
import matplotlib.pyplot as plt

# data_temp = np.load('noah_wrist_manual.npy').astype(np.float32)
data_temp = np.load('heart_rotation.npy').astype(np.float32)


print(data_temp.shape)

data_temp = data_temp/np.amax(data_temp)



def SNR(im):
    r = np.ravel(im)
    m = np.max(r)
    signal = r

    std = np.std(signal)
    return m / std

def phase_correlate_custom(image_1, image_2):
    sz = image_1.get_shape().as_list()

    # calculate hamming window
    ham2d = np.sqrt(np.einsum('i,j->ij',
                              tf.contrib.signal.hamming_window(sz[1]),
                              tf.contrib.signal.hamming_window(sz[2])))

    corr_v = (tf.ifft2d(tf.fft2d(tf.cast(image_1[0, :, :, 0] * ham2d, tf.complex64)) *
                        tf.ifft2d(tf.cast(image_2[0, :, :, 0] * ham2d, tf.complex64))))
    corr_v = tf.real(corr_v)
    max_index = tf.argmax(tf.reshape(corr_v, [-1]))
    max_index = tf.unravel_index(max_index, image_2[0, :, :, 0].shape)

    # convert index shifts from circular to linear
    max_index = tf.where(tf.cast(max_index[0], tf.float32) > (sz[1] / 2), [-1 * sz[1], 0] + max_index, max_index)
    max_index = tf.where(tf.cast(max_index[1], tf.float32) > (sz[2] / 2), [0, -1 * sz[2]] + max_index, max_index)



def find_rotation_and_translation(im_0, im_1, translation=(10, 10), theta=(-5, 5), scale=1.0):
    """Finds displacement and rotation of im_1 relative to im_0.
    """

    sz = im_1.shape
    if sz[0] % 2 == 1:
        im_0 = im_0[:-1]
        im_1 = im_1[:-1]
    if sz[1] % 2 == 1:
        im_0 = im_0[:, :-1]
        im_1 = im_1[:, :-1]

    sz = im_1.shape

    max_vals = {
        'angle': 0,
        'translation': (0, 0)
    }

    best_correlation = 0

    ##downsample ims?
    factor = 1.0 / scale

    im_0 = gaussian_filter(im_0, scale)
    im_1 = gaussian_filter(im_1, scale)

    im_0 = rescale(im_0, factor)
    im_1 = rescale(im_1, factor)

    for rotation_degree in np.arange(theta[0], theta[1] + 1, step=.25):
        # search over rotation degrees
        im_1_rotated = rotate(im_1, rotation_degree, reshape=False)
        im_1_rotated = np.clip(im_1_rotated, 0, a_max=None)

        c = signal.fftconvolve(im_0, im_1_rotated[::-1, ::-1], mode='same')
        # if new max update max_vals
        midpoints = np.array(c.shape) / 2
        peak_location = np.unravel_index(np.argmax(c), c.shape)

        SNR_ = SNR(c[(peak_location[0] - 50) * ((peak_location[0] - 50) > 0): peak_location[0] + 50,
                   (peak_location[1] - 50) * ((peak_location[1] - 50) > 0): peak_location[1] + 50])

        #         print(SNR_)

        if SNR_ > best_correlation:
            best_correlation = SNR_
            translation_a = (peak_location - midpoints) * scale

            max_vals.update({
                'angle': rotation_degree,
                'translation': translation_a
            })

    return max_vals

rotate_angs = []
displacements = []
time_s = time.time()
for image_num in np.arange(8):
    im_a = data_temp[image_num]
    im_b = data_temp[image_num+1]
    out = find_rotation_and_translation(im_a, im_b,theta = (0,15), scale = 4)
    rotate_angs.append(out['angle'])
    displacements.append(out['translation'])

print("total time {}".format(time.time()- time_s))
plt.plot(rotate_angs)
plt.savefig("rotate_angs")

cum_sum_d = np.cumsum(np.array(displacements),axis=0)
cum_sum_d = np.insert(cum_sum_d,0, [0] ,axis = 0)
cum_sum_d = (cum_sum_d  - np.mean(cum_sum_d,axis=0)).astype(np.int)
print(cum_sum_d)
cum_sum_angle = np.cumsum(rotate_angs)
cum_sum_angle = np.insert(cum_sum_angle, 0,[0], axis=0)




# cum_sum_angle = [0,10,20,30,40,50,60,70,80]
cum_sum_angle = cum_sum_angle #* 2 * np.pi / 360
print(cum_sum_angle)

# np.savez("heart_rotation_ig", rotations = cum_sum_angle, translations = cum_sum_d)