import numpy as np
import scipy.io as sio
import scipy.ndimage as ndimage

ang = np.arange(1,10)

file_name = "wrist_data.mat"
temp_data = sio.loadmat(file_name)
raw_data = temp_data['img_int']

raw_data = np.swapaxes(raw_data,2,0)


print(raw_data[0])


# ### need to rescale to 0-255
# max = np.amax(raw_data)
# scaled_data = (raw_data*255/max).astype(np.uint8)
#
# print(scaled_data.shape)
#
#
# print(scaled_data.shape)
# # zoom_ratio = (512 / 1456) * (28 / 23)
# # im_zoomed = ndimage.zoom(scaled_data, zoom=(1, zoom_ratio, 1), order=1)

np.save("wrist_data", raw_data)