import numpy as np
import scipy.io as sio
import scipy.ndimage as ndimage

ang = np.arange(1,10)
raw_data=np.zeros([9, 1456, 512])

for angle in ang:
    file_name = "../../data/3_1_heart/ang{}.mat".format(angle)
    print(file_name)
    temp_data = sio.loadmat(file_name)

    raw_data[angle-1] = temp_data['rfData_env']

### need to rescale to 0-255
max = np.amax(raw_data)
scaled_data = (raw_data*255/max).astype(np.uint8)

zoom_ratio = (512 / 1456) * (28 / 23)

im_zoomed = ndimage.zoom(scaled_data, zoom=(1, zoom_ratio, 1), order=1)

np.save("heart_rotation", im_zoomed)