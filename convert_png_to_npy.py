import imageio
import numpy as np
import matplotlib.pyplot as plt

#load images
face_np = imageio.imread("Mcdonalds.png")
face_np = np.sum(face_np,2)
sz = face_np.shape
print(sz)
face_shift = np.pad(face_np,100,mode='constant')
print(face_shift.shape)
face_shift = face_np[int(sz[0]/2):sz[0],:sz[1]]
print(face_shift.shape)
np.save("test_normal", face_np)
np.save("test_crop", face_shift)
