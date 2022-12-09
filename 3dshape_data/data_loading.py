"""
loading 3d shape dataset
"""
# %%
import h5py
import torch
from torch.utils import data
import numpy as np

# load dataset
# note that images and labels were generated in two separate runs cuz vs crashed when trying to load both

dataset = h5py.File('3dshapes.h5', 'r')
print(dataset.keys())
images = dataset['images'][()]  # array shape [480000,64,64,3], uint8 in range(256)
images = images.transpose((0, 3, 1, 2))
# labels = dataset['labels'][()]  # array shape [480000,6], float64

# save as npy 
np.save('images', images)
# np.save('labels', labels)


# image_shape = images.shape[1:]  # [64,64,3]
# label_shape = labels.shape[1:]  # [6]
# n_samples = labels.shape[0]  # 10*10*10*8*4*15=480000


