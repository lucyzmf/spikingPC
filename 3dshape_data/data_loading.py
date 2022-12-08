"""
loading 3d shape dataset
"""

import h5py

# %%
# load dataset
import torch
from torch.utils import data

dataset = h5py.File('3dshapes.h5', 'r')
print(dataset.keys())
images = dataset['images']  # array shape [480000,64,64,3], uint8 in range(256)
labels = dataset['labels']  # array shape [480000,6], float64
image_shape = images.shape[1:]  # [64,64,3]
label_shape = labels.shape[1:]  # [6]
n_samples = labels.shape[0]  # 10*10*10*8*4*15=480000

_FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                     'orientation']
_NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10,
                          'scale': 8, 'shape': 4, 'orientation': 15}


# %%
class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.file_path = path
        self.dataset = None
        with h5py.File(self.file_path, 'r') as file:
            self.dataset_len = len(file["images"])
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')

    def __getitem__(self, index):
        image = self.dataset['images'][index]
        label = self.dataset['labels'][index]
        return label, image

    def __len__(self):
        return self.dataset_len


# %%
dataset = H5Dataset('3dshapes.h5')

loader_params = {'batch_size': 100, 'shuffle': True, 'num_workers': 0}

data_loader = data.DataLoader(dataset, **loader_params)

# %%
for i, t in enumerate(data_loader):
    print(i, t)
