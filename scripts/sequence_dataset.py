import torch 
import torchvision
from torch.utils.data import Dataset
import numpy as np

# change at fixed or random step 
# the class it changes into is random or fixed (ie, always ascending or random)
# only one switch 

# load mnnist 
# create indices for the entire dataset 
# pass indices to loaded mnist 
# create batches 

class SequenceDataset(Dataset):
    def __init__(self, images, labels, sequence_len, random_switch):
        data = create_sequences(self, images, labels, sequence_len, random_switch)
        
        

    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.images[idx]      
        image = self.transform(np.array(image))
        return image, label

    
    def __len__(self):
        return len(self.labels)

    def create_sequences(
        self, 
        images: torch.Tensor, 
        labels: torch.Tensor, 
        sequence_len: int, 
        random_switch: bool, 
        switch_time: list, 
        num_switch: int):
        """create image sequence 

        Args:
            images (torch.Tensor): images used to create sequences
            labels (torch.Tensor): corresponding labels 
            sequence_len (int): length of sequence created
            random_switch (bool): whether the switch in sequence happen randomly or 
            switch_time (list): provided switch time (can be list of len 1)
            num_switch (int): number of switches in the whole sequence 
        """
        if random_switch: 
            t_switch = np.random.choice(np.arange(sequence_len), size=num_switch, replace=False) 
        else: 
            t_switch = switch_time 

        n_samples = len(labels)
        
        image_idx_byclass = []
        for i in range(len(torch.unique(labels))):
            image_idx_byclass.append(torch.arange(n_samples)[labels==i])
        
        # find indices by class 
        max_num_sequences = int(n_samples / num_switch) 
        sequence_indices = torch.zeros((max_num_sequences, num_switch+1))  # empty tensor containing indices of images for sequence construction

        

