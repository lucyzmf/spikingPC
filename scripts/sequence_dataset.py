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
            image_idx_byclass.append(torch.randperm(torch.arange(n_samples)[labels==i]))
        
        # find indices by class 
        max_num_sequences = int(n_samples / num_switch) 
        sequence_indices = torch.zeros((max_num_sequences, num_switch+1))  # empty tensor containing indices of images for sequence construction
        
        randomised_indices = np.arange(n_samples)

        sequence_indices = randomised_indices[:(max_num_sequences*(num_switch+1))].reshape(max_num_sequences, num_switch+1)
        selected_labels = labels[randomised_indices].reshape(max_num_sequences, num_switch+1)
        
        for i in range(max_num_sequences):
            repeat = 0
            while len(np.unique(selected_labels[i, :])) <= num_switch:  # there are repeats in the sequence
                idx_keep = np.unique(selected_labels[i, :], return_index=True)[1] 
                idx_change = np.setdiff1d(np.arange(num_switch+1), idx_keep) 
                new_sample_idx = randomised_indices[(max_num_sequences*(num_switch+1)+repeat):]
                # update 
                sequence_indices[i, idx_change] = new_sample_idx
                selected_labels[i, idx_change] = labels[sequence_indices[i, ]]
                repeat += 1
        
        


        

