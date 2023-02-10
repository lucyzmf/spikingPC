import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, images: torch.Tensor,
                 labels: torch.Tensor,
                 sequence_len: int,
                 random_switch: bool,
                 switch_time: list,
                 num_switch: int):
        # image_seq, label_seq = create_sequences(images, labels, sequence_len, random_switch, switch_time, num_switch)
        self.image_data = images
        self.label_data = labels
        self.seq_len = sequence_len
        self.random_switch = random_switch
        self.switch_time = switch_time
        self.num_switch = num_switch
        
        self.seq_idx = create_sequences(images, labels, sequence_len, random_switch, switch_time, num_switch)

    def __getitem__(self, idx):
        if self.random_switch:
            # randomly select switch time from t=1 on
            t_switch = np.random.choice(np.arange(1, self.sequence_len), size=self.num_switch, replace=False).tolist()
        else:
            t_switch = self.switch_time
            
        # get img index used in a sequence
        img_idx = self.seq_idx[idx].tolist()
        image_seq = sample_to_seq(self.image_data[img_idx], self.seq_len, t_switch)
        label_seq = sample_to_seq(self.label_data[img_idx], self.seq_len, t_switch)
        return image_seq, label_seq

    def __len__(self):
        return len(self.seq_idx)


def create_sequences(
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
    

    n_samples = len(labels)
    img_dim = tuple(images[0].size())

    # find indices by class
    max_num_sequences = int(n_samples / (num_switch + 1))
    # empty tensor containing indices of images for sequence construction
    sequence_indices = torch.zeros((max_num_sequences, num_switch + 1))
    randomised_indices = np.random.permutation(n_samples)
    mask = []  # selected index

    # create torch tensors that
    image_sequences = torch.zeros((max_num_sequences, sequence_len,) + img_dim)
    label_sequences = torch.zeros((max_num_sequences, sequence_len))

    for s in range(max_num_sequences):
        if s%1000 == 0:
            print(str(s) + 'sequences sampled')
        available_idx = np.setdiff1d(randomised_indices, mask, assume_unique=True)
        selected_idx = np.random.choice(available_idx, size=(num_switch + 1), replace=False)
        # if there's repeat resample
        while len(np.unique(labels[selected_idx])) < (num_switch + 1):
            selected_idx = np.random.choice(available_idx, size=(num_switch + 1), replace=False)
        # update indices that have been selected
        mask = np.concatenate((mask, selected_idx))
        sequence_indices[s, :] = torch.tensor(selected_idx)

        # create sequences
        # sequence_img_sample = sample_to_seq(images[selected_idx], sequence_len, t_switch)
        # sequence_label_sample = sample_to_seq(labels[selected_idx], sequence_len, t_switch)

        # image_sequences[s, :, :, :] = sequence_img_sample
        # label_sequences[s, :] = sequence_label_sample

    return sequence_indices # image_sequences, label_sequences


def sample_to_seq(sample: torch.Tensor, seq_len: int, switch_t: list):
    """
    expand from images into sequences given sequence length and switch time
    :param sample: selected images or used in seq
    :param seq_len: total seq length
    :param switch_t: when to change stimulus
    :return: one sample sequence, tensor by shape seq_len*h*w or if targets seq_len
    """
    ts = [0] + switch_t + [seq_len]
    sequence = []
    for i in range(len(ts) - 1):  # iterate through switch_t elements
        # check whether samples are images or targets
        if len(sample[0].size())>1: 
            sequence.append(sample[i].repeat(int(ts[i + 1] - ts[i]), 1, 1))
        else:
            sequence.append(sample[i].repeat(int(ts[i + 1] - ts[i]), 1))

    sequence = torch.vstack(sequence)

    assert len(sequence) == seq_len

    return sequence.squeeze()
