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
        image_seq, label_seq = create_sequences(images, labels, sequence_len, random_switch, switch_time, num_switch)
        self.image_data = image_seq
        self.label_data = label_seq
        self.img_datasize = image_seq.size()
        self.label_size = label_seq.size()

    def __getitem__(self, idx):
        image_seq = self.image_data[idx, :, :, :]
        label_seq = self.label_data[idx, :, :]
        return image_seq, label_seq

    def __len__(self):
        return len(self.label_data)


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
    if random_switch:
        # randomly select switch time from t=1 on
        t_switch = np.random.choice(np.arange(1, sequence_len), size=num_switch, replace=False).tolist()
    else:
        t_switch = switch_time

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
        selected_idx = np.random.choice(np.setdiff1d(randomised_indices, mask), size=(num_switch + 1))
        # if there's repeat resample
        while len(np.unique(labels[selected_idx])) < (num_switch + 1):
            selected_idx = np.random.choice(np.setdiff1d(randomised_indices, mask), size=(num_switch + 1))
        # update indices that have been selected
        mask = np.concatenate((mask, selected_idx))
        sequence_indices[s, :] = torch.tensor(selected_idx)

        # create sequences
        sequence_img_sample = sample_to_seq(images[selected_idx], sequence_len, t_switch)
        sequence_label_sample = sample_to_seq(labels[selected_idx], sequence_len, t_switch)

        image_sequences[s, :, :, :] = sequence_img_sample
        label_sequences[s, :] = sequence_label_sample

    return image_sequences, label_sequences


def sample_to_seq(sample: torch.Tensor, seq_len: int, switch_t: list):
    """
    expand from images into sequences given sequence length and switch time
    :param sample: selected images or used in seq
    :param seq_len: total seq length
    :param switch_t: when to change stimulus
    :return: one sample sequence, tensor by shape seq_len*h*w or if targets seq_len
    """
    # check whether samples are images or targets
    if len(sample[0].size()) > 1:
        dims = tuple(sample[0].size())
    else:
        dims = (1,)
    ts = [0] + switch_t + [seq_len]
    sequence = []
    for i in range(len(ts) - 1):  # iterate through switch_t elements
        sequence.append(sample[i].repeat((ts[i + 1] - ts[i],) + dims))

    sequence = torch.vstack(sequence)

    assert len(sequence) == seq_len

    return sequence.squeeze()
