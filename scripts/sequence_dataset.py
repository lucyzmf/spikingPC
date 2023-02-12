import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

class SequenceDataset(Dataset):
    def __init__(self, images: torch.Tensor,
                 labels: torch.Tensor,
                 sequence_len: int,
                 random_switch: bool,
                 switch_time: list,
                 num_switch: int,
                 transform=None):
        """
        create sequence dataset from image data
        :param images: images used to create sequences
        :param labels: corresponding labels
        :param sequence_len: length of sequence created
        :param random_switch: whether the change in stimulus happens randomly or at fixed time
        :param switch_time: if not random switch, provide switch time
        :param num_switch: total number of switch times in the sequence
        """

        self.image_data = torch.unsqueeze(images, dim=1)  # unsqueeze for transform
        self.label_data = labels
        self.seq_len = sequence_len
        self.random_switch = random_switch
        self.switch_time = switch_time
        self.num_switch = num_switch
        self.transform = transform

        self.seq_idx = create_sequences(labels, num_switch)
        self.num_samples = len(self.seq_idx)

        print('num of sequences created: %i' % self.num_samples)

    def __getitem__(self, idx):
        if self.random_switch:
            # randomly select switch time from t=1 on
            t_switch = np.random.choice(np.arange(1, self.seq_len), size=self.num_switch, replace=False).tolist()
        else:
            t_switch = self.switch_time

        # get img index used in a sequence
        img_idx = self.seq_idx[idx].tolist()
        image_data_transformed = self.transform(self.image_data[img_idx]).squeeze()  # get rid of channel dim here
        image_seq = sample_to_seq(image_data_transformed, self.seq_len, t_switch)
        label_seq = sample_to_seq(self.label_data[img_idx], self.seq_len, t_switch)
        return image_seq, label_seq

    def __len__(self):
        return len(self.seq_idx)


def create_sequences(
        labels: torch.Tensor,
        num_switch: int):
    """create image sequence idx

    Args:
        labels (torch.Tensor): corresponding labels
        num_switch (int): number of switches in the whole sequence
    """

    n_samples = len(labels)

    # find indices by class
    max_num_sequences = int(n_samples / (num_switch + 1))
    # empty tensor containing indices of images for sequence construction
    sequence_indices = torch.zeros((max_num_sequences, num_switch + 1))
    randomised_indices = np.random.permutation(n_samples)
    mask = []  # selected index

    for s in range(max_num_sequences):
        if s % 1000 == 0:
            print(str(s) + 'sequences sampled')
        available_idx = np.setdiff1d(randomised_indices, mask, assume_unique=True)

        # check towards end of sampling if the remaining targets 
        if (max_num_sequences - s) < 100 and len(np.unique(labels[available_idx])) == 1:
            sequence_indices = sequence_indices[:(s - 1), :]
            break

        selected_idx = np.random.choice(available_idx, size=(num_switch + 1), replace=False)
        # if there's repeat resample
        while len(np.unique(labels[selected_idx])) < (num_switch + 1):
            selected_idx = np.random.choice(available_idx, size=(num_switch + 1), replace=False)
        # update indices that have been selected
        mask = np.concatenate((mask, selected_idx))
        sequence_indices[s, :] = torch.tensor(selected_idx)

    return sequence_indices


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
        if len(sample[0].size()) > 1:
            sequence.append(sample[i].repeat(int(ts[i + 1] - ts[i]), 1, 1))
        else:
            sequence.append(sample[i].repeat(int(ts[i + 1] - ts[i]), 1))

    sequence = torch.vstack(sequence)

    assert len(sequence) == seq_len

    return sequence.squeeze()


class SequenceDatasetPredictable(Dataset):
    def __init__(self, images: torch.Tensor,
                 labels: torch.Tensor,
                 sequence_len: int,
                 random_switch: bool,
                 switch_time: list,
                 num_switch: int,
                 transform=None):
        """
        create sequence dataset from image data
        :param images: images used to create sequences
        :param labels: corresponding labels
        :param sequence_len: length of sequence created
        :param random_switch: whether the change in stimulus happens randomly or at fixed time
        :param switch_time: if not random switch, provide switch time
        :param num_switch: total number of switch times in the sequence
        """

        self.image_data = torch.unsqueeze(images, dim=1)
        self.image_data_trans = self.transform(self.image_data)
        self.label_data = labels
        self.seq_len = sequence_len
        self.random_switch = random_switch
        self.switch_time = switch_time
        self.num_switch = num_switch
        self.transform = transform

        self.num_samples = len(self.label_data)

        print('num of sequences created: %i' % self.num_samples)

    def __getitem__(self, idx):
        if self.random_switch:
            # randomly select switch time from t=1 on
            t_switch = np.random.choice(np.arange(1, self.seq_len), size=self.num_switch, replace=False).tolist()
        else:
            t_switch = self.switch_time

        # get img_idx which is a list containing appropriate indices
        first_label_idx = idx
        first_label = self.label_data[first_label_idx]
        # find index of the second stimulus in sequence
        if first_label != 9:
            second_label_indices = torch.nonzero(self.label_data == (first_label+1)).squeeze()
            second_label_idx = second_label_indices[np.random.randint(len(second_label_indices))].item()
        else:
            second_label_indices = torch.nonzero(self.label_data == 0).squeeze()
            second_label_idx = second_label_indices[np.random.randint(len(second_label_indices))].item()
        seq_indices = [first_label_idx, second_label_idx]

        # transform
        image_data_trans = self.image_data_trans[seq_indices].squeeze()
        image_seq = sample_to_seq(image_data_trans, self.seq_len, t_switch)
        label_seq = sample_to_seq(self.label_data[seq_indices], self.seq_len, t_switch)
        return image_seq, label_seq

    def __len__(self):
        return len(self.label_data)
