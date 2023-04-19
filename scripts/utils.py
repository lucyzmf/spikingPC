# %%
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy
import os
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# function to transform dataset
def image_to_sequence(data, sequence_length, normalise=True):
    """transforms image data into image sequence data 

    Args:
        data (tensor): data tensor, has shape n*dim*dim
        sequence_length (int): number of frames per sequence 
        normalise (boolean): whether to apply normalisation to images or not 

    Returns:
        tensor: transformed new data, has shape n*t*dim*dim
    """
    if normalise:
        data = normalise_inputs(data.unsqueeze(1).double())
    # flatten the image dims
    data = data.view(len(data), -1).double()
    # create new t dim 
    new_data = data.unsqueeze(1).repeat(1, sequence_length, 1)

    return new_data  # shape n*seq len*n pixels


def normalise_inputs(data_tensor):
    """normalise image sequences 

    Args:
        data_tensor (tensor): sequence image tensor, n*n pixels 

    Returns:
        tensor: normalised image data
    """
    norm = transforms.Normalize((0.5), (0.5))
    normalised_data = norm(data_tensor)

    return normalised_data


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def model_save(fn, model, criterion, optimizer):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)


def model_result_dict_load(fn):
    """load tar file with saved model

    Args:
        fn (str): tar file name

    Returns:
        dict: dictornary containing saved results
    """
    with open(fn, 'rb') as f:
        dict = torch.load(f)
    return dict


def save_checkpoint(state, is_best, prefix, filename='_rec2_bias_checkpoint.pth.tar'):
    print('saving at ', prefix + filename)
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + '_rec2_bias_model_best.pth.tar')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# %%
def plot_distribution(param_names, param_dict, keyword):
    """plot distribution of given parameters

    Args:
        param_names (list): list containing param names 
        param_dict (dict): list containing param names and values 
        keyword (str): str of what param to plot, eg. weight, tau
    """

    # find where weight params are stored 
    for name in param_names:
        if keyword in name:
            plt.hist(param_dict[name])
            plt.title(name)
            plt.show()


# %%
def plot_spike_heatmap(spikes):
    """given array of spikes plot heat map along time axis for each neuron 

    Args:
        spikes (np.array): 2d np array containing spike matrices for t time steps, shape num neurons * t 
    """

    fig, ax = plt.subplots(figsize=(5, 20))
    sns.heatmap(spikes, ax=ax)
    plt.show()


# %%
def compute_energy_consumption(all_spikes, weights, alpha=1 / 3):
    """compute the energy consumption of sequence given a batch of spike records and recurrent weights
    of network

    Args:
        all_spikes (np.array): np array containing spike records batch*neuron num*time_steps
        weights (np.array): recurrent weights 
        alpha (float, optional): weighting between spike and synaptic 
            transmition in energy computation. Defaults to 1/3.

    Returns:
        e: energy consumption, np array shaped batch*time_steps, each value is mean energy per sample per time step
    """

    # take mean along dim 1 to avg over all neurons 
    mean_spike = all_spikes.mean(axis=1)  # shape batch*t
    mean_syn_trans = []  # shape batch*t
    for i in range(all_spikes.shape[-1]):
        synp_trans = all_spikes[:, :, i] @ weights  # spikes * recurrent weights
        mean_syn_trans.append(synp_trans.mean(axis=1))
    mean_syn_trans = np.stack(mean_syn_trans).T

    e = alpha * mean_spike + (1 - alpha) * mean_syn_trans
    return mean_spike.T, mean_syn_trans.T, e.T  # final shape t*batch


# %%
def get_internal_drive(spikes, weights):
    """get internal drive per neuron for 2d visualisation 

    Args:
        spikes (np.array): spiking record, neuron*time_steps
        weights (np.array): weight matrix 

    Returns:
        np.array: array containing internal drive at each t
    """
    drive = []
    for i in range(spikes.shape[-1]):
        synp_trans = spikes[:, i] @ weights
        drive.append(synp_trans)

    return np.stack(drive)


# %%
def get_internal_drive_fc(spikes, weights):
    """get internal drive per neuron for 2d visualisation
    spiking projection back to image domain

    Args:
        spikes (np.array): spiking record, neuron*time_steps
        weights (np.array): weight matrix of layer1, contains rec and fc weights

    Returns:
        np.array: array containing internal drive at each t
    """
    drive = []
    d = len(weights)
    rec_weights = weights[:, d:]
    fc_weights = weights[:, :d]

    for i in range(spikes.shape[-1]):
        synp_trans = spikes[:, i] @ rec_weights @ fc_weights.T
        drive.append(synp_trans)

    return np.stack(drive)


# %%
def get_spikes(hiddens):
    """get all spikes from hidden states for the entire sequence 

    Args:
        hiddens (list): list containing hidden states at each time step 
    """
    spikes_all = []
    for i in range(len(hiddens)):
        spikes_all.append(hiddens[i][0][1].detach().cpu().numpy())

    spikes_all = np.stack(spikes_all).transpose((1, 2, 0))
    print(spikes_all.shape)

    return spikes_all


# %%

def normalize(tensor):
    """normalise batch data 

    Args:
        tensor (tensor): b * input dim 
    """
    mean = tensor.mean(dim=1).unsqueeze(dim=1)
    std = tensor.std(dim=1).unsqueeze(dim=1)
    # mean = torch.full(tensor.size(), 0.5)
    # std = torch.full(tensor.size(), 0.5)
    tensor = (tensor - mean) / std

    return tensor


# %%

def shift_input(i, T, data):
    if i < T / 4 and i % 2 == 0:
        data = torch.roll(data, i, -1)
    elif T / 4 <= i < T / 2 and i % 2 == 0:
        data = torch.roll(data, int(T / 2 - i), -1)
    elif T / 2 <= i < 3 * T / 4 and i % 2 == 0:
        data = torch.roll(data, -int(i - T / 2), -1)
    elif i % 2 == 0:
        data = torch.roll(data, i - T, -1)

    return data


# %%
# get all hidden states
def get_all_analysis_data(trained_model, test_loader, device, IN_dim, T, conv=None, batch_no=None, occlusion_p = None, log=True, gaussian_noise=True):
    trained_model.eval()
    test_loss = 0
    correct = 0

    if log == True:
        hiddens_all_ = []
        data_all_ = []  # get transformed data 

    preds_all_ = []  # predictions at all timesptes

    # for data, target in test_loader:
    for i, (data, target) in enumerate(test_loader):
        if batch_no == i:
            break
        b, c, h, w = data.size()

        if log == True:
            data_all_.append(data.data)
        data, target = data.to(device), target.to(device)
        if conv is None:
            data = data.view(-1, IN_dim)
            if occlusion_p is not None:
                data += occlusion_p
                # print(data.mean())

        with torch.no_grad():
            trained_model.eval()
            hidden = trained_model.init_hidden(data.size(0))

            log_softmax_outputs, hidden = trained_model.inference(data, hidden, T)
            if log == True:
                hiddens_all_.append(hidden)

            test_loss += F.nll_loss(log_softmax_outputs[-1], target, reduction='sum').data.item()

            pred = log_softmax_outputs[-1].data.max(1, keepdim=True)[1]
            preds_all_.append(pred)

        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        torch.cuda.empty_cache()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        test_acc))

    preds_all_ = torch.stack(preds_all_).flatten().cpu().numpy()

    if log == True:
        data_all_ = torch.stack(data_all_).reshape(data.size(0)*batch_no, h, w)
        r = [hiddens_all_, preds_all_, data_all_, test_acc]
    else: 
        r = [preds_all_, test_acc]

    return r


# %%
# %%
# get all hiddens and corresponding pred, target, and images into dict

def get_states(hiddens_all_: list, idx: int, hidden_dim_: int, batch_size, T=20, num_samples=10000):
    """
    get a particular internal state depending on index passed to hidden
    :param hidden_dim_: the size of a state, eg. num of r or p neurons
    :param T: total time steps
    :param hiddens_all_: list containing hidden states of all batch and time steps during inference
    :param idx: which index in h is taken out
    :return: np array containing desired states
    """
    all_states = []
    for batch_idx in range(len(hiddens_all_)):  # iterate over batch
        batch_ = []
        for t in range(T):
            seq_ = []
            for b in range(batch_size):
                seq_.append(hiddens_all_[batch_idx][t][idx][b].detach().cpu().numpy())
            seq_ = np.stack(seq_)
            batch_.append(seq_)
        batch_ = np.stack(batch_)
        all_states.append(batch_)

    all_states = np.stack(all_states)

    return all_states.transpose(0, 2, 1, 3).reshape(num_samples, T, hidden_dim_)


# %%
# log seq spiking pattern during training
def plot_spiking_sequence(hidden, target, sample_no=0):
    """
    given t*batch hidden, data, and target, plot one sequence of spiking for logging in wandb during test function call
    :param sample_no: which sample to take in batch
    :param hidden: list containing batch hiddens
    :param target: b*t target for image
    :return: fig object for logging in wandb
    """
    fig, axes = plt.subplots(3, 10, figsize=(40, 3))
    for t in range(10):  # num of time steps
        layer1_spk = hidden[t][1][sample_no].detach().cpu().numpy()
        layer2_spk = hidden[t][5][sample_no].detach().cpu().numpy()
        out_mem = hidden[t][-1][sample_no].detach().cpu().numpy()

        # plot p spiking
        axes[0][t].imshow(layer1_spk.reshape(28, 28))
        axes[0][t].axis('off')
        axes[0][t].set_title('target: %i' % target[sample_no].item())

        # plot r spiking
        axes[1][t].imshow(layer2_spk.reshape(16, 16))
        axes[1][t].axis('off')

        # plot out mem potential 
        pos2 = axes[2][t].imshow(out_mem.reshape(2, 5))
        axes[2][t].axis('off')
        fig.colorbar(pos2, ax=axes[2][t], shrink=0.5)
    
    plt.tight_layout()

    return fig



# %%
