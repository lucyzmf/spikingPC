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
    print('saving at ', prefix+filename)
    torch.save(state, prefix+filename)
    if is_best:
        shutil.copyfile(prefix+filename, prefix+ '_rec2_bias_model_best.pth.tar')


def count_parameters(model):
    return sum(p.numel() for p in model.network.parameters() if p.requires_grad)

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
def compute_energy_consumption(all_spikes, weights, alpha=1/3): 
    """compute the energy consumption of sequence given a batch of spike records and recurrent weights
    of network

    Args:
        all_spikes (np.array): np array containing spike records batch*neuron num*T
        weights (np.array): recurrent weights 
        alpha (float, optional): weighting between spike and synaptic 
            transmition in energy computation. Defaults to 1/3.

    Returns:
        e: energy consumption, np array shaped batch*T, each value is mean energy per sample per time step
    """
    
    # take mean along dim 1 to avg over all neurons 
    mean_spike = all_spikes.mean(axis=1) # shape batch*t
    mean_syn_trans = [] # shape batch*t
    for i in range(all_spikes.shape[-1]):
        synp_trans = all_spikes[:, :, i] @ weights # spikes * recurrent weights 
        mean_syn_trans.append(synp_trans.mean(axis=1))
    mean_syn_trans = np.stack(mean_syn_trans).T
    
    e = alpha*mean_spike + (1-alpha)*mean_syn_trans
    return mean_spike.T, mean_syn_trans.T, e.T # final shape t*batch


# %%
def get_internal_drive(spikes, weights):
    """get internal drive per neuron for 2d visualisation 

    Args:
        spikes (np.array): spiking record, neuron*T
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
        spikes (np.array): spiking record, neuron*T
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
def get_spikes(hiddens, out_tensor=True):
    """get all spikes from hidden states for the entire sequence 

    Args:
        hiddens (list): list containing hidden states at each time step 
        out_tensor: return as tensor or np
    """
    spikes_all = []
    for i in range(len(hiddens)):
        spikes_all.append(hiddens[i][0][1].detach().cpu().numpy())

    spikes_all = np.stack(spikes_all).transpose((1, 2, 0))
    print(spikes_all.shape)

    if out_tensor:
        spikes_all = torch.tensor(spikes_all)

    return spikes_all
# %%
def shift_input(i, T, data):
    if i<T/4:
        data = torch.roll(data, i, -1)
    elif i>=T/4 and i<T/2:
        data = torch.roll(data, int(T/2-i), -1)
    elif i>=T/2 and i<3*T/4:
        data = torch.roll(data, -int(i-T/2), -1)
    else:
        data = torch.roll(data, i-T, -1)

    return data
# %%
# function to creat distance map
def creat_dist_map(img_h, img_w):
    x, y = img_w, img_h
    xv, yv = np.meshgrid(np.arange(x), np.arange(y))
    pos = np.vstack([yv.ravel(), xv.ravel()])

    dist_map = np.zeros((y*x, y*x))
    for i in range(x*y):
        relative_dist = []
        ref_point = pos[:, i]
        for j in range(x*y):
            new_point = pos[:, j]
            dist = np.linalg.norm(ref_point - new_point)
            relative_dist.append(dist)

        dist_map[i, :] += relative_dist

    return dist_map


# %%
