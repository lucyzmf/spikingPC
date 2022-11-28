# %%
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy 
import os
import shutil
import matplotlib.pyplot as plt

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
