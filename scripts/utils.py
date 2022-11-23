# %%
import torch
import numpy 

# function to transform dataset 
def image_to_sequence(data, sequence_length):
    """transforms image data into image sequence data 

    Args:
        data (tensor): data tensor, has shape n*dim*dim
        sequence_length (int): number of frames per sequence 

    Returns:
        tensor: transformed new data, has shape n*t*dim*dim
    """    
    return data.unsqueeze(1).repeat(1, sequence_length, 1, 1)     
    