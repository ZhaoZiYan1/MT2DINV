'''
Data loading (.mat v7/v7.3 compatibility) and Gaussian normalization utilities
'''
import torch
import numpy as np
import scipy.io
import h5py
#################################################
# .mat data loading
#################################################
def loadmat_fields(file_path, fields):
    """
    Support both v7 (scipy) and v7.3 (HDF5) .mat files and load multiple fields at once
    Arrays read from v7.3 files have reversed dimension order relative to MATLAB and must be transposed accordingly

    Args:
        file_path : Path to the .mat file
        fields    : List of field names
    Returns:
        Dictionary mapping field names to ndarrays
    """
    try:
        data = scipy.io.loadmat(file_path)
        return {k: np.asarray(data[k]) for k in fields}
    except NotImplementedError:
        # v7.3 (HDF5) format
        out = {}
        with h5py.File(file_path, 'r') as f:
            for k in fields:
                x = np.asarray(f[k])
                x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))
                out[k] = x
        return out
#################################################
# Gaussian normalization module for multichannel outputs
#################################################
class GaussianNormalizer_out(object):
    # Gaussian normalizer for multichannel outputs
    def __init__(self, x, flag='Point', eps=0.00001):
        self.n_out = x.size()[-1]  # Number of channels
        self.eps = eps  # Small constant to prevent division by zero
        # Compute the mean and standard deviation
        if flag == 'Point':
            # Global normalization: compute the mean and standard deviation of each channel i over all samples and spatial locations
            # The resulting self.mean has shape (n_out,)
            self.mean = torch.stack([torch.mean(x[...,i]) for i in range(self.n_out)])
            self.std = torch.stack([torch.std(x[...,i])  for i in range(self.n_out)])
        elif flag == 'Unit':
            # Unit-wise normalization: preserve spatial dimensions and average only over the batch dimension
            # Typically used for fixed grids
            self.mean = torch.stack([torch.mean(x[...,i],0) for i in range(self.n_out)])
            self.std = torch.stack([torch.std(x[...,i],0)  for i in range(self.n_out)])
        else:
            raise KeyError("invlid flag, must 'Point' or 'Unit'")
    def encode(self, x):
        # Encoding: physical space -> normalized space
        # Formula: x = (x - mean) / std
        # Process each channel independently (clone prevents modification of the caller's original data)
        x = x.clone()
        for i in range(self.n_out):
            mean = self.mean[i,...]
            std = self.std[i,...]
            x[...,i] = (x[...,i] - mean) / (std + self.eps)
        return x
    def decode(self, x, sample_idx=None):
        # Decoding: normalized space -> physical space (for validation and visualization)
        # Formula: x = x * std + mean
        if sample_idx is None:
            # Restore values using global statistics
            std = self.std + self.eps # n
            mean = self.mean
        else:
            # (Less commonly used) Restore values using sample-specific statistics
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]
        # Restore each channel independently (clone prevents modification of the caller's original data)
        x = x.clone()
        for i in range(self.n_out):
            x[...,i] = (x[...,i] * std[i,...]) + mean[i,...]
        return x
    def to(self,device):
        # Move normalization statistics to the GPU/CPU
        self.mean = self.mean.to(device)
        self.std  = self.std.to(device)
class GaussianNormalizer(object):
    # Gaussian normalizer for single-channel data
    def __init__(self, x, flag='Point', eps=0.00001):
        self.eps = eps

        if flag == 'Point':
            # Compute global mean and standard deviation (scalars)
            self.mean = torch.mean(x)
            self.std = torch.std(x)
        elif flag == 'Unit':
            # Compute statistics independently at each spatial location (preserving spatial structure)
            self.mean = torch.mean(x, 0)
            self.std = torch.std(x, 0)
        else:
            raise KeyError("invlid flag, must 'Point' or 'Unit'")
    def encode(self, x):
        # Z-score normalization
        x = (x - self.mean) / (self.std + self.eps)
        return x
    def decode(self, x, sample_idx=None):
        # Restore original values
        x = (x * (self.std + self.eps)) + self.mean
        return x
    def to(self,device):
        self.mean = self.mean.to(device)
        self.std  = self.std.to(device)
    def cuda(self):
        self.mean = self.mean.cuda()
        self.std  = self.std.cuda()
    def cpu(self):
        self.mean = self.mean.cpu()
        self.std  = self.std.cpu()
