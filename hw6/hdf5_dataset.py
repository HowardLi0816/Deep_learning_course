from pathlib import Path

import h5py
import torch
from torch.utils import data

class HDF5Dataset(data.Dataset):
  """Abstract HDF5 dataset
  
  Usage:
    from hdf5_dataset import HDF5Dataset 
    train_set = hdf5_dataset.HDF5Dataset(
      file_path = f"{PATH_TO_HDF5}", data_name = 'xdata', label_name = 'ydata')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True)
    
    ** NOTE: labels is 1-hot encoding, not target label number (as in PyTorch MNIST dataset)
    ** keep in mind when comparing model output to target labels
  
  Input params:
    file_path: Path to the folder containing the dataset
    data_name: name of hd5_file "x" dataset
    data_name: name of hd5_file "y"
  """
  def __init__(self, file_path, data_name, label_name):
    super().__init__()
    self.data = {}
    
    self.data_name = data_name
    self.label_name = label_name

    h5dataset_fp = Path(file_path)
    assert(h5dataset_fp.is_file())
    
    with h5py.File(file_path) as h5_file:
      # iterate datasets
      for dname, ds in h5_file.items():
        self.data[dname] = ds[()]
      
        
  def __getitem__(self, index):
    # get data
    x = self.data[self.data_name][index]
    x = torch.from_numpy(x)

    # get label
    y = self.data[self.label_name][index]
    y = torch.from_numpy(y)
    return (x, y)


  def __len__(self):
    return len(self.data[self.data_name])

