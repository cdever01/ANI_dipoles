import torch
import torchani
import sys
import os
import pyanitools as pyt
import numpy as np
import pickle
from itertools import chain
import warnings
from sklearn import linear_model
from torchani.transforms import AtomicNumbersToIndices, SubtractSAE
import pathlib
from pathlib import Path
warnings.filterwarnings('ignore')



member='0'


prm_name = 'rHCNOFSCl-infR_16-3.8A_a4-8.params'

# device to run the training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# batch size
batch_size = 1280

data_path = '/data/cdever01/2x_data/datasets/ANI2x-wb97md3bj-def2tzvpp/'
data_list=[data_path+'ANI-1x-wB97MD3BJ-def2TZVPP.h5', data_path+'ANI-2x_heavy_and_dimers-wB97MD3BJ-def2TZVPP.h5']

batched_data_path = '/data/cdever01/torch_train/dipol_training/batched_2x_3/some_data/'
energy_shifter, self_energies = torchani.neurochem.load_sae('/data/cdever01/torch_train/dipol_training/SAE_o/sae_wb97m_tz.dat', return_dict=True)

gsaes = []
for key in self_energies:
    gsaes.append(self_energies[key])

if not Path(batched_data_path).resolve().is_dir():
    elements = ('H', 'C', 'N', 'O', 'F', 'S', 'Cl')
    # here we use the GSAEs for self energies
    transform = torchani.transforms.Compose([AtomicNumbersToIndices(elements), SubtractSAE(elements, gsaes)])
    torchani.datasets.create_batched_dataset(data_list,
                                                 dest_path=batched_data_path,
                                                 batch_size=batch_size,
                                                 inplace_transform=transform,
                                                  include_properties=['energies', 'species', 'coordinates', 'dipoles'],
                                                  folds=8       #How many sets to split the data into. 
                                                    )
data = torchani.datasets.AniBatchedDataset(batched_data_path, split='training')

print(len(data))
