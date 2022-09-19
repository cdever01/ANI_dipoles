import torch
import torchani
from torchani.units import hartree2kcalmol


#import ../ensemble_tools.py as et
#from ../ensemble_tools.py import ANIModelDipole_eA
#import models
#from models import ANIModelDipole_eA

import numpy as np

import matplotlib as mpl
mpl.use('Agg') # Must be before importing matplotlib.pyplot or pylab!

import matplotlib.pyplot as plt
import re

import sys
sys.path.append('/data/cdever01/torch_train/dipol_training/sample_net/ANI_dipoles/')
#sys.path.insert(1, '../model_tools/')

import model_tools.read_n_write as rnw
import model_tools.ensemble_tools as et
from model_tools.ensemble_tools import ANIModelDipole_eA


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
species_to_tensor = torchani.utils.ChemicalSymbolsToInts(['H','C','N','O','F','S','Cl'])


def get_Orca_E(f):
        string = open(f, 'r').read()
        regex=re.compile("FINAL SINGLE POINT ENERGY\s*([-]?\d*\.\d*)")
        E=regex.findall(string)
        return np.array(E,dtype=float)



#logs='../../QEQNNP/charge_dist_logs/split/rep/damp/'
logs = '../ANI-2dx/'
res_files=['mem0/','mem1/','mem2/','mem3/','mem4/','mem5/','mem6/','mem7/']
members=[]
for net in range(len(res_files)):
    print(net)
    const_path = logs+res_files[net]+'rHCNOFSCl-infR_16-3.8A_a4-8.params'
    consts = torchani.neurochem.Constants(const_path)
    aev_computer = torchani.AEVComputer(**consts,use_cuda_extension=True).to(device)
    sae_file = 'sae_wb97m_tz.dat'
    energy_shifter = torchani.neurochem.load_sae(logs+res_files[net]+sae_file)

    pt_file=logs+res_files[net]+'/best.pt'
    loaded_file = torch.load(pt_file)
    nn = et.build_model_sp(aev_computer, ANI='2x', cd=True, nmax=loaded_file['nmax'])
    nn.load_state_dict(loaded_file['model'],strict=False)
    members.append(nn)

sae=loaded_file['self_energies']
members = torch.nn.ModuleList(members)
ens=et.Ensemble(members)









f='dimer_starting_coor.xyz'

step=np.array([[-0.1, 0.0,0.0],
              [-0.1, 0.0, 0.0],
              [-0.1, 0.0, 0.0],
              [0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0]])

n_c=0




X, S, Na, ct = rnw.read_xyz(f)
S=S[0]
distances=[]
Xs=[]
c=0
for i in range(0,75):
    Xs.append(X[0]+i*step)
    distances.append(np.abs(Xs[c][0][0]-Xs[c][3][0]))
    c+=1
Xs=np.array(Xs)
    
rnw.write_xyz('water_scan'+str(n_c)+'.xyz', Xs, [S]*len(Xs), cmt='', aw='w')

coordinates=torch.from_numpy(Xs).float().to(device)
coordinates=coordinates.clone().detach().requires_grad_(True)
species = species_to_tensor(''.join(S)).unsqueeze(0).to(device)
mol_sae=sae[species].sum()



species=species.repeat(coordinates.shape[0],1)
_, energy, dipole, coulomb, charges, ens_by_mem, dip_by_mem  = ens((species, coordinates))

ani_energy=energy+mol_sae
ani_energy = ani_energy - max(ani_energy)

atomic_energy = ani_energy-coulomb


ani_energy = hartree2kcalmol(ani_energy)
atomic_energy = hartree2kcalmol(atomic_energy)
coulomb = hartree2kcalmol(coulomb)


dft_energy = rnw.get_Orca_E_array('water_scan_dft.out')
dft_energy = hartree2kcalmol(np.array(dft_energy))
dft_energy -= max(dft_energy)

ani_energy = ani_energy.cpu().detach().numpy()
atomic_energy = atomic_energy.cpu().detach().numpy()
coulomb = coulomb.cpu().detach().numpy()

np.save('ani_energy.npy',ani_energy)
plt.plot(distances, ani_energy, label='ANI-2dx Energy')
#plt.plot(distances, atomic_energy, label='Atomic Energy')
#plt.plot(distances, coulomb, label='Coulomb Energy')
plt.plot(distances, dft_energy, label='DFT Energy')

plt.legend(fontsize=14)
plt.xlabel('O-O Distance', fontsize=16)
plt.ylabel('Energy (kcal/mol)', fontsize=16)
#plt.show()
plt.savefig('water_scan')
 
