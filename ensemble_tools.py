import torch
import torchani
import numpy as np
from torchani.repulsion import RepulsionCalculator, StandaloneRepulsionCalculator
from typing import Tuple, NamedTuple
from torch import Tensor
from torchani.units import ANGSTROM_TO_BOHR


def ea02debye(x):
    return x/0.3934303


def eA2debeye(x):
    return x/0.20819434


class ANIModelDipole_eA(torch.nn.ModuleList):

    """
    coordinates: Angstroms
    damping_b: Angstroms
    Rcr: Angstoms
    dipoles: eA
    energies: Hartrees
    charges: e
    """

    def __init__(self, modules, aev_computer, damping = True, damping_a = -4.0, damping_b = 2.0, repulsion=True, Rcr=5.2, elements=('H', 'C', 'N', 'O')):
        super(ANIModelDipole_eA, self).__init__(modules)
        self.reducer = torch.sum
        self.padding_fill = 0
        self.aev_computer = aev_computer
        self.damping=damping
        self.damping_a=damping_a
        self.damping_b=damping_b
        self.repulsion=repulsion
        self.Rcr=Rcr
        if repulsion:
            self.RR = RepulsionCalculator(cutoff=self.Rcr, cutoff_fn='smooth',elements=elements)


    def get_atom_mask(self, species):
        padding_mask = (species.ne(-1)).float()
        assert padding_mask.sum() > 1.e-6
        padding_mask = padding_mask.unsqueeze(-1)
        return padding_mask

    def dampen(self, distances):
        damped=1.0/(1.0+torch.exp(self.damping_a * (distances - self.damping_b)))
        return damped

    def get_coulomb(self, charges, distances, molecule_indices, atom_index12, ceng):
        coulomb=0
        charges=torch.flatten(charges)
        coulomb=charges[atom_index12[0]]*charges[atom_index12[1]]
        coulomb/=distances
        if self.damping:
            damped = self.dampen(distances)
            coulomb = coulomb*damped
        ceng.index_add_(0, molecule_indices, coulomb)
        return ceng


    def get_dipole(self, xyz, charge):
        charge = charge.unsqueeze(1)
        xyz = xyz.permute(0, 2, 1)
        dipole = charge * xyz
        dipole = dipole.permute(0, 2, 1)
        dipole = torch.sum(dipole, dim=1)
        return dipole

    def forward(self, species_coordinates, total_charge=0):
        species, coordinates = species_coordinates
        atom_index12_d, ssv, diff_vectors_d, distances_d = self.aev_computer.neighborlist(species, coordinates)
        molecule_indices = torch.div(atom_index12_d[0], species.shape[1], rounding_mode='floor')
        mask=torch.zeros_like(species).bool()
        atom_index12, _, diff_vectors, distances = self.aev_computer.neighborlist._screen_with_cutoff(self.Rcr, coordinates, atom_index12_d, ssv, mask)
        aev = self.aev_computer._compute_aev(species, atom_index12, diff_vectors, distances)
        species_ = species.flatten()
        num_atoms = (species.ne(-1)).sum(dim=1, dtype=aev.dtype)

        present_species = torchani.utils.present_species(species)
        aev = aev.flatten(0, 1)
        output = torch.full_like(species_, self.padding_fill, dtype=aev.dtype)
        output_c = torch.full_like(species_, self.padding_fill, dtype=aev.dtype)

        for i in present_species:
            # Check that none of the weights are nan.
            for parameter in self[i].parameters():
                assert not (torch.isnan(parameter)).any()
            mask = (species_ == i)
            input_ = aev.index_select(0, mask.nonzero().squeeze())
            res = self[i](input_.double())
            #print(res)
            #print(res.shape)
            output.masked_scatter_(mask, res[:, 0].squeeze())
            output_c.masked_scatter_(mask, res[:, 1].squeeze())
        output = output.view_as(species)
        output_c = output_c.view_as(species)


        # Maintain conservation of charge
        excess_charge = (
            torch.full_like(
                output_c[:, 0],
                total_charge
            ) - torch.sum(output_c, dim=1)
        ) / num_atoms#.unsqueeze(-1)

        excess_charge = excess_charge.unsqueeze(1)
        output_c = (output_c + excess_charge) * self.get_atom_mask(species).squeeze(-1)
        output = self.reducer(output, dim=1)
        if self.repulsion:
            _, repulsion_energy = self.RR((species.to(output.device),output),atom_index12.to(output.device),distances.to(output.device))


        ceng=torch.zeros_like(output)
        coulomb = self.get_coulomb(output_c, distances_d, molecule_indices, atom_index12_d, ceng)
        coulomb = coulomb / ANGSTROM_TO_BOHR
        output += coulomb

        dipole = self.get_dipole(coordinates, output_c)


        return species, output, dipole, excess_charge, output_c



class ANIModel_charge_dist(torch.nn.ModuleList):

    """
    Very similar to ANIModelDipole_eA. However, here excess charge is distributed according to a learned parameter for each element rather than an even distribution.
    coordinates: Angstroms
    damping_b: Angstroms
    Rcr: Angstoms
    dipoles: eA
    energies: Hartrees
    charges: e
    """

    def __init__(self, modules, aev_computer, damping = True, damping_a = -4.0, damping_b = 2.0, repulsion=True, Rcr=5.2, elements=('H', 'C', 'N', 'O'), nmax=0):
        super(ANIModel_charge_dist, self).__init__(modules)
        self.reducer = torch.sum
        self.padding_fill = 0
        self.aev_computer = aev_computer
        self.damping=damping
        self.damping_a=damping_a
        self.damping_b=damping_b
        self.repulsion=repulsion
        self.Rcr=Rcr
        
        if torch.is_tensor(nmax):
            self.nmax=nmax
        else:
            self.nmax=torch.nn.Parameter(torch.ones(len(elements)))
        if repulsion:
            self.RR = RepulsionCalculator(cutoff=self.Rcr, cutoff_fn='smooth',elements=elements)


    def get_atom_mask(self, species):
        padding_mask = (species.ne(-1)).float()
        assert padding_mask.sum() > 1.e-6
        padding_mask = padding_mask.unsqueeze(-1)
        return padding_mask

    def dampen(self, distances):
        damped=1.0/(1.0+torch.exp(self.damping_a * (distances - self.damping_b)))
        return damped


    def get_correction(self, excess_charge, species):
        nmax_ = self.nmax.unsqueeze(1)
        nmax_matrix = nmax_[species]
        new_nmax_matrix = nmax_matrix *  self.get_atom_mask(species)
        nmax_sum = new_nmax_matrix.sum(dim=1)
        nmax_matrix = nmax_matrix.squeeze(2)
        nmax_sum = nmax_sum.squeeze(1)
        new_nmax_matrix = torch.transpose(new_nmax_matrix, 0, 1).squeeze()
        su=torch.sum(new_nmax_matrix/nmax_sum,dim=0)
        charge_corrections = (new_nmax_matrix/nmax_sum)*excess_charge
        return torch.transpose(charge_corrections, 0, -1)


    def get_coulomb(self, charges, distances, molecule_indices, atom_index12, ceng):
        coulomb=0
        charges=torch.flatten(charges)
        coulomb=charges[atom_index12[0]]*charges[atom_index12[1]]
        coulomb/=distances
        if self.damping:
            damped = self.dampen(distances)
            coulomb = coulomb*damped
        ceng.index_add_(0, molecule_indices, coulomb)
        return ceng


    def get_dipole(self, xyz, charge):
        charge = charge.unsqueeze(1)
        xyz = xyz.permute(0, 2, 1)
        dipole = charge * xyz
        dipole = dipole.permute(0, 2, 1)
        dipole = torch.sum(dipole, dim=1)
        return dipole

    def forward(self, species_coordinates, total_charge=0):
        species, coordinates = species_coordinates
        atom_index12_d, ssv, diff_vectors_d, distances_d = self.aev_computer.neighborlist(species, coordinates)
        molecule_indices = torch.div(atom_index12_d[0], species.shape[1], rounding_mode='floor')
        mask=torch.zeros_like(species).bool()
        atom_index12, _, diff_vectors, distances = self.aev_computer.neighborlist._screen_with_cutoff(self.Rcr, coordinates, atom_index12_d, ssv, mask)
        aev = self.aev_computer._compute_aev(species, atom_index12, diff_vectors, distances)
        species_ = species.flatten()
        num_atoms = (species.ne(-1)).sum(dim=1, dtype=aev.dtype)

        present_species = torchani.utils.present_species(species)
        aev = aev.flatten(0, 1)
        output = torch.full_like(species_, self.padding_fill, dtype=aev.dtype)
        output_c = torch.full_like(species_, self.padding_fill, dtype=aev.dtype)

        for i in present_species:
            # Check that none of the weights are nan.
            for parameter in self[i].parameters():
                assert not (torch.isnan(parameter)).any()
            mask = (species_ == i)
            input_ = aev.index_select(0, mask.nonzero().squeeze())
            res = self[i](input_.double())
            output.masked_scatter_(mask, res[:, 0].squeeze())
            output_c.masked_scatter_(mask, res[:, 1].squeeze())
        output = output.view_as(species)
        output_c = output_c.view_as(species)

        # Maintain conservation of charge
        excess_charge = torch.sum(output_c,dim=1)
        correction = self.get_correction(excess_charge, species)
        output_c*=self.get_atom_mask(species).squeeze(-1)
        output_c-=correction
        csum=torch.sum(output_c)
        output = self.reducer(output, dim=1)
        if self.repulsion:
            _, repulsion_energy = self.RR((species.to(output.device),output),atom_index12.to(output.device),distances.to(output.device))


        ceng=torch.zeros_like(output)
        coulomb = self.get_coulomb(output_c, distances_d, molecule_indices, atom_index12_d, ceng)
        coulomb = coulomb / ANGSTROM_TO_BOHR
        output += coulomb

        dipole = self.get_dipole(coordinates, output_c)


        return species, output, dipole, excess_charge, output_c




class SpeciesEnergiesDipoleCharge(NamedTuple):
    species: Tensor
    energies: Tensor
    dipole: Tensor
    charge: Tensor
    eng_ls: Tensor
    dip_ls: Tensor

class Ensemble(torch.nn.ModuleList):
    #Compute the average output of an ensemble of modules.

    def __init__(self, modules):
        super().__init__(modules)
        self.size = len(modules)

    def forward(self, species_input: Tuple[Tensor, Tensor]) -> SpeciesEnergiesDipoleCharge:
        eng = 0
        dip = 0
        charges=torch.zeros(species_input[1].shape[0],species_input[1].shape[1]).to('cuda')
        eng_ls=[]
        dip_ls=[]
        for x in self:
            species, energy, dipole, excess_charge, charge = x((species_input))
            eng += energy
            dip += dipole
            charges += charge
            eng_ls.append(energy.item())
            dip_ls.append(torch.sqrt(torch.sum(dipole**2)).item())
        species, _ = species_input
        return SpeciesEnergiesDipoleCharge(species, eng / self.size, dip / self.size, charges / self.size, eng_ls, dip_ls)


def make_Hsplit(aev_dim=384,bias=False,activation=torchani.nn.FittedSoftplus()):
    class H_net(torch.nn.Module):
        def __init__(self, activation=activation):
            super(H_net, self).__init__()
            self.activation=activation
            self.linear11 = torch.nn.Linear(aev_dim, 160, bias=bias)
            self.linear12 = torch.nn.Linear(160,128, bias=bias)
            self.linear13 = torch.nn.Linear(128,80, bias=bias)
            self.linear14 = torch.nn.Linear(80,1, bias=bias)

            self.linear21 = torch.nn.Linear(aev_dim, 160, bias=bias)
            self.linear22 = torch.nn.Linear(160,128, bias=bias)
            self.linear23 = torch.nn.Linear(128,80, bias=bias)
            self.linear24 = torch.nn.Linear(80,1, bias=bias)

        def forward(self, x):
            en = self.linear11(x)
            en = self.activation(en)
            en = self.linear12(en)
            en = self.activation(en)
            en = self.linear13(en)
            en = self.activation(en)
            en = self.linear14(en)

            ch = self.linear21(x)
            ch = self.activation(ch)
            ch = self.linear22(ch)
            ch = self.activation(ch)
            ch = self.linear23(ch)
            ch = self.activation(ch)
            ch = self.linear24(ch)

            return torch.hstack((en,ch))
    return H_net()

def make_Csplit(aev_dim=384,bias=False,activation=torchani.nn.FittedSoftplus()):
    class C_net(torch.nn.Module):
        def __init__(self, activation=activation):
            super(C_net, self).__init__()
            self.activation = activation
            self.linear11 = torch.nn.Linear(aev_dim, 160, bias=bias)
            self.linear12 = torch.nn.Linear(160,128, bias=bias)
            self.linear13 = torch.nn.Linear(128,80, bias=bias)
            self.linear14 = torch.nn.Linear(80,1, bias=bias)

            self.linear21 = torch.nn.Linear(aev_dim, 160, bias=bias)
            self.linear22 = torch.nn.Linear(160,128, bias=bias)
            self.linear23 = torch.nn.Linear(128,80, bias=bias)
            self.linear24 = torch.nn.Linear(80,1, bias=bias)

        def forward(self, x):
            en = self.linear11(x)
            en = self.activation(en)
            en = self.linear12(en)
            en = self.activation(en)
            en = self.linear13(en)
            en = self.activation(en)
            en = self.linear14(en)

            ch = self.linear21(x)
            ch = self.activation(ch)
            ch = self.linear22(ch)
            ch = self.activation(ch)
            ch = self.linear23(ch)
            ch = self.activation(ch)
            ch = self.linear24(ch)

            return torch.hstack((en,ch))
    return C_net()

def make_Nsplit(aev_dim=384,bias=False,activation=torchani.nn.FittedSoftplus()):
    class N_net(torch.nn.Module):
        def __init__(self, activation=activation):
            super(N_net, self).__init__()
            self.activation=activation
            self.linear11 = torch.nn.Linear(aev_dim, 128, bias=bias)
            self.linear12 = torch.nn.Linear(128,96, bias=bias)
            self.linear13 = torch.nn.Linear(96,80, bias=bias)
            self.linear14 = torch.nn.Linear(80,1, bias=bias)

            self.linear21 = torch.nn.Linear(aev_dim, 128, bias=bias)
            self.linear22 = torch.nn.Linear(128,96, bias=bias)
            self.linear23 = torch.nn.Linear(96,80, bias=bias)
            self.linear24 = torch.nn.Linear(80,1, bias=bias)

        def forward(self, x):
            en = self.linear11(x)
            en = self.activation(en)
            en = self.linear12(en)
            en = self.activation(en)
            en = self.linear13(en)
            en = self.activation(en)
            en = self.linear14(en)

            ch = self.linear21(x)
            ch = self.activation(ch)
            ch = self.linear22(ch)
            ch = self.activation(ch)
            ch = self.linear23(ch)
            ch = self.activation(ch)
            ch = self.linear24(ch)

            return torch.hstack((en,ch))
    return N_net()

def make_Osplit(aev_dim=384,bias=False,activation=torchani.nn.FittedSoftplus()):
    class O_net(torch.nn.Module):
        def __init__(self, activation=activation):
            super(O_net, self).__init__()
            self.activation=activation
            self.linear11 = torch.nn.Linear(aev_dim, 128, bias=bias)
            self.linear12 = torch.nn.Linear(128,96, bias=bias)
            self.linear13 = torch.nn.Linear(96,80, bias=bias)
            self.linear14 = torch.nn.Linear(80,1, bias=bias)

            self.linear21 = torch.nn.Linear(aev_dim, 128, bias=bias)
            self.linear22 = torch.nn.Linear(128,96, bias=bias)
            self.linear23 = torch.nn.Linear(96,80, bias=bias)
            self.linear24 = torch.nn.Linear(80,1, bias=bias)

        def forward(self, x):
            en = self.linear11(x)
            en = self.activation(en)
            en = self.linear12(en)
            en = self.activation(en)
            en = self.linear13(en)
            en = self.activation(en)
            en = self.linear14(en)

            ch = self.linear21(x)
            ch = self.activation(ch)
            ch = self.linear22(ch)
            ch = self.activation(ch)
            ch = self.linear23(ch)
            ch = self.activation(ch)
            ch = self.linear24(ch)

            return torch.hstack((en,ch))
    return O_net()

def make_Fsplit(aev_dim=384,bias=False,activation=torchani.nn.FittedSoftplus()):
    class F_net(torch.nn.Module):
        def __init__(self, activation=activation):
            super(F_net, self).__init__()
            self.activation=activation
            self.linear11 = torch.nn.Linear(aev_dim, 160, bias=bias)
            self.linear12 = torch.nn.Linear(160,128, bias=bias)
            self.linear13 = torch.nn.Linear(128,96, bias=bias)
            self.linear14 = torch.nn.Linear(96,1, bias=bias)

            self.linear21 = torch.nn.Linear(aev_dim, 160, bias=bias)
            self.linear22 = torch.nn.Linear(160,128, bias=bias)
            self.linear23 = torch.nn.Linear(128,96, bias=bias)
            self.linear24 = torch.nn.Linear(96,1, bias=bias)

        def forward(self, x):
            en = self.linear11(x)
            en = self.activation(en)
            en = self.linear12(en)
            en = self.activation(en)
            en = self.linear13(en)
            en = self.activation(en)
            en = self.linear14(en)

            ch = self.linear21(x)
            ch = self.activation(ch)
            ch = self.linear22(ch)
            ch = self.activation(ch)
            ch = self.linear23(ch)
            ch = self.activation(ch)
            ch = self.linear24(ch)

            return torch.hstack((en,ch))
    return F_net()

def make_Ssplit(aev_dim=384,bias=False,activation=torchani.nn.FittedSoftplus()):
    class S_net(torch.nn.Module):
        def __init__(self, activation=activation):
            super(S_net, self).__init__()
            self.activation=activation
            self.linear11 = torch.nn.Linear(aev_dim, 160, bias=bias)
            self.linear12 = torch.nn.Linear(160,128, bias=bias)
            self.linear13 = torch.nn.Linear(128,96, bias=bias)
            self.linear14 = torch.nn.Linear(96,1, bias=bias)

            self.linear21 = torch.nn.Linear(aev_dim, 160, bias=bias)
            self.linear22 = torch.nn.Linear(160,128, bias=bias)
            self.linear23 = torch.nn.Linear(128,96, bias=bias)
            self.linear24 = torch.nn.Linear(96,1, bias=bias)

        def forward(self, x):
            en = self.linear11(x)
            en = self.activation(en)
            en = self.linear12(en)
            en = self.activation(en)
            en = self.linear13(en)
            en = self.activation(en)
            en = self.linear14(en)

            ch = self.linear21(x)
            ch = self.activation(ch)
            ch = self.linear22(ch)
            ch = self.activation(ch)
            ch = self.linear23(ch)
            ch = self.activation(ch)
            ch = self.linear24(ch)

            return torch.hstack((en,ch))
    return S_net()

def make_Clsplit(aev_dim=384,bias=False,activation=torchani.nn.FittedSoftplus()):
    class Cl_net(torch.nn.Module):
        def __init__(self, activation=activation):
            super(Cl_net, self).__init__()
            self.activation=activation
            self.linear11 = torch.nn.Linear(aev_dim, 160, bias=bias)
            self.linear12 = torch.nn.Linear(160,128, bias=bias)
            self.linear13 = torch.nn.Linear(128,96, bias=bias)
            self.linear14 = torch.nn.Linear(96,1, bias=bias)

            self.linear21 = torch.nn.Linear(aev_dim, 160, bias=bias)
            self.linear22 = torch.nn.Linear(160,128, bias=bias)
            self.linear23 = torch.nn.Linear(128,96, bias=bias)
            self.linear24 = torch.nn.Linear(96,1, bias=bias)

        def forward(self, x):
            en = self.linear11(x)
            en = self.activation(en)
            en = self.linear12(en)
            en = self.activation(en)
            en = self.linear13(en)
            en = self.activation(en)
            en = self.linear14(en)

            ch = self.linear21(x)
            ch = self.activation(ch)
            ch = self.linear22(ch)
            ch = self.activation(ch)
            ch = self.linear23(ch)
            ch = self.activation(ch)
            ch = self.linear24(ch)

            return torch.hstack((en,ch))
    return Cl_net()

def make_H_network(aev_dim=384, bias=False, activation=torchani.nn.FittedSoftplus(), out_size=2):
    Hn = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 160, bias=bias),
    activation,
    torch.nn.Linear(160, 128, bias=bias),
    activation,
    torch.nn.Linear(128, 80, bias=bias),
    activation,
    torch.nn.Linear(80, out_size, bias=bias)
    )
    return Hn

def make_C_network(aev_dim=384, bias=False, activation=torchani.nn.FittedSoftplus(), out_size=2):
    Cn = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 160, bias=bias),
    activation,
    torch.nn.Linear(160, 128, bias=bias),
    activation,
    torch.nn.Linear(128, 80, bias=bias),
    activation,
    torch.nn.Linear(80, out_size, bias=bias)
    )
    return Cn

def make_N_network(aev_dim=384, bias=False, activation=torchani.nn.FittedSoftplus(), out_size=2):
    Nn = N_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 128, bias=bias),
    activation,
    torch.nn.Linear(128, 96, bias=bias),
    activation,
    torch.nn.Linear(96, 80, bias=bias),
    activation,
    torch.nn.Linear(80, out_size, bias=bias)
    )
    return Nn

def make_O_network(aev_dim=384, bias=False, activation=torchani.nn.FittedSoftplus(), out_size=2):
    On = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 128, bias=bias),
    activation,
    torch.nn.Linear(128, 96, bias=bias),
    activation,
    torch.nn.Linear(96, 80, bias=bias),
    activation,
    torch.nn.Linear(80, out_size, bias=bias)
    )
    return On

def make_F_network(aev_dim=384, bias=False, activation=torchani.nn.FittedSoftplus(), out_size=2):
    Fn = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 160, bias=bias),
    activation,
    torch.nn.Linear(160, 128, bias=bias),
    activation,
    torch.nn.Linear(128, 96, bias=bias),
    activation,
    torch.nn.Linear(96, out_size, bias=bias)
    )
    return Fn

def make_S_network(aev_dim=384, bias=False, activation=torchani.nn.FittedSoftplus(), out_size=2):
    Sn = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 160, bias=bias),
    activation,
    torch.nn.Linear(160, 128, bias=bias),
    activation,
    torch.nn.Linear(128, 96, bias=bias),
    activation,
    torch.nn.Linear(96, out_size, bias=bias)
    )
    return Sn

def make_Cl_network(aev_dim=384, bias=False, activation=torchani.nn.FittedSoftplus(), out_size=2):
    Cln = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 160, bias=bias),
    activation,
    torch.nn.Linear(160, 128, bias=bias),
    activation,
    torch.nn.Linear(128, 96, bias=bias),
    activation,
    torch.nn.Linear(96, out_size, bias=bias)
    )
    return Cln


def build_model(aev_computer, ANI='1x', damping = True, damping_a = -4.0, damping_b = 2.0, repulsion=True, Rcr=5.2, cd=False, nmax=0, device='cuda'):
    if ANI=='2x':
        H_network = make_H_network(aev_dim=1008)
        C_network = make_C_network(aev_dim=1008)
        N_network = make_N_network(aev_dim=1008)
        O_network = make_O_network(aev_dim=1008)
        F_network = make_F_network(aev_dim=1008)
        S_network = make_S_network(aev_dim=1008)
        Cl_network = make_Cl_network(aev_dim=1008)
        if cd:
            model = ANIModel_charge_dist([H_network, C_network, N_network, O_network, F_network, S_network, Cl_network], aev_computer, damping = damping, damping_a = -4.0, damping_b = 2.0, repulsion=repulsion, Rcr=5.2, elements=('H', 'C', 'N', 'O', 'F', 'S', 'Cl'), nmax=nmax).to(device)
        else:
            model = ANIModelDipole_eA([H_network, C_network, N_network, O_network, F_network, S_network, Cl_network], aev_computer, damping = damping, damping_a = damping_a, damping_b = damping_b, repulsion=repulsion, Rcr=Rcr, elements=('H', 'C', 'N', 'O', 'F', 'S', 'Cl')).to(device)
        return model.double()
    H_network = make_H_network()
    C_network = make_C_network()
    N_network = make_N_network()
    O_network = make_O_network()
    if cd:
        model = ANIModel_charge_dist([H_network, C_network, N_network, O_network], aev_computer, damping = damping, damping_a = damping_a, damping_b = damping_b, repulsion=repulsion, Rcr=Rcr, nmax=nmax).to(device)
    else:
        model = ANIModelDipole_eA([H_network, C_network, N_network, O_network], aev_computer, damping = damping, damping_a = damping_a, damping_b = damping_b, repulsion=repulsion, Rcr=Rcr).to(device)
    return model#.double()


def build_model_sp(aev_computer, ANI='2x', damping = True, damping_a = -4.0, damping_b = 2.0, repulsion=True, Rcr=5.2, cd=True, nmax=0, device='cuda'):
    if ANI=='2x':
        H_network = make_Hsplit(aev_dim=1008)
        C_network = make_Csplit(aev_dim=1008)
        N_network = make_Nsplit(aev_dim=1008)
        O_network = make_Osplit(aev_dim=1008)
        F_network = make_Fsplit(aev_dim=1008)
        S_network = make_Ssplit(aev_dim=1008)
        Cl_network = make_Clsplit(aev_dim=1008)
        if cd:
            model = ANIModel_charge_dist([H_network, C_network, N_network, O_network, F_network, S_network, Cl_network], aev_computer, damping = damping, damping_a = -4.0, damping_b = 2.0, repulsion=repulsion, Rcr=5.2, elements=('H', 'C', 'N', 'O', 'F', 'S', 'Cl'), nmax=nmax).to(device)
        else:
            model = ANIModelDipole_eA([H_network, C_network, N_network, O_network, F_network, S_network, Cl_network], aev_computer, damping = damping, damping_a = damping_a, damping_b = damping_b, repulsion=repulsion, Rcr=Rcr, elements=('H', 'C', 'N', 'O', 'F', 'S', 'Cl')).to(device)
        return model.double()
    H_network = make_Hsplit()
    C_network = make_Csplit()
    N_network = make_Nsplit()
    O_network = make_Osplit()
    if cd:
        model = ANIModel_charge_dist([H_network, C_network, N_network, O_network], aev_computer, damping = damping, damping_a = damping_a, damping_b = damping_b, repulsion=repulsion, Rcr=Rcr, nmax=nmax).to(device)
    else:
        model = ANIModelDipole_sp([H_network, C_network, N_network, O_network], aev_computer, damping = damping, damping_a = damping_a, damping_b = damping_b, repulsion=repulsion, Rcr=Rcr).to(device)
    return model#.double()

