import torch
import torchani
import math
import tqdm
import shutil
import os
import datetime
from torchani.units import hartree2kcalmol
from nets import ANIModelDipoleTZ
from nets import ANIModelDipoleTZ_damp_repulse_eA
from nets2 import ANIModelDipole_eA_test
from loss import MTLLoss
import ensemble_tools as et
from nn import RNN_S, EMA, MTLLoss
from optim import AdamaxW
import warnings
import pickle
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.filterwarnings('ignore')


###############################################################################
# setting device and loading data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device='cpu'
print(f'... GPU allocated: {torch.cuda.get_device_name(0)}')

#dspath = '../ANI-1x_tz_dipoles_forces.h5'

member='3'
split=True
damping=True
repuls=True


data_path='/data/cdever01/torch_train/dipol_training/batched_2x_3/some_data/'
#data_path='/data/cdever01/torch_train/dipol_training/mini_batch/some_data/'
batched_train_path = data_path# + 'training' + member + '/'
batched_valid_path = data_path# + 'validation' + member + '/'

energy_shifter, self_energies = torchani.neurochem.load_sae('/data/cdever01/torch_train/dipol_training/SAE_o/sae_wb97m_tz.dat', return_dict=True)



#now = datetime.datetime.now()
#date = now.strftime("%Y%m%d_%H%M")
#log = 'ens_logs/' + str(date) + '/'

log = 'charge_dist_logs/'
if split:
    log+='split/'
else:
    log+='no_split/'
if repuls:
    log+='rep/'
else:
    log+='no_rep/'

if damping:
    log+='damp/'
else:
    log+='no_damp/'


log+='mem'+member+'/'
if not os.path.exists(log):
    os.makedirs(log)


Rcr = 5.2
Rca = 3.5
species=['H', 'C', 'N', 'O', 'F', 'S', 'Cl']
#prm = alt.anitrainerparamsdesigner(species, 16, Rcr, 8, 4, Rca, 0.75, ACA=True)
#prm_name = prm.get_filename()
#prm.create_params_file(log)


prm_name = 'rHCNOFSCl-infR_16-3.8A_a4-8.params'
shutil.copy(prm_name, log+prm_name)
#consts = torchani.neurochem.Constants(log+'/'+prm_name)
consts = torchani.neurochem.Constants(prm_name)


aev_computer = torchani.AEVComputer(**consts, use_cuda_extension=True)
#energy_shifter = torchani.utils.EnergyShifter(None)
sae_file = 'sae_wb97m_tz.dat'
energy_shifter = torchani.neurochem.load_sae('../SAE_o/'+sae_file)
shutil.copy('../SAE_o/'+sae_file, log+sae_file)


batch_size = 1024

training = torchani.datasets.AniBatchedDataset(batched_train_path, split='training'+member)
validation = torchani.datasets.AniBatchedDataset(batched_valid_path, split='validation'+member)
training = list(training)
#training = training.collate(batch_size).cache()
#validation = validation.collate(batch_size).cache()
print('Self atomic energies:\n', energy_shifter.self_energies.tolist())

###############################################################################
# Now let's define atomic neural networks.
out_size = 2
aev_dim = aev_computer.aev_length

bias=False

activation=torchani.nn.FittedSoftplus()
#activation=torch.nn.GELU()
#activation=torch.nn.CELU(0.1)
if split:
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
            #en = activation(en)

            ch = self.linear21(x)
            ch = self.activation(ch)
            ch = self.linear22(ch)
            ch = self.activation(ch)
            ch = self.linear23(ch)
            ch = self.activation(ch)
            ch = self.linear24(ch)
            #ch = activation(ch)

            return torch.hstack((en,ch))


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
            #en = activation(en)

            ch = self.linear21(x)
            ch = self.activation(ch)
            ch = self.linear22(ch)
            ch = self.activation(ch)
            ch = self.linear23(ch)
            ch = self.activation(ch)
            ch = self.linear24(ch)
            #ch = activation(ch)

            return torch.hstack((en,ch))

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
            #en = activation(en)

            ch = self.linear21(x)
            ch = self.activation(ch)
            ch = self.linear22(ch)
            ch = self.activation(ch)
            ch = self.linear23(ch)
            ch = self.activation(ch)
            ch = self.linear24(ch)
            #ch = activation(ch)

            return torch.hstack((en,ch))

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
            #en = activation(en)

            ch = self.linear21(x)
            ch = self.activation(ch)
            ch = self.linear22(ch)
            ch = self.activation(ch)
            ch = self.linear23(ch)
            ch = self.activation(ch)
            ch = self.linear24(ch)
            #ch = activation(ch)

            return torch.hstack((en,ch))

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
            #en = activation(en)

            ch = self.linear21(x)
            ch = self.activation(ch)
            ch = self.linear22(ch)
            ch = self.activation(ch)
            ch = self.linear23(ch)
            ch = self.activation(ch)
            ch = self.linear24(ch)
            #ch = activation(ch)

            return torch.hstack((en,ch))

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
            #en = activation(en)

            ch = self.linear21(x)
            ch = self.activation(ch)
            ch = self.linear22(ch)
            ch = self.activation(ch)
            ch = self.linear23(ch)
            ch = self.activation(ch)
            ch = self.linear24(ch)
            #ch = activation(ch)

            return torch.hstack((en,ch))

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
            #en = activation(en)

            ch = self.linear21(x)
            ch = self.activation(ch)
            ch = self.linear22(ch)
            ch = self.activation(ch)
            ch = self.linear23(ch)
            ch = self.activation(ch)
            ch = self.linear24(ch)
            #ch = activation(ch)

            return torch.hstack((en,ch))

    H_network = H_net()
    C_network = C_net()
    N_network = N_net()
    O_network = O_net()
    F_network = F_net()
    S_network = S_net()
    Cl_network = Cl_net()
else:
    H_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 160, bias=bias),
    activation,
    torch.nn.Linear(160, 128, bias=bias),
    activation,
    torch.nn.Linear(128, 80, bias=bias),
    activation,
    torch.nn.Linear(80, out_size, bias=bias)
    )

    C_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 160, bias=bias),
    activation,
    torch.nn.Linear(160, 128, bias=bias),
    activation,
    torch.nn.Linear(128, 80, bias=bias),
    activation,
    torch.nn.Linear(80, out_size, bias=bias)
    )

    N_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 128, bias=bias),
    activation,
    torch.nn.Linear(128, 96, bias=bias),
    activation,
    torch.nn.Linear(96, 80, bias=bias),
    activation,
    torch.nn.Linear(80, out_size, bias=bias)
    )

    O_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 128, bias=bias),
    activation,
    torch.nn.Linear(128, 96, bias=bias),
    activation,
    torch.nn.Linear(96, 80, bias=bias),
    activation,
    torch.nn.Linear(80, out_size, bias=bias)
    )

    F_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 160, bias=bias),
    activation,
    torch.nn.Linear(160, 128, bias=bias),
    activation,
    torch.nn.Linear(128, 96, bias=bias),
    activation,
    torch.nn.Linear(96, out_size, bias=bias)
    )

    S_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 160, bias=bias),
    activation,
    torch.nn.Linear(160, 128, bias=bias),
    activation,
    torch.nn.Linear(128, 96, bias=bias),
    activation,
    torch.nn.Linear(96, out_size, bias=bias)
    )


    Cl_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 160, bias=bias),
    activation,
    torch.nn.Linear(160, 128, bias=bias),
    activation,
    torch.nn.Linear(128, 96, bias=bias),
    activation,
    torch.nn.Linear(96, out_size, bias=bias)
    )

nn = et.ANIModel_charge_dist([H_network, C_network, N_network, O_network, F_network, S_network, Cl_network], aev_computer, damping = damping, damping_a = -4.0, damping_b = 2.0, repulsion=repuls, Rcr=5.2, elements=('H', 'C', 'N', 'O', 'F', 'S', 'Cl'))

def init_params(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, a=1.0)
        #torch.nn.init.zeros_(m.bias)


nn.apply(init_params)
model = nn.to(device).double()

###############################################################################
print(model)
model_weights = []
model_biases = []
for name, param in model.named_parameters():
    if 'bias' in name:
        model_biases.append(param)
    elif 'weight' in name:
        model_weights.append(param)

#assert len(list(model.parameters())) == len(model_biases) + len(model_weights)


optimizer = AdamaxW([
    {'params': model_weights, 'weight_decay': 1e-4},
    {'params': model_biases},
], lr=1e-3)

optimizer.param_groups[0]['params'].append(model.nmax)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.98, patience=1, threshold=0.0001)


mtl = MTLLoss(num_tasks=2).to(device)
optimizer.param_groups[0]['params'].append(mtl.log_sigma)  # avoids LRdecay problem

###############################################################################


latest_checkpoint = log + '/latest.pt'
best_model_checkpoint = log + '/best.pt'
shutil.copy(__file__, log + '/trainer.py')


if os.path.isfile(latest_checkpoint):
    checkpoint = torch.load(latest_checkpoint)
    nn.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    #SGD.load_state_dict(checkpoint['SGD'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    #SGD_scheduler.load_state_dict(checkpoint['SGD_scheduler'])


def save_model(checkpoint):
    torch.save({
        'model': nn.state_dict(),
        'optimizer': optimizer.state_dict(),
        'self_energies': energy_shifter.self_energies,
        'scheduler': scheduler.state_dict(),
        'nmax' : model.nmax
    }, checkpoint)


###############################################################################
def eA2debeye(x):
    return x/0.20819434


def validate():
    # run validation
    mse_sum = torch.nn.MSELoss(reduction='sum')
    total_energy_mse = 0.0
    total_dipole_mse = 0.0
    total_excess_mse = 0.0
    total_force_mse = 0.0
    count = 0
    for properties in validation:
        species = torch.tensor(properties['species']).to(device)
        coordinates = torch.tensor(properties['coordinates']).to(device).double().requires_grad_(True)
        true_energies = torch.tensor(properties['energies']).to(device).double()
        true_dipoles = torch.tensor(properties['dipoles']).to(device).double()  # *0.20819434
        #true_forces = torch.tensor(properties['forces']).to(device).double()
        num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)

        _, predicted_energies, predicted_dipoles, excess_charge, predicted_charges = model((species, coordinates))
        #forces = -torch.autograd.grad(predicted_energies.sum(), coordinates)[0]

        #total_force_mse += (mse(true_forces, forces).sum(dim=(1, 2)) / (3 * num_atoms)).sum()
        total_dipole_mse += mse_sum(predicted_dipoles, true_dipoles).item()
        total_energy_mse += mse_sum(predicted_energies, true_energies).item()
        total_excess_mse += mse_sum(excess_charge, torch.zeros_like(excess_charge)).item()
        count += predicted_energies.shape[0]

    return (
        hartree2kcalmol(math.sqrt(total_energy_mse / count)),
        eA2debeye(math.sqrt(total_dipole_mse / count)),
        math.sqrt(total_excess_mse / count),
        torch.sum(predicted_charges),
        #hartree2kcalmol(math.sqrt(total_force_mse / count))
    )


###############################################################################
mse = torch.nn.MSELoss(reduction='none')
ema = EMA(model, decay=0.999)
#mtl = MTLLoss(num_tasks=3).to(device)
#optimizer.param_groups[0]['params'].append(mtl.log_sigma)  # avoids LRdecay problem

max_epochs = 1500
early_stopping_learning_rate = 1.0E-5
best = 1e3
best_epoch=0
iteration = 0

results_file = open(log+'/results.txt', 'w')

for epoch in range(max_epochs):
    print('Epoch:', epoch)
    ema.assign(model)

    energy_rmse, dipole_rmse, excess_rmse, charges_sum = validate()

    # checkpoint
    if energy_rmse < best:
        print(f'Saving the model, epoch = {epoch}, RMSE = {energy_rmse}')
        save_model(best_model_checkpoint)
        best = energy_rmse
        best_epoch=epoch
    
    ema.resume(model)
    learning_rate = optimizer.param_groups[0]['lr']

    results_file.write('Epoch: %s\n' %epoch)
    results_file.write('ERMSE: %s\n' %energy_rmse)
    results_file.write('DRMSE: %s\n' %dipole_rmse)
    #results_file.write('FRMSE: %s\n' %force_rmse)
    results_file.write('CRMSE: %s\n' %excess_rmse)
    results_file.write('CSUM: %s\n' %charges_sum)
    results_file.write('Nmax: %s\n' %model.nmax)
    results_file.write('Best ERMSE: %s\n' %best)
    results_file.write('Best epcoch: %s\n' %best_epoch)
    results_file.write('\n')


    learning_rate = optimizer.param_groups[0]['lr']
    if learning_rate < early_stopping_learning_rate:
        break

    scheduler.step(energy_rmse)

    for i, properties in tqdm.tqdm(
        enumerate(training),
        total=len(training),
        desc="epoch {}".format(epoch)
    ):
        species = torch.tensor(properties['species']).to(device)
        coordinates = torch.tensor(properties['coordinates']).to(device).double().requires_grad_(True)
        true_energies = torch.tensor(properties['energies']).to(device).double()
        true_dipoles = torch.tensor(properties['dipoles']).to(device).double()  # *0.20819434
        #true_forces = torch.tensor(properties['forces']).to(device).float()
        num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)

        _, predicted_energies, predicted_dipoles, excess_charge, predicted_charges = model((species, coordinates))
        #forces = -torch.autograd.grad(predicted_energies.sum(), coordinates, create_graph=True, retain_graph=True)[0]

        exp_weight = torch.exp(- true_energies / num_atoms / 0.006).clamp_(0.01, 1.0)
        energy_loss = ((mse(predicted_energies, true_energies) / num_atoms.sqrt())*exp_weight).mean()
        #force_loss = ((mse(true_forces, forces).sum(dim=(1, 2)) / (3.0 * num_atoms))*exp_weight).mean()
        dipole_loss = ((torch.sum((mse(predicted_dipoles, true_dipoles))/3.0, dim=1) / num_atoms.sqrt())*exp_weight).mean()
        #loss = mtl(energy_loss, force_loss, dipole_loss)
        loss = mtl(energy_loss, dipole_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        iteration += 1
    save_model(latest_checkpoint)
results_file.close()
#for i in range(8):
#    if str(i)!=member:
#        os.remove('../split_data/ANI-1x_tz_'+str(i)+'.h5')
#os.remove('../split_data/ANI-1x_tz_'+member+'.h5')


