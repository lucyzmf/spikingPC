# %%
# this script contains training with sequence data

import torchvision
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from network_class import *
from sequence_dataset import *
from utils import *

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# set seed
torch.manual_seed(999)

# network hypers
adap_neuron = True  # whether use adaptive neuron or not
dp = 0.5
onetoone = True
num_readout = 10

# seq data set config
seq_len = 20  # sequence length
random_switch = False  # predictable or random switch time
switch_time = [seq_len / 2]  # if not random switch, provide switch time
num_switch = 1  # used when random switch=T

n_classes = 10

# %%
###############################################################
# IMPORT DATASET
###############################################################

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])

batch_size = 200

testdata = torchvision.datasets.MNIST(root='./data', train=False,
                                      download=True, transform=transform)

seq_test = SequenceDataset(testdata.data, testdata.targets, seq_len, random_switch,
                           switch_time, num_switch)

test_loader = torch.utils.data.DataLoader(seq_test, batch_size=batch_size,
                                          shuffle=False, num_workers=3)

for batch_idx, (data, target) in enumerate(test_loader):
    print(data.shape)
    print(target.shape)
    break

# %%
# ##############################################################
# DEFINE NETWORK
# ##############################################################

# set input and t param
IN_dim = 784
hidden_dim = [10 * num_readout, 784]

# define network
model = SnnNetworkSeq(IN_dim, hidden_dim, n_classes, is_adapt=adap_neuron, one_to_one=onetoone,
                      dp_rate=dp)
model.to(device)
print(model)

# define new loss and optimiser
total_params = count_parameters(model)
print('total param count %i' % total_params)

# %%

exp_dir = '/home/lucy/spikingPC/results/Feb-10-2023/fptt_ener_seq_newloss2_outmemtrainable_fpttalpha02_currinit-1_t40/'
saved_dict = model_result_dict_load(exp_dir + 'onelayer_rec_best.pth.tar')

model.load_state_dict(saved_dict['state_dict'])

# %%
# get params and put into dict
param_names = []
param_dict = {}
for name, param in model.named_parameters():
    if param.requires_grad:
        param_names.append(name)
        param_dict[name] = param.detach().cpu().numpy()

print(param_names)

# %%
# plot p to r weights
fig, axs = plt.subplots(1, 10, figsize=(35, 3))
for i in range(10):
    sns.heatmap(model.rout2rin.weight[:, num_readout * i:(i + 1) * num_readout].detach().cpu().numpy().sum(axis=1).reshape(28, 28),
                ax=axs[i])
plt.title('r_out weights to r_in by class')

# plt.show()
plt.savefig(exp_dir + 'r_out weights to r_in by class')
plt.close()

# %% 
# plot p to r exci vs inhi
fig = plt.figure()
x = ['exci', 'inhi']
p2r_w = model.rout2rin.weight
y = [(p2r_w*(p2r_w>0)).detach().cpu().numpy().sum(), -(p2r_w*(p2r_w<0)).detach().cpu().numpy().sum()]
sns.barplot(x=x, y=y)
plt.title('p to r exci vs inhi')
plt.show()
# %%
# weight matrix for p2p

fig = plt.figure()
abs_max = np.max(np.abs(model.r_out_rec.rec_w.weight.detach().cpu().numpy()))
sns.heatmap(model.r_out_rec.rec_w.weight.detach().cpu().numpy(), vmax=abs_max, vmin=-abs_max, cmap='icefire')
plt.title('p2p weights')
# plt.show()
plt.savefig(exp_dir + 'p2p weights')
plt.close()
# %%
# plot output tau 
fig = plt.figure()
plt.plot(np.arange(10), model.output_layer.tau_m.detach().cpu().numpy())
plt.show()
# %%
# weight matrix for r2r

fig = plt.figure()
abs_max = np.max(np.abs(model.r_in_rec.rec_w.weight.detach().cpu().numpy()))
sns.heatmap(model.r_in_rec.rec_w.weight.detach().cpu().numpy(), vmax=abs_max, vmin=-abs_max, cmap='icefire')
plt.title('r2r weights')
plt.show()
# plt.savefig(exp_dir + 'r2r weights')
# plt.close()
# %%
# weight matrix for r2p

fig = plt.figure()
abs_max = np.max(np.abs(model.rin2rout.weight.detach().cpu().numpy()))
sns.heatmap(model.rin2rout.weight.detach().cpu().numpy(), vmax=abs_max, vmin=-abs_max, cmap='icefire')
plt.title('r2p weights')
plt.show()
# plt.savefig(exp_dir + 'r2p weights')
# plt.close()
# %%
#  get_analysisdata_seq
model.eval()
test_loss = 0
correct = 0
correct_end_of_seq = 0

# for data, target in test_loader:
for i, (data, target) in enumerate(test_loader):
    data, target = data.to(device), target.to(device)
    data = data.view(-1, seq_len, model.in_dim)

    with torch.no_grad():
        model.eval()
        hidden = model.init_hidden(data.size(0))

        log_softmax_outputs, hidden, pred_hist = model.inference(data, hidden, seq_len)

        # compute loss at each time step
        for t in range(seq_len):
            test_loss += F.nll_loss(log_softmax_outputs[t], target[:, t], reduction='sum').data.item()

    correct += pred_hist.T.eq(target.data).cpu().sum()
    # only check end of sequence acc 
    correct_end_of_seq += pred_hist.T[:, int(seq_len/2)-1].eq(target[:, int(seq_len/2)-1].data).cpu().sum() 
    correct_end_of_seq += pred_hist.T[:, seq_len-1].eq(target[:, seq_len-1].data).cpu().sum()
    torch.cuda.empty_cache()


test_loss /= len(test_loader.dataset)  # per t loss
test_acc = 100. * correct / len(test_loader.dataset) / seq_len
test_acc_endofseq = 100 * correct_end_of_seq / len(test_loader.dataset) / 2

print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, int(correct / seq_len), len(test_loader.dataset),
    test_acc))

