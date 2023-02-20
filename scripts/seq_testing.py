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
dp = 0.2
onetoone = True
num_readout = 10

# seq data set config
seq_len = 40  # sequence length
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

seq_test = SequenceDatasetPredictable(testdata, testdata.targets, seq_len, random_switch,
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

exp_dir = '/home/lucy/spikingPC/results/Feb-13-2023/fptt_ener_fpttalpha02_curr0/'
saved_dict = model_result_dict_load(exp_dir + 'onelayer_rec_best.pth.tar')

model.load_state_dict(saved_dict['state_dict'])
model.eval()

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
    sns.heatmap(
        model.rout2rin.weight[:, num_readout * i:(i + 1) * num_readout].detach().cpu().numpy().sum(axis=1).reshape(28,
                                                                                                                   28),
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
y = [(p2r_w * (p2r_w > 0)).detach().cpu().numpy().sum(), -(p2r_w * (p2r_w < 0)).detach().cpu().numpy().sum()]
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
# r2r histogram 
plt.hist(model.r_in_rec.rec_w.weight.detach().cpu().numpy().flatten())
plt.title('r2r weight distribution')
plt.show()

# %%
# p2p histogram 
plt.hist(model.r_out_rec.rec_w.weight.detach().cpu().numpy().flatten())
plt.title('p2p weight distribution')
plt.show()

# %%
# p2r dist
plt.hist(model.rout2rin.weight.detach().cpu().numpy().flatten())
plt.title('p2r weight distribution')
plt.show()

# %%
# r2p dist
plt.hist(model.rin2rout.weight.detach().cpu().numpy().flatten())
plt.title('r2p weight distribution')
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
#  get all analysis data in sequence condition
test_loss = 0
correct = 0
correct_end_of_seq = 0

hiddens_all = []
predictions_all = []

# for data, target in test_loader:
for i, (data, target) in enumerate(test_loader):
    data, target = data.to(device), target.to(device)
    data = data.view(-1, seq_len, model.in_dim)

    with torch.no_grad():
        model.eval()
        hidden = model.init_hidden(data.size(0))

        log_softmax_outputs, hidden, pred_hist = model.inference(data, hidden, seq_len)
        hiddens_all.append(hidden)
        predictions_all.append(pred_hist)

        # compute loss at each time step
        for t in range(seq_len):
            test_loss += F.nll_loss(log_softmax_outputs[t], target[:, t], reduction='sum').data.item()

    correct += pred_hist.T.eq(target.data).cpu().sum()
    # only check end of sequence acc 
    correct_end_of_seq += pred_hist.T[:, int(seq_len / 2) - 1].eq(target[:, int(seq_len / 2) - 1].data).cpu().sum()
    correct_end_of_seq += pred_hist.T[:, seq_len - 1].eq(target[:, seq_len - 1].data).cpu().sum()
    torch.cuda.empty_cache()

test_loss /= len(test_loader.dataset)  # per t loss
test_acc = 100. * correct / len(test_loader.dataset) / seq_len
test_acc_endofseq = 100 * correct_end_of_seq / len(test_loader.dataset) / 2

print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), End of seq acc: {:.2f}%\n'.format(
    test_loss, int(correct / seq_len), len(test_loader.dataset),
    test_acc, test_acc_endofseq))

predictions_all = torch.stack(predictions_all).reshape(seq_test.num_samples, seq_test.seq_len)

# %%
# decompose input signals in normal sequence
seq_hidden_batch = hiddens_all[0]
r_from_p_ex = []
r_from_p_inh = []
p_from_r_ex = []
p_from_r_inh = []
p_from_p_ex = []
p_from_p_inh = []
r_from_r_ex = []
r_from_r_inh = []
r_spk_rate = []
p_spk_rate =[]

timesteps = len(seq_hidden_batch)

# only take one sample for the computation
for t in np.arange(1, timesteps):
    # r_from_p_ex.append(((model.rout2rin.weight.ge(0) * model.rout2rin.weight) @ seq_hidden_batch[t - 1][5][0].T).mean().detach().cpu().numpy())
    # r_from_p_inh.append(((model.rout2rin.weight.le(0) * model.rout2rin.weight) @ seq_hidden_batch[t - 1][5][0].T).mean().detach().cpu().numpy())
    p_from_r_ex.append(((model.rin2rout.weight.ge(0) * model.rin2rout.weight) @ seq_hidden_batch[t][1][0].T).mean().detach().cpu().numpy())
    p_from_r_inh.append(((model.rin2rout.weight.le(0) * model.rin2rout.weight) @ seq_hidden_batch[t][1][0].T).mean().detach().cpu().numpy())
    p_from_p_ex.append(((model.r_out_rec.rec_w.weight.ge(0) * model.r_out_rec.rec_w.weight) @ seq_hidden_batch[t - 1][5][0].T).mean().detach().cpu().numpy())
    p_from_p_inh.append(((model.r_out_rec.rec_w.weight.le(0) * model.r_out_rec.rec_w.weight) @ seq_hidden_batch[t - 1][5][0].T).mean().detach().cpu().numpy())
    # r_from_r_ex.append(((model.r_in_rec.rec_w.weight.ge(0) * model.r_in_rec.rec_w.weight) @ seq_hidden_batch[t - 1][1][0].T).mean().detach().cpu().numpy())
    # r_from_r_inh.append(((model.r_in_rec.rec_w.weight.le(0) * model.r_in_rec.rec_w.weight) @ seq_hidden_batch[t - 1][1][0].T).mean().detach().cpu().numpy())
    # r_spk_rate.append(seq_hidden_batch[t][1][0].mean().detach().cpu().numpy())
    p_spk_rate.append(seq_hidden_batch[t][5][0].mean().detach().cpu().numpy())



fig = plt.figure()
x = np.arange(1, timesteps)
plt.plot(x, np.hstack(p_from_r_ex), label='p_from_r_ex')
plt.plot(x, np.hstack(p_from_r_inh), label='p_from_r_inh')
plt.plot(x, np.hstack(p_from_p_ex), label='p_from_p_ex')
plt.plot(x, np.hstack(p_from_p_inh), label='p_from_p_inh')
plt.plot(x, np.hstack(p_spk_rate), label='p spk rate', linestyle='dashed')

# plt.plot(x, np.hstack(r_from_p_ex), label='r_from_p_ex')
# plt.plot(x, np.hstack(r_from_p_inh), label='r_from_p_inh')
# plt.plot(x, np.hstack(r_from_r_ex), label='r_from_r_ex')
# plt.plot(x, np.hstack(r_from_r_inh), label='r_from_r_inh')
# plt.plot(x, (np.hstack(r_from_p_ex)+np.hstack(r_from_p_inh)), label='p2r input')
# plt.plot(x, (np.hstack(r_from_r_ex)+np.hstack(r_from_r_inh)), label='r2r input')
# plt.plot(x, np.hstack(r_spk_rate), label='r spk rate', linestyle='dashed')

plt.legend()
plt.title('mean exhitatory and inhibitory signals by source and target type')
plt.show()
# plt.savefig(exp_dir + 'mean exhitatory and inhibitory signals by source and target type')
# plt.close()

# %%
curr_r = get_states(hiddens_all, 2, hidden_dim[1], batch_size, T=40)
mem_r = get_states(hiddens_all, 0, hidden_dim[1], batch_size, T=40)
adp_thre_r = get_states(hiddens_all, 3, hidden_dim[1], batch_size, T=40)
# %%
fig = plt.figure()
for i in range(3):
    plt.plot(np.arange(timesteps), curr_r[:, :, i+200].mean(axis=0), label='idx %i curr' % (i+200))
    plt.plot(np.arange(timesteps), mem_r[:, :, i+200].mean(axis=0), label='idx %i mem' % (i+200))
    plt.plot(np.arange(timesteps), adp_thre_r[:, :, i+200].mean(axis=0), label='idx %i adp' % (i+200))
plt.legend()
plt.show()
# %%
curr_p = get_states(hiddens_all, 6, hidden_dim[0], batch_size, T=40)
mem_p = get_states(hiddens_all, 4, hidden_dim[0], batch_size, T=40)
adp_thre_p = get_states(hiddens_all, 7, hidden_dim[0], batch_size, T=40)
# %%
fig = plt.figure()
for i in range(1):
    plt.plot(np.arange(timesteps), curr_p[:, :, i+200].mean(axis=0), label='idx %i curr' % (i+200))
    plt.plot(np.arange(timesteps), mem_p[:, :, i+200].mean(axis=0), label='idx %i mem' % (i+200))
    plt.plot(np.arange(timesteps), adp_thre_p[:, :, i+200].mean(axis=0), label='idx %i adp' % (i+200))
plt.legend()
plt.show()