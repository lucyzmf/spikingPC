# %%
# compare models trained with different random:fixed switch time ratios on the same dataset 

import torchvision
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import copy

from network_class import *
from sequence_dataset import *
from utils import *
import glob

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
random_switch = 0.  # predictable or random switch time
switch_time = [seq_len / 2 + 5]  # if not random switch, provide switch time
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

# %%
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
exp_path = '/home/lucy/spikingPC/results/Feb-15-2023/'

# %%
# make list containing all models 
probs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.]
models = []

for p in probs:
    model_ = SnnNetworkSeq(IN_dim, hidden_dim, n_classes, is_adapt=adap_neuron, one_to_one=onetoone,
                      dp_rate=dp)
    model_.to(device)

    model_path = glob.glob(exp_path + '/*%s/*onelayer_rec_best.pth.tar' %str(p))
    print(model_path)
    saved_dict = model_result_dict_load(model_path[0])

    model_.load_state_dict(saved_dict['state_dict'])
    model_.eval()

    models.append(model_)

# %%
# plot p to r exci vs inhi
fig = plt.figure()
x = ['exci', 'inhi'] * len(models)
ys = []
for i in range(len(models)):
    p2r_w = models[i].rout2rin.weight.detach().cpu().numpy()
    y = [(p2r_w * (p2r_w > 0)).sum(), -(p2r_w * (p2r_w < 0)).sum()]
    ys.append(y)
ys = np.concatenate(ys)

exinh = {
    'connectivity type': x, 
    'sum w': ys, 
    'p': np.repeat(probs, 2)
}

df = pd.DataFrame.from_dict(exinh)

ax = sns.barplot(df, x='p', y='sum w', hue='connectivity type')
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.title('p2r sum exci inhi w')
plt.show()

# %%
# taus 
def get_real_constants(pseudo_constants):
    return -1/np.log(1/(1 + np.exp(-pseudo_constants)))

for i in range(len(probs)):
    plt.hist(get_real_constants(models[i].r_out_rec.tau_i.detach().cpu().numpy().flatten()), label='p=%f' %probs[i])
plt.legend()
plt.show()

# %%
def save_to_cpu(hidden_):
    for i in range(len(hidden_)):
        hidden_[i] = list(hidden_[i])
        for j in range(len(hidden_[0])):
            hidden_[i][j] = hidden_[i][j].cpu()

    return hidden_

def get_all_data(model, test_loader, onlyshowone=None):
    #  get all analysis data in sequence condition
    test_loss = 0
    correct = 0
    correct_end_of_seq = 0

    hiddens_all = []
    predictions_all = []
    targets_all = []

    # for data, target in test_loader:
    for i, (data, target) in enumerate(test_loader):
        targets_all.append(target)
        data, target = data.to(device), target.to(device)
        data = data.view(-1, seq_len, model.in_dim)
        if onlyshowone:
            data[:, onlyshowone:, :] = -1

        with torch.no_grad():
            model.eval()
            hidden = model.init_hidden(data.size(0))

            log_softmax_outputs, hidden, pred_hist = model.inference(data, hidden, seq_len)
            hiddens_all.append(save_to_cpu(hidden))
            predictions_all.append(pred_hist.T)

        correct += pred_hist.T.eq(target.data).cpu().sum()
        # only check end of sequence acc 
        # correct_end_of_seq += pred_hist.T[:, int(seq_len / 2) - 1].eq(target[:, int(seq_len / 2) - 1].data).cpu().sum()
        # correct_end_of_seq += pred_hist.T[:, seq_len - 1].eq(target[:, seq_len - 1].data).cpu().sum()
        torch.cuda.empty_cache()

    test_acc = 100. * correct / len(test_loader.dataset) / seq_len
    test_acc_endofseq = 100 * correct_end_of_seq / len(test_loader.dataset) / 2

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), End of seq acc: {:.2f}%\n'.format(
        test_loss, int(correct / seq_len), len(test_loader.dataset),
        test_acc, test_acc_endofseq))

    predictions_all = torch.stack(predictions_all).reshape(test_loader.dataset.num_samples, seq_test.seq_len).cpu()
    targets_all = torch.stack(targets_all).reshape(test_loader.dataset.num_samples, seq_test.seq_len).cpu()

    return hiddens_all, targets_all, predictions_all


# %%
# get all hiddens, targets, preds 
hiddens, targets, predictions = [], [], []
selected = np.arange(0, 6, 2)
stopshow = None

# %%
for i in selected:
    hidden_, targets_, predictions_ = get_all_data(models[i], test_loader, onlyshowone=stopshow)

    hiddens.append(hidden_), 
    targets.append(targets_)
    predictions.append(predictions_)

# %% 
# change targets to so that between stopshow and 20 targets are moved to the next because fo p drift 
# targets_edited = copy.deepcopy(targets)

# for i in range(len(targets_edited)):
#     targets_edited[i][:, stopshow:int(seq_len/2)] = targets_edited[i][:, stopshow:int(seq_len/2)]+1
#     targets_edited[i][targets_edited[i]==10] = 0  # get rid of 10s 

# targets_edited[0][0, :]

# %%
# plot acc curves 
fig = plt.figure()
t = np.arange(seq_len)
for i in range(len(hiddens)):
    plt.plot(t, (predictions[i] == targets[i]).float().mean(dim=0), label='p= %f' % probs[selected[i]])
plt.xlabel('t')
plt.ylabel('acc')
plt.vlines(stopshow, ymin=0, ymax=1, colors='red')
plt.legend()
plt.show()

# %%
# compute p spk rate for each step 
pred_spk_based = []
for i in range(len(hiddens)):
    p_spk = get_states(hiddens[i], 5, hidden_dim[0], batch_size, T=40)
    mean_bypop = p_spk.reshape(10000, seq_len, 10, 10).mean(axis=-1)
    preds = torch.tensor(mean_bypop).data.max(-1, keepdim=True)[1]
    pred_spk_based.append(preds.squeeze())
    

# %%
# plot acc curves spk based evaluation
fig = plt.figure()
t = np.arange(seq_len)
for i in range(len(hiddens)):
    plt.plot(t, (pred_spk_based[i] == targets[i]).float().mean(dim=0), label='p= %f' % probs[selected[i]])
plt.xlabel('t')
plt.ylabel('acc spk based')
plt.vlines(stopshow, ymin=0, ymax=1, colors='red')
plt.legend()
plt.show()

# %%
# change in proportion of output class based on spk output 
one_model_preds = pred_spk_based[0]

df = {
    't': np.repeat(np.arange(seq_len), 10000, axis=0).flatten(),
    'predicted labels': one_model_preds[0].flatten()
}

df = pd.DataFrame.from_dict(df)

ax = sns.histplot(df, x='t', hue='predicted labels', multiple='stack')
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.show()

# %%
# quantify switch speed 

# %%
# spk mean p 
for i in range(len(hiddens)):
    p_spk_all = get_states(hiddens[i], 5, hidden_dim[0], batch_size, T=40)
    plt.plot(np.arange(seq_len), p_spk_all.mean(axis=0).mean(axis=1), label='p= %f' % probs[selected[i]])
plt.legend()
plt.xlabel('t')
plt.ylabel('p spk rate')
plt.vlines(stopshow, ymin=0, ymax=0.4, colors='red')
plt.show()

# %%

# %%
