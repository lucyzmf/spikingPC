import torch
import torch.nn.functional as F
import wandb
from FTTP import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
def train_fptt_conv(epoch, batch_size, log_interval,
                    train_loader, model, named_params,
                    time_steps, k_updates, omega, optimizer,
                    clf_alpha, energy_alpha, clip, lr):
    train_loss = 0
    total_clf_loss = 0
    total_regularizaton_loss = 0
    total_energy_loss = 0
    correct = 0
    model.train()

    # for each batch
    for batch_idx, (data, target) in enumerate(train_loader):

        # to device and reshape
        data, target = data.to(device), target.to(device)
        B = target.size()[0]

        for p in range(time_steps):

            if p == 0:
                h = model.init_hidden(B)
            elif p % omega == 0:
                h = tuple(v.detach() for v in h)

            o, h = model.forward(data, h)

            # get prediction
            if p == (time_steps - 1):
                pred = o.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            if p % omega == 0 and p > 0:
                optimizer.zero_grad()

                # classification loss
                clf_loss = (p + 1) / k_updates * F.nll_loss(o, target)
                # clf_loss = snr*F.cross_entropy(output, target,reduction='none')
                # clf_loss = torch.mean(clf_loss)

                # regularizer loss
                regularizer = get_regularizer_named_params(named_params, _lambda=1.0)

                # mem potential loss take l1 norm / num of neurons /batch size
                energy = (p + 1) / k_updates * ((torch.norm(h[1], p=1) + torch.norm(h[5], p=1) + torch.norm(h[9], p=1)
                                                 + torch.norm(h[13], p=1)) / B / model.neuron_count)
                # overall loss
                loss = clf_alpha * clf_loss + regularizer + energy_alpha * energy

                loss.backward()

                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

                optimizer.step()
                post_optimizer_updates(named_params)

                train_loss += loss.item()
                total_clf_loss += clf_loss.item()
                total_regularizaton_loss += regularizer  # .item()
                total_energy_loss += energy.item()

        if batch_idx > 0 and batch_idx % log_interval == (log_interval - 1):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tlr: {:.6f}\ttrain acc:{:.4f}\tLoss: {:.6f}\
                \tClf: {:.6f}\tReg: {:.6f}\tFr_p: {:.6f}\tFr_r: {:.6f}'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), lr, 100 * correct / (log_interval * B),
                       train_loss / log_interval,
                       total_clf_loss / log_interval, total_regularizaton_loss / log_interval,
                       model.pc_layer.fr_p / time_steps / log_interval,
                       model.pc_layer.fr_r / time_steps / log_interval))

            wandb.log({
                'clf_loss': total_clf_loss / log_interval / k_updates,
                'train_acc': 100 * correct / (log_interval * B),
                'regularisation_loss': total_regularizaton_loss / log_interval / k_updates,
                'energy_loss': total_energy_loss / log_interval / k_updates,
                'total_loss': train_loss / log_interval / k_updates,
                'pred spiking freq': model.pc_layer.fr_p / time_steps / log_interval,  # firing per time step
                'rep spiking fr': model.pc_layer.fr_r / time_steps / log_interval,
                'conv1 spk fr': model.fr_conv1 / time_steps / log_interval, 
                'conv2 spk fr': model.fr_conv2 / time_steps / log_interval, 
            })

            train_loss = 0
            total_clf_loss = 0
            total_regularizaton_loss = 0
            total_energy_loss = 0
            correct = 0
        # model.network.fr = 0
        model.pc_layer.fr_p = 0
        model.pc_layer.fr_r = 0

        model.fr_conv1 = 0
        model.fr_conv2 = 0


def train_bp_conv(epoch, batch_size, log_interval,
                    train_loader, model, named_params,
                    time_steps, k_updates, omega, optimizer,
                    clf_alpha, energy_alpha, clip, lr):
    train_loss = 0
    total_clf_loss = 0
    total_regularizaton_loss = 0
    total_energy_loss = 0
    correct = 0
    model.train()

    # for each batch
    for batch_idx, (data, target) in enumerate(train_loader):

        # to device and reshape
        data, target = data.to(device), target.to(device)
        B = target.size()[0]

        for p in range(time_steps):

            if p == 0:
                h_conv = model.init_hidden(B)
                h_pc = model.pc_layer.init_hidden(B)
            elif p % omega == 0:
                h_conv = tuple(v.detach() for v in h_conv)
                h_pc = tuple(v.detach() for v in h_pc)

            o, h_conv, h_pc = model.forward(data, h_conv, h_pc)

            # get prediction
            if p == (time_steps-1): 
                pred = o.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            optimizer.zero_grad()

            # classification loss
            clf_loss = (p + 1) / k_updates * F.nll_loss(o, target)
            # clf_loss = snr*F.cross_entropy(output, target,reduction='none')
            # clf_loss = torch.mean(clf_loss)

            # mem potential loss take l1 norm / num of neurons /batch size
            energy = (torch.norm(h_pc[1], p=1) + torch.norm(h_pc[5], p=1)) / B / (model.input_to_pc_sz + model.classify_population_sz)

            # overall loss
            loss = clf_alpha * clf_loss + energy_alpha * energy

            loss.backward()

            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()
            post_optimizer_updates(named_params)

            train_loss += loss.item()
            total_clf_loss += clf_loss.item()
            total_energy_loss += energy.item()

        if batch_idx > 0 and batch_idx % log_interval == (log_interval - 1):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tlr: {:.6f}\ttrain acc:{:.4f}\tLoss: {:.6f}\
                \tClf: {:.6f}\tReg: {:.6f}\tFr_p: {:.6f}\tFr_r: {:.6f}'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), lr, 100 * correct / (log_interval * B),
                       train_loss / log_interval,
                       total_clf_loss / log_interval, total_regularizaton_loss / log_interval,
                       model.pc_layer.fr_p / time_steps / log_interval,
                       model.pc_layer.fr_r / time_steps / log_interval))

            wandb.log({
                'clf_loss': total_clf_loss / log_interval / k_updates,
                'train_acc': 100 * correct / (log_interval * B),
                'regularisation_loss': total_regularizaton_loss / log_interval / k_updates,
                'energy_loss': total_energy_loss / log_interval / k_updates,
                'total_loss': train_loss / log_interval / k_updates,
                'pred spiking freq': model.pc_layer.fr_p / time_steps / log_interval,  # firing per time step
                'rep spiking fr': model.pc_layer.fr_r / time_steps / log_interval,
                'conv1 spk fr': model.fr_conv1 / time_steps / log_interval, 
                'conv2 spk fr': model.fr_conv2 / time_steps / log_interval, 
            })

            train_loss = 0
            total_clf_loss = 0
            total_regularizaton_loss = 0
            total_energy_loss = 0
            correct = 0
        # model.network.fr = 0
        model.pc_layer.fr_p = 0
        model.pc_layer.fr_r = 0

        model.fr_conv1 = 0
        model.fr_conv2 = 0




# test function
def test_conv(model, test_loader, time_steps):
    model.eval()
    test_loss = 0
    correct = 0

    # for data, target in test_loader:
    for i, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)

        with torch.no_grad():
            model.eval()
            h_conv = model.init_hidden(data.size(0))
            h_pc = model.pc_layer.init_hidden(data.size(0))

            log_softmax_outputs, _, _ = model.inference(data, h_conv, h_pc, time_steps)

            test_loss += F.nll_loss(log_softmax_outputs[-1], target, reduction='sum').data.item()

            pred = log_softmax_outputs[-1].data.max(1, keepdim=True)[1]

        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        torch.cuda.empty_cache()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    wandb.log({
        'test_loss': test_loss,
        'test_acc': test_acc
    })

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        test_acc))

    return test_loss, 100. * correct / len(test_loader.dataset)

# %%
