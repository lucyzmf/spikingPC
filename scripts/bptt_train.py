# %%
###############################################################################################
##########################                  BPTT                ###############################
###############################################################################################

# fptt parameters
import torch
import torch.nn.functional as F
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_bptt(epoch, batch_size, log_interval,
               train_loader, model, 
               time_steps, optimizer,
               clf_alpha, energy_alpha, spike_alpha, clip, lr):
    train_loss = 0
    total_clf_loss = 0
    total_regularizaton_loss = 0
    total_energy_loss = 0
    total_spike_loss = 0
    correct = 0
    model.train()

    # for each batch
    for batch_idx, (data, target) in enumerate(train_loader):

        loss = 0

        # to device and reshape
        data, target = data.to(device), target.to(device)
        data = data.view(-1, model.in_dim)

        B = target.size()[0]

        optimizer.zero_grad()

        for p in range(time_steps):

            if p == 0:
                h = model.init_hidden(data.size(0))

            o, h = model.forward(data, h)
            # wandb.log({
            #         'rec layer adap threshold': h[5].detach().cpu().numpy(),
            #         'rec layer mem potential': h[3].detach().cpu().numpy()
            #     })

            # get prediction
            if p == (time_steps - 1):
                pred = o.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()


            # classification loss
            clf_loss = (p + 1) / time_steps * F.nll_loss(o, target)
            # clf_loss = snr*F.cross_entropy(output, target,reduction='none')
            # clf_loss = torch.mean(clf_loss)

            # mem potential loss take l1 norm / num of neurons /batch size
            if len(model.hidden_dims) == 2:
                energy = (torch.sum(model.error1 ** 2) + torch.sum(model.error2 ** 2)) / B / sum(model.hidden_dims)
                spike_loss = (torch.sum(h[1]) + torch.sum(h[5])) / B / sum(model.hidden_dims)
            elif len(model.hidden_dims) == 3:
                # energy = (torch.sum(model.error1 ** 2) + torch.sum(model.error2 ** 2) + torch.sum(model.error3 ** 2)) / B / sum(model.hidden_dims)
                energy = (torch.sum(torch.abs(model.error1)) + torch.sum(torch.abs(model.error2)) + torch.sum(torch.abs(model.error3))) / B / sum(model.hidden_dims)
                spike_loss = (torch.sum(h[1]) + torch.sum(h[5]) + torch.sum(h[9])) / B / sum(model.hidden_dims)

            # overall loss
            loss += clf_alpha * clf_loss + energy_alpha * energy + spike_alpha * spike_loss


            wandb.log({
                'layer1 error':(model.error1.detach().cpu().numpy() ** 2).mean(), 
                'layer2 error':(model.error2.detach().cpu().numpy() ** 2).mean(), 
            })

            model.error1 = 0
            model.error2 = 0    
            if len(model.hidden_dims) == 3:
                    model.error3 = 0

        loss.backward()

        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        train_loss += loss.item()
        total_clf_loss += clf_loss.item()
        total_energy_loss += energy.item()
        total_spike_loss += spike_loss.item()


        if batch_idx > 0 and batch_idx % log_interval == (log_interval - 1):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tlr: {:.6f}\ttrain acc:{:.4f}\tLoss: {:.6f}\
                \tClf: {:.6f}\tReg: {:.6f}\tFr_p: {:.6f}\tFr_r: {:.6f}'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), lr, 100 * correct / (log_interval * B),
                       train_loss / log_interval,
                       total_clf_loss / log_interval, total_regularizaton_loss / log_interval,
                       model.fr_layer2 / time_steps / log_interval,
                       model.fr_layer1 / time_steps / log_interval))

            wandb.log({
                'clf_loss': total_clf_loss / log_interval / time_steps,
                'train_acc': 100 * correct / (log_interval * B),
                'regularisation_loss': total_regularizaton_loss / log_interval / time_steps,
                'energy_loss': total_energy_loss / log_interval / time_steps,
                'spike loss': total_spike_loss / log_interval / time_steps,
                'total_loss': train_loss / log_interval / time_steps,
                'l2 fr': model.fr_layer2 / time_steps / log_interval,  # firing per time step
                'l1 fr': model.fr_layer1 / time_steps / log_interval,
            })

            if len(model.hidden_dims) == 3:
                wandb.log({'l3 fr': model.fr_layer3  / time_steps / log_interval})
                model.fr_layer3 = 0

            train_loss = 0
            total_clf_loss = 0
            total_regularizaton_loss = 0
            total_energy_loss = 0
            total_spike_loss = 0
            correct = 0
            # model.network.fr = 0
            model.fr_layer2 = 0
            model.fr_layer1 = 0


