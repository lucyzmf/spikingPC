# %%
###############################################################################################
##########################                  FTTP                ###############################
###############################################################################################

# fptt parameters
import torch
import torch.nn.functional as F
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

alpha = .2
beta = .5
rho = 0.


# %%
def get_stats_named_params(model):
    named_params = {}
    for name, param in model.named_parameters():
        sm, lm, dm = param.detach().clone(), 0.0 * param.detach().clone(), 0.0 * param.detach().clone()
        named_params[name] = (param, sm, lm, dm)
    return named_params


def post_optimizer_updates(named_params):
    for name in named_params:
        param, sm, lm, dm = named_params[name]
        lm.data.add_(-alpha * (param - sm))
        sm.data.mul_((1.0 - beta))
        sm.data.add_(beta * param - (beta / alpha) * lm)


def get_regularizer_named_params(named_params, _lambda=1.0):
    regularization = torch.zeros([], device=device)
    for name in named_params:
        param, sm, lm, dm = named_params[name]
        regularization += (rho - 1.) * torch.sum(param * lm)
        r_p = _lambda * 0.5 * alpha * torch.sum(torch.square(param - sm))
        regularization += r_p
        # print(name,r_p)
    return regularization


def reset_named_params(named_params):
    for name in named_params:
        param, sm, lm, dm = named_params[name]
        param.data.copy_(sm.data)


def train_fptt(epoch, batch_size, log_interval,
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
        data = data.view(-1, model.in_dim)

        B = target.size()[0]

        for p in range(time_steps):

            if p == 0:
                h = model.init_hidden(data.size(0))
            elif p % omega == 0:
                h = tuple(v.detach() for v in h)

            o, h = model.forward(data, h)
            # wandb.log({
            #         'rec layer adap threshold': h[5].detach().cpu().numpy(),
            #         'rec layer mem potential': h[3].detach().cpu().numpy()
            #     })

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
                energy = (torch.norm(h[1], p=1) + torch.norm(h[5], p=1)) / B / (784 + 100)

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
                       model.fr_p / time_steps / log_interval,
                       model.fr_r / time_steps / log_interval))

            wandb.log({
                'clf_loss': total_clf_loss / log_interval / k_updates,
                'train_acc': 100 * correct / (log_interval * B),
                'regularisation_loss': total_regularizaton_loss / log_interval / k_updates,
                'energy_loss': total_energy_loss / log_interval / k_updates,
                'total_loss': train_loss / log_interval / k_updates,
                'pred spiking freq': model.fr_p / time_steps / log_interval,  # firing per time step
                'rep spiking fr': model.fr_r / time_steps / log_interval,
            })

            train_loss = 0
            total_clf_loss = 0
            total_regularizaton_loss = 0
            total_energy_loss = 0
            correct = 0
        # model.network.fr = 0
        model.fr_p = 0
        model.fr_r = 0


def train_fptt_seq(epoch, batch_size, log_interval,
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
        data = data.view(-1, time_steps, model.in_dim)

        B = target.size()[0]

        for t in range(time_steps):

            if t == 0:
                h = model.init_hidden(data.size(0))
            elif t % omega == 0:
                h = tuple(v.detach() for v in h)

            o, h = model.forward(data[:, t, :], h)

            pred = o.data.max(1, keepdim=True)[1]
            correct += pred.eq(target[:, t].data.view_as(pred)).cpu().sum()

            if t % omega == 0 and t > 0:
                optimizer.zero_grad()

                # classification loss
                clf_loss = F.nll_loss(o, target[:, t])
                # clf_loss = (t % int(k_updates/2) + 1) / (int(k_updates/2)) *F.nll_loss(o, target[:, t])
                # clf_loss = snr*F.cross_entropy(output, target,reduction='none')
                # clf_loss = torch.mean(clf_loss)

                # regularizer loss
                regularizer = get_regularizer_named_params(named_params, _lambda=1.0)

                # mem potential loss take l1 norm / num of neurons /batch size
                energy = (torch.norm(h[1], p=1) + torch.norm(h[5], p=1)) / B / sum(model.hidden_dims)

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
                100. * batch_idx / len(train_loader), lr, 100 * correct / (log_interval * B * time_steps),
                train_loss / log_interval / B,
                total_clf_loss / log_interval / B, total_regularizaton_loss / log_interval / B,
                model.fr_p / time_steps / log_interval,
                model.fr_r / time_steps / log_interval))

            wandb.log({
                'clf_loss': total_clf_loss / log_interval / B,
                'train_acc': 100 * correct / (log_interval * B * time_steps),
                'regularisation_loss': total_regularizaton_loss / log_interval / B,
                'energy_loss': total_energy_loss / log_interval / B,
                'total_loss': train_loss / log_interval / B,
                'pred spiking freq': model.fr_p / time_steps / log_interval,  # firing per time step
                'rep spiking fr': model.fr_r / time_steps / log_interval,
            })

            train_loss = 0
            total_clf_loss = 0
            total_regularizaton_loss = 0
            total_energy_loss = 0
            correct = 0
        # model.network.fr = 0
        model.fr_p = 0
        model.fr_r = 0
