# %%
import torch
import wandb
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###############################################################################################
##########################          Test function             ###############################
###############################################################################################
# test function
def test(model, test_loader, time_steps):
    model.eval()
    test_loss = 0
    correct = 0

    # for data, target in test_loader:
    for i, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(-1, model.in_dim)

        with torch.no_grad():
            model.eval()
            hidden = model.init_hidden(data.size(0))

            log_softmax_outputs, hidden = model.inference(data, hidden, time_steps)

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


# test function for sequence data
def test_seq(model, test_loader, time_steps):
    model.eval()
    test_loss = 0
    correct = 0

    # for data, target in test_loader:
    for i, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(-1, model.in_dim)

        with torch.no_grad():
            model.eval()
            hidden = model.init_hidden(data.size(0))

            log_softmax_outputs, hidden, pred_hist = model.inference(data, hidden, time_steps)

            # compute loss at each time step
            for t in range(time_steps):
                test_loss += F.nll_loss(log_softmax_outputs[t], target, reduction='sum').data.item()

        correct += pred_hist.eq(target.data.view_as(pred_hist)).cpu().sum()
        torch.cuda.empty_cache()

    test_loss /= len(test_loader.dataset)  # per t loss
    test_acc = 100. * correct / len(test_loader.dataset) / time_steps
    wandb.log({
        'test_loss': test_loss,
        'test_acc': test_acc
    })

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        test_acc))

    return test_loss, 100. * correct / len(test_loader.dataset)
