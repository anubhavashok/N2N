import torch
from torchvision import datasets, transforms, models
from torch import nn
import torch.optim as optim
from torch.autograd import Variable

net = None
best_accuracy = 0
batch_size = 200
kwargs = {'num_workers': 1, 'pin_memory': True}

args = lambda: None
args.cuda = True
args.batch_size = 128

def target_transform(target):
    return int(target[0]) - 1

train_loader = torch.utils.data.DataLoader(
    datasets.SVHN('./', split='train', download=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ]),
                   target_transform=target_transform
    ),
    batch_size=batch_size, shuffle=True, **kwargs)


test_loader = torch.utils.data.DataLoader(
    datasets.SVHN('./', split='test', download=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ]),
                   target_transform=target_transform
    ),
    batch_size=batch_size, shuffle=False, **kwargs)

def test():
    net.eval()
    global best_accuracy
    correct = 0
    for idx, (data, target) in enumerate(test_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)

        # do the forward pass
        score = net.forward(data)
        pred = score.data.max(1)[1] # got the indices of the maximum, match them
        correct += pred.eq(target.data).cpu().sum()

    print("predicted {} out of {}".format(correct, len(test_loader.dataset)))
    val_accuracy = correct / float(len(test_loader.dataset)) * 100.0
    print("accuracy = {:.2f}".format(val_accuracy))

    # now save the model if it has better accuracy than the best model seen so forward
    return val_accuracy/100.0
