import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torchvision import datasets, transforms
from torchvision import models

# Training settings
parser = argparse.ArgumentParser('PyTorch ImageNet Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='batch size of train')
parser.add_argument('--epochs', type=int, default=5, metavar='N', help='number of epochs to train for')
parser.add_argument('--learning-rate', type=float, default=1e-3, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='percentage of past parameters to store')
parser.add_argument('--no-cuda', action='store_true', default=False, help='use cuda for training')
parser.add_argument('--log-schedule', type=int, default=10, metavar='N', help='number of epochs to save snapshot after')
parser.add_argument('--seed', type=int, default=1, help='set seed to some constant value to reproduce experiments')
parser.add_argument('--model_name', type=str, default=None, help='Use a pretrained model')
parser.add_argument('--want_to_test', type=bool, default=False, help='make true if you just want to test')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    print('Using cuda')
    torch.cuda.manual_seed(args.seed)
    #torch.cuda.set_device(1)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
traindir = '/mnt/sdb1/anubhava/ILSVRC2012_img_train'
valdir = '/mnt/sdb1/anubhava/ILSVRC2012_img_val'

train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        **kwargs)

test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        **kwargs)


avg_loss = list()
best_accuracy = 0.0

# train the network
optimizer = None
def train(epoch):
    global optimizer
    if epoch == 1:
        #optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
        optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    global avg_loss
    correct = 0
    net.train()
    for b_idx, (data, targets) in enumerate(train_loader):
        if b_idx >= 250000:
            break
        if args.cuda:
            data, targets = data.cuda(), targets.cuda()
        # convert the data and targets into Variable and cuda form
        data, targets = Variable(data), Variable(targets)

        # train the network
        optimizer.zero_grad()
        scores = net.forward(data)
        loss = F.nll_loss(scores, targets)

        # compute the accuracy
        pred = scores.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(targets.data).cpu().sum()

        avg_loss.append(loss.data[0])
        loss.backward()
        optimizer.step()

        if b_idx % args.log_schedule == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (b_idx+1) * len(data), len(train_loader.dataset),
                100. * (b_idx+1)*len(data) / len(train_loader.dataset), loss.data[0]))

    # now that the epoch is completed plot the accuracy
    train_accuracy = correct / float(len(train_loader.dataset))
    print("training accuracy ({:.2f}%)".format(100*train_accuracy))
    return (train_accuracy*100.0)


def test():
    net.eval()
    global best_accuracy
    correct = 0
    for idx, (data, target) in enumerate(test_loader):
        if idx >= 100000:
            break
        if args.cuda:
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

