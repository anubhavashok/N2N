import torch
from torch.utils import data
from torch import nn
from torchvision import transforms
from torch.autograd import Variable
from os.path import join
from PIL import Image
from glob import glob
import numpy as np
from torch import optim

args = lambda: None
args.cuda = True
args.batch_size = 128

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img.resize((224, 224))

mean = torch.FloatTensor([0.485, 0.456, 0.406])
std = torch.FloatTensor([0.229, 0.224, 0.225])
img_transform = transforms.Compose([
transforms.Scale(256),
transforms.RandomCrop(224),
transforms.RandomHorizontalFlip(),
transforms.ToTensor(),
transforms.Normalize(mean,std)
])

class Caltech256(data.Dataset):
    def __init__(self, split='train', path='/home/anubhava/ArchSearch/data/caltech256/256_ObjectCategories'):
        super(Caltech256, self).__init__()
        self.path = path
        self.split = split
        self.filepaths = glob(join(self.path, '*/*.jpg'))
        n = len(self.filepaths)
        train_paths, test_paths = self.get_splits(self.path, 1001)
        if split == "train":
            self.filepaths = train_paths#list(map(lambda i: self.filepaths[i], train_paths))
        else:
            #test_choices = filter(lambda i: i not in train_choices, range(len(self.filepaths)))
            self.filepaths = test_paths#list(map(lambda i: self.filepaths[i], test_paths))
        self.targets = [f.split('/')[-1] for f in glob(join(self.path, '*'))]
    
    def get_splits(self, base_path, seed=1000):
        np.random.seed(seed)
        train_files = []
        test_files = []
        # From each class select 10% at random
        classes = [f.split('/')[-1] for f in glob(join(base_path, '*'))]
        for c in classes:
            files = glob(join(base_path, c, '*'))
            n = len(files)
            #train = np.random.choice(files, int(n*0.8), replace=False)
            train = np.random.choice(files, n - 15, replace=False)
            test = filter(lambda x: x not in train, files)
            train_files.extend(train)
            test_files.extend(test)
        return train_files, test_files
    
    def __getitem__(self, index):
        filepath = self.filepaths[index]
        img = img_transform(load_img(filepath))
        # Scale and convert to tensor
        target = torch.Tensor([self.targets.index(filepath.split('/')[-2])])
        return img, target
    
    def __len__(self):
        return len(self.filepaths)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(Caltech256(split='train'), batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(Caltech256(split='test'), batch_size=args.batch_size, **kwargs) 
optimizer = None
ceLoss = nn.CrossEntropyLoss()
lr = 0.01
lr_decay = 10

def lr_schedule(optimizer, epoch):
    new_lr = lr / pow(10, epoch // lr_decay)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return optimizer

def train(epoch):
    global optimizer
    global avg_loss
    if epoch == 1:
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, nesterov=True)
        #optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    optimizer = lr_schedule(optimizer, epoch)
    correct = 0
    net.train()
    for b_idx, (data, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        data, targets = Variable(data).cuda(), Variable(targets.long().squeeze()).cuda().detach()
        output = net(data)
        loss = ceLoss(output, targets)
        loss.backward()
        optimizer.step()
        
        # compute the accuracy
        pred = output.data.max(1)[1].squeeze() # get the index of the max log-probability
        correct += pred.eq(targets.data).sum()
        
        if b_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (b_idx+1) * len(data), len(train_loader.dataset),
                100. * (b_idx+1)*len(data) / len(train_loader.dataset), loss.data[0]))
    # now that the epoch is completed plot the accuracy
    train_accuracy = correct / float(len(train_loader.dataset))
    print("training accuracy ({:.2f}%)".format(100*train_accuracy))
    return (train_accuracy*100.0)


best_accuracy = 0.0

def test():
    net.eval()
    global best_accuracy
    correct = 0
    for idx, (data, target) in enumerate(test_loader):
        data, target = Variable(data, volatile=True).cuda(), Variable(target.long().squeeze()).cuda()

        # do the forward pass
        score = net.forward(data)
        pred = score.data.max(1)[1] # got the indices of the maximum, match them
        correct += pred.eq(target.data).cpu().sum()

    print("predicted {} out of {}".format(correct, len(test_loader.dataset)))
    val_accuracy = correct / float(len(test_loader.dataset)) * 100.0
    print("accuracy = {:.2f}".format(val_accuracy))

    # now save the model if it has better accuracy than the best model seen so forward
    return val_accuracy/100.0
 
