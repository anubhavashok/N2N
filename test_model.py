import torch
import argparse
import warnings
warnings.filterwarnings('ignore')
with warnings.catch_warnings():
    warnings.simplefilter('ignore')

parser = argparse.ArgumentParser(description='Test model')
parser.add_argument('model', type=str,
                    help='Path to model')
parser.add_argument('--cuda', type=str, default=True,
                    help='Use GPU or not')
parser.add_argument('dataset', type=str, choices=['mnist', 'cifar10', 'cifar10_old', 'cifar100', 'svhn', 'caltech256'],
                    help='Name of dataset')
args = parser.parse_args()

# ----DATASETS----
if args.dataset == 'mnist':
    import datasets.mnist as dataset
elif args.dataset == 'cifar10':
    import datasets.cifar10 as dataset
elif args.dataset == 'cifar10_old':
    import datasets.cifar10_old as dataset
elif args.dataset == 'cifar100':
    import datasets.cifar100 as dataset
elif args.dataset == 'svhn':
    import datasets.svhn as dataset
elif args.dataset == 'caltech256':
    import datasets.caltech256 as dataset
elif args.dataset == 'imagenet':
    import datasets.imagenet as dataset
else:
    print('Dataset not found: ' + args.dataset)
    quit()


model = torch.load(args.model)
dataset.net = model.cuda() if args.cuda else model

acc = dataset.test()
