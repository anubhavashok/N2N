import argparse
parser = argparse.ArgumentParser(description='N2N: Network to Network Compressin using Policy Gradient Reinforcement Learning')


parser.add_argument('mode', type=str, choices=['removal', 'shrinkage'],
                    help='Which mode to run the program')
parser.add_argument('dataset', type=str, choices=['mnist', 'cifar10', 'cifar100', 'svhn', 'caltech256'],
                    help='Name of dataset')
parser.add_argument('teacherModel', type=str,
                    help='Path to teacher model')
parser.add_argument('--model', type=str, required=False,
                    help='Path to base model architecture if different from teacherModel')
parser.add_argument('--cuda', type=bool, required=False, default=True,
                    help='Use GPU or not')
parser.add_argument('--gpuids', type=list, required=False, default=[0],
                    help='Which GPUs to use')

args = parser.parse_args()

# Load dataset
if args.dataset == 'mnist':
    import datasets.mnist as dataset
elif args.dataset == 'cifar10':
    import datasets.cifar10 as dataset
elif args.dataset == 'cifar100':
    import datasets.cifar100 as dataset
elif args.dataset == 'svhn':
    import datasets.svhn as dataset
elif args.dataset == 'caltech256':
    import datasets.caltech256 as dataset
else:
    print('Dataset not found: ' + args.dataset)

# Load teacherModel
# Load baseModel (if available)
# Initialize controller based on mode
