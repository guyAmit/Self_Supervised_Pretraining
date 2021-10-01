import argparse

import torch

from resnets import build_net
from utils import train

parser = argparse.ArgumentParser(
    description='PyTorch Self-Supervised Pretraining')
parser.add_argument('-type',  default='SimCLR',
                    help='Select the pretraining type',
                    choices=['SimCLR', 'InPainting'])
parser.add_argument('--dataset', default='stl10',
                    help='dataset name(stl10, cifar10, ...), \
                    if costum data, place it in the data folder')
parser.add_argument('--arch', default='Resnet18',
                    choices=['Resnet18', 'Resnet34'], help='achitecture type')

parser.add_argument('--workers', default=6, type=int,
                    help='number of data loading workers for each loader')
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128), this is the total')

parser.add_argument('--opti', default='SGD',
                    help='Select the otimizer type',
                    choices=['SGD', 'Adam'])

parser.add_argument('--lr', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='SGD optimizer momentum (default:0.9 for SGD)',)
parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n_views', default=2, type=int,
                    help='Number of views for contrastive learning.')
parser.add_argument('--projection_size', default=128, type=int,
                    help='Size of projection head for SimCLR (default: 128)')
parser.add_argument('--mask_size', default=8, type=int,
                    help='Maksed area size (size, size).')

parser.add_argument('--device', default=0, type=int, help='Gpu index.')


def main():
    args = parser.parse_args()
    device = (torch.device(f"cuda:{args.device}")
              if torch.cuda.is_available() else "cpu")
    net = build_net(args).to(device)
    train(net, device, args)


if __name__ == "__main__":
    main()
