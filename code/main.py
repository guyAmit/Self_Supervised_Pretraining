import argparse

from Self_Supervised_Pretraining.code.utils import train
from .resnets import build_net

parser = argparse.ArgumentParser(
    description='PyTorch Self-Supervised Pretraining')
parser.add_argument('-type',  default='SimCLR',
                    help='Select the pretraining type',
                    choices=['SimCLR', 'InPainting'])
parser.add_argument('--dataset-name', default='stl10',
                    help='dataset name', choices=['stl10', 'cifar10'])
parser.add_argument('--arch', metavar='ARCH', default='resnet18',
                    choices=['resnet18', 'resnet34'], help='achitecture type')

parser.add_argument('--workers', default=6, type=int,
                    help='number of data loading workers for each loader')
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total')

parser.add_argument('--opti', '--optimizer', default='SGD',
                    help='Select the otimizer type',
                    choices=['SGD', 'Adam'])

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='SGD optimizer momentum (default:0.9 for SGD)',)
parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n_views', default=2, type=int,
                    help='Number of views for contrastive learning.')
parser.add_argument('--mask_size', default=8, type=int,
                    help='Maksed area size (size, size).')

parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')


def main():
    args = parser.parse_args()
    device = (torch.device(f"cuda:{args.device}")
              if torch.cuda.is_available() else "cpu")
    net = build_net(args).to(device)
    train(net, device, args)


if __name__ == "__main__":
    main()
