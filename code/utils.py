import numpy as np

from torch import optim
from torch.cuda.amp import GradScaler

from .datasets import get_dataloaders
from .loss_functions import SimClr_loss, InPainting_Loss
from .training_utils import train_epoch, test_net


def get_optimizer(net, args):
    if args.opti == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
        return optimizer
    elif args.opti == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
        return optimizer
    else:
        print('Not Implemented')
        exit(-1)


def get_loss_function(net, device, args):
    if args.type == 'SimCLR':
        return SimClr_loss(net, device, args)
    elif args.type == 'InPainting':
        return InPainting_Loss(net, device, args)
    else:
        print('Not Implemented')
        quit(-1)


def train(net, device, args):
    best_loss = np.inf
    optimizer = get_optimizer(net, args)
    trainloader, validloader = get_dataloaders(args)
    loss_func = get_loss_function(net, device, args)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=len(
                                                               trainloader),
                                                           eta_min=0,
                                                           last_epoch=-1)
    scaler = GradScaler(enabled=True)
    for epoch in range(args.epochs):
        train_loss = train_epoch(net, trainloader,
                                 loss_func, optimizer,
                                 scaler, device)
        test_loss = test_net(net, validloader,
                             loss_func, device)
        print(
            f'epoch ({epoch+1})| Train loss {round(train_loss, 3)} | Test loss {round(test_loss, 3)}')
        if best_loss > test_loss:
            print('Saving model...')
            model_state = {'net': net.state_dict(),
                           'loss': test_loss, 'epoch': epoch}
            torch.save(model_state, f'./models/{args.type}.ckpt.pth')
            best_loss = test_loss
        if epoch <= 10:
            scheduler.step()
