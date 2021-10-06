import logging

import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler

from datasets import get_dataloaders
from lars import LARS
from loss_functions import (InPainting_Loss, SimClr_2views_loss, SimClr_loss,
                            VICReg_Loss)
from training_utils import test_net, train_epoch
from consts import train_msg, save_path


def get_optimizer(net, args):
    if args.opti == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.opti == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
    elif args.opti == 'LARS':
        optimizer = LARS(net.parameters(), lr=args.batch_size/(args.lr*256),
                         weight_decay=args.weight_decay)
    else:
        print('Not Implemented')
        exit(-1)
    return optimizer


def get_loss_function(net, device, args):
    if args.type == 'SimCLR':
        if args.n_views == 2:
            return SimClr_2views_loss(net, device, args)
        elif args.n_views > 2:
            return SimClr_loss(net, device, args)
        else:
            print('number of views must be at least 2')
    elif args.type == 'InPainting':
        return InPainting_Loss(net, device, args)
    elif args.type == 'VICReg':
        return VICReg_Loss(net, device, args)
    else:
        print('Not Implemented')
        quit(-1)


def save_moedel(net, test_loss, epoch, args):
    print('Saving model...')
    logging.info('Saving model...')
    model_state = {'net': net.state_dict(),
                   'loss': test_loss, 'epoch': epoch}
    f'./models/{args.type}_{args.arch}_{args.dataset}.ckpt.pth'
    torch.save(
        model_state, save_path.format(args.type, args.arch,
                                      args.dataset))


def train(net, device, args):
    logging.baseconfig(filename=f'{args.type}_{args.arch}_{args.dataset}.log',
                       level=logging.INFO)
    best_loss = np.inf
    optimizer = get_optimizer(net, args)
    trainloader, validloader = get_dataloaders(args)
    loss_func = get_loss_function(net, device, args)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=len(
                                                               trainloader),
                                                           eta_min=0,
                                                           last_epoch=-1)
    scaler = GradScaler(enabled=args.grade_scale)
    for epoch in range(args.epochs):
        train_loss = train_epoch(net, trainloader,
                                 loss_func, optimizer,
                                 scaler, args)
        test_loss = test_net(net, validloader,
                             loss_func)
        print(train_msg.format(epoch+1, round(train_loss, 6),
                               round(test_loss, 6)))
        logging.info(train_msg.format(epoch+1, round(train_loss, 6),
                                      round(test_loss, 6)))
        if best_loss > test_loss:
            save_moedel(net, test_loss, epoch, args)
            best_loss = test_loss
        if epoch <= 10:
            scheduler.step()
