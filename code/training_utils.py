import torch
from torch.cuda.amp import autocast


def train_epoch(net, trainloader, loss_func,
                optimizer, scaler, args):

    net.train()
    train_loss = 0
    for batch_idx, inputs in enumerate(trainloader):
        batch_size = (inputs.size(0) if type(inputs)
                      is not list else inputs[0].size(0))
        optimizer.zero_grad()
        with autocast(enabled=args.grade_scale):
            loss = loss_func(inputs)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
    train_loss = train_loss/(batch_idx+1)
    return train_loss


def test_net(net, testloader, loss_func):
    net.eval()
    test_loss = 0
    total = 0
    with torch.no_grad():
        for batch_idx, inputs in enumerate(testloader):
            batch_size = (inputs.size(0) if type(inputs)
                          is not list else inputs[0].size(0))
            loss = loss_func(inputs)
            test_loss += loss.item()
            total += batch_size
    test_loss = test_loss/(batch_idx+1)
    return test_loss
