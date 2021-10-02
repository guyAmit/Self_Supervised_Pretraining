import torch
import torch.nn as nn


class SimClr_loss(nn.Module):

    def __init__(self, net, device, args):
        super(SimClr_loss, self).__init__()
        self.net = net
        self.n_views = args.n_views
        self.temperature = args.temperature
        self.device = device

    def forward(self, inputs):
        inputs = torch.cat(inputs, dim=0)
        _, features = self.net(inputs.to(self.device))
        batch_size = features.size(0) // self.n_views

        sim_matrix = torch.matmul(features, features.T)

        sim_ij = torch.diag(sim_matrix, batch_size)
        sim_ji = torch.diag(sim_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        negatives_mask = (~torch.eye(batch_size * 2, batch_size * 2,
                                     dtype=bool, device=self.device)).float()

        nominator = torch.exp(positives / self.temperature)
        denominator = negatives_mask * torch.exp(sim_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * batch_size)
        return loss


class InPainting_Loss(nn.Module):

    def __init__(self, net, device, args):
        super(InPainting_Loss, self).__init__()
        self.net = net
        self.device = device
        self.criterion = nn.MSELoss()

    def forward(self, inputs):
        inputs, masks = inputs
        inputs, masks = inputs.to(self.device), masks.to(self.device)
        _, preds = self.net(inputs)
        return self.criterion(preds, masks)
