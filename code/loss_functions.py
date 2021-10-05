import torch
import torch.nn as nn


class SimClr_loss(nn.Module):

    def __init__(self, net, device, args):
        super(SimClr_loss, self).__init__()
        self.net = net
        self.n_views = args.n_views
        self.temperature = args.temperature
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, inputs):
        inputs = torch.cat(inputs, dim=0)
        _, features = self.net(inputs.to(self.device))
        batch_size = features.size(0) // self.n_views

        labels = torch.cat([torch.arange(batch_size)
                            for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1))
        labels = labels.to(self.device)

        sim_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = ~torch.eye(labels.shape[0], dtype=torch.bool,
                          device=self.device)
        labels = labels[mask].view(labels.shape[0], -1)
        sim_matrix = sim_matrix[mask].view(sim_matrix.shape[0], -1)

        # select positives
        positives = sim_matrix[labels].view(labels.shape[0], -1)
        # select negatives
        negatives = sim_matrix[~labels].view(sim_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        logits = logits / self.temperature

        labels = torch.zeros(logits.shape[0], dtype=torch.long,
                             device=self.device)
        return self.criterion(logits, labels)


class SimClr_2views_loss(nn.Module):

    def __init__(self, net, device, args):
        super(SimClr_2views_loss, self).__init__()
        self.net = net
        self.n_views = args.n_views
        self.temperature = args.temperature
        self.device = device

    def forward(self, inputs):
        inputs = torch.cat(inputs, dim=0)
        _, features = self.net(inputs.to(self.device))
        batch_size = features.size(0) // 2

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


class VICReg_Loss(nn.Module):

    def __init__(self, net, device, args):
        super(VICReg_Loss, self).__init__()
        self.net = net
        self.device = device
        self.lambd = args.lambd
        self.mu = args.mu
        self.nu = args.nu

        self.mse_loss = nn.MSELoss()
        self.relu = nn.ReLU()

    def std_loss(self, z):
        var = z.var(dim=0)
        loss = self.relu(1-var).mean()
        return loss

    def cov_loss(self, z):
        batch_size = z.size(0)
        Z = z - z.mean(dim=0, keepdims=True)
        cov = Z@Z.T / (batch_size-1)
        mask = (~torch.eye(batch_size, dtype=bool, device=self.device)).float()
        loss = ((mask*cov)**2).sum() / z.size(1)
        return loss

    def forward(self, inputs):
        inputs = torch.cat(inputs, dim=0)
        _, features = self.net(inputs.to(self.device))
        batch_size = features.size(0) // 2
        z_a = features[:batch_size]
        z_b = features[batch_size:]

        sim_loss = self.mse_loss(z_a, z_b)
        std_loss = self.std_loss(z_a)+self.std_loss(z_b)
        cov_loss = self.cov_loss(z_a)+self.cov_loss(z_b)

        return self.lambd*sim_loss + self.mu*std_loss+self.nu*cov_loss


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
