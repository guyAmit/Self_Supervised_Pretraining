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
        assert sim_matrix.shape == (self.n_views*batch_size,
                                    self.n_views*batch_size)
        assert sim_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool,
                         device=self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        sim_matrix = sim_matrix[~mask].view(sim_matrix.shape[0], -1)
        assert sim_matrix.shape == labels.shape

        # select positives
        positives = sim_matrix[labels].view(labels.shape[0], -1)
        # select negatives
        negatives = sim_matrix[~labels].view(sim_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        logits = logits / self.temperature

        labels = torch.zeros(logits.shape[0], dtype=torch.long,
                             device=self.device)
        return self.criterion(logits, labels)


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
