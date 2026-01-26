import torch
import math
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LogMSELoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(LogMSELoss, self).__init__()

    def forward(self, inputs, targets, max_num, min_num):
        targets = (torch.log(targets+1) - min_num) / (max_num - min_num)
        inputs = inputs.reshape(len(targets))
        loss = (targets - inputs)**2
        return torch.mean(loss)

class DataUncertaintyLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DataUncertaintyLoss, self).__init__()

    def forward(self, pred, va, targets, max_num, min_num):
        targets = (torch.log(targets+1) - min_num) / (max_num - min_num)
        pred = pred.reshape(len(targets))
        loss = torch.log(va)/2 + torch.square(pred-targets)/(2*va) + math.log(2*math.pi)/2
        return torch.mean(loss)

class PairRankingLoss(torch.nn.Module):
    def __init__(self, margin=0.0, use_upper_triangle=True, reduction="mean", exp_clip=50.0):
        super().__init__()
        self.margin = margin
        self.use_upper_triangle = use_upper_triangle
        self.reduction = reduction
        self.exp_clip = exp_clip  # to avoid overflow in exp

    def forward(self, cost, targets, max_num, min_num):
        n = len(targets)

        targets = (torch.log(targets + 1) - min_num) / (max_num - min_num)
        cost = cost.reshape(n)

        targets_m = targets[:, None] - targets[None, :]
        cost_m    = cost[:, None] - cost[None, :]

        y = torch.sign(targets_m).detach()
        valid = (y != 0)

        if self.use_upper_triangle:
            tri = torch.triu(torch.ones_like(valid, dtype=torch.bool), diagonal=1)
            valid = valid & tri

        yv = y[valid]
        dv = cost_m[valid]

        s = yv * dv
        wrong = (s < 0)

        # only penalize wrong pairs
        z = (self.margin - s).clamp(max=self.exp_clip)
        loss = torch.where(wrong, torch.exp(z)-math.exp(self.margin), torch.zeros_like(z))

        if self.reduction == "mean":
            return loss.mean() if loss.numel() > 0 else dv.new_tensor(0.0)
        if self.reduction == "sum":
            return loss.sum()
        return loss

class ExplanationLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(ExplanationLoss, self).__init__()
        self.explanation_loss = torch.nn.MSELoss()

    def forward(self, expl, local_labels):
        expl = expl.reshape(-1)
        loss = self.explanation_loss(expl, local_labels)

        return loss


