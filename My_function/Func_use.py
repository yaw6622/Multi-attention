import torch
import torch.nn as nn
def N_Parameter(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def value(self):
        return self.avg if self.count > 0 else 0

class IoU:
    def __init__(self, num_classes, ignore_index=None, device=None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = device
        self.conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.float32, device=device)

    def add(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)

        if self.ignore_index is not None:
            mask = target != self.ignore_index
            pred = pred[mask]
            target = target[mask]

        conf_matrix = self._generate_matrix(pred, target)
        self.conf_matrix += conf_matrix

    def _generate_matrix(self, pred, target):
        mask = (target >= 0) & (target < self.num_classes)
        conf_matrix = torch.bincount(
            self.num_classes * target[mask] + pred[mask], minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes).float()
        return conf_matrix

    def get_miou_acc(self):
        intersection = torch.diag(self.conf_matrix)
        union = self.conf_matrix.sum(1) + self.conf_matrix.sum(0) - intersection
        miou = (intersection / union).mean().item()
        acc = intersection.sum().item() / self.conf_matrix.sum().item()
        return miou, acc

    def confusion_matrix(self):
        return self.conf_matrix.clone()

    def reset(self):
        self.conf_matrix.zero_()