import math
from torch.optim.lr_scheduler import LRScheduler

from ..core import register


@register()
class CosineAnnealingRatioLR(LRScheduler):
    # Cosine annealing per param group: lr_i(t) = base_i * (r + (1 - r) * (1 + cos(pi * t / T_max)) / 2).
    # Stepped per epoch by the repo solver (only after LinearWarmup has finished).
    def __init__(self, optimizer, T_max, min_lr_ratio=0.1, last_epoch=-1):
        self.T_max = max(1, int(T_max))
        self.min_lr_ratio = float(min_lr_ratio)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = min(max(self.last_epoch, 0), self.T_max)
        f = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * t / self.T_max))
        return [base * f for base in self.base_lrs]
