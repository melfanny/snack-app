
import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class SPDConv(nn.Module):
    """Space-to-depth lalu konvolusi."""

    def __init__(self, c1, c2=256, *args, **kwargs):
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k=3, s=1, p=1)

    def forward(self, x):
        h, w = x.shape[-2:]

        if h % 2 or w % 2:
            raise ValueError(
                f"SPDConv membutuhkan H dan W genap: {(h, w)}"
            )

        x = torch.cat((
            x[..., 0::2, 0::2],
            x[..., 1::2, 0::2],
            x[..., 0::2, 1::2],
            x[..., 1::2, 1::2],
        ), dim=1)

        return self.conv(x)


class OmniKernelBlock(nn.Module):

    def __init__(self, c, e=0.5):
        super().__init__()

        hidden = max(8, int(c * e))

        self.branch3 = Conv(c, hidden, k=3, s=1, p=1)
        self.branch5 = Conv(c, hidden, k=5, s=1, p=2)
        self.fuse = Conv(hidden * 2, c, k=1, s=1)

    def forward(self, x):
        y = torch.cat((
            self.branch3(x),
            self.branch5(x)
        ), dim=1)

        return x + self.fuse(y)


class CSPOK(nn.Module):

    def __init__(
        self, c1, c2=256, n=1, e=0.5, *args, **kwargs
    ):
        super().__init__()

        hidden = max(8, int(c2 * e))
        n = max(1, int(n))

        self.cv1 = Conv(c1, hidden, k=1, s=1)
        self.cv2 = Conv(c1, hidden, k=1, s=1)

        self.m = nn.Sequential(
            *(OmniKernelBlock(hidden) for _ in range(n))
        )

        self.cv3 = Conv(hidden * 2, c2, k=1, s=1)

    def forward(self, x):
        a = self.m(self.cv1(x))
        b = self.cv2(x)

        return self.cv3(torch.cat((a, b), dim=1))
