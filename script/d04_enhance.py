
import torch.nn as nn


class D04EnhanceStem(nn.Module):
    """
    D04 detection-embedded enhancement stem.
    Preserve the original YOLO Conv parameter names and
    initialize the residual enhancement branch near identity.
    """

    def __init__(self, c1, c2):
        super().__init__()

        # Keep names compatible with Ultralytics Conv:
        # conv.weight, bn.weight, bn.bias, etc.
        self.conv = nn.Conv2d(
            c1, c2, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

        self.refine = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
            nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c2),
        )

        # Start with a zero residual so the module initially
        # behaves like the original Conv as closely as possible.
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)


    def forward(self, x):
        base = self.bn(self.conv(x))
        y = self.act(base)
        return self.act(base + self.refine(y))
