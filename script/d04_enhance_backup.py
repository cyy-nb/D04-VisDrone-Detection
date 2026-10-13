import torch
import torch.nn as nn


class D04EnhanceStem(nn.Module):
    """
    D04 Detection-Embedded Enhancement Stem

    用于 YOLOv8s 前端的轻量特征增强。
    输入:
        [B, 3, H, W]

    输出:
        [B, c2, H/2, W/2]

    作用:
        1. 完成 YOLO 原始第一层的下采样
        2. 对浅层特征进行进一步增强
        3. 使用残差结构保留原始信息
    """

    def __init__(self, c1, c2):
        super().__init__()

        # 主分支：保持 YOLO 第一层的基本功能
        self.main = nn.Sequential(
            nn.Conv2d(
                c1,
                c2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

        # 特征细化分支
        self.refine = nn.Sequential(
            nn.Conv2d(
                c2,
                c2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),

            nn.Conv2d(
                c2,
                c2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(c2),
        )

        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        # YOLO 第一层特征
        y = self.main(x)

        # 特征增强
        r = self.refine(y)

        # Residual enhancement
        y = y + r

        return self.act(y)

