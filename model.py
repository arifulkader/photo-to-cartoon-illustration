"""The network used by both models (main model and face refiner). Inference only."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3), nn.InstanceNorm2d(ch), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3), nn.InstanceNorm2d(ch),
        )

    def forward(self, x):
        return x + self.body(x)


def conv_block(cin, cout, stride=1):
    return nn.Sequential(
        nn.ReflectionPad2d(1), nn.Conv2d(cin, cout, 3, stride=stride),
        nn.InstanceNorm2d(cout), nn.ReLU(True),
    )


class UNetGenerator(nn.Module):
    """3 downsamples (bottleneck at 1/8 resolution) with skip connections.

    depth=4 adds a fourth downsample: the ResBlocks then work at 1/16 resolution,
    so each output pixel is decided from twice as wide an area of the photo
    (a large face fits in view), and 2 extra ResBlocks refine at 1/8.
    """

    def __init__(self, ngf=32, n_blocks=6, depth=3):
        super().__init__()
        self.depth = depth
        c1, c2, c3, c4 = ngf, ngf * 2, ngf * 4, ngf * 8
        self.stem = nn.Sequential(
            nn.ReflectionPad2d(3), nn.Conv2d(3, c1, 7), nn.InstanceNorm2d(c1), nn.ReLU(True),
            conv_block(c1, c1),
        )
        self.down1 = nn.Sequential(conv_block(c1, c2, 2), conv_block(c2, c2))
        self.down2 = nn.Sequential(conv_block(c2, c3, 2), conv_block(c3, c3))
        self.down3 = conv_block(c3, c4, 2)
        self.blocks = nn.Sequential(*[ResBlock(c4) for _ in range(n_blocks)])
        if depth == 4:
            self.down4 = conv_block(c4, c4, 2)
            self.up4 = conv_block(c4, c4)
            self.dec4 = nn.Sequential(conv_block(c4 * 2, c4), ResBlock(c4), ResBlock(c4))
        self.up3 = conv_block(c4, c3)
        self.dec3 = nn.Sequential(conv_block(c3 * 2, c3), conv_block(c3, c3))
        self.up2 = conv_block(c3, c2)
        self.dec2 = nn.Sequential(conv_block(c2 * 2, c2), conv_block(c2, c2))
        self.up1 = conv_block(c2, c1)
        self.dec1 = nn.Sequential(conv_block(c1 * 2, c1), conv_block(c1, c1))
        self.out = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(c1, 3, 7), nn.Tanh())

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        s1 = self.stem(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        if self.depth == 4:
            s4 = self.down3(s3)
            b = self.blocks(self.down4(s4))
            b = self.dec4(torch.cat([self.up4(self._up(b, s4)), s4], 1))
        else:
            b = self.blocks(self.down3(s3))
        d3 = self.dec3(torch.cat([self.up3(self._up(b, s3)), s3], 1))
        d2 = self.dec2(torch.cat([self.up2(self._up(d3, s2)), s2], 1))
        d1 = self.dec1(torch.cat([self.up1(self._up(d2, s1)), s1], 1))
        return self.out(d1)


