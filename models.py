import torch
from torch import nn
import torch.nn.functional as F


def conv_block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                         nn.GroupNorm(min(8, cout), cout), nn.LeakyReLU(.2, inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                         nn.GroupNorm(min(8, cout), cout), nn.LeakyReLU(.2, inplace=True))


class ResidualUNet(nn.Module):
    """Five down/up levels; bilinear upsampling avoids transposed-conv checkerboards."""
    def __init__(self, base=64):
        super().__init__()
        if base < 8 or base % 8:
            raise ValueError('base must be a positive multiple of 8')
        self.base = base
        channels = [base, base*2, base*4, base*8, base*8]
        self.enc = nn.ModuleList([conv_block(a, b) for a,b in zip([1]+channels[:-1], channels)])
        self.bottleneck = conv_block(channels[-1], channels[-1])
        self.dec = nn.ModuleList()
        previous = channels[-1]
        for width in reversed(channels):
            self.dec.append(conv_block(previous+width, width))
            previous = width
        self.out = nn.Conv2d(base, 1, 1)
        # Identity start: network initially returns the limited-angle reconstruction.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        original, skips = x, []
        for block in self.enc:
            x = block(x)
            skips.append(x)
            x = F.avg_pool2d(x, 2)
        x = self.bottleneck(x)
        for block, skip in zip(self.dec, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        # No clamp during training: saturated values must retain gradients.
        return original + self.out(x)


class PatchGAN(nn.Module):
    """70x70 receptive field: kernel4 strides 2,2,2,1,1, including logits head.

    Four feature convolutions use 64,128,256,512 channels. Making all four
    stride 2 would yield a 142x142 discriminator after the output head.
    """
    def __init__(self, base=64):
        super().__init__()
        layers, cin = [], 2
        for i, cout in enumerate([base, base*2, base*4, base*8]):
            layers.append(nn.Conv2d(cin, cout, 4, stride=2 if i < 3 else 1, padding=1))
            if i:
                layers.append(nn.InstanceNorm2d(cout))
            layers.append(nn.LeakyReLU(.2, inplace=True))
            cin = cout
        layers.append(nn.Conv2d(cin, 1, 4, 1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, condition, candidate):
        return self.net(torch.cat([condition, candidate], dim=1))


def ssim_loss(x, y):
    # Local 11x11 SSIM with fixed data range 1 (training surrogate).
    ux, uy = F.avg_pool2d(x, 11, 1), F.avg_pool2d(y, 11, 1)
    vx = F.avg_pool2d(x*x, 11, 1)-ux*ux
    vy = F.avg_pool2d(y*y, 11, 1)-uy*uy
    cov = F.avg_pool2d(x*y, 11, 1)-ux*uy
    score = ((2*ux*uy+.01**2)*(2*cov+.03**2) /
             ((ux*ux+uy*uy+.01**2)*(vx+vy+.03**2)))
    return 1-score.mean()
