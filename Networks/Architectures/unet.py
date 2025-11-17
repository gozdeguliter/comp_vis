import torch
import torch.nn as nn
import torch.nn.functional as F

######################################################################################
#
# U-Net style architecture for semantic segmentation
# An instance of Net is created in model.py
#
######################################################################################


class DoubleConv(nn.Module):
    """
    (Conv2d -> BN -> ReLU) x 2
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """
    Downscaling with maxpool then double conv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    """
    Upscaling then double conv.
    Uses bilinear upsampling + conv for simplicity.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # in_channels = channels of (skip + upsampled) concatenated
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1 = deeper feature, x2 = skip connection
        x1 = self.up(x1)

        # Pad x1 if needed to match x2 (in case of odd dimensions)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)

        x1 = F.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2,
             diff_y // 2, diff_y - diff_y // 2]
        )

        # Concatenate along channel dim
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Final 1x1 conv to get logits for each class
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net style network.
    Input:  (B, 3, H, W)
    Output: (B, nb_classes, H, W)
    """

    def __init__(self, param):
        super().__init__()

        # Base number of channels from your YAML (e.g. 32 or 64)
        base_ch = param["MODEL"]["NB_CHANNEL"]
        # Number of output classes (0..4 for your dataset)
        self.nb_classes = param["MODEL"].get("NB_CLASSES", 5)

        # Encoder
        self.inc   = DoubleConv(3, base_ch)              # 3 -> base
        self.down1 = Down(base_ch, base_ch * 2)          # base -> 2base
        self.down2 = Down(base_ch * 2, base_ch * 4)      # 2base -> 4base
        self.down3 = Down(base_ch * 4, base_ch * 8)      # 4base -> 8base

        # Bottleneck
        self.down4 = Down(base_ch * 8, base_ch * 16)     # 8base -> 16base

        # Decoder
        self.up1 = Up(base_ch * 16 + base_ch * 8, base_ch * 8)   # (16+8)base -> 8base
        self.up2 = Up(base_ch * 8 + base_ch * 4,  base_ch * 4)   # (8+4)base  -> 4base
        self.up3 = Up(base_ch * 4 + base_ch * 2,  base_ch * 2)   # (4+2)base  -> 2base
        self.up4 = Up(base_ch * 2 + base_ch,      base_ch)       # (2+1)base  -> base

        self.outc = OutConv(base_ch, self.nb_classes)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)    # (B, base, H,   W)
        x2 = self.down1(x1) # (B, 2b,  H/2, W/2)
        x3 = self.down2(x2) # (B, 4b,  H/4, W/4)
        x4 = self.down3(x3) # (B, 8b,  H/8, W/8)
        x5 = self.down4(x4) # (B, 16b, H/16,W/16)

        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)

        logits = self.outc(x)  # (B, nb_classes, H, W)
        return logits
