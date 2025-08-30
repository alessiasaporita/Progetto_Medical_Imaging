""" Full assembly of the parts to form the complete network """

from unet_parts import *

class UNet2D(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet2D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv2D(n_channels, 64))
        self.down1 = (Down2D(64, 128))
        self.down2 = (Down2D(128, 256))
        self.down3 = (Down2D(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down2D(512, 1024 // factor))
        self.up1 = (Up2D(1024, 512 // factor, bilinear))
        self.up2 = (Up2D(512, 256 // factor, bilinear))
        self.up3 = (Up2D(256, 128 // factor, bilinear))
        self.up4 = (Up2D(128, 64, bilinear))
        self.outc = (OutConv2D(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)



""" Full assembly of the parts to form the complete network """

class DoubleConv3D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, eps=1e-5):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm3d(mid_channels, affine=True, eps=eps),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm3d(mid_channels, affine=True, eps=eps),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down3D(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up3D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, skip_channels=None):
        super().__init__()
        if not skip_channels:
            skip_channels = in_channels // 2
        self.up = nn.ConvTranspose3d(in_channels, skip_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv3D(skip_channels + skip_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Pad x_dec to match x_skip along D,H,W (pad order: W_left, W_right, H_left, H_right, D_left, D_right)
        diffD = x2.size(2) - x1.size(2)
        diffH = x2.size(3) - x1.size(3)
        diffW = x2.size(4) - x1.size(4)
        x1 = F.pad(
            x1,
            [   diffW // 2, diffW - diffW // 2,
                diffH // 2, diffH - diffH // 2,
                diffD // 2, diffD - diffD // 2]
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# --- nnU-Net plans–style UNet3D (3d_fullres) ---
class UNet3D_nnunet(nn.Module):
    """
    6-stage 3D U-Net matching nnU-Net plans:
      features = [32, 64, 128, 256, 320, 320]
      two convs per stage, InstanceNorm3d + LeakyReLU (as in your DoubleConv3D)
    Uses your DoubleConv3D, Down3D, OutConv3D implementations.
    """
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # ----- Encoder -----
        self.inc   = DoubleConv3D(n_channels, 32)
        self.down1 = Down3D(32,   64)
        self.down2 = Down3D(64,   128)
        self.down3 = Down3D(128,  256)
        self.down4 = Down3D(256,  320)
        self.down5 = Down3D(320,  320)  

        # ----- Decoder (mirror) -----
        self.up1 = Up3D(in_channels=320, skip_channels=320, out_channels=320)  
        self.up2 = Up3D(in_channels=320, skip_channels=256, out_channels=256)
        self.up3 = Up3D(in_channels=256, skip_channels=128, out_channels=128)
        self.up4 = Up3D(in_channels=128, skip_channels=64,  out_channels=64)
        self.up5 = Up3D(in_channels=64,  skip_channels=32,  out_channels=32)

        self.outc = OutConv3D(32, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # ----- Encoder -----
        x1 = self.inc(x)      # (B, 32, 128, 128, 128)
        x2 = self.down1(x1)   # (B, 64, 64, 64, 64)
        x3 = self.down2(x2)   # (B, 128, 32, 32, 32)
        x4 = self.down3(x3)   # (B, 256, 16, 16, 16)
        x5 = self.down4(x4)   # (B, 320, 8, 8, 8)
        x6 = self.down5(x5)   # (B, 320, 4, 4, 4)

        # ----- Decoder -----
        y = self.up1(x6, x5)  # (B, 320, 8, 8, 8)
        y = self.up2(y, x4)   # (B, 256, 16, 16, 16)
        y = self.up3(y, x3)   # (B, 128, 32, 32, 32)
        y = self.up4(y, x2)   # (B, 64, 64, 64, 64)
        y = self.up5(y, x1)   # (B, 32, 128, 128, 128)

        y = self.outc(y)   # logits: (B, n_classes, D, H, W)
        return self.softmax(y)


    # (optional) checkpoint convenience
    def use_checkpointing(self):
        self.inc   = torch.utils.checkpoint.checkpoint_sequential(self.inc, 1)
        self.down1 = torch.utils.checkpoint.checkpoint_sequential(self.down1, 1)
        self.down2 = torch.utils.checkpoint.checkpoint_sequential(self.down2, 1)
        self.down3 = torch.utils.checkpoint.checkpoint_sequential(self.down3, 1)
        self.down4 = torch.utils.checkpoint.checkpoint_sequential(self.down4, 1)
        self.down5 = torch.utils.checkpoint.checkpoint_sequential(self.down5, 1)
        self.up1   = torch.utils.checkpoint.checkpoint_sequential(self.up1, 1)
        self.up2   = torch.utils.checkpoint.checkpoint_sequential(self.up2, 1)
        self.up3   = torch.utils.checkpoint.checkpoint_sequential(self.up3, 1)
        self.up4   = torch.utils.checkpoint.checkpoint_sequential(self.up4, 1)
        self.up5   = torch.utils.checkpoint.checkpoint_sequential(self.up5, 1)
        self.outc  = torch.utils.checkpoint.checkpoint_sequential(self.outc, 1)

class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = (DoubleConv3D(n_channels, 64))
        self.down1 = (Down3D(64, 128))
        self.down2 = (Down3D(128, 256))
        self.down3 = (Down3D(256, 512))

        factor = 2 
        self.up1 = (Up3D(512, 512 // factor))
        self.up2 = (Up3D(256, 256 // factor))
        self.up3 = (Up3D(128, 128 // factor))

        self.outc = (OutConv3D(64, n_classes))

    def forward(self, x):
        #Encoder
        x1 = self.inc(x)           #(B, 64, D, H, W)

        x2 = self.down1(x1)        #(B, 128, D/2, H/2, W/2)
        x3 = self.down2(x2)        #(B, 256, D/4, H/4, W/4)
        x4 = self.down3(x3)        #(B, 512, D/8, H/8, W/8) -> bottleneck

        #Decoder
        x = self.up1(x4, x3)       #(B, 256, D/4, H/4, W/4)
        x = self.up2(x, x2)        #(B, 128, D/2, H/2, W/2)
        x = self.up3(x, x1)        #(B, 64, D, H, W)  

        logits = self.outc(x)      #(B, n_classes, D, H, W)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.outc = torch.utils.checkpoint(self.outc)


model = UNet3D_nnunet(4, 3)
x = torch.rand(2, 3, 128, 128, 128)
model.cuda()
print(model)
x = x.to("cuda")
x = model.forward(x)
x = x.to("cpu")
print(x.size())

"""
model = UNet3D(4,3)
x = torch.rand(2, 3, 128, 128, 128)
model.cuda()
print(model)
x = x.to("cuda")
x = model.forward(x)
x = x.to("cpu")
print(x.size())
"""