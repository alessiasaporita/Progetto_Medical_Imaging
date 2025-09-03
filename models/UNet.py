
import torch
import torch.nn as nn
import torch.nn.functional as F

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
            nn.InstanceNorm3d(out_channels, affine=True, eps=eps),
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

class final_patch_expanding(nn.Module):
    def __init__(self,dim,num_class,patch_size):
        super().__init__()
        self.up=nn.ConvTranspose3d(dim, num_class, patch_size, patch_size)
      
    def forward(self,x):
        return self.up(x) 

class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes, deep_supervision=False):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.do_ds = deep_supervision

        self.inc = (DoubleConv3D(n_channels, 64))
        self.down1 = (Down3D(64, 128))
        self.down2 = (Down3D(128, 256))
        self.down3 = (Down3D(256, 512))

        factor = 2 
        self.up1 = (Up3D(512, 512 // factor))
        self.up2 = (Up3D(256, 256 // factor))
        self.up3 = (Up3D(128, 128 // factor))

        self.final_heads = nn.ModuleList()
        if self.do_ds:
            self.final_heads.append(final_patch_expanding(256, n_classes, patch_size=1))  
            self.final_heads.append(final_patch_expanding(128, n_classes, patch_size=1))  
            self.final_heads.append(final_patch_expanding( 64, n_classes, patch_size=1))  
        else:
            self.final_heads.append(final_patch_expanding(64, n_classes, patch_size=1))  


    def forward(self, x):
        #Encoder
        x1 = self.inc(x)           #(B, 64, D, H, W)
        x2 = self.down1(x1)        #(B, 128, D/2, H/2, W/2)
        x3 = self.down2(x2)        #(B, 256, D/4, H/4, W/4)
        x4 = self.down3(x3)        #(B, 512, D/8, H/8, W/8) -> bottleneck

        # Decoder
        y1 = self.up1(x4, x3)      # (B, 256, D/4, H/4, W/4)
        y2 = self.up2(y1, x2)      # (B, 128, D/2, H/2, W/2)
        y3 = self.up3(y2, x1)      # (B,  64, D,   H,   W)

        # Deep supervision
        if self.do_ds:
            l0 = self.final_heads[0](y1)    # (B, n_classes, 32, 32, 32)
            l1 = self.final_heads[1](y2)    # (B, n_classes, 64, 64, 64)
            l2 = self.final_heads[2](y3)    # (B, n_classes, 128, 128, 128) 
            return [l2, l1, l0]  # from high resolution to low resolution
        else:
            return self.final_heads[0](y3)



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