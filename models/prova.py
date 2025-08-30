""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.factor = 2

        self.inc = (DoubleConv3D(n_channels, 64))
        self.down1 = (Down3D(64, 128))
        self.down2 = (Down3D(128, 256))
        self.down3 = (Down3D(256, 512))
        self.down4 = (Down3D(512, 512))

        self.up1 = (Up3D(512, 512))
        self.up2 = (Up3D(512, 256))
        self.up3 = (Up3D(256, 128))
        self.up4 = (Up3D(128, 64))
        self.outc = (OutConv3D(64, n_classes))

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


model = UNet3D(4, 3)
print(model)
input = torch.randn(1, 4, 128, 128, 128)
output = model(input)
print(output.shape)
