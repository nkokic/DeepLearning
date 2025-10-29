from LV2_dataload import *

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=4):
        super(UNet, self).__init__()

        # Downsampling (Contracting path)
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Upsampling (Expanding path)
        self.up4 = self.upconv_block(1024, 512)
        self.up3 = self.upconv_block(512, 256)
        self.up2 = self.upconv_block(256, 128)
        self.up1 = self.upconv_block(128, 64)

        # Final 1x1 Convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        # Decoder
        up4 = self.up4(bottleneck)
        up4 = torch.cat([up4, enc4], dim=1)  # Concatenate skip connections
        up4 = self.conv_block(up4.size(1), 512)(up4)  # Reduce channels to 512

        up3 = self.up3(up4)
        up3 = torch.cat([up3, enc3], dim=1)
        up3 = self.conv_block(up3.size(1), 256)(up3)  # Reduce channels to 256

        up2 = self.up2(up3)
        up2 = torch.cat([up2, enc2], dim=1)
        up2 = self.conv_block(up2.size(1), 128)(up2)  # Reduce channels to 128

        up1 = self.up1(up2)
        up1 = torch.cat([up1, enc1], dim=1)
        up1 = self.conv_block(up1.size(1), 64)(up1)  # Reduce channels to 64

        # Final convolution
        return self.final_conv(up1)