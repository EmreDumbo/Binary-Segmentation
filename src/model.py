import torch
import torch.nn as nn
import torch.nn.functional as F

class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unet, self).__init__()

        self.encoder = nn.Sequential(
            self.conv_block(in_channels, 64),
            self.conv_block(64, 128),
            self.conv_block(128, 256),
            self.conv_block(256, 512),
        )

        self.middle = self.conv_block(512, 1024)

        self.decoder = nn.Sequential(
            self.deconv_block(1024, 512),
            self.deconv_block(512, 256),
            self.deconv_block(256, 128),
            self.deconv_block(128, 64)
        )

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.adjust_channels_1 = nn.Conv2d(1024, 512, kernel_size=1)
        self.adjust_channels_2 = nn.Conv2d(512, 256, kernel_size=1)
        self.adjust_channels_3 = nn.Conv2d(256, 128, kernel_size=1)
        self.adjust_channels_4 = nn.Conv2d(128, 64, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    
    def forward(self, x):
        enc1 = self.encoder[0](x)
        enc2 = self.encoder[1](enc1)
        enc3 = self.encoder[2](enc2)
        enc4 = self.encoder[3](enc3)

        middle = self.middle(enc4)

        dec1 = self.decoder[0](middle)
        dec1 = torch.cat([F.interpolate(dec1, size=enc4.shape[2:], mode="bilinear", align_corners=False), enc4], dim=1)
        dec1 = self.adjust_channels_1(dec1)

        dec2 = self.decoder[1](dec1)
        dec2 = torch.cat([F.interpolate(dec2, size=enc3.shape[2:], mode="bilinear", align_corners=False), enc3], dim=1)
        dec2 = self.adjust_channels_2(dec2)

        dec3 = self.decoder[2](dec2)
        dec3 = torch.cat([F.interpolate(dec3, size=enc2.shape[2:], mode="bilinear", align_corners=False), enc2], dim=1)
        dec3 = self.adjust_channels_3(dec3)

        dec4 = self.decoder[3](dec3)
        dec4 = torch.cat([F.interpolate(dec4, size=enc1.shape[2:], mode="bilinear", align_corners=False), enc1], dim=1)
        dec4 = self.adjust_channels_4(dec4)
        
        logits = self.final_conv(dec4)
        return torch.softmax(logits, dim=1)
    
model = Unet(in_channels=3, out_channels=2)

