import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetAutoencoder(nn.Module):
    def __init__(self):
        super(UNetAutoencoder, self).__init__()

        # Encoder
        self.enc1 = self.contracting_block(1, 64)
        self.enc2 = self.contracting_block(64, 128)
        self.enc3 = self.contracting_block(128, 256)

        # Bottleneck
        self.bottleneck = self.contracting_block(256, 512)

        # Decoder
        self.dec3 = self.expansive_block(512, 256)      # input: bottleneck output
        self.dec2 = self.expansive_block(512, 128)      # input: dec3 output + enc3 (256+256)
        self.dec1 = self.expansive_block(256, 64)       # input: dec2 output + enc2 (128+128)

        self.final_conv = nn.Conv2d(128, 1, kernel_size=1)  # input: dec1 output + enc1 (64+64)

    def contracting_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def expansive_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = F.max_pool2d(e1, 2)

        e2 = self.enc2(p1)
        p2 = F.max_pool2d(e2, 2)

        e3 = self.enc3(p2)
        p3 = F.max_pool2d(e3, 2)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder
        d3 = self.dec3(b)
        d3 = self.crop_and_concat(e3, d3)

        d2 = self.dec2(d3)
        d2 = self.crop_and_concat(e2, d2)

        d1 = self.dec1(d2)
        d1 = self.crop_and_concat(e1, d1)

        out = self.final_conv(d1)
        return out

    def crop_and_concat(self, enc_feature, dec_feature):
        # Crop enc_feature if size mismatch
        if enc_feature.size()[2:] != dec_feature.size()[2:]:
            diffY = enc_feature.size(2) - dec_feature.size(2)
            diffX = enc_feature.size(3) - dec_feature.size(3)
            enc_feature = enc_feature[:, :, diffY // 2 : enc_feature.size(2) - diffY // 2, diffX // 2 : enc_feature.size(3) - diffX // 2]
        return torch.cat([enc_feature, dec_feature], dim=1)
