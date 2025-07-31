import torch
from tqdm import tqdm
import torch.nn as nn
from torch.nn.functional import relu
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from linformer_pytorch import Linformer

class U_Net(nn.Module):
    def __init__(self, in_channels=3, seg_out_channels=1):
        super(U_Net, self).__init__()

        def double_conv(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        # Downsampling path
        self.dconv_down1 = double_conv(in_channels, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)
        self.maxpool = nn.MaxPool2d(2)

        # Bottleneck processing
        self.bottleneck_pool = nn.MaxPool2d(2)


        self.transformer_encoder = Linformer(
            input_size=16 * 16,      # 16 * 16 = 256
            channels=512,           # CNN output channels
            dim_k=64,               # Projection dimension
            dim_ff=512,             # Feedforward dimension
            dim_d=512,              # Per-head dimension
            dropout_ff=0.15,
            nhead=4,                # Number of attention heads
            depth=1,                # Number of Linformer layers
            dropout=0.1,
            activation="gelu",
            checkpoint_level="C0",
            parameter_sharing="layerwise",
            k_reduce_by_layer=0,
            full_attention=False,
            include_ff=True,
            w_o_intermediate_dim=None,
            decoder_mode=False,
            causal=False,
            method="learnable",
            ff_intermediate=None
        )

        # Segmentation Decoder (fixed skip connections)
        self.upsample3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=4)  # Adjusted to upsample 16x16→64x64
        self.dconv_up3 = double_conv(512, 256)
        self.upsample2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dconv_up2 = double_conv(256, 128)
        self.upsample1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dconv_up1 = double_conv(128, 64)
        self.seg_head = nn.Conv2d(64, seg_out_channels, 1)

        # Classification Head (dimension-correct)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 128),  # Input must match transformer output
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # Encoder steps
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        x = self.dconv_down4(x)

        # Bottleneck processing 
        x = self.bottleneck_pool(x)  # Reduces to 16x16
        batch, C, H, W = x.shape     # Should be [B, 512, 16, 16]
        x_flat = x.flatten(2).permute(0, 2, 1)  # [B, 256, 512]
        x_transformed = self.transformer_encoder(x_flat)  # Still [B, 256, 512]
        x_transformed = x_transformed.permute(0, 2, 1).view(batch, C, H, W)  # Reshape to [B, 512, 16, 16]

        # Segmentation Decoder 
        x = self.upsample3(x_transformed)  # Now 256 channels, 64x64
        x = torch.cat([x, conv3], dim=1)    # Concatenate with conv3 (256 channels, 64x64)
        x = self.dconv_up3(x)
        x = self.upsample2(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample1(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        seg_out = torch.sigmoid(self.seg_head(x))

        # Classification
        cls_out = self.cls_head(x_transformed)

        return seg_out, cls_out