import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class DifferenceModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, pre_feat, post_feat):
        # Difference highlighting change
        diff = torch.abs(pre_feat - post_feat)
        diff = self.conv(diff)
        # Concatenate or sum? User asked to "concatenate it with the original features".
        # Which original? "highlights change while keeping identity".
        # Let's concatenate with post_feat (damage status).
        return torch.cat([post_feat, diff], dim=1)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, skip=None):
        if skip is not None:
            # Check shapes and handle mismatch if needed (MaxViT might be tricky with shapes)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        if skip is None: # Initial upsample if needed or last layer
             pass 
        return self.up(x) # Upsample for next stage

class DualAxisGeoFormer(nn.Module):
    def __init__(self, num_classes=4, backbone='maxvit_base_tf_224.in1k'):
        super().__init__()
        # Load MaxViT
        # 'features_only=True' to get intermediate layers
        self.encoder = timm.create_model(backbone, pretrained=True, features_only=True)
        
        # Get channel counts from encoder
        # Dummy forward to check channels if not available in config
        dummy = torch.randn(1, 3, 224, 224)
        features = self.encoder(dummy)
        self.enc_channels = [f.shape[1] for f in features] 
        # Example channels for maxvit_base: [64, 64, 128, 256, 512] (Approx, need verification)
        
        # Difference Modules for each scale? Or just at the bottleneck?
        # "Siamese Encoder" -> shared weights.
        # "Difference Module" -> usually at each stage or bottleneck. 
        # Let's apply difference at each stage for skip connections to Fuse Pre/Post info early.
        
        self.diff_modules = nn.ModuleList([
            DifferenceModule(c) for c in self.enc_channels
        ])
        
        # Decoder
        # Input to decoder is fused features. Each skip connection is double channels (post + diff).
        # We start from the deepest layer.
        
        decoder_channels = [256, 128, 64, 32] # define decoder width
        
        self.center = nn.Conv2d(self.enc_channels[-1] * 2, decoder_channels[0], 3, padding=1) # *2 for post+diff
        
        self.dec_blocks = nn.ModuleList()
        # Iterate backwards
        for i in range(len(self.enc_channels)-2, -1, -1):
            # Input is prev_decoder_out + skip_connection (which is enc_channels[i]*2)
            in_ch = decoder_channels[0] + self.enc_channels[i] * 2
            out_ch = decoder_channels[1] if i > 0 else 32
            self.dec_blocks.append(DecoderBlock(in_ch, out_ch))
            decoder_channels.pop(0)

        # Final segmentation head
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)
        
    def forward(self, pre_img, post_img):
        # Siamese Encoder
        enc_pre = self.encoder(pre_img)
        enc_post = self.encoder(post_img)
        
        # Fusion
        fused_features = []
        for i, (p, po) in enumerate(zip(enc_pre, enc_post)):
            # Apply Difference Module
            # Post features + Difference features
            fused = self.diff_modules[i](p, po)
            fused_features.append(fused)
            
        # Decoder
        # Start from bottleneck (last feature map)
        x = fused_features[-1] 
        x = self.center(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True) # Initial up from bottleneck
        
        # Decode with skips
        # fused_features: [f1, f2, f3, f4, f5]
        # dec_blocks loops from idx 3 down to 0
        for i, block in enumerate(self.dec_blocks):
            skip_idx = len(fused_features) - 2 - i
            skip = fused_features[skip_idx]
            x = block(x, skip)
            
        # Final prediction
        # Check output size, might need final resize to match input
        x = self.final_conv(x)
        x = F.interpolate(x, size=pre_img.shape[2:], mode='bilinear', align_corners=True)
        
        return x

if __name__ == '__main__':
    # Verification
    model = DualAxisGeoFormer()
    x1 = torch.randn(2, 3, 224, 224) # Pre
    x2 = torch.randn(2, 3, 224, 224) # Post
    y = model(x1, x2)
    print(f"Output shape: {y.shape}")
