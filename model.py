"""
Dual-Axis GeoFormer -- architecture prototype.

This is a real, runnable PyTorch implementation of the architecture proposed
in the thesis: a Siamese encoder built from simplified MaxViT-style blocks
(MBConv + windowed Block Attention + strided Grid Attention), a bi-temporal
Difference Module, a U-shaped decoder with skip connections, and a 4-channel
Geo-Head.

Scope honesty, stated up front (say this out loud when you present it):
  - The attention mechanism (block-local + strided-global, linear complexity)
    is implemented faithfully to Tu et al.'s MaxViT design -- this is the
    actual mechanism the thesis argues for, not a stand-in.
  - By default (`GeoFormerConfig.pretrained_backbone=None`) this is a smaller,
    randomly-initialized version of the same block structure, sized to run a
    forward pass on a CPU in seconds -- not the full MaxViT-Base backbone from
    timm/ImageNet-21k weights (a ~120M-parameter network training this
    session's CPU environment can't run at real scale). Phase 2 is now real,
    not just described: `pretrained_backbone="efficientnet_b0"` (or any timm
    model with a 4-stage `features_only` output at strides 4/8/16/32) swaps
    in ImageNet-1k-pretrained features as the encoder's input to the SAME
    MaxViTBlock attention stages, via a 1x1 projection per stage -- see
    `SiameseMaxViTEncoder`. EfficientNet-B0, not MaxViT-Base, because it's
    the CPU-tractable pretrained option available here; a MaxViT-Base swap
    is a one-line `pretrained_backbone` change for whoever has the compute.
  - The model has NOT been trained. Running `demo.py` proves the architecture
    builds, the tensor shapes are correct end-to-end, and the pipeline
    (encoder -> diff module -> decoder -> geo-head -> skeleton bridging)
    executes on a real image pair. It does not produce a trained flood
    prediction -- there is no SpaceNet-8 data or training loop here yet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Multi-axis attention: Block (local) + Grid (global, strided) attention.
# This is the mechanism the thesis's Section 2.4 / 3 Phase-1 argument rests
# on: grid attention gives every layer a full-image receptive field at
# linear cost, instead of only after several downsampling stages.
# --------------------------------------------------------------------------

class _MultiHeadAttention(nn.Module):
    """Plain multi-head self-attention over a set of tokens."""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, N, C) -> returns (out, attn) where attn is (B, heads, N, N)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B, heads, N, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out, attn


class BlockAttention(nn.Module):
    """Local windowed self-attention -- resolves fine, sharp building edges."""

    def __init__(self, dim: int, window: int = 8, num_heads: int = 4):
        super().__init__()
        self.window = window
        self.norm = nn.LayerNorm(dim)
        self.attn = _MultiHeadAttention(dim, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        w = self.window
        pad_h, pad_w = (w - H % w) % w, (w - W % w) % w
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[-2], x.shape[-1]

        # partition into (Hp/w * Wp/w) windows of w*w tokens each
        xt = x.permute(0, 2, 3, 1)  # B, Hp, Wp, C
        xt = xt.reshape(B, Hp // w, w, Wp // w, w, C)
        xt = xt.permute(0, 1, 3, 2, 4, 5).reshape(-1, w * w, C)

        residual = xt
        xt = self.norm(xt)
        out, _attn = self.attn(xt)
        xt = residual + out

        xt = xt.reshape(B, Hp // w, Wp // w, w, w, C)
        xt = xt.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)
        xt = xt.permute(0, 3, 1, 2)
        return xt[:, :, :H, :W]


class GridAttention(nn.Module):
    """
    Strided global attention -- one token sampled every `grid` steps across
    the ENTIRE feature map, per group. This is what gives the network
    full-image reach from the earliest stage: a flood cue on one side of the
    tile can attend directly to a road segment on the other side, in one
    layer, at linear cost (unlike full dense attention over all H*W tokens).
    """

    def __init__(self, dim: int, grid: int = 8, num_heads: int = 4):
        super().__init__()
        self.grid = grid
        self.norm = nn.LayerNorm(dim)
        self.attn = _MultiHeadAttention(dim, num_heads)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, H, W) -> returns (out, saliency_map (B, H, W))
        B, C, H, W = x.shape
        g = self.grid
        pad_h, pad_w = (g - H % g) % g, (g - W % g) % g
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[-2], x.shape[-1]

        # group into g*g groups, each group holds (Hp/g * Wp/g) tokens strided
        # across the whole map -- the "grid" in grid attention.
        xt = x.permute(0, 2, 3, 1)  # B, Hp, Wp, C
        xt = xt.reshape(B, Hp // g, g, Wp // g, g, C)
        xt = xt.permute(0, 2, 4, 1, 3, 5).reshape(-1, (Hp // g) * (Wp // g), C)

        residual = xt
        xt = self.norm(xt)
        out, attn = self.attn(xt)
        xt = residual + out

        xt = xt.reshape(B, g, g, Hp // g, Wp // g, C)
        xt = xt.permute(0, 3, 1, 4, 2, 5).reshape(B, Hp, Wp, C)
        xt = xt.permute(0, 3, 1, 2)[:, :, :H, :W]

        # saliency: how much attention each token *receives* on average,
        # averaged over heads and groups -- this is what Phase 4's road
        # bridging reuses as a "does the model already think these two
        # points are related" signal.
        # attn is (B*g*g, heads, N, N) -- the leading dim is B*g*g, not just
        # g*g, so B must be carried through this reshape explicitly or it
        # silently only works at batch size 1 (which is how this bug hid in
        # demo.py: that script always runs batch size 1).
        received = attn.mean(dim=1).mean(dim=1)  # (B*g*g, N)
        n_h, n_w = Hp // g, Wp // g
        received = received.reshape(B, g, g, n_h, n_w)
        received = received.permute(0, 3, 1, 4, 2).reshape(B, Hp, Wp)
        received = received[:, :H, :W]
        return xt, received


class MBConv(nn.Module):
    """Inverted-residual conv block (the "MB" half of MaxViT's hybrid block)."""

    def __init__(self, dim: int, expand: int = 4):
        super().__init__()
        hidden = dim * expand
        self.block = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class MaxViTBlock(nn.Module):
    """One MBConv -> Block Attention -> Grid Attention stage unit.

    `use_grid_attention=False` gives the "GeoFormer - grid attention"
    ablation from the thesis proposal's Table 2 -- block attention (purely
    local) still runs, but the strided global attention that's this
    architecture's whole argument for road connectivity is skipped. This is
    what isolates whether grid attention specifically, not just "the model,"
    is responsible for a connectivity result -- see docs/MANUAL.md's
    experimental design section.
    """

    def __init__(self, dim: int, window: int = 8, grid: int = 8, num_heads: int = 4,
                 use_grid_attention: bool = True):
        super().__init__()
        self.use_grid_attention = use_grid_attention
        self.mbconv = MBConv(dim)
        self.block_attn = BlockAttention(dim, window, num_heads)
        self.grid_attn = GridAttention(dim, grid, num_heads) if use_grid_attention else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.mbconv(x)
        x = self.block_attn(x)
        if self.grid_attn is None:
            # No global attention ran -- there is no real saliency signal to
            # report. A flat zero map (rather than skipping the return value
            # entirely) keeps DualAxisGeoFormer.forward's output shape
            # consistent whether or not this ablation is active, and reads
            # as "no signal" rather than a fabricated one.
            saliency = torch.zeros(x.shape[0], x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
            return x, saliency
        x, saliency = self.grid_attn(x)
        return x, saliency


# --------------------------------------------------------------------------
# Siamese encoder: shared-weight MaxViT stages over pre- and post-event tiles
# --------------------------------------------------------------------------

@dataclass
class GeoFormerConfig:
    in_channels: int = 3
    stem_channels: int = 32
    stage_dims: tuple[int, ...] = (64, 128, 256, 512)
    stage_windows: tuple[int, ...] = (8, 8, 4, 2)
    stage_grids: tuple[int, ...] = (8, 4, 2, 1)
    num_heads: int = 4
    num_classes: int = 4  # background, building, road, flooded
    use_grid_attention: bool = True  # False = the Table 2 "- grid attention" ablation
    # Phase 2 (thesis): an ImageNet-pretrained CNN backbone (any timm model
    # supporting features_only=True with 4 stages at strides 4/8/16/32,
    # e.g. "efficientnet_b0") replaces the from-scratch stem+downsample
    # convs as the source of per-stage features -- the existing
    # MaxViTBlock attention stages (block + grid) still run on top of
    # those features unchanged, so the thesis's actual proposed attention
    # mechanism and Phase 4's grid-saliency signal are untouched. None
    # (default) keeps the original from-scratch, randomly-initialized
    # stem -- exact prior behavior, verified by test.
    pretrained_backbone: str | None = None
    # Load real ImageNet weights (the actual point of pretrained_backbone in
    # production). False skips the download and randomly initializes the
    # backbone instead -- exists so tests can exercise the wiring (shapes,
    # gradient flow, projection channels) fast and without a network call,
    # not something a real training run should ever set.
    pretrained: bool = True
    # Decouple `flooded` from the joint 4-way softmax (docs/MANUAL.md
    # S12.14 item 1, S12.17): False (default, exact prior behavior) keeps
    # one Geo-Head producing all 4 classes in direct competition. True
    # splits the head into a 3-way structure classifier
    # (background/building/road) and an independent binary flood
    # classifier, each with its own final conv layer and its own loss
    # term (see losses.py, train.py) -- giving `flooded` its own gradient
    # pathway into the shared decoder features instead of one that has to
    # also serve background's overwhelming per-pixel dominance. A
    # synthesized 4-channel `logits` tensor is still returned for
    # backward compatibility with every existing consumer (evaluate.py,
    # demo.py, serve.py, ConfusionAccumulator, the dashboard) -- none of
    # them need to change to support this.
    separate_flood_head: bool = False


class SiameseMaxViTEncoder(nn.Module):
    """Shared-weight encoder run once per timestamp (pre / post).

    WHY THIS CHANGE (Phase 2, docs/MANUAL.md S12.8/S12.10): three straight
    real-data training configurations (plain, oversampled, oversampled +
    class-weighted loss) each produced SOME partial class collapse under
    this project's from-scratch, randomly-initialized encoder -- a
    different one each time, never all three foreground classes learning
    together. Researching the actual SpaceNet-8 winning solutions found
    they credited a pretrained backbone with "significantly" improving
    their score. `pretrained_backbone` wires that in as an alternative
    feature source for the SAME downstream pipeline (DiffModule, decoder,
    Geo-Head, grid-saliency-based Phase 4 bridging), not a redesign.
    """

    def __init__(self, cfg: GeoFormerConfig):
        super().__init__()
        self.cfg = cfg
        self.pretrained_backbone_name = cfg.pretrained_backbone

        if cfg.pretrained_backbone:
            # Imported lazily -- timm is only a hard dependency when this
            # config option is actually used, not for the default
            # from-scratch path (keeps the base install lighter and
            # existing synthetic/CPU-smoke-test runs unaffected).
            import timm
            self.backbone = timm.create_model(
                cfg.pretrained_backbone, pretrained=cfg.pretrained, features_only=True,
                out_indices=(1, 2, 3, 4), in_chans=cfg.in_channels,
            )
            backbone_channels = self.backbone.feature_info.channels()
            if len(backbone_channels) != len(cfg.stage_dims):
                raise ValueError(
                    f"'{cfg.pretrained_backbone}' produced {len(backbone_channels)} feature "
                    f"stages, expected {len(cfg.stage_dims)} (one per cfg.stage_dims entry)."
                )
            # 1x1 convs project the pretrained backbone's own channel
            # counts onto cfg.stage_dims, so every downstream module
            # (DiffModule, decoder, Geo-Head) is completely unaware
            # whether its input came from this backbone or the
            # from-scratch stem below.
            self.projections = nn.ModuleList([
                nn.Conv2d(backbone_channels[i], cfg.stage_dims[i], 1)
                for i in range(len(cfg.stage_dims))
            ])
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(cfg.in_channels, cfg.stem_channels, 3, stride=2, padding=1),
                nn.BatchNorm2d(cfg.stem_channels),
                nn.GELU(),
            )
            dims = (cfg.stem_channels,) + cfg.stage_dims
            self.downsamples = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], 3, stride=2, padding=1),
                    nn.BatchNorm2d(dims[i + 1]),
                    nn.GELU(),
                )
                for i in range(len(cfg.stage_dims))
            ])

        self.stages = nn.ModuleList([
            MaxViTBlock(cfg.stage_dims[i], cfg.stage_windows[i], cfg.stage_grids[i], cfg.num_heads,
                        use_grid_attention=cfg.use_grid_attention)
            for i in range(len(cfg.stage_dims))
        ])
        self._backbone_frozen = False

    def set_backbone_frozen(self, frozen: bool) -> None:
        """Freeze/unfreeze the pretrained backbone -- standard transfer-
        learning practice, and a real efficiency lever: forward() then
        runs the backbone under torch.no_grad() (skips building its
        backward graph entirely -- less compute AND less activation
        memory, not merely "the optimizer won't step these"), and puts
        it in eval() mode so BatchNorm running stats stop drifting on
        data this training run may see in a different distribution than
        ImageNet did. No-op if there's no backbone (from-scratch path).
        """
        if not self.pretrained_backbone_name:
            return
        self._backbone_frozen = frozen
        for p in self.backbone.parameters():
            p.requires_grad = not frozen
        self.backbone.train(not frozen)

    def forward(self, x: torch.Tensor):
        feats, saliencies = [], []
        if self.pretrained_backbone_name:
            if self._backbone_frozen:
                with torch.no_grad():
                    backbone_feats = self.backbone(x)
            else:
                backbone_feats = self.backbone(x)
            for proj, bf, stage in zip(self.projections, backbone_feats, self.stages):
                f, sal = stage(proj(bf))
                feats.append(f)
                saliencies.append(sal)
        else:
            x = self.stem(x)
            for down, stage in zip(self.downsamples, self.stages):
                x = down(x)
                x, sal = stage(x)
                feats.append(x)
                saliencies.append(sal)
        return feats, saliencies  # each a list of per-stage tensors


# --------------------------------------------------------------------------
# Bi-temporal Difference Module
# --------------------------------------------------------------------------

class DiffModule(nn.Module):
    """D = |F_pre - F_post|, fused with F_post via 1x1 conv (Phase 1)."""

    def __init__(self, dim: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

    def forward(self, f_pre: torch.Tensor, f_post: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(f_pre - f_post)
        return self.fuse(torch.cat([diff, f_post], dim=1))


# --------------------------------------------------------------------------
# U-shaped decoder + Geo-Head
# --------------------------------------------------------------------------

class UpBlock(nn.Module):
    def __init__(self, in_dim: int, skip_dim: int, out_dim: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_dim, out_dim, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_dim + skip_dim, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DualAxisGeoFormer(nn.Module):
    """Full pipeline: Siamese MaxViT encoder -> Diff modules -> U-decoder -> Geo-Head."""

    def __init__(self, cfg: GeoFormerConfig | None = None):
        super().__init__()
        self.cfg = cfg or GeoFormerConfig()
        cfg = self.cfg
        self.encoder = SiameseMaxViTEncoder(cfg)
        self.diffs = nn.ModuleList([DiffModule(d) for d in cfg.stage_dims])

        dims = list(cfg.stage_dims)  # e.g. [64, 128, 256, 512], shallow -> deep
        self.up_blocks = nn.ModuleList([
            UpBlock(dims[i + 1], dims[i], dims[i]) for i in reversed(range(len(dims) - 1))
        ])
        self.final_up = nn.ConvTranspose2d(dims[0], dims[0] // 2, 2, stride=2)
        if cfg.separate_flood_head:
            if cfg.num_classes != 4:
                raise ValueError("separate_flood_head assumes the 4-class "
                                  "background/building/road/flooded scheme")
            # A separate trunk+heads module tree (not reusing `geo_head`'s
            # name/shape) -- keeping the default path's module structure
            # byte-for-byte unchanged from before this feature existed is
            # what lets every EXISTING checkpoint still load with
            # separate_flood_head=False, its default. Renaming/restructuring
            # geo_head itself for both modes would silently break
            # load_state_dict for every checkpoint this project has ever
            # produced.
            self.split_trunk = nn.Sequential(
                nn.Conv2d(dims[0] // 2, dims[0] // 2, 3, padding=1),
                nn.BatchNorm2d(dims[0] // 2),
                nn.GELU(),
            )
            self.structure_head = nn.Conv2d(dims[0] // 2, 3, 1)  # background, building, road
            self.flood_head = nn.Conv2d(dims[0] // 2, 1, 1)      # binary: flooded or not
        else:
            self.geo_head = nn.Sequential(
                nn.Conv2d(dims[0] // 2, dims[0] // 2, 3, padding=1),
                nn.BatchNorm2d(dims[0] // 2),
                nn.GELU(),
                nn.Conv2d(dims[0] // 2, cfg.num_classes, 1),
            )

    def forward(self, pre: torch.Tensor, post: torch.Tensor):
        feats_pre, _ = self.encoder(pre)
        feats_post, saliencies_post = self.encoder(post)

        fused = [diff(fp, fq) for diff, fp, fq in zip(self.diffs, feats_pre, feats_post)]

        x = fused[-1]
        for i, up in enumerate(self.up_blocks):
            skip = fused[-(i + 2)]
            x = up(x, skip)

        x = self.final_up(x)
        x = F.interpolate(x, size=pre.shape[-2:], mode="bilinear", align_corners=False)

        out = {
            # BUG THIS FIXES: this read saliencies_post[-2] with a comment
            # claiming "deepest usable stage" -- but the actual deepest
            # stage (-1) IS usable (verified: correct shape, no NaN, at
            # every config tried). -2 was one stage shallower than the
            # comment's own stated intent, an off-by-one against the design
            # rationale (the deepest stage's attention has the most
            # downsampling behind it, so the most globally-contextualized
            # signal -- exactly what Phase 4 bridging wants).
            "grid_saliency": saliencies_post[-1],  # deepest stage, for Phase 4 bridging
        }

        if self.cfg.separate_flood_head:
            trunk_out = self.split_trunk(x)
            structure_logits = self.structure_head(trunk_out)  # (B, 3, H, W): bg/building/road
            flood_logit = self.flood_head(trunk_out)           # (B, 1, H, W): flooded or not
            out["structure_logits"] = structure_logits
            out["flood_logit"] = flood_logit
            # A synthesized 4-channel tensor so every EXISTING consumer
            # (evaluate.py, demo.py, serve.py, ConfusionAccumulator, the
            # dashboard) keeps working unchanged -- argmax naturally picks
            # the flooded channel whenever the independent flood head is
            # more confident than any structure class, the same real
            # decision the original single softmax made, just no longer
            # sharing gradients to get there.
            out["logits"] = torch.cat([structure_logits, flood_logit], dim=1)
        else:
            out["logits"] = self.geo_head(x)  # (B, num_classes, H, W)

        return out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    cfg = GeoFormerConfig()
    model = DualAxisGeoFormer(cfg)
    pre = torch.randn(1, 3, 256, 256)
    post = torch.randn(1, 3, 256, 256)
    out = model(pre, post)
    print("logits:", tuple(out["logits"].shape))
    print("grid_saliency:", tuple(out["grid_saliency"].shape))
    print(f"parameters: {model.num_parameters():,}")
