import math
import torch
from torch import nn
from torch.nn import functional as F


class MultiHeadLocalAttention(nn.Module):
    """Multi-head local attention over non-overlapping windows."""

    def __init__(self, channels, heads=4, window_size=8):
        super().__init__()
        if channels % heads != 0:
            raise ValueError("channels must be divisible by heads")
        self.channels = channels
        self.heads = heads
        self.window_size = window_size
        self.scale = (channels // heads) ** -0.5
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        batch, channels, height, width = x.shape
        win = self.window_size
        pad_h = (win - height % win) % win
        pad_w = (win - width % win) % win
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        _, _, padded_h, padded_w = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        q = self._reshape_windows(q, padded_h, padded_w, win)
        k = self._reshape_windows(k, padded_h, padded_w, win)
        v = self._reshape_windows(v, padded_h, padded_w, win)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = self._restore_windows(out, padded_h, padded_w, win, batch)
        out = self.proj(out)
        if pad_h or pad_w:
            out = out[:, :, :height, :width]
        return out

    def _reshape_windows(self, x, padded_h, padded_w, win):
        batch, channels, _, _ = x.shape
        head_dim = channels // self.heads
        x = x.view(batch, self.heads, head_dim, padded_h, padded_w)
        x = x.unfold(3, win, win).unfold(4, win, win)
        x = x.contiguous().view(batch, self.heads, head_dim, -1, win * win)
        x = x.permute(0, 1, 3, 4, 2)
        return x

    def _restore_windows(self, x, padded_h, padded_w, win, batch):
        _, heads, windows, tokens, head_dim = x.shape
        x = x.permute(0, 1, 4, 2, 3)
        x = x.contiguous().view(batch, heads * head_dim, windows, win, win)
        grid_h = padded_h // win
        grid_w = padded_w // win
        x = x.view(batch, heads * head_dim, grid_h, grid_w, win, win)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(batch, heads * head_dim, padded_h, padded_w)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with MHLA and conv paths (1x1/3x3) as in the diagram."""

    def __init__(self, channels, heads=4, window_size=8):
        super().__init__()
        self.mhla = MultiHeadLocalAttention(channels, heads=heads, window_size=window_size)
        self.conv_1x1_a = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_1x1_b = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_3x3_a = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv_3x3_b = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.GELU()

    def forward(self, x):
        attn = self.mhla(x)
        x = x + attn
        branch = self.conv_1x1_a(x)
        branch = self.act(branch)
        branch = self.conv_1x1_b(branch)
        branch = self.act(branch)
        branch = self.conv_3x3_a(branch)
        branch = self.act(branch)
        branch = self.conv_3x3_b(branch)
        branch = self.act(branch)
        return x + branch


class DownsampleTransformerBlock(nn.Module):
    """Downsample block with a transformer block and strided conv."""

    def __init__(self, input_nc, output_nc, heads=4, window_size=8):
        super().__init__()
        self.proj = nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=2, padding=1)
        self.tb = TransformerBlock(output_nc, heads=heads, window_size=window_size)

    def forward(self, x):
        x = self.proj(x)
        return self.tb(x)


class RegionSeparatedAttention(nn.Module):
    """Region-separated attention (RSAM) with q/k/alpha/beta linear maps and masking."""

    def __init__(self, channels, grid_size=2):
        super().__init__()
        self.grid_size = grid_size
        self.q_map = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_map = nn.Conv2d(channels, channels, kernel_size=1)
        self.alpha_map = nn.Conv2d(channels, channels, kernel_size=1)
        self.beta_map = nn.Conv2d(channels, channels, kernel_size=1)
        self.out_map = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        batch, channels, height, width = x.shape
        grid = self.grid_size
        pad_h = (grid - height % grid) % grid
        pad_w = (grid - width % grid) % grid
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        _, _, padded_h, padded_w = x.shape
        q = self.q_map(x)
        k = self.k_map(x)
        alpha = self.alpha_map(x)
        beta = self.beta_map(x)
        region_h = padded_h // grid
        region_w = padded_w // grid
        outputs = []
        for row in range(grid):
            row_outputs = []
            for col in range(grid):
                h0 = row * region_h
                w0 = col * region_w
                h1 = h0 + region_h
                w1 = w0 + region_w
                q_r = q[:, :, h0:h1, w0:w1]
                k_r = k[:, :, h0:h1, w0:w1]
                a_r = alpha[:, :, h0:h1, w0:w1]
                b_r = beta[:, :, h0:h1, w0:w1]
                q_flat = q_r.flatten(2)
                k_flat = k_r.flatten(2)
                attn = torch.matmul(q_flat.transpose(1, 2), k_flat) / math.sqrt(channels)
                attn = attn.softmax(dim=-1)
                a_flat = a_r.flatten(2)
                b_flat = b_r.flatten(2)
                out_r = torch.matmul(a_flat, attn) + torch.matmul(b_flat, attn.transpose(-2, -1))
                out_r = out_r.view(batch, channels, region_h, region_w)
                row_outputs.append(out_r)
            outputs.append(torch.cat(row_outputs, dim=3))
        out = torch.cat(outputs, dim=2)
        out = self.out_map(out)
        if pad_h or pad_w:
            out = out[:, :, :height, :width]
        return out


class DualStreamGatedFeatureFusion(nn.Module):
    """DSGF: dual-stream gated fusion with l/k/v mappings and sigmoid gates."""

    def __init__(self, in_primary, in_advanced, out_channels):
        super().__init__()
        self.primary_proj = nn.Conv2d(in_primary, out_channels, kernel_size=1)
        self.advanced_proj = nn.Conv2d(in_advanced, out_channels, kernel_size=1)
        self.l_map = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.k_map = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.v_map = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
        self.gate_l = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
        self.gate_k = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, primary_feat, advanced_feat):
        fp = self.primary_proj(primary_feat)
        fa = self.advanced_proj(advanced_feat)
        concat = torch.cat([fp, fa], dim=1)
        gate_l = torch.sigmoid(self.gate_l(concat))
        gate_k = torch.sigmoid(self.gate_k(concat))
        fused_l = fp + gate_l * fa
        fused_k = fa + gate_k * fp
        fused = torch.cat([fused_l, fused_k], dim=1)
        v = self.v_map(fused)
        return v
