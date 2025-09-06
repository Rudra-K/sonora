import torch
from torch import nn
import torch.nn.functional as F
import math
from einops import rearrange
import logging


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Takes a tensor of shape (batch_size, 1) and returns a sinusoidal
    position embedding of shape (batch_size, dim). Used to embed the
    noise timestep t.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    """A basic building block for the U-Net: Conv -> GroupNorm -> Activation."""
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class ResBlock(nn.Module):
    """A ResNet-style block with a residual connection."""
    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        self.block1 = Block(in_channels, out_channels)
        self.block2 = Block(out_channels, out_channels)
        
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, out_channels))
        
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)
        if time_emb is not None:
            time_embedding = self.time_mlp(time_emb)
            # Add time embedding to the feature map
            h = h + time_embedding.unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h)
        return h + self.res_conv(x)

class CrossAttention(nn.Module):
    """
    A cross-attention layer that allows the model to attend to a context vector.
    """
    def __init__(self, dim, context_dim=None, heads=4, dim_head=32):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else dim
        
        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, context=None):
        h = self.heads
        q = self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

class AttentionBlock(nn.Module):
    """Combines a GroupNorm with a CrossAttention layer."""
    def __init__(self, dim, context_dim=None, heads=4, dim_head=32):
        super().__init__()
        self.norm = nn.GroupNorm(8, dim)
        self.attn = CrossAttention(dim, context_dim, heads, dim_head)

    def forward(self, x, context=None):
        x_norm = self.norm(x)
        # Permute from (batch, channel, height, width) to (batch, sequence, channel)
        x_norm = rearrange(x_norm, 'b c h w -> b (h w) c')
        out = self.attn(x_norm, context)
        # Permute back
        out = rearrange(out, 'b (h w) c -> b c h w', h=x.shape[2])
        return x + out

class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
    
    def forward(self, x):
        return self.conv(self.upsample(x))


class GeneratorUNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        model_channels=64,
        channel_mults=(1, 2, 4),
        time_emb_dim=128,
        text_emb_dim=384
    ):
        super().__init__()
        
        self.model_channels = model_channels
        dims = [model_channels] + [model_channels * m for m in channel_mults]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4), nn.GELU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        self.init_conv = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        # --- Downsampling Path ---
        for i, (dim_in, dim_out) in enumerate(in_out):
            is_last = i >= (len(in_out) - 1)
            self.downs.append(nn.ModuleList([
                ResBlock(dim_in, dim_out, time_emb_dim),
                AttentionBlock(dim_out, context_dim=text_emb_dim),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        # --- Middle Block ---
        mid_dim = dims[-1]
        self.mid_block1 = ResBlock(mid_dim, mid_dim, time_emb_dim)
        self.mid_attn = AttentionBlock(mid_dim, context_dim=text_emb_dim)
        self.mid_block2 = ResBlock(mid_dim, mid_dim, time_emb_dim)

        # --- Upsampling Path ---
        for i, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = i >= (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                ResBlock(dim_out * 2, dim_in, time_emb_dim),
                AttentionBlock(dim_in, context_dim=text_emb_dim),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        # --- Final Output Block ---
        self.final_conv = nn.Sequential(
            Block(model_channels, model_channels),
            nn.Conv2d(model_channels, out_channels, kernel_size=1)
        )

    def forward(self, x, time, text_embedding):
        original_width = x.shape[-1]
        # Ensure input width is divisible by total downsampling factor
        mod = original_width % 4
        if mod != 0:
            padding = 4 - mod
            x = F.pad(x, (0, padding))

        x = self.init_conv(x)
        t = self.time_mlp(time)
        skip_connections = []
        
        # Add a sequence dimension for attention
        text_embedding = text_embedding.unsqueeze(1)

        # Downsampling
        for resblock, attention, downsample in self.downs:
            x = resblock(x, t)
            x = attention(x, text_embedding)
            skip_connections.append(x)
            x = downsample(x)
        
        # Middle
        x = self.mid_block1(x, t)
        x = self.mid_attn(x, text_embedding)
        x = self.mid_block2(x, t)

        # Upsampling
        for resblock, attention, upsample in self.ups:
            s = skip_connections.pop()
            x = torch.cat((x, s), dim=1)
            x = resblock(x, t)
            x = attention(x, text_embedding)
            x = upsample(x)
        
        output = self.final_conv(x)

        # Remove padding to match original input size
        return output[:, :, :, :original_width]


if __name__ == '__main__':
    logging.info("Running model test...")
    model = GeneratorUNet()
    
    dummy_spectrogram = torch.randn(4, 1, 128, 862) # Batch of 4, 1 channel, 128 mels, 862 time frames
    dummy_time = torch.randint(0, 1000, (4,)).long()
    dummy_text_embedding = torch.randn(4, 384) # Batch of 4, 384 dims
    
    output = model(dummy_spectrogram, dummy_time, dummy_text_embedding)
    
    print("\n--- Model Test ---")
    print(f"Input spectrogram shape: {dummy_spectrogram.shape}")
    print(f"Output spectrogram shape: {output.shape}")
    
    assert output.shape == dummy_spectrogram.shape, "Output shape does not match input shape!"
    print("\nModel test passed")