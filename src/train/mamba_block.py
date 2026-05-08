"""
Pure PyTorch Mamba block (no CUDA kernels needed).

Adapted from johnma2006/mamba-minimal. Single MambaBlock = selective state-space
core (S6) wrapped with input/output projections, 1D causal conv, and gated SiLU.

For our task (max_seq_len=50, small batch), the explicit Python sequential scan
is tractable (~50 iters per forward).

Bidirectional version: forward Mamba on x + backward Mamba on flipped x,
concat → 2× hidden dim output. Each direction has hidden_dim // 2 output.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    """Selective state-space block (Mamba/S6).

    Args:
        d_model: input/output channel dim
        d_state: state-space inner dim (default 16)
        d_conv: 1D conv kernel size (default 4)
        expand: input projection expansion factor (default 2 → d_inner = 2*d_model)
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank=None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16) if dt_rank is None else dt_rank

        # Input projection: x → (x_, z) of size d_inner each
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Causal 1D conv (depthwise, groups=d_inner)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True,
        )

        # SSM parameters: x → (delta, B, C) via single linear
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        # delta projection (low-rank)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize delta_proj bias for stable training (per Mamba paper Appendix D)
        dt_init_std = self.dt_rank ** -0.5 * 1.0
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        # delta bias init: softplus^-1 of dt sampled from log-uniform
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # State-space matrices A (negated, log-parameterized) and D (skip)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))  # (d_inner, d_state)
        self.D = nn.Parameter(torch.ones(self.d_inner))  # (d_inner,)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x, mask=None):
        """x: (B, L, d_model) → (B, L, d_model)"""
        B, L, _ = x.shape
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # Causal 1D conv (truncate to L)
        x_ = x_.transpose(1, 2)  # (B, d_inner, L)
        x_ = self.conv1d(x_)[:, :, :L]
        x_ = x_.transpose(1, 2)  # (B, L, d_inner)
        x_ = F.silu(x_)

        # Selective scan
        y = self.selective_scan(x_)

        # Gated output
        y = y * F.silu(z)
        return self.out_proj(y)

    def selective_scan(self, x):
        """Selective state-space scan (sequential).

        x: (B, L, d_inner) → y: (B, L, d_inner)
        """
        B, L, d_inner = x.shape
        d_state = self.A_log.shape[1]

        # SSM parameters
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state), negative for stability
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        dt_proj_in, B_in, C_in = torch.split(
            x_dbl, [self.dt_rank, d_state, d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt_proj_in))  # (B, L, d_inner)

        # Discretize: dA = exp(dt * A), dB = dt * B
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, d_inner, d_state)
        dB = dt.unsqueeze(-1) * B_in.unsqueeze(2)  # (B, L, d_inner, d_state)

        # Sequential scan over L
        h = torch.zeros(B, d_inner, d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            # output y_t = C_t @ h_t per channel
            y_t = (h * C_in[:, t].unsqueeze(1)).sum(-1)  # (B, d_inner)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # (B, L, d_inner)

        return y + x * D


class BidirectionalMamba(nn.Module):
    """Forward Mamba + reverse Mamba, concat outputs (drop-in for BiLSTM).

    Total output channels = d_model (split d_model//2 per direction in MambaBlock).
    """
    def __init__(self, d_model, num_layers=2, d_state=16, d_conv=4, expand=2, dropout=0.3):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for bidirectional split"
        per_dir = d_model // 2
        self.layers_fwd = nn.ModuleList([
            MambaBlock(per_dir, d_state, d_conv, expand) for _ in range(num_layers)
        ])
        self.layers_bwd = nn.ModuleList([
            MambaBlock(per_dir, d_state, d_conv, expand) for _ in range(num_layers)
        ])
        # Project full d_model input to per-direction d_model//2
        self.fwd_proj = nn.Linear(d_model, per_dir)
        self.bwd_proj = nn.Linear(d_model, per_dir)
        self.norm_fwd = nn.LayerNorm(per_dir)
        self.norm_bwd = nn.LayerNorm(per_dir)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """x: (B, L, d_model) → (B, L, d_model)"""
        # Forward direction
        h_f = self.fwd_proj(x)
        for layer in self.layers_fwd:
            h_f = h_f + self.dropout(layer(self.norm_fwd(h_f), mask))

        # Backward direction (flip, run, flip back)
        h_b = self.bwd_proj(x).flip(dims=[1])
        for layer in self.layers_bwd:
            h_b = h_b + self.dropout(layer(self.norm_bwd(h_b), mask))
        h_b = h_b.flip(dims=[1])

        return torch.cat([h_f, h_b], dim=-1)
