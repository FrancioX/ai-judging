"""Lightweight ST-GCN + MLP regressor for freeride score prediction.

Architecture:
  Input: (B, 3, T, 17)  — batch × xyz × time × joints

  ST-GCN Block 1:  SpatialGraphConv(3→64)   + TemporalConv1d(k=9)            + BN + ReLU + Drop(0.2)
  ST-GCN Block 2:  SpatialGraphConv(64→128)  + TemporalConv1d(k=9, stride=2)  + BN + ReLU + Drop(0.2)
  ST-GCN Block 3:  SpatialGraphConv(128→256) + TemporalConv1d(k=9, stride=2)  + BN + ReLU + Drop(0.2)

  Temporal Pooling:  mean + std over T → concat → (B, 512, 17)
  Spatial Pooling:   mean over joints  → (B, 512)

  MLP Head: Linear(512→128) → ReLU → Drop(0.3) → Linear(128→1)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.scoring.graph import NUM_JOINTS, build_adjacency


class SpatialGraphConv(nn.Module):
    """Spatial graph convolution using centripetal partitioning (K=3 subsets)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.K = 3  # number of partitions: self, root-ward, leaf-ward
        # One linear transform per partition
        self.W = nn.ModuleList([
            nn.Linear(in_channels, out_channels, bias=False) for _ in range(self.K)
        ])
        # Learnable attention mask per partition
        self.alpha = nn.ParameterList([
            nn.Parameter(torch.ones(NUM_JOINTS, NUM_JOINTS)) for _ in range(self.K)
        ])

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  (B, C_in, T, N)
        A : torch.Tensor  (K, N, N)  — normalized adjacency matrices

        Returns
        -------
        torch.Tensor  (B, C_out, T, N)
        """
        B, C, T, N = x.shape
        out = None
        for k in range(self.K):
            # Masked adjacency
            A_k = A[k] * torch.sigmoid(self.alpha[k])  # (N, N)
            # x: (B, C, T, N) → (B*T, N, C)
            x_flat = x.permute(0, 2, 3, 1).reshape(B * T, N, C)
            # Graph aggregation: (B*T, N, C) × (N, N)^T → (B*T, N, C)
            # A_k[i,j] means node j contributes to node i
            agg = torch.einsum("ni,bic->bnc", A_k, x_flat)  # (B*T, N, C)
            # Linear transform
            agg = self.W[k](agg)  # (B*T, N, C_out)
            agg = agg.reshape(B, T, N, -1).permute(0, 3, 1, 2)  # (B, C_out, T, N)
            out = agg if out is None else out + agg
        return out


class TemporalConv(nn.Module):
    """Temporal 1D convolution over the time dimension."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 9,
        stride: int = 1,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            channels, channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(padding, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T, N) → (B, C, T', N)"""
        return self.conv(x)


class STGCNBlock(nn.Module):
    """One ST-GCN block: spatial graph conv + temporal conv + residual + BN + ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.spatial = SpatialGraphConv(in_channels, out_channels)
        self.temporal = TemporalConv(out_channels, kernel_size=9, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

        # Residual connection
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.spatial(x, A)
        x = self.temporal(x)
        x = self.bn(x)
        x = self.relu(x + res)
        return self.drop(x)


class STGCNRegressor(nn.Module):
    """Full ST-GCN regressor: 3 blocks → temporal pooling → spatial pooling → MLP."""

    def __init__(self, dropout_mlp: float = 0.3) -> None:
        super().__init__()
        A = torch.from_numpy(build_adjacency()).float()
        self.register_buffer("A", A)  # (3, 17, 17)

        self.block1 = STGCNBlock(3, 64, stride=1, dropout=0.2)
        self.block2 = STGCNBlock(64, 128, stride=2, dropout=0.2)
        self.block3 = STGCNBlock(128, 256, stride=2, dropout=0.2)

        # MLP head: input is 512 (mean+std over T, then mean over joints)
        self.mlp = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_mlp),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  (B, 3, T, 17)

        Returns
        -------
        torch.Tensor  (B,)  predicted scores
        """
        A = self.A

        x = self.block1(x, A)   # (B, 64,  T,   17)
        x = self.block2(x, A)   # (B, 128, T/2, 17)
        x = self.block3(x, A)   # (B, 256, T/4, 17)

        # Temporal pooling: mean + std → (B, 512, 17)
        t_mean = x.mean(dim=2)                   # (B, 256, 17)
        t_std = x.std(dim=2).nan_to_num(0.0)     # (B, 256, 17)
        x = torch.cat([t_mean, t_std], dim=1)    # (B, 512, 17)

        # Spatial pooling: mean over joints → (B, 512)
        x = x.mean(dim=2)

        # MLP
        return self.mlp(x).squeeze(-1)  # (B,)
