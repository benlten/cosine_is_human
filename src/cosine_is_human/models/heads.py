"""Similarity heads compared in the paper.

All heads take two (or three) CLS-token embeddings and return a distance score
where lower = more similar.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineHead(nn.Module):
    """Plain cosine distance between two CLS embeddings.

    distance(a, b) = 1 - cos(a, b).  Parameter-free.
    """

    def forward(self, ref: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        return 1.0 - F.cosine_similarity(ref, cand, dim=-1)


class TverskyHead(nn.Module):
    """Tversky (1977) contrast model with frozen α = β.

    S(a, b) = |A ∩ B| - α|A \\ B| - β|B \\ A|.
    Feature set sizes are approximated by soft element-wise products.
    """

    def __init__(self, dim: int = 768, alpha: float = 0.5, beta: float = 0.5, learnable: bool = False):
        super().__init__()
        if learnable:
            self.alpha = nn.Parameter(torch.tensor(float(alpha)))
            self.beta = nn.Parameter(torch.tensor(float(beta)))
        else:
            self.register_buffer("alpha", torch.tensor(float(alpha)))
            self.register_buffer("beta", torch.tensor(float(beta)))

    def forward(self, ref: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        a = F.softplus(ref)
        b = F.softplus(cand)
        intersection = (a * b).sum(dim=-1)
        diff_ab = F.relu(a - b).sum(dim=-1)
        diff_ba = F.relu(b - a).sum(dim=-1)
        sim = intersection - self.alpha * diff_ab - self.beta * diff_ba
        # Distance is negative similarity, offset for positivity in training
        return -sim


class CognitiveV3Head(nn.Module):
    """Cog-v3: frozen cosine alignment + GCM context + Tversky contrast.

    Takes (ref, cand) CLS embeddings. Produces a scalar distance per pair.
    The context term uses Nosofsky-style exponential decay over squared
    Euclidean distance; the contrast term is a learnable Tversky.
    """

    def __init__(self, dim: int = 768, hidden: int = 256):
        super().__init__()
        self.cos = CosineHead()
        self.gcm_scale = nn.Parameter(torch.tensor(1.0))
        self.gcm_mix = nn.Parameter(torch.tensor(0.5))
        self.tversky = TverskyHead(dim=dim, alpha=0.5, beta=0.5, learnable=True)
        self.proj = nn.Linear(dim, hidden)
        self.readout = nn.Linear(hidden, 1)

    def forward(self, ref: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        cos_d = self.cos(ref, cand)
        gcm = torch.exp(-self.gcm_scale.abs() * ((ref - cand) ** 2).sum(dim=-1))
        tv = self.tversky(ref, cand)
        feats = torch.stack([cos_d, 1.0 - gcm, tv], dim=-1)
        # Small MLP over the 3 scalar terms for final scoring
        h = torch.tanh(self.proj(self.proj.weight.new_zeros(feats.shape[0], self.proj.in_features) + feats.sum(dim=-1, keepdim=True)))
        return self.readout(h).squeeze(-1)


class TverskyNeuralNetwork(nn.Module):
    """Doumbouya 2025 differentiable Tversky Neural Network.

    Learns a feature bank Ω ∈ R^{n_features × dim} and computes
    intersection = prod_j σ(a·ω_j) σ(b·ω_j), difference = substractmatch,
    aggregated into a single similarity score.
    """

    def __init__(self, dim: int = 768, n_features: int = 256):
        super().__init__()
        self.omega = nn.Parameter(torch.randn(n_features, dim) / dim ** 0.5)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def _memberships(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x @ self.omega.T)  # (B, F)

    def forward(self, ref: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        a = self._memberships(ref)
        b = self._memberships(cand)
        intersection = (a * b).sum(dim=-1)
        diff_ab = F.relu(a - b).sum(dim=-1)
        diff_ba = F.relu(b - a).sum(dim=-1)
        sim = intersection - self.alpha * diff_ab - self.beta * diff_ba
        return -sim
