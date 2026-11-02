"""AST-extracted network/math only; no driver or test-scoring routine."""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def site_lhat(x, site_codes):
    """Scaled symmetric Laplacian (lambda_max=2) for same-site cliques.

    The original graph has A_ij=1 when site_i==site_j and i!=j.
    Therefore Lhat=-D^(-1/2) A D^(-1/2), with zero isolated rows.
    """
    n_groups = getattr(site_codes, '_report_n_groups', None)
    if n_groups is None:
        n_groups = int(site_codes.max().item()) + 1
    membership = getattr(site_codes, '_report_site_membership', None)
    if membership is not None:
        # A precomputed grouping matrix avoids atomic scatter nondeterminism.
        sums = membership @ x
    else:
        sums = torch.zeros((n_groups, x.shape[1]), dtype=x.dtype, device=x.device)
        sums.index_add_(0, site_codes, x)
    counts = getattr(site_codes, '_report_site_counts', None)
    if counts is None:
        counts = torch.bincount(site_codes, minlength=n_groups).to(x.dtype)
    denominator = (counts[site_codes] - 1).clamp_min(1).unsqueeze(1)
    return -(sums[site_codes] - x) / denominator

class SiteChebK2(nn.Module):
    """K=2 ChebConv for the exact graph used by the archive's GCN.

    Identical operator, Xavier-uniform (Glorot) weights and zero bias to the
    documented PyG construction. This does not promise identical RNG consumption
    to a particular PyG version's constructors/scatter reduction order.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.lins = nn.ModuleList([nn.Linear(input_dim, output_dim, bias=False)
                                   for _ in range(2)])
        self.bias = nn.Parameter(torch.zeros(output_dim))
        for layer in self.lins:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x, site_codes):
        return self.lins[0](x) + self.lins[1](site_lhat(x, site_codes)) + self.bias

class SiteGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.conv1 = SiteChebK2(input_dim, hidden_dim)
        self.conv2 = SiteChebK2(hidden_dim, hidden_dim)
        self.conv3 = SiteChebK2(hidden_dim, 2)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, site_codes):
        h = self.relu(self.dropout(self.conv1(x, site_codes)))
        h = self.relu(self.dropout(self.conv2(h, site_codes)))
        return F.log_softmax(self.conv3(h, site_codes), dim=1)
