"""AST-extracted network/math only; no driver or test-scoring routine."""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class ROIProfileNetwork(nn.Module):
    """Atlas ROI tokens; fixed nonnegative gate; exact empty-D scalar anchor."""
    def __init__(self, classifier, rois, cfg):
        super().__init__()
        hidden = cfg['hidden']
        self.classifier = classifier
        self.direction_weight = cfg['direction_weight']
        self.ic_projection = nn.Linear(rois, hidden)
        # No bias: zero D adds exactly zero, without dividing IC by (1+gate).
        self.d_projection = nn.Linear(rois, hidden, bias=False)
        self.input_norm = nn.LayerNorm(hidden)
        self.input_dropout = nn.Dropout(cfg['dropout'])
        if classifier == 'lstm':
            self.encoder = nn.LSTM(hidden, hidden, num_layers=1, batch_first=True)
            self.position = None
        else:
            self.position = nn.Parameter(torch.empty(1, rois, hidden))
            nn.init.normal_(self.position, mean=0., std=.02)
            # Construct separately; TransformerEncoder clones would share initial values.
            self.encoder = nn.ModuleList([
                nn.TransformerEncoderLayer(d_model=hidden, nhead=cfg['heads'], dim_feedforward=2 * hidden,
                    dropout=cfg['dropout'], activation='gelu', batch_first=True, norm_first=True)
                for _ in range(cfg['layers'])])
        self.output_norm = nn.LayerNorm(hidden)
        self.output_dropout = nn.Dropout(cfg['dropout'])
        self.output = nn.Linear(hidden, 1)

    def forward(self, ic, d):
        tokens = self.ic_projection(ic) + self.direction_weight * self.d_projection(d)
        if self.position is not None:
            tokens = tokens + self.position
        tokens = self.input_dropout(F.gelu(self.input_norm(tokens)))
        if self.classifier == 'lstm':
            tokens, _ = self.encoder(tokens)
        else:
            for layer in self.encoder:
                tokens = layer(tokens)
        pooled = self.output_dropout(self.output_norm(tokens.mean(dim=1)))
        return self.output(pooled).squeeze(-1)
