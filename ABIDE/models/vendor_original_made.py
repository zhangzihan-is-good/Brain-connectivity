"""AST-extracted network/math only; no driver or test-scoring routine."""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class OriginalAE(nn.Module):
    def __init__(self, input_size, code_size):
        super().__init__()
        bound = 6.0 / math.sqrt(input_size + code_size)
        self.W_enc = nn.Parameter(torch.empty(input_size, code_size).uniform_(-bound, bound))
        self.b_enc = nn.Parameter(torch.zeros(code_size))
        self.b_dec = nn.Parameter(torch.zeros(input_size))
        self.W_dec = nn.Parameter(torch.empty(code_size, input_size).uniform_(-bound, bound))

    def encode(self, x):
        return torch.tanh(x @ self.W_enc + self.b_enc)

    def loss(self, x, corruption, sparse_p=None, sparse_coefficient=0.):
        corrupted = x * torch.rand_like(x) * (1. - corruption) if corruption > 0 else x
        hidden = self.encode(corrupted)
        reconstruction = hidden @ self.W_dec + self.b_dec
        loss = torch.sqrt(torch.mean((x - reconstruction).square()))
        if sparse_p is not None:
            p_hat = hidden.abs().mean(0)
            kl = sparse_p * torch.log(sparse_p / p_hat) + (1. - sparse_p) * torch.log((1. - sparse_p) / (1. - p_hat))
            loss = loss + sparse_coefficient * kl.sum()
        return loss

class OriginalMLP(nn.Module):
    def __init__(self, ae1, ae2, extra_size, hidden_size, keep_probs):
        super().__init__()
        n_conn, width1 = ae1.W_enc.shape
        width2 = ae2.W_enc.shape[1]
        input_size = n_conn + extra_size
        bound = 6.0 / math.sqrt(input_size + width1)
        # Shape repair: use the trained connection block and fresh Glorot rows
        # only for genuinely supplied auxiliary SEX/AGE input columns.
        w1 = torch.empty(input_size, width1, device=ae1.W_enc.device, dtype=ae1.W_enc.dtype).uniform_(-bound, bound)
        w1[:n_conn].copy_(ae1.W_enc.detach())
        self.W1 = nn.Parameter(w1)
        self.b1 = nn.Parameter(ae1.b_enc.detach().clone())
        self.W2 = nn.Parameter(ae2.W_enc.detach().clone())
        self.b2 = nn.Parameter(ae2.b_enc.detach().clone())
        w3 = (np.random.randn(width2, hidden_size) / 10000.).astype(np.float32)
        self.W3 = nn.Parameter(torch.from_numpy(w3).to(device=ae1.W_enc.device, dtype=ae1.W_enc.dtype))
        self.b3 = nn.Parameter(ae2.b_enc[:hidden_size].detach().clone())
        bound_out = 3.0 / math.sqrt(hidden_size + 2)
        self.W_out = nn.Parameter(torch.empty(hidden_size, 2, device=ae1.W_enc.device, dtype=ae1.W_enc.dtype).uniform_(-bound_out, bound_out))
        self.b_out = nn.Parameter(torch.zeros(2, device=ae1.W_enc.device, dtype=ae1.W_enc.dtype))
        self.keep_probs = keep_probs

    def forward(self, x):
        for weight, bias, keep in ((self.W1, self.b1, self.keep_probs[0]),
                                   (self.W2, self.b2, self.keep_probs[1]),
                                   (self.W3, self.b3, self.keep_probs[2])):
            x = torch.tanh(x @ weight + bias)
            x = F.dropout(x, p=1. - keep, training=self.training)
        return x @ self.W_out + self.b_out

def _copy_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}

def train_original_ae(Xtr, Xva, *, code_size, epochs, batch_size, lr,
                      corruption, sparse_p=None, sparse_coefficient=0.):
    """Original contiguous, nonshuffled full minibatches; discard remainder.

    Checkpoint cost is the mean of fresh noisy validation losses after EVERY
    training minibatch within an epoch, as in nn_new.py, not just epoch-end cost.
    Test reconstruction monitoring is removed; no test labels or inputs needed.
    """
    model = OriginalAE(Xtr.shape[1], code_size).to(Xtr.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    batches = len(Xtr) // batch_size
    if batches < 1:
        raise ValueError("AE needs at least one complete original minibatch")
    best_cost, best_state, best_epoch = float("inf"), None, -1
    history = []
    for epoch in range(1, int(epochs) + 1):
        # Source stores per-batch float32 costs in np.zeros(float64) before mean.
        train_sum = torch.zeros((), device=Xtr.device, dtype=torch.float64)
        valid_sum = torch.zeros((), device=Xtr.device, dtype=torch.float64)
        for start in range(0, batches * batch_size, batch_size):
            optimizer.zero_grad()
            loss = model.loss(Xtr[start:start + batch_size], corruption, sparse_p, sparse_coefficient)
            loss.backward()
            optimizer.step()
            train_sum += loss.detach()
            with torch.no_grad():
                valid_sum += model.loss(Xva, corruption, sparse_p, sparse_coefficient)
        train_cost, valid_cost = (float(v.item()) / batches for v in (train_sum, valid_sum))
        if not math.isfinite(train_cost) or not math.isfinite(valid_cost):
            raise FloatingPointError(f"Original AE nonfinite cost at epoch {epoch}")
        improved = valid_cost < best_cost
        if improved:
            best_cost, best_epoch, best_state = valid_cost, epoch, _copy_state(model)
        history.append({"epoch": epoch, "train_cost": train_cost,
                        "validation_cost": valid_cost, "selected": improved})
    model.load_state_dict(best_state)
    return model, history, best_epoch
