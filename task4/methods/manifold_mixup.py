"""Manifold mixup between different classes for PROSER."""
import torch


def sample_different_class_partners(labels):
    """Pick a random different-class partner for each sample; returns partner index and valid mask."""
    diff = (labels[:, None] != labels[None, :]).float()
    valid = diff.sum(1) > 0
    diff[~valid] = 1.0  # avoid an all-zero row; these samples are masked out by `valid`
    partner = torch.multinomial(diff, 1).squeeze(1)
    return partner, valid


def manifold_mixup(h, labels, alpha=2.0):
    """Mix h (B, C, H, W) after layer2 with a different-class partner; returns h_mix and valid mask."""
    partner, valid = sample_different_class_partners(labels)
    beta = torch.distributions.Beta(torch.tensor(float(alpha)), torch.tensor(float(alpha)))
    lam = beta.sample((h.size(0),)).to(device=h.device, dtype=h.dtype).view(-1, 1, 1, 1)
    h_mix = lam * h + (1.0 - lam) * h[partner]
    return h_mix, valid