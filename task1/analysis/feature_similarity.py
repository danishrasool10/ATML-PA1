"""Cosine stability between clean and transformed representations:

    I_T = (1/N) * sum_i  f(x_i) . f(T(x_i)) / (||f(x_i)|| * ||f(T(x_i))||)
"""
import torch


def cosine_stability(feat_clean: torch.Tensor, feat_transformed: torch.Tensor) -> dict:
    """`feat_clean` and `feat_transformed` must be row-aligned (same order,
    same images). Returns the mean cosine stability I_T and the per-sample
    cosine similarities."""
    assert feat_clean.shape[0] == feat_transformed.shape[0], "features must be row-aligned"
    fc = torch.nn.functional.normalize(feat_clean, dim=-1)
    ft = torch.nn.functional.normalize(feat_transformed, dim=-1)
    per_sample = (fc * ft).sum(dim=-1)
    return {
        "mean_cosine_stability": per_sample.mean().item(),
        "per_sample": per_sample.numpy(),
    }