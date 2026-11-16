"""Ranking objectives used in neural adaptation experiments."""
from __future__ import annotations

import math

import torch
from torch.nn.functional import softplus


def lambda_ndcg30_loss(scores, target, valid):
    """Pairwise logistic loss weighted by NDCG@30 swaps for binary labels."""
    scores = scores.float().masked_fill(~valid, 0.0)
    width = scores.shape[1]
    ranks = torch.arange(width, device=scores.device)
    discount = torch.where(ranks < 30, 1 / (ranks + 2).float().log2(), 0.0)
    positive = valid & (target > 0)
    negative = valid & ~positive

    with torch.no_grad():
        # Stable sorting puts negative items first when model scores tie.
        by_label = positive.to(torch.int8).argsort(dim=1, stable=True)
        ordered_scores = scores.masked_fill(~valid, -torch.inf).gather(1, by_label)
        order = by_label.gather(1, ordered_scores.argsort(dim=1, descending=True, stable=True))
        by_item = torch.zeros_like(scores).scatter_(1, order, discount.expand_as(scores))
        idcg = ((ranks < positive.sum(1, keepdim=True)) * discount).sum(1).clamp_min(1)
        swaps = (by_item[:, :, None] - by_item[:, None, :]).abs()
        weights = swaps * (positive[:, :, None] & negative[:, None, :]) / idcg[:, None, None]

    margins = scores[:, :, None] - scores[:, None, :]
    losses = (softplus(-margins) * weights).sum(dim=(1, 2))
    return (losses / valid.sum(1).clamp_min(1)).mean()


def boost_hard_negative_logits(scores, target, valid, reference_scores):
    """Triple the softmax mass of up to 20 highest-ranked labeled negatives."""
    negative = valid & (target == 0)
    with torch.no_grad():
        reference = reference_scores.masked_fill(~negative, -torch.inf)
        indices = reference.topk(min(20, scores.shape[1]), dim=1).indices
        hard = torch.zeros_like(negative).scatter_(1, indices, True) & negative
    return scores.float() + hard * math.log(3.0)
