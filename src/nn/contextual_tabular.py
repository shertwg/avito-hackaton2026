"""Early donor interactions and within-category candidate competition."""
import torch
from torch import nn


class ContextualTabular(nn.Module):
    def __init__(self, width, d_model, donor_widths):
        super().__init__()
        self.groups = tuple(name for name, size in donor_widths.items() if size)
        self.signal_indices = tuple(i for i in (5, 6, 7, 8, 9, 15, 17, 18) if i < width)
        total = width + sum(donor_widths.values()) + len(self.signal_indices) + 1
        self.project = nn.Sequential(nn.Linear(total, 96), nn.SiLU(), nn.Linear(96, d_model))
        nn.init.zeros_(self.project[-1].weight)
        nn.init.zeros_(self.project[-1].bias)

    @staticmethod
    def category_context(signals, categories, valid):
        # Compact category IDs independently per feed, regardless of vocabulary size.
        ordered, perm = categories.sort(dim=1)
        starts = torch.ones_like(ordered)
        starts[:, 1:] = (ordered[:, 1:] != ordered[:, :-1]).long()
        groups = torch.empty_like(perm).scatter_(1, perm, starts.cumsum(1) - 1)
        mask = valid.unsqueeze(-1)
        index = groups.unsqueeze(-1)
        counts = signals.new_zeros((*valid.shape, 1)).scatter_add_(1, index, mask.float())
        count = counts.gather(1, index).clamp_min(1)
        index = index.expand_as(signals)
        sums = torch.zeros_like(signals).scatter_add_(1, index, signals * mask)
        centered = (signals - sums.gather(1, index) / count) * mask
        squares = torch.zeros_like(signals).scatter_add_(1, index, centered.square())
        z = centered * torch.rsqrt(squares.gather(1, index) / count + 1e-5)
        share = count / mask.sum(1, keepdim=True).clamp_min(1)
        return torch.cat((z.clamp(-6, 6), share * mask), dim=-1)

    def forward(self, batch):
        features = batch["cand_features"].float().clamp(-8, 8)
        signals = features[..., self.signal_indices]
        relative = self.category_context(signals, batch["cand_category_id"], batch["candidate_mask"])
        donors = [batch[group].float().clamp(-8, 8) for group in self.groups]
        return self.project(torch.cat([features, relative, *donors], dim=-1))
