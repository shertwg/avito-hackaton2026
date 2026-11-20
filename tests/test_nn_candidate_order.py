"""Regression coverage for position-free candidate features."""
import numpy as np
import torch

from src.nn.dataset import CAND_FEAT_NAMES, compute_candidate_features
from src.nn.train_cv import shuffle_candidates


def test_features_follow_candidates_without_a_position_signal():
    items = np.array([20, 42, 80])
    categories = np.array([1, 0, 1])
    row = {"f1": [0.3, 0.1, 0.2]}
    empty = np.array([], dtype=int)
    clean = compute_candidate_features(row, items, categories, empty, empty)
    order = [2, 0, 1]
    reordered = compute_candidate_features(
        {"f1": np.asarray(row["f1"])[order]}, items[order], categories[order], empty, empty
    )
    np.testing.assert_allclose(reordered, clean[order], atol=1e-6)
    assert np.count_nonzero(clean[:, CAND_FEAT_NAMES.index("reserved_zero")]) == 0


def test_training_permutation_keeps_labels_and_donor_features_aligned():
    items = torch.tensor([[4, 7, 8], [3, 2, 0]])
    mask = items != 0
    batch = {
        "candidate_mask": mask,
        "cand_item_id": items,
        "target": (items == 7).float(),
        "pos": torch.where(mask, items * 2, -1),
        "itemstats": items.unsqueeze(-1).float(),
        "hist_item_id": torch.tensor([[9, 10, 11], [12, 13, 14]]),
    }
    torch.manual_seed(43)
    shuffled = shuffle_candidates(batch)
    shuffled_items = shuffled["cand_item_id"]
    assert not torch.equal(shuffled_items, items)
    assert torch.equal(shuffled["candidate_mask"], mask)
    assert torch.equal(shuffled["target"], (shuffled_items == 7).float())
    assert torch.equal(shuffled["pos"], torch.where(mask, shuffled_items * 2, -1))
    assert torch.equal(shuffled["itemstats"].squeeze(-1), shuffled_items.float())
    assert torch.equal(shuffled["hist_item_id"], batch["hist_item_id"])
