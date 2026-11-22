"""Checks for the optional numeric and attention paths in the neural ranker."""
import unittest
try:
    import torch
    import pytest
except ImportError as exc:
    raise unittest.SkipTest(f"optional NN test dependency missing: {exc}") from exc

from src.nn.model import FeedRankingModel, HistoryAttentionBias


def make_batch():
    generator = torch.Generator().manual_seed(33)
    return {
        "user_emb": torch.randn(2, 128, generator=generator),
        "cand_emb": torch.randn(2, 3, 128, generator=generator),
        "cand_category_id": torch.tensor([[1, 7, 8], [1, 7, 0]]),
        "cand_features": torch.randn(2, 3, 24, generator=generator),
        "candidate_mask": torch.tensor([[True, True, True], [True, True, False]]),
        "cand_has_emb_mask": torch.tensor([[True, False, True], [True, True, False]]),
        "hist_emb": torch.randn(2, 4, 128, generator=generator),
        "hist_category_id": torch.tensor([[1, 7, 8, 0], [0, 0, 0, 0]]),
        "hist_event_type": torch.zeros(2, 4, dtype=torch.long),
        "hist_event_source": torch.zeros(2, 4, dtype=torch.long),
        "history_mask": torch.tensor([[True, True, True, False], [False, False, False, False]]),
        "hist_has_emb_mask": torch.tensor([[True, True, True, False], [False, False, False, False]]),
    }


@pytest.mark.parametrize("option", ["use_periodic_numeric", "use_history_attention_bias", "use_contextual_tabular"])
def test_new_paths_start_from_same_scores_and_receive_gradients(option):
    batch = make_batch()
    torch.manual_seed(43)
    base = FeedRankingModel(d_model=32, dropout=0.0).eval()
    torch.manual_seed(43)
    variant = FeedRankingModel(d_model=32, dropout=0.0, **{option: True}).eval()
    for key, tensor in base.state_dict().items():
        torch.testing.assert_close(tensor, variant.state_dict()[key], rtol=0, atol=0)
    _, base_scores = base(batch)
    _, scores = variant(batch)
    torch.testing.assert_close(scores, base_scores)
    assert torch.isfinite(scores).all()

    scores[batch["candidate_mask"]].square().mean().backward()
    extra_gradients = [p.grad for key, p in variant.named_parameters() if key not in base.state_dict()]
    assert all(torch.isfinite(gradient).all() for gradient in extra_gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in extra_gradients)


def test_history_bias_masks_padding_and_missing_embedding():
    batch = make_batch()
    safe = batch["history_mask"].clone()
    safe[1, 0] = True
    layer = HistoryAttentionBias(nhead=1)
    with torch.no_grad():
        layer.category_scale.fill_(1)
        layer.cosine_scale.fill_(1)
        bias = layer(batch, safe, torch.float32)
    assert bias[0, 1, 1] == 1  # category match; no candidate embedding
    assert bias[0, 1, 0] == 0
    assert torch.isneginf(bias[0, :, 3]).all()
    assert (bias[1, :, 0] == 0).all()  # dummy key for the empty history
    assert torch.isneginf(bias[1, :, 1:]).all()


def test_learned_paths_remain_equivariant_to_candidate_permutation():
    batch = make_batch()
    model = FeedRankingModel(
        d_model=32, dropout=0.0, use_periodic_numeric=True, use_history_attention_bias=True, use_contextual_tabular=True
    ).eval()
    permutation = torch.tensor([1, 2, 0])
    permuted = {
        key: value[:, permutation] if key.startswith("cand_") or key == "candidate_mask" else value
        for key, value in batch.items()
    }
    with torch.no_grad():
        model.contextual_tabular.project[-1].weight.fill_(0.02)
        model.periodic_numeric.project[-1].weight.fill_(0.01)
        model.history_attention_bias.category_scale.fill_(0.7)
        model.history_attention_bias.cosine_scale.fill_(0.5)
        _, original = model(batch)
        _, changed = model(permuted)
    torch.testing.assert_close(changed, original[:, permutation])


def test_category_context_excludes_padding_and_handles_singletons():
    from src.nn.contextual_tabular import ContextualTabular

    features = torch.tensor([[[1.], [4.], [3.], [8.]]])
    categories = torch.tensor([[99999, 0, 99999, 99999]])
    mask = torch.tensor([[True, True, True, False]])
    result = ContextualTabular.category_context(features, categories, mask)
    expected = torch.tensor([[[-1., 2/3], [0., 1/3], [1., 2/3], [0., 0.]]])
    torch.testing.assert_close(result, expected, atol=1e-5, rtol=0)


def test_contextual_donor_influences_candidate_query():
    batch = make_batch()
    batch["itemstats"] = torch.ones(2, 3, 2)
    model = FeedRankingModel(d_model=32, dropout=0, num_itemstats=2, use_contextual_tabular=True).eval()
    with torch.no_grad():
        layer = model.contextual_tabular.project
        layer[0].weight.zero_()
        layer[0].bias.zero_()
        layer[0].weight[0, -1] = 1
        layer[-1].weight[0, 0] = 1
        original = model.contextual_tabular(batch)
        batch["itemstats"][0, 0, -1] = 2
        changed = model.contextual_tabular(batch)
    difference = changed - original
    assert difference[0, 0, 0] > 0
    assert difference.count_nonzero() == 1
