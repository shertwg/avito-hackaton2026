"""Продвинутая нейросетевая модель ранжирования рекомендаций Avito.

Архитектура:
1. History Transformer Encoder:
   - Товарные эмбеддинги (memmap) + category_id + event_type + event_source + positional_embedding.
   - Multi-Head Self-Attention слой с маскированием паддингов (history_mask).
2. Candidate Encoder:
   - Товарные эмбеддинги + category_id + числовые признаки (f1..f5) -> MLP в d_model.
3. Candidate-to-History Cross-Attention:
   - Query: кандидаты [B, M, d_model]
   - Key / Value: история [B, H, d_model]
   - Находит релевантные события истории для каждого отдельного кандидата.
4. User Interaction & Ranking Head:
   - Объединяет представление пользователя, кандидата, контекста истории и их dot-product взаимодействия.
5. Positional Debias Head (Two-Tower Debias):
   - Отдельная голова для моделирования position bias по фиче `pos` во время train.
   - На валидации/тесте голова позиции отключается, гарантируя несмещенный релевантный скор.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nn.contextual_tabular import ContextualTabular
from src.nn.dataset import CAND_FEAT_NAMES


EVENT_COSINE_TYPES = 10
EVENT_COSINE_WINDOW = 50


def event_cosine_features(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Maximum candidate cosine per event type over the latest history window."""
    history = batch["hist_emb"][:, :EVENT_COSINE_WINDOW]
    event_types = batch["hist_event_type"][:, :EVENT_COSINE_WINDOW]
    valid_history = (
        batch["history_mask"][:, :EVENT_COSINE_WINDOW]
        & batch["hist_has_emb_mask"][:, :EVENT_COSINE_WINDOW]
        & (event_types >= 1)
        & (event_types <= EVENT_COSINE_TYPES)
    )
    candidates = batch["cand_emb"]
    with torch.autocast(device_type=candidates.device.type, enabled=False):
        cand_unit = F.normalize(candidates.float(), dim=-1)
        hist_unit = F.normalize(history.float(), dim=-1)
        cosine = torch.bmm(cand_unit, hist_unit.transpose(1, 2)).clamp(-1, 1)
    groups = event_types.masked_fill(~valid_history, 0)
    features = cosine.new_full((*candidates.shape[:2], EVENT_COSINE_TYPES + 1), -torch.inf)
    features.scatter_reduce_(
        2, groups[:, None, :].expand_as(cosine), cosine, reduce="amax",
    )
    features = features[..., 1:]
    valid_candidate = batch["candidate_mask"] & batch["cand_has_emb_mask"]
    return features.masked_fill(~features.isfinite() | ~valid_candidate.unsqueeze(-1), 0.0)


class HistoryMatchHead(nn.Module):
    """Supervised candidate-to-event similarity with recent-history pooling."""

    WINDOW = 20

    def __init__(self, emb_dim: int, num_categories: int, width: int = 32) -> None:
        super().__init__()
        self.candidate_projection = nn.Linear(emb_dim, width)
        self.event_projection = nn.Linear(emb_dim, width)
        self.category_embedding = nn.Embedding(num_categories, width, padding_idx=0)
        self.event_embedding = nn.Embedding(32, width, padding_idx=0)
        self.recency_bias = nn.Parameter(-0.2 * torch.arange(self.WINDOW).float().log1p())
        self.output = nn.Linear(2, 1, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        window = slice(None, self.WINDOW)
        history = batch["hist_emb"][:, window]
        types = batch["hist_event_type"][:, window].clamp(0, self.event_embedding.num_embeddings - 1)
        hist_categories = batch["hist_category_id"][:, window].clamp(0, self.category_embedding.num_embeddings - 1)
        cand_categories = batch["cand_category_id"].clamp(0, self.category_embedding.num_embeddings - 1)
        candidate = self.candidate_projection(batch["cand_emb"]) + self.category_embedding(cand_categories)
        event = self.event_projection(history) + self.category_embedding(hist_categories) + self.event_embedding(types)
        valid = batch["history_mask"][:, window] & batch["hist_has_emb_mask"][:, window]
        valid_candidate = batch["candidate_mask"] & batch["cand_has_emb_mask"]

        query = F.normalize(candidate.float(), dim=-1)
        key = F.normalize(event.float(), dim=-1)
        pairs = 4 * torch.bmm(query, key.transpose(1, 2)).float()
        pairs = pairs + self.recency_bias[:history.shape[1]]
        pairs = pairs.masked_fill(~valid[:, None, :], -1e4)
        count = valid.sum(dim=1, keepdim=True)
        maximum = pairs.amax(dim=2)
        logmeanexp = pairs.logsumexp(dim=2) - count.clamp_min(1).float().log()
        pooled = torch.stack((maximum, logmeanexp), dim=-1)
        pooled = pooled.masked_fill(~(valid_candidate & (count > 0)).unsqueeze(-1), 0.0)
        return self.output(pooled).squeeze(-1)


class HistoryTransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_hist_len: int = 256,
        num_categories: int = 1000,
        num_event_types: int = 32,
        num_event_sources: int = 64,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Эмбеддинги признаков истории
        self.cat_emb = nn.Embedding(num_categories, 32, padding_idx=0)
        self.ev_type_emb = nn.Embedding(num_event_types, 16, padding_idx=0)
        self.ev_source_emb = nn.Embedding(num_event_sources, 16, padding_idx=0)
        self.pos_emb = nn.Embedding(max_hist_len + 1, 32)

        in_dim = 128 + 32 + 16 + 16 + 32
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

    def forward(
        self,
        hist_emb: torch.Tensor,
        hist_cat: torch.Tensor,
        hist_ev_type: torch.Tensor,
        hist_ev_source: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        hist_emb: [B, H, 128]
        hist_cat, hist_ev_type, hist_ev_source: [B, H]
        history_mask: [B, H] (bool, True = валидное событие)
        
        Returns:
            encoded_history: [B, H, d_model]
        """
        b, h, _ = hist_emb.shape

        # Позиции в истории: 0 = самое свежее, 1, 2...
        pos_idx = torch.arange(h, device=hist_emb.device).unsqueeze(0).expand(b, h)
        pos_idx = torch.clamp(pos_idx, 0, self.pos_emb.num_embeddings - 1)

        c_emb = self.cat_emb(torch.clamp(hist_cat, 0, self.cat_emb.num_embeddings - 1))
        et_emb = self.ev_type_emb(torch.clamp(hist_ev_type, 0, self.ev_type_emb.num_embeddings - 1))
        es_emb = self.ev_source_emb(torch.clamp(hist_ev_source, 0, self.ev_source_emb.num_embeddings - 1))
        p_emb = self.pos_emb(pos_idx)

        x = torch.cat([hist_emb, c_emb, et_emb, es_emb, p_emb], dim=-1)
        x = self.input_proj(x)  # [B, H, d_model]

        # Защита от NaN: если у пользователя вообще нет истории (вся строка mask = False),
        # временно открываем первый токен, а затем зануляем выход
        has_any = history_mask.any(dim=1, keepdim=True)  # [B, 1]
        safe_mask = history_mask.clone()
        safe_mask[:, 0] = safe_mask[:, 0] | (~has_any.squeeze(1))

        # В PyTorch src_key_padding_mask: True = игнорировать позицию
        padding_mask = ~safe_mask

        encoded = self.transformer(x, src_key_padding_mask=padding_mask)  # [B, H, d_model]

        # Зануляем паддинги
        encoded = encoded * history_mask.unsqueeze(-1).float()
        return encoded


class CandidateEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        num_categories: int = 1000,
        num_cand_feats: int = 24,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cat_emb = nn.Embedding(num_categories, 32, padding_idx=0)
        in_dim = 128 + 32 + num_cand_feats

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        cand_emb: torch.Tensor,
        cand_cat: torch.Tensor,
        cand_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        cand_emb: [B, M, 128]
        cand_cat: [B, M]
        cand_feats: [B, M, num_cand_feats]
        
        Returns:
            [B, M, d_model]
        """
        cat_e = self.cat_emb(torch.clamp(cand_cat, 0, self.cat_emb.num_embeddings - 1))
        x = torch.cat([cand_emb, cat_e, cand_feats], dim=-1)
        return self.mlp(x)


class PeriodicCandidateEncoder(nn.Module):
    """Learn periodic embeddings of normalized numeric signals."""

    FEATURES = (
        "f1_z", "f2_z", "f3_z", "f4_z", "f5_z", "cos_user", "cos_hist_mean",
        "cos_hist_max", "cos_hist_recent_max", "cos_hist_recent_mean",
    )

    def __init__(self, d_model: int, num_cand_feats: int) -> None:
        super().__init__()
        indices = [CAND_FEAT_NAMES.index(name) for name in self.FEATURES]
        if max(indices) >= num_cand_feats:
            raise ValueError("Periodic encoding requires the standard candidate feature layout")
        self.register_buffer("indices", torch.tensor(indices), persistent=False)
        frequencies = torch.linspace(0.125, 1.0, 8).repeat(len(indices), 1)
        self.frequencies = nn.Parameter(frequencies)
        self.project = nn.Sequential(
            nn.Linear(len(indices) * frequencies.shape[1] * 2, 64),
            nn.SiLU(),
            nn.Linear(64, d_model),
        )
        nn.init.zeros_(self.project[-1].weight)
        nn.init.zeros_(self.project[-1].bias)

    def forward(self, candidate_features: torch.Tensor) -> torch.Tensor:
        selected = candidate_features.index_select(-1, self.indices).float().clamp(-5, 5)
        phase = selected.unsqueeze(-1) * (2 * math.pi * self.frequencies.float())
        periodic = torch.cat((phase.sin(), phase.cos()), dim=-1).flatten(-2)
        return self.project(periodic)


class HistoryAttentionBias(nn.Module):
    """Per-head preferences for category matches and candidate/event similarity."""

    def __init__(self, nhead: int) -> None:
        super().__init__()
        self.category_scale = nn.Parameter(torch.zeros(nhead))
        self.cosine_scale = nn.Parameter(torch.zeros(nhead))

    def forward(self, batch: dict[str, torch.Tensor], safe_history: torch.Tensor,
                dtype: torch.dtype) -> torch.Tensor:
        candidate_category = batch["cand_category_id"][:, :, None]
        history_category = batch["hist_category_id"][:, None, :]
        same_category = (candidate_category == history_category) & (candidate_category > 0)
        category_valid = batch["candidate_mask"][:, :, None] & batch["history_mask"][:, None, :]
        candidate = F.normalize(batch["cand_emb"].float(), dim=-1)
        history = F.normalize(batch["hist_emb"].float(), dim=-1)
        cosine = candidate @ history.transpose(-1, -2)
        cosine_valid = batch["cand_has_emb_mask"][:, :, None] & batch["hist_has_emb_mask"][:, None, :]
        cosine = cosine.masked_fill(~(category_valid & cosine_valid), 0.0)
        same_category = (same_category & category_valid).to(cosine.dtype)
        bias = (
            same_category[:, None] * self.category_scale[None, :, None, None]
            + cosine[:, None] * self.cosine_scale[None, :, None, None]
        )
        return bias.masked_fill(~safe_history[:, None, None, :], -torch.inf).to(dtype).flatten(0, 1)


class FeedRankingModel(nn.Module):
    """Полнофункциональная модель ранжирования ленты с Cross-Attention и дебиасингом."""

    def __init__(
        self,
        emb_dim: int = 128,
        d_model: int = 128,
        nhead: int = 4,
        num_categories: int = 1000,
        num_cand_feats: int = 24,
        max_hist_len: int = 256,
        max_position: int = 512,
        dropout: float = 0.15,
        use_set_attention: bool = True,
        use_numeric_head: bool = False,
        use_history_context: bool = True,
        num_itemstats: int = 0,
        num_item2item: int = 0,
        num_time: int = 0,
        use_event_cosine: bool = False,
        use_history_match: bool = False,
        use_periodic_numeric: bool = False,
        use_history_attention_bias: bool = False,
        use_contextual_tabular: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.use_set_attention = use_set_attention
        self.use_history_context = use_history_context

        # 1. Энкодер истории
        self.hist_encoder = HistoryTransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=1,
            dropout=dropout,
            max_hist_len=max_hist_len,
            num_categories=num_categories,
        )

        # 2. Энкодер кандидатов
        self.cand_encoder = CandidateEncoder(
            d_model=d_model,
            num_categories=num_categories,
            num_cand_feats=num_cand_feats,
            dropout=dropout,
        )

        # 3. Проекция пользователя
        self.user_proj = nn.Sequential(
            nn.Linear(emb_dim, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
        )

        # 4. Cross-Attention: Query = Кандидаты, Key/Value = История
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(d_model)

        # 4.5. Listwise Set-Attention: Self-Attention между кандидатами внутри ленты
        if self.use_set_attention:
            self.set_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True,
            )
            self.set_norm = nn.LayerNorm(d_model)
            self.set_dropout = nn.Dropout(dropout)

        # 5. Основная голова скоринга (Контентная релевантность)
        head_in_dim = d_model * 3 + 3 + 1 + num_cand_feats
        self.main_head = nn.Sequential(
            nn.Linear(head_in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # 6. Голова дебиасинга позиции (Position Bias Tower)
        # Обучается только на train по фиче pos
        self.pos_bias_emb = nn.Embedding(max_position, 32)
        self.pos_bias_head = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.use_numeric_head = use_numeric_head
        if use_numeric_head:
            # Keep the shared model and dropout RNG aligned for paired runs.
            with torch.random.fork_rng(devices=[]):
                self.numeric_head = nn.Sequential(
                    nn.Linear(num_cand_feats, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                )
                # Start with the existing score and learn a residual from tabular features.
                nn.init.zeros_(self.numeric_head[-1].weight)
                nn.init.zeros_(self.numeric_head[-1].bias)


        self.itemstats_head = self._donor_head(num_itemstats)
        self.item2item_head = self._donor_head(num_item2item)
        self.time_head = self._donor_head(num_time)

        self.event_cosine_head = None
        if use_event_cosine:
            with torch.random.fork_rng(devices=[]):
                self.event_cosine_head = nn.Sequential(
                    nn.Linear(EVENT_COSINE_TYPES, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                )
                nn.init.zeros_(self.event_cosine_head[-1].weight)
                nn.init.zeros_(self.event_cosine_head[-1].bias)

        self.history_match_head = None
        if use_history_match:
            with torch.random.fork_rng(devices=[]):
                self.history_match_head = HistoryMatchHead(emb_dim, num_categories)

        self.periodic_numeric = None
        if use_periodic_numeric:
            # Preserve initialization and dropout draws of all shared layers.
            with torch.random.fork_rng(devices=[]):
                self.periodic_numeric = PeriodicCandidateEncoder(d_model, num_cand_feats)
        self.history_attention_bias = HistoryAttentionBias(nhead) if use_history_attention_bias else None

        self.contextual_tabular = None
        if use_contextual_tabular:
            with torch.random.fork_rng(devices=[]):
                widths = {"itemstats": num_itemstats, "item2item": num_item2item, "time": num_time}
                self.contextual_tabular = ContextualTabular(num_cand_feats, d_model, widths)

    @staticmethod
    def _donor_head(width):
        if not width:
            return None
        with torch.random.fork_rng(devices=[]):
            head = nn.Sequential(nn.Linear(width, 64), nn.ReLU(), nn.Linear(64, 1))
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        return head

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        use_pos_bias: bool = False,
        return_features: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            batch: словарь тензоров от collate_ranking_batch.
            use_pos_bias: если True и в batch есть 'pos', к скору добавляется position bias.
                          На val/test должно быть False!
            return_features: если True, дополнительно возвращает словарь косинусных сходств.
                          
        Returns:
            scores: [B, M] — итоговые скоры для обучения (с позицией, если use_pos_bias=True).
            main_scores: [B, M] — чистые контентные скоры без позиции (для валидации и сабмита).
        """
        user_emb = batch["user_emb"]            # [B, 128]
        cand_emb = batch["cand_emb"]            # [B, M, 128]
        cand_cat = batch["cand_category_id"]    # [B, M]
        cand_feats = batch["cand_features"]     # [B, M, num_feats]
        cand_mask = batch["candidate_mask"]     # [B, M]
        cand_has_emb = batch["cand_has_emb_mask"] # [B, M]

        b, m, _ = cand_emb.shape

        # 1. Энкодинг пользователя
        user_repr = self.user_proj(user_emb)    # [B, d_model]

        # 2. Энкодинг кандидатов
        cand_repr = self.cand_encoder(cand_emb, cand_cat, cand_feats) # [B, M, d_model]
        if self.periodic_numeric is not None:
            cand_repr = cand_repr + self.periodic_numeric(cand_feats)

        if self.contextual_tabular is not None:
            cand_repr = cand_repr + self.contextual_tabular(batch)

        # 3. Энкодинг истории
        history_mask = batch["history_mask"]    # [B, H]
        hist_encoded = self.hist_encoder(
            hist_emb=batch["hist_emb"],
            hist_cat=batch["hist_category_id"],
            hist_ev_type=batch["hist_event_type"],
            hist_ev_source=batch["hist_event_source"],
            history_mask=history_mask,
        ) # [B, H, d_model]

        # 4. Cross-Attention (Query = cand_repr, Key/Value = hist_encoded)
        has_any_hist = history_mask.any(dim=1, keepdim=True) # [B, 1]
        safe_hist_mask = history_mask.clone()
        safe_hist_mask[:, 0] = safe_hist_mask[:, 0] | (~has_any_hist.squeeze(1))
        padding_mask = ~safe_hist_mask
        attention_mask = None
        if self.history_attention_bias is not None:
            attention_mask = self.history_attention_bias(batch, safe_hist_mask, cand_repr.dtype)
            padding_mask = None  # The additive bias already masks padded keys.

        attn_out, _ = self.cross_attn(
            query=cand_repr,
            key=hist_encoded,
            value=hist_encoded,
            key_padding_mask=padding_mask,
            attn_mask=attention_mask,
        ) # [B, M, d_model]

        attn_out = attn_out * has_any_hist.unsqueeze(-1).float()
        # Keep attention dropout draws aligned in the history ablation.
        if not self.use_history_context:
            attn_out = torch.zeros_like(attn_out)
        cand_hist_ctx = self.cross_norm(cand_repr + attn_out) # Residual

        # 4.5. Listwise Set-Attention между кандидатами внутри ленты
        if self.use_set_attention:
            has_any_cand = cand_mask.any(dim=1, keepdim=True) # [B, 1]
            safe_cand_mask = cand_mask.clone()
            safe_cand_mask[:, 0] = safe_cand_mask[:, 0] | (~has_any_cand.squeeze(1))

            set_out, _ = self.set_attn(
                query=cand_hist_ctx,
                key=cand_hist_ctx,
                value=cand_hist_ctx,
                key_padding_mask=~safe_cand_mask,
            ) # [B, M, d_model]
            set_out = set_out * cand_mask.unsqueeze(-1).float()
            cand_hist_ctx = self.set_norm(cand_hist_ctx + self.set_dropout(set_out))
            cand_hist_ctx = cand_hist_ctx * cand_mask.unsqueeze(-1).float()

        # 5. Скалярные произведения и сходства
        # FP16 rounds the normalization epsilon to zero for zero vectors.
        u_norm = F.normalize(user_repr.float(), p=2, dim=-1).unsqueeze(1) # [B, 1, d_model]
        c_norm = F.normalize(cand_repr.float(), p=2, dim=-1)             # [B, M, d_model]
        h_norm = F.normalize(cand_hist_ctx.float(), p=2, dim=-1)         # [B, M, d_model]

        dot_u = (c_norm * u_norm).sum(dim=-1, keepdim=True)      # [B, M, 1]
        dot_h = (c_norm * h_norm).sum(dim=-1, keepdim=True)      # [B, M, 1]
        dot_uh = (u_norm * h_norm).sum(dim=-1, keepdim=True)     # [B, M, 1]
        has_emb_f = cand_has_emb.unsqueeze(-1).float()           # [B, M, 1]

        # 6. Основной контентный скор (Main Head)
        user_rep_exp = user_repr.unsqueeze(1).expand(-1, m, -1)
        head_input = torch.cat([
            cand_hist_ctx,
            user_rep_exp,
            cand_repr,
            dot_u,
            dot_h,
            dot_uh,
            has_emb_f,
            cand_feats,
        ], dim=-1)

        main_scores = self.main_head(head_input).squeeze(-1) # [B, M]

        if self.use_numeric_head:
            main_scores = main_scores + self.numeric_head(cand_feats.float()).squeeze(-1)

        if self.itemstats_head is not None:
            main_scores = main_scores + self.itemstats_head(batch["itemstats"]).squeeze(-1)

        if self.item2item_head is not None:
            main_scores = main_scores + self.item2item_head(batch["item2item"]).squeeze(-1)

        if self.time_head is not None:
            main_scores = main_scores + self.time_head(batch["time"]).squeeze(-1)

        if self.event_cosine_head is not None:
            main_scores = main_scores + self.event_cosine_head(event_cosine_features(batch)).squeeze(-1)

        if self.history_match_head is not None:
            main_scores = main_scores + self.history_match_head(batch)

        # 7. Positional Debias Tower
        if use_pos_bias and "pos" in batch and batch["pos"] is not None:
            raw_pos = batch["pos"]
            pos_valid = (raw_pos >= 0).to(main_scores.dtype)
            pos_indices = torch.clamp(raw_pos, 0, self.pos_bias_emb.num_embeddings - 1)
            pos_emb = self.pos_bias_emb(pos_indices) # [B, M, 32]
            pos_bias = self.pos_bias_head(pos_emb).squeeze(-1) * pos_valid # [B, M]
            total_scores = main_scores + pos_bias
        else:
            total_scores = main_scores

        # Маскируем паддинги большим отрицательным значением (в float32 для совместимости с AMP/fp16)
        total_scores = total_scores.float().masked_fill(~cand_mask, -1e9)
        main_scores = main_scores.float().masked_fill(~cand_mask, -1e9)

        if return_features:
            feats_dict = {
                "dot_user": dot_u.squeeze(-1).masked_fill(~cand_mask, 0.0),
                "dot_hist": dot_h.squeeze(-1).masked_fill(~cand_mask, 0.0),
                "dot_uh": dot_uh.squeeze(-1).masked_fill(~cand_mask, 0.0),
            }
            return total_scores, main_scores, feats_dict

        return total_scores, main_scores
