from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from teenyzero.alphafold.features import build_square_pair_feature_tensor
from teenyzero.alphazero.config import (
    INPUT_PLANES,
    MODEL_CHANNELS,
    MODEL_RES_BLOCKS,
    POLICY_HEAD_CHANNELS,
    VALUE_HEAD_HIDDEN,
)


POLICY_SIZE = 4672


def _resolve_num_heads(channels: int, requested_heads: int) -> int:
    requested = max(1, int(requested_heads))
    for candidate in (requested, 8, 6, 4, 3, 2, 1):
        if channels % candidate == 0:
            return candidate
    return 1


class FeedForward(nn.Module):
    def __init__(self, channels: int, hidden_scale: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = channels * hidden_scale
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.fc1(tokens)
        tokens = F.gelu(tokens)
        tokens = self.dropout(tokens)
        tokens = self.fc2(tokens)
        return self.dropout(tokens)


class PairBiasSelfAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int, pair_feature_dim: int, dropout: float = 0.0):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}")

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        self.pair_bias_proj = nn.Linear(pair_feature_dim, num_heads, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, pair_features: torch.Tensor, return_attention: bool = False):
        batch_size, token_count, _ = tokens.shape

        query = self.q_proj(tokens).view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(tokens).view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(tokens).view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)

        logits = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        pair_bias = self.pair_bias_proj(pair_features).permute(2, 0, 1).unsqueeze(0)
        logits = logits + pair_bias

        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        merged = torch.matmul(attn, value)
        merged = merged.transpose(1, 2).contiguous().view(batch_size, token_count, self.channels)
        output = self.out_proj(merged)
        if return_attention:
            return output, attn
        return output


class RelationBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int, pair_feature_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn_norm = nn.LayerNorm(channels)
        self.attn = PairBiasSelfAttention(
            channels=channels,
            num_heads=num_heads,
            pair_feature_dim=pair_feature_dim,
            dropout=dropout,
        )
        self.ff_norm = nn.LayerNorm(channels)
        self.ff = FeedForward(channels=channels, hidden_scale=4, dropout=dropout)

    def forward(self, tokens: torch.Tensor, pair_features: torch.Tensor, return_attention: bool = False):
        attn_result = self.attn(self.attn_norm(tokens), pair_features, return_attention=return_attention)
        if return_attention:
            attn_out, attention = attn_result
        else:
            attn_out = attn_result
            attention = None
        tokens = tokens + attn_out
        tokens = tokens + self.ff(self.ff_norm(tokens))
        if return_attention:
            return tokens, attention
        return tokens


class AlphaFoldBoardModel(nn.Module):
    supports_geometry_aux = True

    def __init__(
        self,
        input_planes: int = INPUT_PLANES,
        channels: int = MODEL_CHANNELS,
        num_relation_blocks: int = MODEL_RES_BLOCKS,
        policy_head_channels: int = POLICY_HEAD_CHANNELS,
        value_hidden: int = VALUE_HEAD_HIDDEN,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_relation_blocks <= 0:
            raise ValueError("num_relation_blocks must be positive")
        num_heads = _resolve_num_heads(channels, num_heads)

        pair_features = build_square_pair_feature_tensor()
        pair_feature_dim = int(pair_features.shape[-1])

        self.input_planes = input_planes
        self.channels = channels
        self.num_relation_blocks = num_relation_blocks
        self.num_heads = num_heads
        self.policy_head_channels = policy_head_channels
        self.value_hidden = value_hidden

        self.input_proj = nn.Conv2d(input_planes, channels, kernel_size=1)
        self.input_mix = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.position_embed = nn.Parameter(torch.zeros(1, 64, channels))
        self.register_buffer("pair_features", pair_features, persistent=False)

        self.blocks = nn.ModuleList(
            [
                RelationBlock(
                    channels=channels,
                    num_heads=num_heads,
                    pair_feature_dim=pair_feature_dim,
                    dropout=dropout,
                )
                for _ in range(num_relation_blocks)
            ]
        )
        self.final_norm = nn.LayerNorm(channels)

        self.policy_token_proj = nn.Linear(channels, policy_head_channels)
        self.policy_out = nn.Linear(64 * policy_head_channels, POLICY_SIZE)

        self.value_fc1 = nn.Linear(channels, value_hidden)
        self.value_fc2 = nn.Linear(value_hidden, 1)

        self.friendly_attack_head = nn.Linear(channels, 1)
        self.enemy_attack_head = nn.Linear(channels, 1)
        self.friendly_king_pressure_head = nn.Linear(channels, 1)
        self.enemy_king_pressure_head = nn.Linear(channels, 1)

        nn.init.normal_(self.position_embed, mean=0.0, std=0.02)
        nn.init.zeros_(self.value_fc2.weight)
        nn.init.zeros_(self.value_fc2.bias)

    def _encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = F.gelu(x)
        x = self.input_mix(x)
        x = F.gelu(x)
        tokens = x.flatten(2).transpose(1, 2)
        return tokens + self.position_embed

    def _run_relation_trunk(self, tokens: torch.Tensor, return_attention: bool = False):
        pair_features = self.pair_features.to(device=tokens.device, dtype=tokens.dtype)
        last_attention = None
        for block in self.blocks:
            if return_attention:
                tokens, last_attention = block(tokens, pair_features, return_attention=True)
            else:
                tokens = block(tokens, pair_features)
        if return_attention:
            return self.final_norm(tokens), last_attention
        return self.final_norm(tokens)

    def _policy_head(self, tokens: torch.Tensor) -> torch.Tensor:
        policy_tokens = self.policy_token_proj(tokens)
        policy_tokens = F.gelu(policy_tokens)
        return self.policy_out(policy_tokens.reshape(policy_tokens.size(0), -1))

    def _value_head(self, tokens: torch.Tensor) -> torch.Tensor:
        pooled = tokens.mean(dim=1)
        pooled = F.gelu(self.value_fc1(pooled))
        return torch.tanh(self.value_fc2(pooled))

    def _aux_heads(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "friendly_attack": self.friendly_attack_head(tokens).squeeze(-1).view(-1, 8, 8),
            "enemy_attack": self.enemy_attack_head(tokens).squeeze(-1).view(-1, 8, 8),
            "friendly_king_pressure": self.friendly_king_pressure_head(tokens).squeeze(-1).view(-1, 8, 8),
            "enemy_king_pressure": self.enemy_king_pressure_head(tokens).squeeze(-1).view(-1, 8, 8),
        }

    def forward(self, x: torch.Tensor, return_aux: bool = False, return_attention: bool = False):
        tokens = self._encode_tokens(x)
        if return_attention:
            tokens, attention = self._run_relation_trunk(tokens, return_attention=True)
        else:
            tokens = self._run_relation_trunk(tokens)
            attention = None
        policy_logits = self._policy_head(tokens)
        value = self._value_head(tokens)

        if not return_aux and not return_attention:
            return policy_logits, value

        outputs = [policy_logits, value]
        if return_aux:
            outputs.append(self._aux_heads(tokens))
        if return_attention:
            outputs.append(attention.mean(dim=1) if attention is not None else None)
        return tuple(outputs)
