# Copyright 2025 AVAR Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
AVAR Visual Attention Reward for RL Stage

This module implements the visual attention reward shaping for the AVAR framework
during reinforcement learning (GRPO/PPO) stage.

Reference: AVAR: Attention-Guided Visual-Augmented Reasoning

Key Components:
1. r_visual: Visual attention reward (ratio of image token attention to system token attention)
2. r_total: Total reward combining accuracy, visual attention, and format rewards
"""

from typing import Optional

import torch


def compute_visual_attention_reward(
    attention_weights: list[torch.Tensor],
    image_token_mask: torch.Tensor,
    system_token_mask: torch.Tensor,
    layer_indices: Optional[list[int]] = None,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    r"""Compute visual attention reward for AVAR.

    Formula from AVAR paper:
        r_visual = 1/|T| · Σ_{t∈T} ( Σ_{l∈L} Σ_{k∈K_img} A^l / Σ_{l∈L} Σ_{k∈K_sys} A^l )

    This computes the ratio of total attention to image tokens vs system tokens
    across all layers and timesteps.

    Args:
        attention_weights: List of attention weight tensors from each layer.
            Each tensor has shape (batch_size, num_heads, seq_len, seq_len).
        image_token_mask: Boolean mask for image tokens, shape (batch_size, seq_len).
        system_token_mask: Boolean mask for system tokens, shape (batch_size, seq_len).
        layer_indices: Optional list of layer indices to use. If None, uses all layers.
        epsilon: Numerical stability constant.

    Returns:
        Visual attention reward tensor of shape (batch_size,).
    """
    if layer_indices is not None:
        attention_weights = [attention_weights[i] for i in layer_indices if i < len(attention_weights)]

    if len(attention_weights) == 0:
        return torch.zeros(image_token_mask.shape[0], device=image_token_mask.device)

    batch_size = image_token_mask.shape[0]

    # Sum attention to image tokens and system tokens across all layers
    total_img_attention = torch.zeros(batch_size, device=image_token_mask.device)
    total_sys_attention = torch.zeros(batch_size, device=image_token_mask.device)

    for attn_weights in attention_weights:
        # attn_weights shape: (batch_size, num_heads, seq_len, seq_len)
        # We want attention FROM response tokens TO image/system tokens

        # Expand masks for broadcasting
        # key_mask: (batch, 1, 1, seq) - which positions are image/system tokens
        k_img_mask = image_token_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq)
        k_sys_mask = system_token_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq)

        # Sum attention weights to image tokens: Σ_{l∈L} Σ_{k∈K_img} A^l
        # Sum over heads and key positions
        img_attn = (attn_weights * k_img_mask.float()).sum(dim=(1, 2, 3))  # (batch,)
        total_img_attention = total_img_attention + img_attn

        # Sum attention weights to system tokens: Σ_{l∈L} Σ_{k∈K_sys} A^l
        sys_attn = (attn_weights * k_sys_mask.float()).sum(dim=(1, 2, 3))  # (batch,)
        total_sys_attention = total_sys_attention + sys_attn

    # Compute ratio: total_img_attention / total_sys_attention
    r_visual = total_img_attention / (total_sys_attention + epsilon)

    return r_visual


def compute_avar_total_reward(
    accuracy_reward: torch.Tensor,
    format_reward: torch.Tensor,
    visual_attention_reward: torch.Tensor,
    lambda_v: float = 0.3,
    lambda_f: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    r"""Compute total AVAR reward.

    Formula:
        r_total = r_accuracy + λ_v · r_visual + λ_f · r_format

    Note: r_visual is only applied when accuracy_reward > 0 (correct answer).

    Args:
        accuracy_reward: Accuracy reward tensor, shape (batch_size,).
        format_reward: Format reward tensor, shape (batch_size,).
        visual_attention_reward: Visual attention reward tensor, shape (batch_size,).
        lambda_v: Visual attention reward weight (default: 0.3).
        lambda_f: Format reward weight (default: 0.1).

    Returns:
        Tuple of (total_reward, reward_dict) where reward_dict contains individual rewards.
    """
    batch_size = accuracy_reward.shape[0]

    # Visual attention reward is only applied for correct answers
    # r_visual = 0 if rollout is incorrect
    r_visual = visual_attention_reward * (accuracy_reward > 0).float()

    # Total reward
    r_total = accuracy_reward + lambda_v * r_visual + lambda_f * format_reward

    reward_dict = {
        "accuracy": accuracy_reward.mean().item(),
        "format": format_reward.mean().item(),
        "visual_attn": r_visual.mean().item(),
        "total": r_total.mean().item(),
    }

    return r_total, reward_dict


def get_image_token_indices(
    input_ids: torch.Tensor,
    image_token_id: int,
) -> list[list[tuple[int, int]]]:
    r"""Get indices of image token spans in input sequences.

    Args:
        input_ids: Input token IDs, shape (batch_size, seq_len).
        image_token_id: Token ID for image tokens.

    Returns:
        List of lists of (start, end) tuples for each sample in batch.
    """
    batch_size, seq_len = input_ids.shape
    all_image_spans = []

    for b in range(batch_size):
        sample_ids = input_ids[b]
        image_spans = []
        in_image = False
        start_idx = 0

        for i in range(seq_len):
            token_id = sample_ids[i].item()

            if token_id == image_token_id:
                if not in_image:
                    in_image = True
                    start_idx = i
            else:
                if in_image:
                    image_spans.append((start_idx, i))
                    in_image = False

        # Handle case where image spans to the end
        if in_image:
            image_spans.append((start_idx, seq_len))

        all_image_spans.append(image_spans)

    return all_image_spans


def create_token_masks(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    image_token_id: Optional[int],
    ignore_index: int = -100,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Create image token mask and system token mask.

    Args:
        input_ids: Input token IDs, shape (batch_size, seq_len).
        labels: Label tensor, shape (batch_size, seq_len). Used to find response start.
        image_token_id: Token ID for image tokens.
        ignore_index: Index to ignore in labels (typically -100).
        device: Device for output tensors.

    Returns:
        Tuple of (image_token_mask, system_token_mask).
        Each mask has shape (batch_size, seq_len).
    """
    if device is None:
        device = input_ids.device

    batch_size, seq_len = input_ids.shape

    # Initialize masks
    image_token_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    system_token_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

    # Find image token spans
    if image_token_id is not None:
        image_spans = get_image_token_indices(input_ids, image_token_id)
        for b, spans in enumerate(image_spans):
            for start, end in spans:
                if start < seq_len and end <= seq_len:
                    image_token_mask[b, start:end] = True

    # System tokens are before the first non-ignored position in labels
    # (i.e., the prompt part before the response)
    for b in range(batch_size):
        sample_labels = labels[b]

        # Find the first non-ignored position in labels
        non_ignored = (sample_labels != ignore_index).nonzero(as_tuple=True)[0]
        if len(non_ignored) > 0:
            response_start = non_ignored[0].item()
        else:
            response_start = seq_len

        # System tokens are typically the first few tokens of the prompt
        # We use a heuristic: first 20% of prompt is system
        # This can be customized based on template
        system_end = max(1, response_start // 5)  # 20% of prompt
        system_token_mask[b, :system_end] = True

    return image_token_mask, system_token_mask


def parse_layer_indices(layers_str: Optional[str]) -> Optional[list[int]]:
    r"""Parse layer indices from comma-separated string.

    Args:
        layers_str: Comma-separated string like "0,1,2,3" or None.

    Returns:
        List of integers or None.
    """
    if layers_str is None or layers_str.strip() == "":
        return None

    try:
        return [int(x.strip()) for x in layers_str.split(",")]
    except ValueError:
        return None
