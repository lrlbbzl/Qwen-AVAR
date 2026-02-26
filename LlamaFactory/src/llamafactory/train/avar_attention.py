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
AVAR Attention Guidance Loss Implementation

This module implements the attention guidance losses for the AVAR framework:
- L_enhance-img: Image attention enhancement loss
- L_suppress-sys: System token attention suppression loss

Reference: AVAR: Attention-Guided Visual-Augmented Reasoning
"""

from typing import Optional

import torch
import torch.nn.functional as F


def compute_attention_guidance_loss(
    attention_weights: list[torch.Tensor],
    image_token_mask: torch.Tensor,
    system_token_mask: torch.Tensor,
    query_token_mask: torch.Tensor,
    layer_indices: Optional[list[int]] = None,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute AVAR attention guidance losses.

    Args:
        attention_weights: List of attention weight tensors from each layer.
            Each tensor has shape (batch_size, num_heads, seq_len, seq_len).
        image_token_mask: Boolean mask for image tokens, shape (batch_size, seq_len).
            True for image tokens, False otherwise.
        system_token_mask: Boolean mask for system tokens, shape (batch_size, seq_len).
            True for system prompt tokens, False otherwise.
        query_token_mask: Boolean mask for query tokens, shape (batch_size, seq_len).
            True for query tokens (user instruction tokens), False otherwise.
        layer_indices: Optional list of layer indices to apply attention guidance.
            If None, applies to all layers.
        epsilon: Numerical stability constant.

    Returns:
        Tuple of (L_enhance_img, L_suppress_sys) losses.
    """
    if layer_indices is not None:
        # Filter attention weights to only specified layers
        attention_weights = [attention_weights[i] for i in layer_indices if i < len(attention_weights)]
    
    if len(attention_weights) == 0:
        return torch.tensor(0.0, device=image_token_mask.device), torch.tensor(0.0, device=image_token_mask.device)
    
    num_layers = len(attention_weights)
    L_enhance_img = torch.tensor(0.0, device=image_token_mask.device)
    L_suppress_sys = torch.tensor(0.0, device=image_token_mask.device)
    
    for attn_weights in attention_weights:
        # attn_weights shape: (batch_size, num_heads, seq_len, seq_len)
        # A^{l,h}_{q,k}: attention weight from query position q to key position k
        
        batch_size, num_heads, seq_len, _ = attn_weights.shape
        
        # Expand masks for broadcasting
        # query_mask: (batch_size, 1, seq_len, 1) - which positions are query tokens
        # key_mask: (batch_size, 1, 1, seq_len) - which positions are image/system tokens
        q_mask = query_token_mask.unsqueeze(1).unsqueeze(-1)  # (batch, 1, seq, 1)
        k_img_mask = image_token_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq)
        k_sys_mask = system_token_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq)
        
        # ========== L_enhance_img ==========
        # L_enhance-img = - (1/|L|) · Σ_{l∈L} (1/H) · Σ_{h=1}^{H} 
        #                 log( (1/(|Q|·|K_img|)) · Σ_{q∈Q} Σ_{k∈K_img} A^{l,h}_{q,k} )
        
        # Get attention weights from query tokens to image tokens
        # Shape: (batch, num_heads, seq, seq)
        attn_q_to_img = attn_weights * q_mask.float() * k_img_mask.float()
        
        # Count query tokens and image tokens per batch
        num_query = query_token_mask.sum(dim=-1, keepdim=True).float()  # (batch, 1)
        num_img = image_token_mask.sum(dim=-1, keepdim=True).float()  # (batch, 1)
        
        # Sum attention weights: Σ_{q∈Q} Σ_{k∈K_img} A^{l,h}_{q,k}
        # Shape: (batch, num_heads)
        attn_sum_img = attn_q_to_img.sum(dim=(2, 3))
        
        # Compute mean: (1/(|Q|·|K_img|)) · Σ_{q∈Q} Σ_{k∈K_img} A^{l,h}_{q,k}
        # Avoid division by zero
        mean_attn_img = attn_sum_img / (num_query * num_img + epsilon)
        
        # Compute log and average over heads
        # Shape: (batch,)
        layer_enhance_loss = -torch.log(mean_attn_img + epsilon).mean(dim=1)
        
        # Add to total loss
        L_enhance_img = L_enhance_img + layer_enhance_loss.mean()
        
        # ========== L_suppress_sys ==========
        # L_suppress-sys = (1/|L|) · Σ_{l∈L} (1/H) · Σ_{h=1}^{H} 
        #                 log( (1/(|Q|·|K_sys|)) · Σ_{q∈Q} Σ_{k∈K_sys} A^{l,h}_{q,k} + ε )
        
        # Get attention weights from query tokens to system tokens
        attn_q_to_sys = attn_weights * q_mask.float() * k_sys_mask.float()
        
        # Count system tokens per batch
        num_sys = system_token_mask.sum(dim=-1, keepdim=True).float()  # (batch, 1)
        
        # Sum attention weights: Σ_{q∈Q} Σ_{k∈K_sys} A^{l,h}_{q,k}
        attn_sum_sys = attn_q_to_sys.sum(dim=(2, 3))
        
        # Compute mean: (1/(|Q|·|K_sys|)) · Σ_{q∈Q} Σ_{k∈K_sys} A^{l,h}_{q,k}
        mean_attn_sys = attn_sum_sys / (num_query * num_sys + epsilon)
        
        # Compute log and average over heads
        layer_suppress_loss = torch.log(mean_attn_sys + epsilon).mean(dim=1)
        
        # Add to total loss
        L_suppress_sys = L_suppress_sys + layer_suppress_loss.mean()
    
    # Average over layers
    L_enhance_img = L_enhance_img / num_layers
    L_suppress_sys = L_suppress_sys / num_layers
    
    return L_enhance_img, L_suppress_sys


def compute_avar_total_loss(
    lm_loss: torch.Tensor,
    attention_weights: list[torch.Tensor],
    image_token_mask: torch.Tensor,
    system_token_mask: torch.Tensor,
    query_token_mask: torch.Tensor,
    alpha: float = 0.15,
    beta: float = 0.15,
    layer_indices: Optional[list[int]] = None,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    r"""Compute total AVAR loss.

    L_total = L_LM + α · L_enhance-img + β · L_suppress-sys

    Args:
        lm_loss: Standard language modeling loss.
        attention_weights: List of attention weight tensors from each layer.
        image_token_mask: Boolean mask for image tokens.
        system_token_mask: Boolean mask for system tokens.
        query_token_mask: Boolean mask for query tokens.
        alpha: Weight for image attention enhancement loss.
        beta: Weight for system token attention suppression loss.
        layer_indices: Layer indices to apply attention guidance.
        epsilon: Numerical stability constant.

    Returns:
        Tuple of (total_loss, loss_dict) where loss_dict contains individual loss values.
    """
    L_enhance_img, L_suppress_sys = compute_attention_guidance_loss(
        attention_weights=attention_weights,
        image_token_mask=image_token_mask,
        system_token_mask=system_token_mask,
        query_token_mask=query_token_mask,
        layer_indices=layer_indices,
        epsilon=epsilon,
    )
    
    total_loss = lm_loss + alpha * L_enhance_img + beta * L_suppress_sys
    
    loss_dict = {
        "lm_loss": lm_loss.item(),
        "enhance_img_loss": L_enhance_img.item(),
        "suppress_sys_loss": L_suppress_sys.item(),
        "total_loss": total_loss.item(),
    }
    
    return total_loss, loss_dict


def create_token_masks_from_indices(
    seq_len: int,
    image_token_indices: list[tuple[int, int]],  # List of (start, end) tuples
    system_token_indices: tuple[int, int],  # (start, end)
    query_token_indices: list[tuple[int, int]],  # List of (start, end) tuples
    batch_size: int = 1,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Create token masks from token index ranges.

    Args:
        seq_len: Sequence length.
        image_token_indices: List of (start, end) tuples for image token spans.
        system_token_indices: (start, end) tuple for system token span.
        query_token_indices: List of (start, end) tuples for query token spans.
        batch_size: Batch size.
        device: Device to create tensors on.

    Returns:
        Tuple of (image_token_mask, system_token_mask, query_token_mask).
        Each mask has shape (batch_size, seq_len).
    """
    if device is None:
        device = torch.device("cpu")
    
    # Create masks
    image_token_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    system_token_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    query_token_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    
    # Set image tokens
    for start, end in image_token_indices:
        if start < seq_len and end <= seq_len:
            image_token_mask[:, start:end] = True
    
    # Set system tokens
    sys_start, sys_end = system_token_indices
    if sys_start < seq_len and sys_end <= seq_len:
        system_token_mask[:, sys_start:sys_end] = True
    
    # Set query tokens
    for start, end in query_token_indices:
        if start < seq_len and end <= seq_len:
            query_token_mask[:, start:end] = True
    
    return image_token_mask, system_token_mask, query_token_mask
