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
Attention Extractor for AVAR Framework

This module provides utilities to extract attention weights from transformer models.
It handles the conversion between Flash Attention and Eager attention modes.
"""

from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn


class AttentionWeightExtractor:
    r"""Extract attention weights from transformer models.

    This class provides functionality to:
    1. Temporarily switch from Flash Attention to Eager attention for weight extraction
    2. Register forward hooks to capture attention weights
    3. Store and return attention weights for computing AVAR losses
    """

    def __init__(self, model: nn.Module, layer_indices: Optional[list[int]] = None):
        r"""Initialize the attention weight extractor.

        Args:
            model: The transformer model to extract attention weights from.
            layer_indices: Specific layer indices to extract attention from.
                          If None, extracts from all layers.
        """
        self.model = model
        self.layer_indices = layer_indices
        self.attention_weights = []
        self.hooks = []
        self._original_attn_impl = None

    def _find_attention_modules(self) -> list[tuple[str, nn.Module]]:
        r"""Find all attention modules in the model.

        Returns:
            List of (name, module) tuples for attention modules.
        """
        attention_modules = []
        for name, module in self.model.named_modules():
            # Common attention module names
            if any(attn_name in name.lower() for attn_name in ["attention", "attn", "self_attn"]):
                if hasattr(module, "forward") and "layer" in name:
                    attention_modules.append((name, module))
        return attention_modules

    def _get_layer_index(self, name: str) -> Optional[int]:
        r"""Extract layer index from module name.

        Args:
            name: Module name.

        Returns:
            Layer index if found, else None.
        """
        import re

        # Match patterns like "layers.0", "layer.1", etc.
        match = re.search(r"layers?\.(\d+)", name)
        if match:
            return int(match.group(1))
        return None

    def _create_hook(self, layer_idx: int):
        r"""Create a forward hook to capture attention weights.

        Args:
            layer_idx: Index of the layer.

        Returns:
            Hook function.
        """

        def hook(module, input, output):
            # Check if output contains attention weights
            # Different models return different outputs
            if isinstance(output, tuple):
                if len(output) >= 2:
                    # Common case: (hidden_states, attention_weights, ...)
                    # But often attention_weights is None when using Flash Attention
                    attn_weights = output[1] if len(output) > 1 else None
                else:
                    attn_weights = None
            elif isinstance(output, dict):
                attn_weights = output.get("attentions", None)
            else:
                attn_weights = None

            if attn_weights is not None:
                self.attention_weights.append((layer_idx, attn_weights))

        return hook

    def _eager_attention_forward(
        self,
        module,
        query,
        key,
        value,
        attention_mask=None,
        head_mask=None,
        **kwargs,
    ):
        r"""Implement eager attention to get attention weights.

        This is a fallback for models using Flash Attention,
        which doesn't return attention weights.
        """
        # Get dimensions
        batch_size, num_heads, seq_len, head_dim = query.shape

        # Compute attention scores: Q @ K^T / sqrt(d)
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores / (head_dim**0.5)

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Apply head mask if provided
        if head_mask is not None:
            attention_weights = attention_weights * head_mask

        # Compute output: attention_weights @ V
        context = torch.matmul(attention_weights, value)

        return context, attention_weights

    def register_hooks(self):
        r"""Register forward hooks on attention modules."""
        attention_modules = self._find_attention_modules()
        self.hooks = []

        for name, module in attention_modules:
            layer_idx = self._get_layer_index(name)
            if layer_idx is not None:
                if self.layer_indices is None or layer_idx in self.layer_indices:
                    hook = module.register_forward_hook(self._create_hook(layer_idx))
                    self.hooks.append(hook)

    def remove_hooks(self):
        r"""Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def clear_attention_weights(self):
        r"""Clear stored attention weights."""
        self.attention_weights = []

    def get_attention_weights(self) -> list[torch.Tensor]:
        r"""Get extracted attention weights sorted by layer index.

        Returns:
            List of attention weight tensors, one per layer.
        """
        if not self.attention_weights:
            return []

        # Sort by layer index and extract weights
        sorted_weights = sorted(self.attention_weights, key=lambda x: x[0])
        return [w for _, w in sorted_weights]

    @contextmanager
    def extract_attention(self):
        r"""Context manager to extract attention weights during forward pass.

        Usage:
            with extractor.extract_attention():
                outputs = model(**inputs)
            attention_weights = extractor.get_attention_weights()
        """
        self.clear_attention_weights()
        self.register_hooks()
        try:
            yield self
        finally:
            self.remove_hooks()


def get_image_token_indices(
    input_ids: torch.Tensor,
    image_token_id: int,
    vision_bos_token_id: Optional[int] = None,
    vision_eos_token_id: Optional[int] = None,
) -> list[list[tuple[int, int]]]:
    r"""Get indices of image tokens in input sequences.

    Args:
        input_ids: Input token IDs, shape (batch_size, seq_len).
        image_token_id: Token ID for image tokens.
        vision_bos_token_id: Optional beginning-of-sequence token for vision.
        vision_eos_token_id: Optional end-of-sequence token for vision.

    Returns:
        List of lists of (start, end) tuples for each sample in batch.
        Each tuple represents a span of image tokens.
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

            # Check for vision BOS token
            if vision_bos_token_id is not None and token_id == vision_bos_token_id:
                in_image = True
                start_idx = i
            # Check for vision EOS token
            elif vision_eos_token_id is not None and token_id == vision_eos_token_id:
                if in_image:
                    image_spans.append((start_idx, i + 1))
                    in_image = False
            # Check for direct image token
            elif token_id == image_token_id:
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


def get_system_token_indices(
    input_ids: torch.Tensor,
    tokenizer,
    template_format: Optional[str] = None,
) -> list[tuple[int, int]]:
    r"""Get indices of system prompt tokens in input sequences.

    This function identifies system prompt tokens based on the template format.
    System prompts typically appear at the beginning of the sequence.

    Args:
        input_ids: Input token IDs, shape (batch_size, seq_len).
        tokenizer: Tokenizer to decode tokens.
        template_format: Template format name (e.g., "qwen", "llama3").

    Returns:
        List of (start, end) tuples for each sample in batch.
    """
    batch_size, seq_len = input_ids.shape
    system_spans = []

    # Common patterns for system prompts
    # These are typically at the start of the sequence, before the first user message

    for b in range(batch_size):
        sample_ids = input_ids[b]

        # Try to find system prompt boundaries
        # Most templates have system prompts at the start (after BOS if present)
        # System prompt typically ends when user message starts

        # Default: assume first few tokens are system prompt
        # This is a simplified approach; actual implementation may need template-specific logic

        # Look for common user message start patterns
        # E.g., "<|im_start|>user" for Qwen, "<|start_header_id|>user" for Llama3

        # For now, use a simple heuristic:
        # Find the first occurrence of user role marker

        # This is template-specific and may need customization
        system_end = 0  # Default: no system prompt

        # Try to decode and find user marker
        try:
            decoded = tokenizer.decode(sample_ids[:100], skip_special_tokens=False)

            # Common patterns
            user_markers = [
                "<|im_start|>user",
                "<|start_header_id|>user",
                "[INST]",
                "user\n",
                "User:",
            ]

            for marker in user_markers:
                idx = decoded.find(marker)
                if idx != -1:
                    # Find token index corresponding to this position
                    # Approximate by finding the position in original text
                    system_end = idx
                    break

        except Exception:
            pass

        if system_end > 0:
            # Estimate token index (approximate)
            # This is a rough estimate; for accurate results, use template-specific parsing
            system_spans.append((0, system_end))
        else:
            # No system prompt found
            system_spans.append((0, 0))

    return system_spans


def create_attention_mask_from_spans(
    seq_len: int,
    spans: list[tuple[int, int]],
    batch_size: int = 1,
    device: torch.device = None,
) -> torch.Tensor:
    r"""Create attention mask from token spans.

    Args:
        seq_len: Sequence length.
        spans: List of (start, end) tuples for token spans.
        batch_size: Batch size.
        device: Device to create tensor on.

    Returns:
        Boolean mask tensor of shape (batch_size, seq_len).
    """
    if device is None:
        device = torch.device("cpu")

    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

    for start, end in spans:
        if start < seq_len and end <= seq_len and start < end:
            mask[:, start:end] = True

    return mask
