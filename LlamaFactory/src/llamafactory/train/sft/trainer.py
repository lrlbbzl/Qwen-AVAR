# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from ..avar_attention import compute_avar_total_loss
from ..avar_extractor import get_image_token_indices

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)
        if finetuning_args.use_avar_attn_guide:
            self._configure_avar_attention()
    

    def _configure_avar_attention(self):
        r"""Configure model for AVAR attention guidance.

        This method ensures the model can return attention weights during training.
        For models using Flash Attention, this may require switching to eager attention.
        """
        import warnings

        # Check current attention implementation
        config = self.model.config
        attn_implementation = getattr(config, "_attn_implementation", None)

        if attn_implementation == "flash_attention_2" or attn_implementation == "flash_attention_3":
            warnings.warn(
                "AVAR attention guidance requires eager attention for weight extraction. "
                "Consider setting `flash_attn=disabled` or `flash_attn=sdpa` in model args. "
                "Note: Using eager attention may increase memory usage and slow down training."
            )

        # Store original config for reference
        self._original_attn_implementation = attn_implementation

        # Log AVAR configuration
        logger.info_rank0(
            f"AVAR attention guidance enabled with alpha={self.finetuning_args.avar_attn_alpha}, "
            f"beta={self.finetuning_args.avar_attn_beta}"
        )

        if self.finetuning_args.avar_attn_layer_indices is not None:
            logger.info_rank0(
                f"Applying attention guidance to layers: {self.finetuning_args.avar_attn_layer_indices}"
            )
        else:
            logger.info_rank0("Applying attention guidance to all layers")

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler()

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        # kwargs.update({'return_outputs': True})
        # inputs.update({'output_attentions': True})
        # loss, outputs = super().compute_loss(model, inputs, *args, **kwargs)
        # import torch.distributed as dist
        # import pdb
        # if dist.is_initialized():
        #     if dist.get_rank() == 0:  # 只在主进程启动pdb
        #         pdb.set_trace()
        #     else:
        #         dist.barrier()  # 其他进程在此等待，防止不同步
        # else:
        #     pdb.set_trace()
        # return loss
        if self.finetuning_args.use_avar_attn_guide:
            return self._compute_avar_loss(model, inputs)
        return super().compute_loss(model, inputs, *args, **kwargs)

    def _compute_avar_loss(self, model, inputs):
        r"""Compute loss with AVAR attention guidance.

        L_total = L_LM + α · L_enhance-img + β · L_suppress-sys
        """
        # Forward pass with attention outputs
        # Need to use eager attention or output_attentions=True
        outputs = model(
            **inputs,
            output_attentions=True,
            output_hidden_states=False,
        )

        # Get the language modeling loss
        lm_loss = outputs.loss if outputs.loss is not None else self._compute_lm_loss(outputs, inputs["labels"])

        # Get attention weights
        attentions = outputs.attentions  # Tuple of attention weights per layer
        if attentions is None or len(attentions) == 0:
            # Fall back to just LM loss if attention weights are not available
            # This can happen with Flash Attention
            return lm_loss

        # Convert attention tuple to list
        attention_weights = list(attentions)

        # Create token masks for AVAR loss computation
        image_token_mask, system_token_mask, query_token_mask = self._create_avar_token_masks(
            inputs, attention_weights[0].device if attention_weights else lm_loss.device
        )

        # Check if we have image tokens in the batch
        if not image_token_mask.any():
            # No image tokens, return just LM loss
            return lm_loss

        # Compute AVAR total loss
        total_loss, loss_dict = compute_avar_total_loss(
            lm_loss=lm_loss,
            attention_weights=attention_weights,
            image_token_mask=image_token_mask,
            system_token_mask=system_token_mask,
            query_token_mask=query_token_mask,
            alpha=self.finetuning_args.avar_attn_alpha,
            beta=self.finetuning_args.avar_attn_beta,
            layer_indices=self.finetuning_args.avar_attn_layer_indices,
            epsilon=self.finetuning_args.avar_attn_epsilon,
        )

        # Log individual losses
        if self.state.is_world_process_zero:
            self.log(loss_dict)

        return total_loss

    def _compute_lm_loss(self, outputs, labels):
        r"""Compute language modeling loss from outputs."""
        logits = outputs.get("logits")
        if logits is None:
            return torch.tensor(0.0, device=labels.device)

        logits = logits.float()
        vocab_size = logits.size(-1)

        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)

        # Compute cross entropy loss
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        loss = loss_fct(shift_logits, shift_labels)

        return loss

    def _create_avar_token_masks(self, inputs, device):
        r"""Create token masks for AVAR attention guidance.

        Returns:
            image_token_mask: Boolean mask for image tokens
            system_token_mask: Boolean mask for system tokens
            query_token_mask: Boolean mask for query (user instruction) tokens
        """
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        batch_size, seq_len = input_ids.shape

        # Initialize masks
        image_token_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        system_token_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        query_token_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

        # Get image token ID from tokenizer or processor
        image_token_id = self._get_image_token_id()

        # Get vision BOS/EOS tokens if available
        vision_bos_token_id = getattr(self.processing_class, "vision_bos_token_id", None)
        vision_eos_token_id = getattr(self.processing_class, "vision_eos_token_id", None)

        # Find image token spans
        if image_token_id is not None:
            image_spans = get_image_token_indices(
                input_ids,
                image_token_id,
                vision_bos_token_id,
                vision_eos_token_id,
            )

            for b, spans in enumerate(image_spans):
                for start, end in spans:
                    if start < seq_len and end <= seq_len:
                        image_token_mask[b, start:end] = True

        # Find system token spans
        # System tokens are typically at the beginning of the sequence
        # They are part of the prompt (labels = IGNORE_INDEX)
        for b in range(batch_size):
            sample_labels = labels[b]
            sample_input_ids = input_ids[b]

            # Find the first non-ignored position in labels
            # This is typically where the response starts
            non_ignored = (sample_labels != IGNORE_INDEX).nonzero(as_tuple=True)[0]
            if len(non_ignored) > 0:
                response_start = non_ignored[0].item()
            else:
                response_start = seq_len

            # System tokens are before the first user message
            # We need to identify where the first user message starts
            # For most templates, this is after the system prompt

            # Try to find user role markers
            user_start = self._find_user_message_start(sample_input_ids, response_start)

            if user_start > 0:
                # Tokens from 0 to user_start are system tokens
                # But we need to exclude BOS token if present
                bos_token_id = self.processing_class.bos_token_id
                start_idx = 1 if bos_token_id is not None and input_ids[b, 0].item() == bos_token_id else 0
                system_token_mask[b, start_idx:user_start] = True

            # Query tokens are the user instructions
            # These are the prompt tokens that are not system tokens
            # Query tokens are from user_start to response_start
            if user_start < response_start:
                query_token_mask[b, user_start:response_start] = True

        return image_token_mask, system_token_mask, query_token_mask

    def _get_image_token_id(self) -> Optional[int]:
        r"""Get the image token ID from tokenizer/processor."""
        # Try different ways to get image token ID
        processing_class = self.processing_class

        # Method 1: Check for common image token attributes
        for attr in ["image_token_id", "image_token", "vision_token_id"]:
            if hasattr(processing_class, attr):
                token_id = getattr(processing_class, attr)
                if isinstance(token_id, int):
                    return token_id

        # Method 2: Convert token string to ID
        for token_str in ["<image>", "<|image|>", "<|vision_start|>", "<|IMAGE|>"]:
            try:
                token_id = processing_class.convert_tokens_to_ids(token_str)
                if token_id != processing_class.unk_token_id:
                    return token_id
            except Exception:
                continue

        # Method 3: Check processor
        if hasattr(processing_class, "image_token"):
            image_token = processing_class.image_token
            try:
                return processing_class.convert_tokens_to_ids(image_token)
            except Exception:
                pass

        return None

    def _find_user_message_start(self, input_ids, response_start):
        r"""Find the start of the first user message in the sequence.

        Args:
            input_ids: Token IDs for the sample
            response_start: Index where response starts

        Returns:            Index where user message starts
        """
        # Try to decode and find user marker
        try:
            # Decode first part of sequence
            decoded = self.processing_class.decode(input_ids[:response_start], skip_special_tokens=False)

            # Common user message markers
            user_markers = [
                "<|im_start|>user",
                "<|start_header_id|>user",
                "[INST]",
                "user\n",
                "User:",
                "<|user|>",
            ]

            for marker in user_markers:
                idx = decoded.find(marker)
                if idx != -1:
                    # Estimate token index
                    # This is approximate - for accurate results use template-specific parsing
                    # Estimate: each character is roughly 0.3-0.5 tokens
                    estimated_token_idx = int(idx * 0.4)
                    return min(estimated_token_idx, response_start)

        except Exception:
            pass

        # Default: assume first 10% of prompt is system, rest is user
        return min(max(10, response_start // 10), response_start)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
