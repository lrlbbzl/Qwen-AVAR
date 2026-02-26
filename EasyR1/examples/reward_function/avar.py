# Copyright 2025 Qwen Team, Alibaba Inc.
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
AVAR Reward Function for GRPO/PPO Training

This module implements the reward function for AVAR framework:
r_total = r_accuracy + lambda_v * r_visual + lambda_f * r_format
"""

import re
from typing import Any

from mathruler.grader import grade_answer


# Metadata
REWARD_NAME = "avar"
REWARD_TYPE = "sequential"


def format_reward(response: str) -> float:
    """Check if response follows the expected format."""
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    """Check if the answer is correct."""
    try:
        content_match = re.search(r"<answer>(.*?)</answer>", response)
        given_answer = content_match.group(1).strip() if content_match else response.strip()
        if grade_answer(given_answer, ground_truth.strip()):
            return 1.0
    except Exception:
        pass
    return 0.0


def compute_score(reward_input: dict[str, Any], format_weight: float = 0.1) -> dict[str, float]:
    """Compute reward score for a single sample."""
    format_score = format_reward(reward_input["response"])
    accuracy_score = accuracy_reward(reward_input["response"], reward_input["ground_truth"])
    return {
        "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
        "format": format_score,
        "accuracy": accuracy_score,
    }


def compute_score_with_visual_attention(
    reward_input: dict[str, Any],
    visual_attention_reward: float,
    lambda_v: float = 0.3,
    lambda_f: float = 0.1,
) -> dict[str, float]:
    """Compute reward score with visual attention for AVAR.

    This implements the AVAR total reward:
    r_total = r_accuracy + lambda_v * r_visual + lambda_f * r_format

    Note: r_visual is only applied when accuracy > 0 (correct answer).
    """
    format_score = format_reward(reward_input["response"])
    accuracy_score = accuracy_reward(reward_input["response"], reward_input["ground_truth"])

    # Visual attention reward is only applied for correct answers
    visual_score = visual_attention_reward if accuracy_score > 0 else 0.0

    # Total reward
    total_score = accuracy_score + lambda_v * visual_score + lambda_f * format_score

    return {
        "overall": total_score,
        "format": format_score,
        "accuracy": accuracy_score,
        "visual": visual_score,
    }
