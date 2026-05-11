# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
OPSD Trainer: Hint-Guided On-Policy Self-Distillation with Self-Evolving Navigator.

Extends RayPPOTrainer with:
  1. Multi-round generation: Student → Navigator → Teacher
  2. Navigator GRPO advantage from Teacher trajectory correctness
  3. OPSD forward batch assembly for joint training
  4. Curriculum learning for ground-truth dropping
"""

import json
import time
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..utils import torch_functional as VF
from ..utils.py_functional import convert_dict_to_str, timer, unflatten_dict
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from .config import PPOConfig
from .metrics import (
    compute_data_metrics,
    compute_length_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from .core_algos import AdvantageEstimator, compute_advantage_return
from .opsd_algos import (
    compute_curriculum_p_drop,
    compute_navigator_advantage,
    construct_navigator_prompts,
    construct_sdpo_teacher_prompts,
    construct_teacher_prompts,
    decode_navigator_hints,
    truncate_text_by_tokens,
)
from .opsd_config import OPSDConfig
from .ray_trainer import RayPPOTrainer, ResourcePoolManager, Role, apply_kl_penalty, compute_advantage


def _concat_prompt_and_response(
    prompt_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    prompt_position_ids: torch.Tensor,
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenate prompt and response following the rollout engine's position-id logic."""
    input_ids = torch.cat([prompt_ids, response_ids], dim=1)
    attention_mask = torch.cat([prompt_attention_mask, response_mask], dim=1)

    response_length = response_ids.size(1)
    batch_size = prompt_ids.size(0)
    delta_position_id = torch.arange(1, response_length + 1, device=prompt_position_ids.device)
    delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
    if prompt_position_ids.ndim == 3:
        delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(
            batch_size, prompt_position_ids.size(1), -1
        )
    response_position_ids = prompt_position_ids[..., -1:] + delta_position_id
    position_ids = torch.cat([prompt_position_ids, response_position_ids], dim=-1)
    return input_ids, attention_mask, position_ids


def _mask_prefix_response_tokens(response_mask: torch.Tensor, prefix_tokens: int) -> torch.Tensor:
    """Zero out the first N valid response tokens in each sequence."""
    if prefix_tokens <= 0:
        return response_mask
    masked = response_mask.clone()
    batch_size = masked.shape[0]
    for idx in range(batch_size):
        valid_positions = torch.nonzero(masked[idx] > 0, as_tuple=False).squeeze(-1)
        if valid_positions.numel() == 0:
            continue
        cutoff = min(prefix_tokens, int(valid_positions.numel()))
        masked[idx, valid_positions[:cutoff]] = 0
    return masked


class RayOPSDTrainer(RayPPOTrainer):
    """OPSD Trainer with multi-round generation and joint Navigator + Reasoner update."""

    def __init__(self, config, *args, **kwargs):
        # OPSD does not rely on PPO/GRPO's rollout.n validation in parent trainer.
        # Temporarily set n=2 to bypass the parent check, then restore it.
        original_n = config.worker.rollout.n
        if original_n == 1:
            config.worker.rollout.n = 2  # bypass parent check
        super().__init__(config, *args, **kwargs)
        config.worker.rollout.n = original_n  # restore
        self.opsd_config: OPSDConfig = self.config.opsd

        # Set opsd_enabled flag on actor config so workers instantiate the right actor class
        self.config.worker.actor.opsd_enabled = True
        self.config.worker.actor.freeze_teacher_model = self.opsd_config.freeze_teacher_model

        # Load navigator and teacher templates
        self._nav_template_path = self.opsd_config.navigator_prompt_template
        self._teacher_template_path = self.opsd_config.teacher_hint_template
        self._sdpo_teacher_template_path = self.opsd_config.sdpo_teacher_template

        if self.opsd_config.enable_sdpo and self.opsd_config.use_gt_as_hint:
            raise ValueError("opsd.enable_sdpo and opsd.use_gt_as_hint cannot both be enabled.")

        if self.opsd_config.enable_sdpo:
            self.opsd_config.num_hints = 1
            self.opsd_config.lambda_nav = 0.0
            self.opsd_config.alpha = 1.0
            print("[OPSD] Mode: SDPO self-distillation")
            print("[OPSD]   - Navigator rollout: SKIPPED")
            print(f"[OPSD]   - candidate samples per prompt: {self.opsd_config.sdpo_num_candidates}")
            print("[OPSD]   - num_hints forced to 1, lambda_nav forced to 0.0, alpha forced to 1.0")
        elif self.opsd_config.use_gt_as_hint:
            # Actually force the config values, not just print about it
            self.opsd_config.num_hints = 1
            self.opsd_config.lambda_nav = 0.0
            self.opsd_config.alpha = 1.0
            print("[OPSD] Mode: Self-Distilled Reasoner (use_gt_as_hint=True)")
            print("[OPSD]   - Navigator rollout: SKIPPED (using GT answers as hints)")
            print("[OPSD]   - num_hints forced to 1, lambda_nav forced to 0.0, alpha forced to 1.0")
            print("[OPSD]   - p_drop forced to 0 (Teacher always sees GT answer)")
            if self.opsd_config.self_distill_negative_off_policy:
                print(
                    "[OPSD]   - negative off-policy enabled: "
                    f"coef={self.opsd_config.self_distill_negative_off_policy_coef}"
                )
        else:
            print(f"[OPSD] Navigator hints per question (K): {self.opsd_config.num_hints}")
            print(f"[OPSD] Lambda_nav: {self.opsd_config.lambda_nav}")
        print(f"[OPSD] Alpha (on/off-policy weight): {self.opsd_config.alpha}")
        print(f"[OPSD] Distillation loss type: {self.opsd_config.distillation_loss_type}")
        print(f"[OPSD] Student rollout n: {self.opsd_config.student_rollout_n}")
        print(
            f"[OPSD] Navigator include student trajectory: "
            f"{self.opsd_config.navigator_include_student_trajectory}"
        )
        print(
            f"[OPSD] Navigator max trajectory length: "
            f"{self.opsd_config.navigator_max_trajectory_length}"
        )
        print(
            f"[OPSD] Teacher max trajectory length: "
            f"{self.opsd_config.teacher_max_trajectory_length}"
        )
        print(f"[OPSD] Teacher context mode: {self.opsd_config.teacher_context_mode}")
        print(
            f"[OPSD] Teacher reference context length: "
            f"{self.opsd_config.teacher_reference_context_length}"
        )
        print(
            f"[OPSD] Teacher student context length: "
            f"{self.opsd_config.teacher_student_context_length}"
        )
        print(f"[OPSD] Dynamic sample hint: {self.opsd_config.dynamic_sample_hint}")
        print(f"[OPSD] Teacher rollout n per hint: {self.opsd_config.teacher_rollout_n}")
        print(f"[OPSD] Distill token selection: {self.opsd_config.distill_token_selection}")
        print(f"[OPSD] Distill token keep ratio: {self.opsd_config.distill_token_keep_ratio}")
        print(f"[OPSD] Curriculum warmup steps: {self.opsd_config.curriculum_warmup_steps}")
        print(f"[OPSD] Off-policy include GT answer: {self.opsd_config.off_policy_include_gt_answer}")
        print(
            "[OPSD] Self-distill negative off-policy: "
            f"{self.opsd_config.self_distill_negative_off_policy}"
        )
        print(
            "[OPSD] Self-distill masked prefix tokens: "
            f"{self.opsd_config.self_distill_mask_prefix_tokens}"
        )
        print(f"[OPSD] KL penalty type: {self.opsd_config.opsd_kl_penalty}")
        if self.opsd_config.distillation_topk is not None:
            print(f"[OPSD] Top-k KL approximation: k={self.opsd_config.distillation_topk}")
            print(f"[OPSD] Top-k source: {self.opsd_config.distillation_topk_source}")

        self._nav_hint_log_samples = 4
        self._nav_hint_log_history_limit = 20
        self._nav_hint_log_history: list[dict[str, Any]] = []
        self._nav_hint_log_path = Path(self.config.trainer.save_checkpoint_path) / "navigator_hint_samples.json"
        self._student_generation_log_samples = 4
        self._student_generation_log_dir = Path(self.config.trainer.save_checkpoint_path) / "student_generation_samples"

    def _get_teacher_questions(self, batch: DataProto) -> np.ndarray:
        questions = batch.non_tensor_batch.get("teacher_question")
        if questions is None:
            questions = batch.non_tensor_batch["prompt_text"]
        return questions

    def _log_navigator_hints(self, batch: DataProto, hints: list[list[str]]) -> None:
        """Persist a small rolling window of navigator hint samples for inspection."""
        if not hints:
            return

        prompt_text = batch.non_tensor_batch.get("prompt_text")
        ground_truth = batch.non_tensor_batch.get("ground_truth")
        uids = batch.non_tensor_batch.get("uid")

        sample_count = min(self._nav_hint_log_samples, len(hints))
        samples = []
        for idx in range(sample_count):
            question = "" if prompt_text is None else str(prompt_text[idx])
            samples.append(
                {
                    "uid": None if uids is None else str(uids[idx]),
                    "question": question[:500],
                    "ground_truth": None if ground_truth is None else str(ground_truth[idx]),
                    "hints": [str(h) for h in hints[idx]],
                }
            )

        self._nav_hint_log_history.append(
            {
                "step": self.global_step,
                "num_questions": len(hints),
                "num_hints_per_question": len(hints[0]) if hints else 0,
                "samples": samples,
            }
        )
        self._nav_hint_log_history = self._nav_hint_log_history[-self._nav_hint_log_history_limit :]

        self._nav_hint_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._nav_hint_log_path.write_text(
            json.dumps(self._nav_hint_log_history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _log_student_generations(self, batch: DataProto, student_output: DataProto) -> None:
        """Persist a few sampled student generations for the current step."""
        if len(student_output) == 0:
            return

        prompt_text = batch.non_tensor_batch.get("prompt_text")
        ground_truth = batch.non_tensor_batch.get("ground_truth")
        uids = batch.non_tensor_batch.get("uid")
        problem_type = batch.non_tensor_batch.get("problem_type")

        sample_count = min(self._student_generation_log_samples, len(student_output))
        sampled_indices = np.random.choice(len(student_output), size=sample_count, replace=False)

        samples = []
        response_ids = student_output.batch["responses"]
        response_mask = student_output.batch["response_mask"]
        for idx in sampled_indices.tolist():
            cur_response_length = int(response_mask[idx].sum().item())
            valid_response_ids = response_ids[idx][:cur_response_length]
            response_text = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)
            question = "" if prompt_text is None else str(prompt_text[idx])
            samples.append(
                {
                    "uid": None if uids is None else str(uids[idx]),
                    "problem_type": None if problem_type is None else str(problem_type[idx]),
                    "ground_truth": None if ground_truth is None else str(ground_truth[idx]),
                    "response_length": cur_response_length,
                    "question": question,
                    "response": response_text,
                }
            )

        payload = {
            "step": self.global_step,
            "num_samples_in_batch": len(student_output),
            "num_logged_samples": sample_count,
            "samples": samples,
        }

        self._student_generation_log_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._student_generation_log_dir / f"step_{self.global_step:06d}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _decode_student_trajectories(
        self,
        student_output: DataProto,
        max_tokens: Optional[int] = None,
    ) -> list[str]:
        """Decode student rollout responses for optional Navigator conditioning."""
        trajectories: list[str] = []
        response_ids = student_output.batch["responses"]
        response_mask = student_output.batch["response_mask"]

        for idx in range(len(student_output)):
            cur_response_length = int(response_mask[idx].sum().item())
            valid_response_ids = response_ids[idx][:cur_response_length]
            response_text = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)
            trajectories.append(truncate_text_by_tokens(response_text, self.tokenizer, max_tokens))

        return trajectories

    def _extract_teacher_context_snippet(
        self,
        text: Any,
        max_tokens: Optional[int],
        *,
        core_only: bool,
    ) -> str:
        if text is None:
            return ""
        text_str = str(text)
        if not text_str:
            return ""
        if not max_tokens or max_tokens <= 0:
            return ""
        if not core_only:
            return truncate_text_by_tokens(text_str, self.tokenizer, max_tokens)

        token_ids = self.tokenizer.encode(text_str, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            return text_str
        if max_tokens <= 32:
            return self.tokenizer.decode(token_ids[-max_tokens:], skip_special_tokens=False)

        head_len = max(1, max_tokens // 2)
        tail_len = max(1, max_tokens - head_len)
        head_text = self.tokenizer.decode(token_ids[:head_len], skip_special_tokens=False).strip()
        tail_text = self.tokenizer.decode(token_ids[-tail_len:], skip_special_tokens=False).strip()
        return (
            "[Earlier Key Step]\n"
            f"{head_text}\n\n"
            "[Conclusion / Final Segment]\n"
            f"{tail_text}"
        )

    def _build_teacher_context_trajectories(
        self,
        batch: DataProto,
        student_output: Optional[DataProto] = None,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        mode = self.opsd_config.teacher_context_mode
        if mode == "hint":
            return None, None

        core_only = mode == "hint_plus_core_trajectories"
        ref_max_tokens = self.opsd_config.teacher_reference_context_length
        student_max_tokens = self.opsd_config.teacher_student_context_length

        reference_trajectories: list[str] = []
        raw_references = batch.non_tensor_batch.get("trajectory")
        for idx in range(len(batch)):
            raw_reference = "" if raw_references is None else raw_references[idx]
            reference_trajectories.append(
                self._extract_teacher_context_snippet(
                    raw_reference,
                    ref_max_tokens,
                    core_only=core_only,
                )
            )

        student_trajectories = None
        if student_output is not None:
            decoded_student_trajectories = self._decode_student_trajectories(student_output)
            student_trajectories = [
                self._extract_teacher_context_snippet(
                    trajectory,
                    student_max_tokens,
                    core_only=core_only,
                )
                for trajectory in decoded_student_trajectories
            ]

        return (
            np.array(reference_trajectories, dtype=object),
            None if student_trajectories is None else np.array(student_trajectories, dtype=object),
        )

    def _get_self_distill_hints(self, batch: DataProto) -> list[list[str]]:
        """Use trajectory as teacher context when available, otherwise fall back to GT."""
        trajectories = batch.non_tensor_batch.get("trajectory")
        hints = []
        for idx, gt in enumerate(batch.non_tensor_batch["ground_truth"]):
            trajectory = ""
            if trajectories is not None:
                trajectory = truncate_text_by_tokens(
                    trajectories[idx],
                    self.tokenizer,
                    self.opsd_config.teacher_max_trajectory_length,
                )
            hints.append([trajectory] if trajectory else [str(gt)])
        return hints

    def _construct_teacher_fallback_prompts(
        self,
        batch: DataProto,
        repeat_k: int,
        student_output: Optional[DataProto] = None,
    ) -> tuple[DataProto, int]:
        """Build fallback teacher prompts using trajectory first and GT only as a final fallback."""
        trajectories = batch.non_tensor_batch.get("trajectory")
        fallback_hints: list[list[str]] = []
        fallback_gt_answers: list[str] = []
        trajectory_missing = 0

        for idx, gt in enumerate(batch.non_tensor_batch["ground_truth"]):
            trajectory = ""
            if trajectories is not None:
                trajectory = truncate_text_by_tokens(
                    trajectories[idx],
                    self.tokenizer,
                    self.opsd_config.teacher_max_trajectory_length,
                )

            if trajectory:
                fallback_hints.append([trajectory] * repeat_k)
                fallback_gt_answers.append("")
            else:
                fallback_hints.append([""] * repeat_k)
                fallback_gt_answers.append(str(gt))
                trajectory_missing += 1

        reference_trajectories, student_trajectories = self._build_teacher_context_trajectories(
            batch,
            student_output=student_output,
        )
        fallback_teacher_gen = construct_teacher_prompts(
            questions=self._get_teacher_questions(batch),
            hints=fallback_hints,
            gt_answers=np.array(fallback_gt_answers, dtype=object),
            p_drop=0.0,
            tokenizer=self.tokenizer,
            processor=self.processor,
            reference_trajectories=reference_trajectories,
            student_trajectories=student_trajectories,
            template_path=self._teacher_template_path,
            max_prompt_length=self.config.data.max_prompt_length,
            multi_modal_data=batch.non_tensor_batch.get("multi_modal_data"),
            image_min_pixels=self.config.data.image_min_pixels,
            image_max_pixels=self.config.data.image_max_pixels,
            video_min_pixels=self.config.data.video_min_pixels,
            video_max_pixels=self.config.data.video_max_pixels,
            video_fps=self.config.data.video_fps,
            video_max_frames=self.config.data.video_max_frames,
            apply_chat_template_kwargs=self.config.data.apply_chat_template_kwargs,
        )
        return fallback_teacher_gen, trajectory_missing

    def _construct_on_policy_teacher_prompts(
        self,
        batch: DataProto,
        hints: list[list[str]],
        student_output: Optional[DataProto] = None,
    ) -> DataProto:
        """Build teacher prompts for on-policy distillation with GT always retained."""
        reference_trajectories, student_trajectories = self._build_teacher_context_trajectories(
            batch,
            student_output=student_output,
        )
        return construct_teacher_prompts(
            questions=self._get_teacher_questions(batch),
            hints=hints,
            gt_answers=batch.non_tensor_batch["ground_truth"],
            p_drop=0.0,
            tokenizer=self.tokenizer,
            processor=self.processor,
            reference_trajectories=reference_trajectories,
            student_trajectories=student_trajectories,
            template_path=self._teacher_template_path,
            max_prompt_length=self.config.data.max_prompt_length,
            multi_modal_data=batch.non_tensor_batch.get("multi_modal_data"),
            image_min_pixels=self.config.data.image_min_pixels,
            image_max_pixels=self.config.data.image_max_pixels,
            video_min_pixels=self.config.data.video_min_pixels,
            video_max_pixels=self.config.data.video_max_pixels,
            video_fps=self.config.data.video_fps,
            video_max_frames=self.config.data.video_max_frames,
            apply_chat_template_kwargs=self.config.data.apply_chat_template_kwargs,
        )

    def _select_dynamic_hint_targets(
        self,
        rollout_correctness: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select the best hint group and one correct rollout from it for each question."""
        batch_size, _, _ = rollout_correctness.shape
        device = rollout_correctness.device
        hint_group_scores = rollout_correctness.float().mean(dim=-1)
        selected_hint_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
        selected_rollout_indices = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        selected_has_correct = torch.zeros(batch_size, dtype=torch.bool, device=device)
        selected_group_scores = torch.zeros(batch_size, dtype=hint_group_scores.dtype, device=device)

        for i in range(batch_size):
            hint_scores = hint_group_scores[i]
            max_score = hint_scores.max()
            best_hint_candidates = torch.nonzero(torch.isclose(hint_scores, max_score), as_tuple=True)[0]
            chosen_hint = best_hint_candidates[np.random.randint(best_hint_candidates.numel())]
            selected_hint_indices[i] = chosen_hint
            selected_group_scores[i] = hint_scores[chosen_hint]

            correct_rollout_candidates = torch.nonzero(rollout_correctness[i, chosen_hint] > 0, as_tuple=True)[0]
            if correct_rollout_candidates.numel() > 0:
                selected_has_correct[i] = True
                selected_rollout_indices[i] = correct_rollout_candidates[
                    np.random.randint(correct_rollout_candidates.numel())
                ]

        return (
            hint_group_scores,
            selected_hint_indices,
            selected_rollout_indices,
            selected_has_correct,
            selected_group_scores,
        )

    def _build_student_gen_batch(self, batch: DataProto) -> DataProto:
        """Build rollout prompts from the current batch without removing prompt text fields."""
        student_gen = batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=[
                "raw_prompt_ids",
                "multi_modal_data",
                "has_offline_trajectory",
                "offline_output",
            ],
            meta_info_keys=[
                "image_min_pixels", "image_max_pixels",
                "video_min_pixels", "video_max_pixels",
                "video_fps", "video_max_frames",
            ],
        )
        if "raw_prompt" in batch.non_tensor_batch:
            student_gen.non_tensor_batch["raw_prompt"] = batch.non_tensor_batch["raw_prompt"].copy()
        return student_gen

    def _compute_response_rewards(
        self,
        source_batch: DataProto,
        rollout_output: DataProto,
        repeat: int = 1,
    ) -> tuple[torch.Tensor, dict[str, list[float]]]:
        reward_batch = DataProto.from_dict(
            tensors={
                "responses": rollout_output.batch["responses"],
                "response_mask": rollout_output.batch["response_mask"],
            },
            non_tensors={},
        )
        for key, value in source_batch.non_tensor_batch.items():
            if key in ("multi_modal_data", "raw_prompt_ids"):
                continue
            reward_batch.non_tensor_batch[key] = np.repeat(value, repeat) if repeat > 1 else value
        return ray.get(self.reward_fn.compute_reward.remote(reward_batch))

    def _select_sdpo_feedback(
        self,
        candidate_output: DataProto,
        candidate_scores: torch.Tensor,
        batch_size: int,
    ) -> tuple[list[str], list[str], dict[str, float]]:
        candidate_n = self.opsd_config.sdpo_num_candidates
        response_ids = candidate_output.batch["responses"]
        response_mask = candidate_output.batch["response_mask"]
        response_lengths = response_mask.sum(dim=-1).to(torch.long)

        correct_solutions: list[str] = []
        incorrect_solutions: list[str] = []
        num_with_correct = 0
        num_with_incorrect = 0

        for i in range(batch_size):
            start = i * candidate_n
            end = start + candidate_n
            group_scores = candidate_scores[start:end]
            group_lengths = response_lengths[start:end]

            correct_idx = (group_scores > 0).nonzero(as_tuple=True)[0]
            incorrect_idx = (group_scores <= 0).nonzero(as_tuple=True)[0]

            selected_correct = ""
            if len(correct_idx) > 0:
                best_rel = correct_idx[group_lengths[correct_idx].argmin()].item()
                best_idx = start + best_rel
                cur_len = int(response_lengths[best_idx].item())
                selected_correct = self.tokenizer.decode(
                    response_ids[best_idx][:cur_len], skip_special_tokens=True
                ).strip()
                selected_correct = truncate_text_by_tokens(
                    selected_correct,
                    self.tokenizer,
                    self.opsd_config.teacher_max_trajectory_length,
                )
                num_with_correct += 1

            selected_incorrect = ""
            if len(incorrect_idx) > 0:
                best_rel = incorrect_idx[group_lengths[incorrect_idx].argmin()].item()
                best_idx = start + best_rel
                cur_len = int(response_lengths[best_idx].item())
                selected_incorrect = self.tokenizer.decode(
                    response_ids[best_idx][:cur_len], skip_special_tokens=True
                ).strip()
                selected_incorrect = truncate_text_by_tokens(
                    selected_incorrect,
                    self.tokenizer,
                    self.opsd_config.teacher_max_trajectory_length,
                )
                num_with_incorrect += 1

            correct_solutions.append(selected_correct)
            incorrect_solutions.append(selected_incorrect)

        metrics = {
            "sdpo/num_prompts_with_correct_candidate": float(num_with_correct),
            "sdpo/num_prompts_with_incorrect_candidate": float(num_with_incorrect),
            "sdpo/correct_candidate_coverage": float(num_with_correct / max(batch_size, 1)),
            "sdpo/incorrect_candidate_coverage": float(num_with_incorrect / max(batch_size, 1)),
        }
        return correct_solutions, incorrect_solutions, metrics

    def _make_opsd_batch(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Generate a full OPSD batch with multi-round generation.

        Steps:
            1. Load data batch with trajectories
            2. Student on-policy rollout (n=opsd.student_rollout_n)
            3. Navigator hint generation (n=K)
            4. Teacher trajectory generation (n=1, B*K prompts)
            5. Evaluate Teacher trajectories for binary rewards
            6. Compute Navigator GRPO advantage
            7. Assemble OPSD forward pass batch

        Returns:
            dict with all data needed for joint update.
        """
        timing_raw = {}

        # === Step 0: Load data ===
        try:
            batch_dict = next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.train_dataloader)
            batch_dict = next(self.data_iterator)

        meta_info = {
            "image_min_pixels": self.config.data.image_min_pixels,
            "image_max_pixels": self.config.data.image_max_pixels,
            "video_min_pixels": self.config.data.video_min_pixels,
            "video_max_pixels": self.config.data.video_max_pixels,
            "video_fps": self.config.data.video_fps,
            "video_max_frames": self.config.data.video_max_frames,
        }
        batch = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
        batch_size = len(batch.batch)
        batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(batch_size)], dtype=object
        )

        use_gt_as_hint = self.opsd_config.use_gt_as_hint
        enable_sdpo = self.opsd_config.enable_sdpo
        self_distill_negative_off_policy = (
            use_gt_as_hint and self.opsd_config.self_distill_negative_off_policy
        )
        student_rollout_n = int(self.opsd_config.student_rollout_n)
        dynamic_sample_hint = bool(self.opsd_config.dynamic_sample_hint)
        teacher_rollout_n = int(self.opsd_config.teacher_rollout_n)
        K = 1 if (use_gt_as_hint or enable_sdpo) else self.opsd_config.num_hints
        skip_nav_grpo = (self.opsd_config.lambda_nav == 0) or use_gt_as_hint or enable_sdpo
        skip_off_policy = (
            ((self.opsd_config.alpha >= 1.0) and (not self_distill_negative_off_policy))
            or (use_gt_as_hint and (not self_distill_negative_off_policy))
            or enable_sdpo
        )

        student_gen_base = self._build_student_gen_batch(batch)
        sdpo_correct_solutions: Optional[list[str]] = None
        sdpo_incorrect_solutions: Optional[list[str]] = None

        # === Step 1: Student On-Policy Rollout ===
        if enable_sdpo:
            candidate_n = self.opsd_config.sdpo_num_candidates
            with timer("sdpo_candidate_rollout", timing_raw):
                sdpo_candidate_gen = deepcopy(student_gen_base)
                sdpo_candidate_gen.meta_info.update(
                    {
                        "n": candidate_n,
                        "temperature": self.config.worker.rollout.temperature,
                        "top_p": self.config.worker.rollout.top_p,
                        "max_tokens": self.config.data.max_response_length,
                        "image_min_pixels": self.config.data.image_min_pixels,
                        "image_max_pixels": self.config.data.image_max_pixels,
                        "video_min_pixels": self.config.data.video_min_pixels,
                        "video_max_pixels": self.config.data.video_max_pixels,
                        "video_fps": self.config.data.video_fps,
                        "video_max_frames": self.config.data.video_max_frames,
                    }
                )
                sdpo_candidate_gen, sdpo_candidate_pad_size = pad_dataproto_to_divisor(
                    sdpo_candidate_gen, self.actor_rollout_ref_wg.world_size
                )
                sdpo_candidate_output = self.actor_rollout_ref_wg.generate_sequences(sdpo_candidate_gen)
                sdpo_candidate_output = unpad_dataproto(
                    sdpo_candidate_output, pad_size=sdpo_candidate_pad_size * candidate_n
                )

            with timer("sdpo_candidate_reward", timing_raw):
                sdpo_candidate_reward_tensor, sdpo_candidate_reward_metrics = self._compute_response_rewards(
                    batch, sdpo_candidate_output, repeat=candidate_n
                )
                metrics.update(
                    {
                        f"sdpo_candidate_reward/{k}": v
                        for k, v in reduce_metrics(sdpo_candidate_reward_metrics).items()
                    }
                )
                sdpo_candidate_scores = sdpo_candidate_reward_tensor.sum(dim=-1)
                (
                    sdpo_correct_solutions,
                    sdpo_incorrect_solutions,
                    sdpo_selection_metrics,
                ) = self._select_sdpo_feedback(sdpo_candidate_output, sdpo_candidate_scores, batch_size)
                metrics.update(sdpo_selection_metrics)

            with timer("student_rollout", timing_raw):
                student_gen = deepcopy(student_gen_base)
                student_gen.meta_info.update(
                    {
                        "n": student_rollout_n,
                        "temperature": self.config.worker.rollout.temperature,
                        "top_p": self.config.worker.rollout.top_p,
                        "max_tokens": self.config.data.max_response_length,
                        "image_min_pixels": self.config.data.image_min_pixels,
                        "image_max_pixels": self.config.data.image_max_pixels,
                        "video_min_pixels": self.config.data.video_min_pixels,
                        "video_max_pixels": self.config.data.video_max_pixels,
                        "video_fps": self.config.data.video_fps,
                        "video_max_frames": self.config.data.video_max_frames,
                    }
                )
                student_gen, student_pad_size = pad_dataproto_to_divisor(
                    student_gen, self.actor_rollout_ref_wg.world_size
                )
                student_output = self.actor_rollout_ref_wg.generate_sequences(student_gen)
                student_output = unpad_dataproto(
                    student_output, pad_size=student_pad_size * student_rollout_n
                )
                student_gen = unpad_dataproto(student_gen, pad_size=student_pad_size)
        else:
            with timer("student_rollout", timing_raw):
                student_gen = deepcopy(student_gen_base)
                student_gen.meta_info.update(
                    {
                        "n": student_rollout_n,
                        "temperature": self.config.worker.rollout.temperature,
                        "top_p": self.config.worker.rollout.top_p,
                        "max_tokens": self.config.data.max_response_length,
                        "image_min_pixels": self.config.data.image_min_pixels,
                        "image_max_pixels": self.config.data.image_max_pixels,
                        "video_min_pixels": self.config.data.video_min_pixels,
                        "video_max_pixels": self.config.data.video_max_pixels,
                        "video_fps": self.config.data.video_fps,
                        "video_max_frames": self.config.data.video_max_frames,
                    }
                )
                student_gen, student_pad_size = pad_dataproto_to_divisor(
                    student_gen, self.actor_rollout_ref_wg.world_size
                )
                student_output = self.actor_rollout_ref_wg.generate_sequences(student_gen)
                student_output = unpad_dataproto(
                    student_output, pad_size=student_pad_size * student_rollout_n
                )
                student_gen = unpad_dataproto(student_gen, pad_size=student_pad_size)

        expected_student_bs = batch_size * student_rollout_n
        if len(student_output) != expected_student_bs:
            raise RuntimeError(
                f"Unexpected student rollout batch size. expected={expected_student_bs}, got={len(student_output)}. "
                f"base_batch_size={batch_size}, student_rollout_n={student_rollout_n}"
            )

        if student_rollout_n > 1:
            batch = batch.repeat(repeat_times=student_rollout_n, interleave=True)
            # Keep each repeated student sample independent in downstream grouping/metrics.
            batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
            )
            if enable_sdpo:
                if sdpo_correct_solutions is not None:
                    sdpo_correct_solutions = np.repeat(
                        np.array(sdpo_correct_solutions, dtype=object), student_rollout_n
                    ).tolist()
                if sdpo_incorrect_solutions is not None:
                    sdpo_incorrect_solutions = np.repeat(
                        np.array(sdpo_incorrect_solutions, dtype=object), student_rollout_n
                    ).tolist()

        # Merge student output back
        batch = batch.union(student_output)
        batch_size = len(batch.batch)
        metrics["opsd/student_rollout_n"] = float(student_rollout_n)
        metrics["opsd/student_rollout_effective_batch"] = float(batch_size)
        self._log_student_generations(batch, student_output)

        print(
            f"[OPSD] Step 1: Student rollout complete. "
            f"base_batch_size={expected_student_bs // student_rollout_n}, "
            f"student_rollout_n={student_rollout_n}, effective_batch_size={batch_size}"
        )

        student_old_lp = None
        student_outcome_advantages = None
        student_outcome_response_mask = None
        student_distill_response_mask = None
        reward_tensor = None
        reward_metrics = None
        if use_gt_as_hint or enable_sdpo:
            with timer("student_reward", timing_raw):
                reward_tensor, reward_metrics = self._compute_response_rewards(batch, student_output)
                metrics.update(
                    {
                        f"student_reward/{k}": v
                        for k, v in reduce_metrics(reward_metrics).items()
                    }
                )

            if self.opsd_config.outcome_ppo_coef > 0:
                with timer("student_old_logprobs", timing_raw):
                    student_logprob_batch = DataProto.from_dict(
                        tensors={
                            "input_ids": student_output.batch["input_ids"],
                            "attention_mask": student_output.batch["attention_mask"],
                            "position_ids": student_output.batch["position_ids"],
                            "responses": student_output.batch["responses"],
                        },
                        non_tensors={
                            "multi_modal_data": batch.non_tensor_batch.get(
                                "multi_modal_data",
                                np.array([{}] * student_output.batch["input_ids"].shape[0], dtype=object),
                            ),
                            "uid": batch.non_tensor_batch["uid"],
                        },
                        meta_info={
                            "temperature": self.config.worker.rollout.temperature,
                            "image_min_pixels": self.config.data.image_min_pixels,
                            "image_max_pixels": self.config.data.image_max_pixels,
                            "video_min_pixels": self.config.data.video_min_pixels,
                            "video_max_pixels": self.config.data.video_max_pixels,
                            "video_fps": self.config.data.video_fps,
                            "video_max_frames": self.config.data.video_max_frames,
                        },
                    )
                    old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(student_logprob_batch)
                    student_old_lp = old_log_probs.batch["old_log_probs"]

                student_outcome_advantages, _ = compute_advantage_return(
                    AdvantageEstimator.REINFORCE_PLUS_PLUS,
                    token_level_rewards=reward_tensor,
                    response_mask=student_output.batch["response_mask"],
                    gamma=torch.as_tensor(
                        self.config.algorithm.gamma,
                        dtype=reward_tensor.dtype,
                        device=reward_tensor.device,
                    ),
                )
                student_outcome_response_mask = student_output.batch["response_mask"]
                student_distill_response_mask = student_output.batch["response_mask"]
                if self.opsd_config.outcome_ppo_incorrect_only:
                    if "accuracy" in reward_metrics:
                        accuracy = torch.as_tensor(
                            reward_metrics["accuracy"],
                            dtype=student_outcome_advantages.dtype,
                            device=student_outcome_advantages.device,
                        )
                        incorrect_mask = (accuracy <= 0).to(student_output.batch["response_mask"].dtype)
                    else:
                        outcome_scores = reward_tensor.sum(dim=-1)
                        incorrect_mask = (outcome_scores <= 0).to(student_output.batch["response_mask"].dtype)
                    student_outcome_response_mask = (
                        student_output.batch["response_mask"] * incorrect_mask.unsqueeze(-1)
                    )
                    student_distill_response_mask = (
                        student_output.batch["response_mask"] * (1 - incorrect_mask).unsqueeze(-1)
                    )
                    metrics["opsd/outcome_num_incorrect"] = float(incorrect_mask.sum().item())
                    metrics["opsd/outcome_num_correct"] = float(
                        incorrect_mask.numel() - incorrect_mask.sum().item()
                    )
            if use_gt_as_hint and self.opsd_config.self_distill_mask_prefix_tokens > 0:
                student_distill_response_mask = _mask_prefix_response_tokens(
                    student_distill_response_mask
                    if student_distill_response_mask is not None
                    else student_output.batch["response_mask"],
                    self.opsd_config.self_distill_mask_prefix_tokens,
                )
                masked_prefix_tokens = (
                    student_output.batch["response_mask"].sum() - student_distill_response_mask.sum()
                ).item()
                metrics["opsd/self_distill_masked_prefix_tokens"] = float(masked_prefix_tokens)
                metrics["opsd/self_distill_mask_prefix_tokens"] = float(
                    self.opsd_config.self_distill_mask_prefix_tokens
                )
            metrics["opsd/outcome_reward_mean"] = (
                (reward_tensor * student_output.batch["response_mask"]).sum()
                / student_output.batch["response_mask"].sum().clamp(min=1)
            ).item()

        # === Step 2: Navigator Hint Generation (or SDPO/GT-as-hint) ===
        if enable_sdpo:
            nav_output = None
            print(
                "[OPSD] Step 2: SDPO candidate selection complete. "
                f"correct_cov={metrics['sdpo/correct_candidate_coverage']:.3f}, "
                f"incorrect_cov={metrics['sdpo/incorrect_candidate_coverage']:.3f}"
            )
        elif use_gt_as_hint:
            # Self-Distilled Reasoner: skip Navigator rollout. Use the preprocessed
            # reference trajectory directly as teacher-side extra context; fall back
            # to GT only if the dataset does not provide one.
            trajectories = batch.non_tensor_batch.get("trajectory")
            hints = []
            used_trajectory = 0
            for idx, gt in enumerate(batch.non_tensor_batch["ground_truth"]):
                trajectory = ""
                if trajectories is not None:
                    trajectory = truncate_text_by_tokens(
                        trajectories[idx],
                        self.tokenizer,
                        self.opsd_config.teacher_max_trajectory_length,
                    )
                if trajectory:
                    hints.append([trajectory])
                    used_trajectory += 1
                else:
                    hints.append([str(gt)])
            nav_output = None
            print(
                f"[OPSD] Step 2: Using reference trajectories as teacher context where available "
                f"(fallback to GT for {batch_size - used_trajectory}/{batch_size}). B={batch_size}"
            )
        else:
            navigator_student_trajectories = None
            if self.opsd_config.navigator_include_student_trajectory:
                navigator_student_trajectories = self._decode_student_trajectories(
                    student_output,
                    max_tokens=self.opsd_config.navigator_max_trajectory_length,
                )

            with timer("navigator_rollout", timing_raw):
                nav_gen = construct_navigator_prompts(
                    questions=batch.non_tensor_batch["prompt_text"],
                    trajectories=batch.non_tensor_batch["trajectory"],
                    student_trajectories=navigator_student_trajectories,
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    template_path=self._nav_template_path,
                    max_traj_len=self.opsd_config.navigator_max_trajectory_length,
                    max_prompt_length=self.config.data.max_prompt_length,
                    multi_modal_data=batch.non_tensor_batch.get("multi_modal_data"),
                    image_min_pixels=self.config.data.image_min_pixels,
                    image_max_pixels=self.config.data.image_max_pixels,
                    video_min_pixels=self.config.data.video_min_pixels,
                    video_max_pixels=self.config.data.video_max_pixels,
                    video_fps=self.config.data.video_fps,
                    video_max_frames=self.config.data.video_max_frames,
                    apply_chat_template_kwargs=self.config.data.apply_chat_template_kwargs,
                )
                nav_gen.meta_info.update({
                    "n": K,
                    "temperature": self.config.worker.rollout.temperature,
                    "top_p": self.config.worker.rollout.top_p,
                    "max_tokens": self.opsd_config.navigator_response_length,
                    "image_min_pixels": self.config.data.image_min_pixels,
                    "image_max_pixels": self.config.data.image_max_pixels,
                    "video_min_pixels": self.config.data.video_min_pixels,
                    "video_max_pixels": self.config.data.video_max_pixels,
                    "video_fps": self.config.data.video_fps,
                    "video_max_frames": self.config.data.video_max_frames,
                })

                nav_gen, nav_pad_size = pad_dataproto_to_divisor(
                    nav_gen, self.actor_rollout_ref_wg.world_size
                )
                nav_output = self.actor_rollout_ref_wg.generate_sequences(nav_gen)
                nav_output = unpad_dataproto(nav_output, pad_size=nav_pad_size * K)

            # Decode hints
            nav_output.meta_info["n"] = K
            hints = decode_navigator_hints(nav_output, self.tokenizer)  # B × K
            self._log_navigator_hints(batch, hints)

            print(f"[OPSD] Step 2: Navigator generated {len(hints)} × {K} hints")

        # === Step 3: Teacher ===
        if enable_sdpo:
            p_drop = 0.0
            metrics["curriculum/p_drop"] = p_drop

            with timer("teacher_prompt_tokenize", timing_raw):
                teacher_gen = construct_sdpo_teacher_prompts(
                    questions=self._get_teacher_questions(batch),
                    correct_solutions=sdpo_correct_solutions or [""] * batch_size,
                    incorrect_solutions=sdpo_incorrect_solutions or [""] * batch_size,
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    template_path=self._sdpo_teacher_template_path,
                    max_prompt_length=self.config.data.max_prompt_length,
                    multi_modal_data=batch.non_tensor_batch.get("multi_modal_data"),
                    image_min_pixels=self.config.data.image_min_pixels,
                    image_max_pixels=self.config.data.image_max_pixels,
                    video_min_pixels=self.config.data.video_min_pixels,
                    video_max_pixels=self.config.data.video_max_pixels,
                    video_fps=self.config.data.video_fps,
                    video_max_frames=self.config.data.video_max_frames,
                    apply_chat_template_kwargs=self.config.data.apply_chat_template_kwargs,
                )
            teacher_output = None
            teacher_uid = batch.non_tensor_batch["uid"]
            hint_correct = torch.ones(batch_size)
            metrics["nav/hint_accuracy"] = 1.0
            print(f"[OPSD] Step 3: SDPO teacher prompt tokenized (no generation). B={batch_size}")
            print("[OPSD] Step 4: Skipped (SDPO, no Teacher trajectories)")
        elif use_gt_as_hint:
            # Self-Distilled Reasoner: skip Teacher generation entirely.
            # Only tokenize Teacher prompts (with GT answer as hint, p_drop=0).
            # The Teacher forward pass happens during actor update, not here.
            p_drop = 0.0
            metrics["curriculum/p_drop"] = p_drop

            with timer("teacher_prompt_tokenize", timing_raw):
                # In self-distillation mode, GT is already provided as `hint`.
                # Pass empty reference answers so the teacher prompt cannot
                # accidentally include a second copy of the ground truth.
                empty_gt_answers = np.array([""] * batch_size, dtype=object)
                teacher_gen = construct_teacher_prompts(
                    questions=self._get_teacher_questions(batch),
                    hints=hints,
                    gt_answers=empty_gt_answers,
                    p_drop=p_drop,
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    template_path=self._teacher_template_path,
                    max_prompt_length=self.config.data.max_prompt_length,
                    multi_modal_data=batch.non_tensor_batch.get("multi_modal_data"),
                    image_min_pixels=self.config.data.image_min_pixels,
                    image_max_pixels=self.config.data.image_max_pixels,
                    video_min_pixels=self.config.data.video_min_pixels,
                    video_max_pixels=self.config.data.video_max_pixels,
                    video_fps=self.config.data.video_fps,
                    video_max_frames=self.config.data.video_max_frames,
                    apply_chat_template_kwargs=self.config.data.apply_chat_template_kwargs,
                )
            teacher_on_policy_gen = teacher_gen
            teacher_output = None
            if self_distill_negative_off_policy:
                with timer("teacher_rollout", timing_raw):
                    teacher_gen.meta_info.update({
                        "n": 1,
                        "temperature": self.config.worker.rollout.temperature,
                        "top_p": self.config.worker.rollout.top_p,
                        "image_min_pixels": self.config.data.image_min_pixels,
                        "image_max_pixels": self.config.data.image_max_pixels,
                        "video_min_pixels": self.config.data.video_min_pixels,
                        "video_max_pixels": self.config.data.video_max_pixels,
                        "video_fps": self.config.data.video_fps,
                        "video_max_frames": self.config.data.video_max_frames,
                    })
                    teacher_gen, teacher_pad_size = pad_dataproto_to_divisor(
                        teacher_gen, self.actor_rollout_ref_wg.world_size
                    )
                    teacher_output = self.actor_rollout_ref_wg.generate_sequences(teacher_gen)
                    teacher_output = unpad_dataproto(teacher_output, pad_size=teacher_pad_size)
                print(
                    "[OPSD] Step 3: Teacher rollout complete for self-distill negative off-policy. "
                    f"B={batch_size}"
                )
            else:
                print(f"[OPSD] Step 3: Teacher prompt tokenized (no generation). B={batch_size}")

            # Step 4: Skip Teacher evaluation. Self-distill does not reward/filter
            # Teacher trajectories, even when a negative off-policy branch is enabled.
            teacher_uid = batch.non_tensor_batch["uid"]
            hint_correct = torch.ones(batch_size)  # GT hint → assume correct
            metrics["nav/hint_accuracy"] = 1.0
            if self_distill_negative_off_policy:
                print("[OPSD] Step 4: Skipped (Self-Distilled Reasoner negative off-policy, no Teacher filtering)")
            else:
                print("[OPSD] Step 4: Skipped (Self-Distilled Reasoner, no Teacher trajectories)")

        else:
            # Full OPSD / Static Navigator: Teacher generates trajectories
            # Curriculum is temporarily disabled: always keep GT available to Teacher.
            p_drop = 0.0
            metrics["curriculum/p_drop"] = p_drop
            teacher_reference_trajectories, teacher_student_trajectories = (
                self._build_teacher_context_trajectories(batch, student_output=student_output)
            )
            rollout_gt_answers = (
                batch.non_tensor_batch["ground_truth"]
                if self.opsd_config.off_policy_include_gt_answer
                else np.array([""] * batch_size, dtype=object)
            )

            with timer("teacher_rollout", timing_raw):
                teacher_gen = construct_teacher_prompts(
                    questions=self._get_teacher_questions(batch),
                    hints=hints,
                    gt_answers=rollout_gt_answers,
                    p_drop=p_drop,
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    reference_trajectories=teacher_reference_trajectories,
                    student_trajectories=teacher_student_trajectories,
                    template_path=self._teacher_template_path,
                    max_prompt_length=self.config.data.max_prompt_length,
                    multi_modal_data=batch.non_tensor_batch.get("multi_modal_data"),
                    image_min_pixels=self.config.data.image_min_pixels,
                    image_max_pixels=self.config.data.image_max_pixels,
                    video_min_pixels=self.config.data.video_min_pixels,
                    video_max_pixels=self.config.data.video_max_pixels,
                    video_fps=self.config.data.video_fps,
                    video_max_frames=self.config.data.video_max_frames,
                    apply_chat_template_kwargs=self.config.data.apply_chat_template_kwargs,
                )
                teacher_gen.meta_info.update({
                    "n": teacher_rollout_n,
                    "temperature": self.config.worker.rollout.temperature,
                    "top_p": self.config.worker.rollout.top_p,
                    "image_min_pixels": self.config.data.image_min_pixels,
                    "image_max_pixels": self.config.data.image_max_pixels,
                    "video_min_pixels": self.config.data.video_min_pixels,
                    "video_max_pixels": self.config.data.video_max_pixels,
                    "video_fps": self.config.data.video_fps,
                    "video_max_frames": self.config.data.video_max_frames,
                })

                teacher_gen, teacher_pad_size = pad_dataproto_to_divisor(
                    teacher_gen, self.actor_rollout_ref_wg.world_size
                )
                teacher_output = self.actor_rollout_ref_wg.generate_sequences(teacher_gen)
                teacher_output = unpad_dataproto(
                    teacher_output, pad_size=teacher_pad_size * teacher_rollout_n
                )

            print(f"[OPSD] Step 3: Teacher rollout complete. B*K*M = {batch_size * K * teacher_rollout_n}")

            teacher_on_policy_gen = teacher_gen
            if not self.opsd_config.off_policy_include_gt_answer:
                with timer("teacher_on_policy_prompt_tokenize", timing_raw):
                    teacher_on_policy_gen = self._construct_on_policy_teacher_prompts(
                        batch,
                        hints,
                        student_output=student_output,
                    )

            # === Step 4: Evaluate Teacher Trajectories ===
            with timer("teacher_reward", timing_raw):
                teacher_repeat = K * teacher_rollout_n
                teacher_gt = np.repeat(batch.non_tensor_batch["ground_truth"], teacher_repeat)
                teacher_rollout_uid = np.repeat(batch.non_tensor_batch["uid"], teacher_repeat)

                teacher_reward_batch = DataProto.from_dict(
                    tensors={
                        "responses": teacher_output.batch["responses"],
                        "response_mask": teacher_output.batch["response_mask"],
                    },
                    non_tensors={
                        "ground_truth": teacher_gt,
                        "uid": teacher_rollout_uid,
                    },
                )
                for key in batch.non_tensor_batch:
                    if key not in teacher_reward_batch.non_tensor_batch and key not in ("multi_modal_data", "raw_prompt_ids"):
                        teacher_reward_batch.non_tensor_batch[key] = np.repeat(
                            batch.non_tensor_batch[key], teacher_repeat
                        )

                reward_tensor, reward_metrics = ray.get(
                    self.reward_fn.compute_reward.remote(teacher_reward_batch)
                )

                teacher_rollout_rewards = reward_tensor.sum(dim=-1)
                if "accuracy" in reward_metrics:
                    rollout_correctness = torch.as_tensor(
                        reward_metrics["accuracy"],
                        dtype=teacher_rollout_rewards.dtype,
                        device=teacher_rollout_rewards.device,
                    )
                else:
                    rollout_correctness = (teacher_rollout_rewards > 0).to(teacher_rollout_rewards.dtype)

                rollout_correctness = rollout_correctness.view(batch_size, K, teacher_rollout_n)
                if dynamic_sample_hint:
                    (
                        hint_group_scores,
                        selected_hint_indices,
                        selected_rollout_indices,
                        selected_has_correct,
                        selected_group_scores,
                    ) = self._select_dynamic_hint_targets(rollout_correctness)
                    hint_correct = hint_group_scores.reshape(-1)
                    teacher_uid = np.repeat(batch.non_tensor_batch["uid"], K)
                else:
                    hint_correct = rollout_correctness.reshape(batch_size * K, teacher_rollout_n).mean(dim=-1)
                    teacher_uid = np.repeat(batch.non_tensor_batch["uid"], K)

                reward_metrics_reduced = {
                    f"teacher_reward/{k}": v
                    for k, v in reduce_metrics(reward_metrics).items()
                }
                metrics.update(reward_metrics_reduced)

            metrics["nav/hint_accuracy"] = hint_correct.mean().item()
            if dynamic_sample_hint:
                metrics["nav/best_hint_accuracy"] = selected_group_scores.mean().item()
                metrics["opsd/dynamic_selected_hint_has_correct_rate"] = (
                    selected_has_correct.float().mean().item()
                )
                print(
                    "[OPSD] Step 4: Teacher reward computed. "
                    f"hint_group_acc={metrics['nav/hint_accuracy']:.3f}, "
                    f"best_hint_acc={metrics['nav/best_hint_accuracy']:.3f}"
                )
            else:
                print(f"[OPSD] Step 4: Teacher reward computed. Hint accuracy: {metrics['nav/hint_accuracy']:.3f}")

        # === Step 5: Navigator GRPO Advantage ===
        if nav_output is not None:
            nav_resp_len = nav_output.batch["responses"].shape[1]
            nav_response_mask = nav_output.batch["response_mask"]
        else:
            # SDPO / use_gt_as_hint: no Navigator output, use minimal placeholders
            nav_resp_len = 1
            nav_response_mask = torch.zeros(batch_size * K, nav_resp_len)

        if not skip_nav_grpo:
            nav_advantages = compute_navigator_advantage(
                hint_rewards=hint_correct,
                hint_uids=teacher_uid,
                K=K,
            )
            nav_advantages_expanded = nav_advantages.unsqueeze(-1).expand(-1, nav_resp_len) * nav_response_mask
            metrics["nav/advantage_std"] = nav_advantages.std().item()
            metrics["nav/advantage_mean"] = nav_advantages.mean().item()
        else:
            # Placeholder: zero advantages (won't be used)
            nav_advantages_expanded = torch.zeros(batch_size * K, nav_resp_len)
            metrics["nav/advantage_std"] = 0.0
            metrics["nav/advantage_mean"] = 0.0

        # === Step 6: Compute Navigator old_log_probs ===
        if not skip_nav_grpo:
            with timer("nav_old_logprobs", timing_raw):
                nav_logprob_batch = DataProto.from_dict(
                    tensors={
                        "input_ids": nav_output.batch["input_ids"],
                        "attention_mask": nav_output.batch["attention_mask"],
                        "position_ids": nav_output.batch["position_ids"],
                        "responses": nav_output.batch["responses"],
                    },
                    non_tensors={
                        "multi_modal_data": nav_output.non_tensor_batch.get(
                            "multi_modal_data",
                            np.array([{}] * nav_output.batch["input_ids"].shape[0], dtype=object),
                        ),
                        "uid": np.repeat(batch.non_tensor_batch["uid"], K),
                    },
                    meta_info={
                        "temperature": self.config.worker.rollout.temperature,
                        "image_min_pixels": self.config.data.image_min_pixels,
                        "image_max_pixels": self.config.data.image_max_pixels,
                        "video_min_pixels": self.config.data.video_min_pixels,
                        "video_max_pixels": self.config.data.video_max_pixels,
                        "video_fps": self.config.data.video_fps,
                        "video_max_frames": self.config.data.video_max_frames,
                    },
                )
                nav_old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(nav_logprob_batch)
                nav_old_lp = nav_old_log_probs.batch["old_log_probs"]
        else:
            # Placeholder: zeros (won't be used)
            nav_old_lp = torch.zeros(batch_size * K, nav_resp_len)
            if enable_sdpo:
                print("[OPSD] Skipping Navigator (enable_sdpo=True)")
            elif use_gt_as_hint:
                print("[OPSD] Skipping Navigator (use_gt_as_hint=True, Self-Distilled Reasoner)")
            else:
                print("[OPSD] Skipping Navigator old_log_probs (lambda_nav=0, static navigator)")

        # === Step 7: Assemble combined batch for joint update ===
        # We pack all data with prefix keys into a single DataProto

        # --- Navigator data (included even when static, but lambda_nav=0 skips gradient) ---
        if nav_output is not None:
            nav_tensors = {
                "nav_input_ids": nav_output.batch["input_ids"],
                "nav_attention_mask": nav_output.batch["attention_mask"],
                "nav_position_ids": nav_output.batch["position_ids"],
                "nav_responses": nav_output.batch["responses"],
                "nav_response_mask": nav_response_mask,
                "nav_old_log_probs": nav_old_lp,
                "nav_advantages": nav_advantages_expanded,
            }
        else:
            # use_gt_as_hint: no Navigator data, use minimal placeholders
            nav_tensors = {
                "nav_input_ids": torch.zeros(batch_size * K, 1, dtype=torch.long),
                "nav_attention_mask": torch.zeros(batch_size * K, 1, dtype=torch.long),
                "nav_position_ids": torch.zeros(batch_size * K, 1, dtype=torch.long),
                "nav_responses": torch.zeros(batch_size * K, 1, dtype=torch.long),
                "nav_response_mask": nav_response_mask,
                "nav_old_log_probs": nav_old_lp,
                "nav_advantages": nav_advantages_expanded,
            }

        if dynamic_sample_hint and teacher_output is not None:
            on_indices = torch.arange(batch_size, device=student_output.batch["responses"].device)
            on_question_indices = on_indices
            num_valid_on_policy = len(on_indices)

            student_input_ids_rep = student_output.batch["input_ids"]
            student_attention_mask_rep = student_output.batch["attention_mask"]
            student_position_ids_rep = student_output.batch["position_ids"]
            student_responses_rep = student_output.batch["responses"]
            student_response_mask_rep = student_output.batch["response_mask"]

            selected_hint_indices_dev = selected_hint_indices.to(device=on_indices.device)
            selected_has_correct_dev = selected_has_correct.to(device=on_indices.device)
            selected_rollout_indices_dev = selected_rollout_indices.to(device=on_indices.device)
            selected_prompt_rows = on_question_indices * K + selected_hint_indices_dev

            teacher_prompts = teacher_on_policy_gen.batch["input_ids"][selected_prompt_rows]
            teacher_prompt_mask = teacher_on_policy_gen.batch["attention_mask"][selected_prompt_rows]
            teacher_prompt_position_ids = teacher_on_policy_gen.batch["position_ids"][selected_prompt_rows]

            invalid_hint_mask = ~selected_has_correct_dev
            if torch.any(invalid_hint_mask):
                with timer("teacher_on_policy_fallback_prompt_tokenize", timing_raw):
                    fallback_teacher_gen, trajectory_missing = self._construct_teacher_fallback_prompts(
                        batch,
                        repeat_k=1,
                        student_output=student_output,
                    )

                fallback_prompts = fallback_teacher_gen.batch["input_ids"]
                fallback_prompt_mask = fallback_teacher_gen.batch["attention_mask"]
                fallback_prompt_position_ids = fallback_teacher_gen.batch["position_ids"]

                teacher_prompts = teacher_prompts.clone()
                teacher_prompt_mask = teacher_prompt_mask.clone()
                teacher_prompt_position_ids = teacher_prompt_position_ids.clone()
                teacher_prompts[invalid_hint_mask] = fallback_prompts[invalid_hint_mask]
                teacher_prompt_mask[invalid_hint_mask] = fallback_prompt_mask[invalid_hint_mask]
                teacher_prompt_position_ids[invalid_hint_mask] = fallback_prompt_position_ids[invalid_hint_mask]
                metrics["opsd/on_policy_trajectory_fallback_count"] = float(invalid_hint_mask.sum().item())
                metrics["opsd/on_policy_fallback_missing_trajectory_count"] = float(trajectory_missing)
                metrics["opsd/on_policy_gt_fallback_count"] = float(trajectory_missing)
            else:
                metrics["opsd/on_policy_trajectory_fallback_count"] = 0.0
                metrics["opsd/on_policy_fallback_missing_trajectory_count"] = 0.0
                metrics["opsd/on_policy_gt_fallback_count"] = 0.0

            on_teacher_input_ids, on_teacher_attention_mask, on_teacher_position_ids = _concat_prompt_and_response(
                prompt_ids=teacher_prompts,
                prompt_attention_mask=teacher_prompt_mask,
                prompt_position_ids=teacher_prompt_position_ids,
                response_ids=student_responses_rep,
                response_mask=student_response_mask_rep,
            )
            on_policy_tensors = {
                "on_student_input_ids": student_input_ids_rep,
                "on_student_attention_mask": student_attention_mask_rep,
                "on_student_position_ids": student_position_ids_rep,
                "on_student_responses": student_responses_rep,
                "on_student_response_mask": student_response_mask_rep,
                "on_teacher_input_ids": on_teacher_input_ids,
                "on_teacher_attention_mask": on_teacher_attention_mask,
                "on_teacher_position_ids": on_teacher_position_ids,
                "on_teacher_responses": student_responses_rep,
                "on_teacher_response_mask": student_response_mask_rep,
                "on_student_distill_response_mask": (
                    student_distill_response_mask if student_distill_response_mask is not None else student_response_mask_rep
                ),
            }
            if student_old_lp is not None and student_outcome_advantages is not None:
                on_policy_tensors["on_student_old_log_probs"] = student_old_lp
                on_policy_tensors["on_student_outcome_advantages"] = student_outcome_advantages
                on_policy_tensors["on_student_outcome_response_mask"] = student_outcome_response_mask

            nonempty_on_mask = (
                (student_attention_mask_rep.sum(dim=-1) > 0)
                & (student_response_mask_rep.sum(dim=-1) > 0)
                & (on_teacher_attention_mask.sum(dim=-1) > 0)
            )
            if not torch.all(nonempty_on_mask):
                on_indices = on_indices[nonempty_on_mask]
                on_question_indices = on_question_indices[nonempty_on_mask]
                selected_has_correct_dev = selected_has_correct_dev[nonempty_on_mask]
                selected_rollout_indices_dev = selected_rollout_indices_dev[nonempty_on_mask]
                selected_hint_indices_dev = selected_hint_indices_dev[nonempty_on_mask]
                num_valid_on_policy = int(nonempty_on_mask.sum().item())
                on_policy_tensors = {
                    key: tensor[nonempty_on_mask] for key, tensor in on_policy_tensors.items()
                }

            on_teacher_attention_mask = on_policy_tensors["on_teacher_attention_mask"]
            off_policy_tensors = {}
            valid_question_indices = on_question_indices[selected_has_correct_dev]
            num_valid = len(valid_question_indices)
            if not skip_off_policy and num_valid > 0:
                valid_hint_rows = valid_question_indices * K + selected_hint_indices_dev[selected_has_correct_dev]
                valid_rollout_rows = valid_hint_rows * teacher_rollout_n + selected_rollout_indices_dev[selected_has_correct_dev]
                teacher_correct_responses = teacher_output.batch["responses"][valid_rollout_rows]
                teacher_correct_resp_mask = teacher_output.batch["response_mask"][valid_rollout_rows]

                student_prompts_valid = student_output.batch["prompts"][valid_question_indices]
                student_prompt_len = student_prompts_valid.shape[1]
                student_prompt_mask_valid = student_output.batch["attention_mask"][valid_question_indices, :student_prompt_len]
                student_prompt_position_ids_valid = student_output.batch["position_ids"][
                    valid_question_indices, ..., :student_prompt_len
                ]
                off_student_input_ids, off_student_attention_mask, off_student_position_ids = _concat_prompt_and_response(
                    prompt_ids=student_prompts_valid,
                    prompt_attention_mask=student_prompt_mask_valid,
                    prompt_position_ids=student_prompt_position_ids_valid,
                    response_ids=teacher_correct_responses,
                    response_mask=teacher_correct_resp_mask,
                )

                off_policy_tensors = {
                    "off_student_input_ids": off_student_input_ids,
                    "off_student_attention_mask": off_student_attention_mask,
                    "off_student_position_ids": off_student_position_ids,
                    "off_student_responses": teacher_correct_responses,
                    "off_student_response_mask": teacher_correct_resp_mask,
                    "off_teacher_input_ids": teacher_output.batch["input_ids"][valid_rollout_rows],
                    "off_teacher_attention_mask": teacher_output.batch["attention_mask"][valid_rollout_rows],
                    "off_teacher_position_ids": teacher_output.batch["position_ids"][valid_rollout_rows],
                    "off_teacher_responses": teacher_correct_responses,
                    "off_teacher_response_mask": teacher_correct_resp_mask,
                }
            elif skip_off_policy:
                print("[OPSD] Skipping off-policy batch assembly (alpha=1.0, on-policy only)")

            metrics["opsd/num_valid_hints"] = num_valid
            metrics["opsd/num_total_hints"] = batch_size * K
            metrics["opsd/dynamic_selected_hint_count"] = float(batch_size)
        else:
            on_indices = torch.arange(batch_size * K, device=student_output.batch["responses"].device)
            if use_gt_as_hint or enable_sdpo:
                on_positive_mask = torch.ones_like(on_indices, dtype=torch.bool)
            else:
                on_positive_mask = hint_correct.to(device=on_indices.device) > 0
            num_valid_on_policy = len(on_indices)

            student_input_ids_rep = student_output.batch["input_ids"].repeat_interleave(K, dim=0)[on_indices]
            student_attention_mask_rep = student_output.batch["attention_mask"].repeat_interleave(K, dim=0)[on_indices]
            student_position_ids_rep = student_output.batch["position_ids"].repeat_interleave(K, dim=0)[on_indices]
            student_responses_rep = student_output.batch["responses"].repeat_interleave(K, dim=0)[on_indices]
            student_response_mask_rep = student_output.batch["response_mask"].repeat_interleave(K, dim=0)[on_indices]
            on_question_indices = on_indices // K if num_valid_on_policy > 0 else on_indices

            on_policy_use_fallback_prompt = False
            if teacher_output is not None:
                teacher_prompts = teacher_on_policy_gen.batch["input_ids"][on_indices]
                teacher_prompt_mask = teacher_on_policy_gen.batch["attention_mask"][on_indices]
                teacher_prompt_position_ids = teacher_on_policy_gen.batch["position_ids"][on_indices]

                invalid_hint_mask = ~on_positive_mask
                if (not use_gt_as_hint) and (not enable_sdpo) and torch.any(invalid_hint_mask):
                    with timer("teacher_on_policy_fallback_prompt_tokenize", timing_raw):
                        fallback_teacher_gen, trajectory_missing = self._construct_teacher_fallback_prompts(
                            batch,
                            repeat_k=K,
                            student_output=student_output,
                        )

                    fallback_prompts = fallback_teacher_gen.batch["input_ids"][on_indices]
                    fallback_prompt_mask = fallback_teacher_gen.batch["attention_mask"][on_indices]
                    fallback_prompt_position_ids = fallback_teacher_gen.batch["position_ids"][on_indices]

                    teacher_prompts = teacher_prompts.clone()
                    teacher_prompt_mask = teacher_prompt_mask.clone()
                    teacher_prompt_position_ids = teacher_prompt_position_ids.clone()
                    teacher_prompts[invalid_hint_mask] = fallback_prompts[invalid_hint_mask]
                    teacher_prompt_mask[invalid_hint_mask] = fallback_prompt_mask[invalid_hint_mask]
                    teacher_prompt_position_ids[invalid_hint_mask] = fallback_prompt_position_ids[invalid_hint_mask]
                    on_policy_use_fallback_prompt = True
                    metrics["opsd/on_policy_trajectory_fallback_count"] = float(invalid_hint_mask.sum().item())
                    metrics["opsd/on_policy_fallback_missing_trajectory_count"] = float(trajectory_missing)
                    metrics["opsd/on_policy_gt_fallback_count"] = float(trajectory_missing)
                else:
                    metrics["opsd/on_policy_trajectory_fallback_count"] = 0.0
                    metrics["opsd/on_policy_fallback_missing_trajectory_count"] = 0.0
                    metrics["opsd/on_policy_gt_fallback_count"] = 0.0
            else:
                teacher_prompts = teacher_gen.batch["input_ids"]
                teacher_prompt_mask = teacher_gen.batch["attention_mask"]
                teacher_prompt_position_ids = teacher_gen.batch["position_ids"]
                metrics["opsd/on_policy_trajectory_fallback_count"] = 0.0
                metrics["opsd/on_policy_fallback_missing_trajectory_count"] = 0.0
                metrics["opsd/on_policy_gt_fallback_count"] = 0.0

            student_resp_for_teacher = (
                student_responses_rep if teacher_output is not None else student_output.batch["responses"]
            )
            student_resp_mask_for_teacher = (
                student_response_mask_rep if teacher_output is not None else student_output.batch["response_mask"]
            )

            on_teacher_input_ids, on_teacher_attention_mask, on_teacher_position_ids = _concat_prompt_and_response(
                prompt_ids=teacher_prompts,
                prompt_attention_mask=teacher_prompt_mask,
                prompt_position_ids=teacher_prompt_position_ids,
                response_ids=student_resp_for_teacher,
                response_mask=student_resp_mask_for_teacher,
            )
            on_teacher_responses = student_resp_for_teacher
            on_teacher_response_mask = student_resp_mask_for_teacher

            if student_distill_response_mask is not None:
                on_student_distill_response_mask = student_distill_response_mask[on_indices]
            else:
                if on_policy_use_fallback_prompt:
                    on_student_distill_response_mask = student_response_mask_rep
                else:
                    on_student_distill_response_mask = (
                        student_response_mask_rep
                        * on_positive_mask[on_indices].unsqueeze(-1).to(student_response_mask_rep.dtype)
                    )

            on_policy_tensors = {
                "on_student_input_ids": student_input_ids_rep,
                "on_student_attention_mask": student_attention_mask_rep,
                "on_student_position_ids": student_position_ids_rep,
                "on_student_responses": student_responses_rep,
                "on_student_response_mask": student_response_mask_rep,
                "on_teacher_input_ids": on_teacher_input_ids,
                "on_teacher_attention_mask": on_teacher_attention_mask,
                "on_teacher_position_ids": on_teacher_position_ids,
                "on_teacher_responses": on_teacher_responses,
                "on_teacher_response_mask": on_teacher_response_mask,
                "on_student_distill_response_mask": on_student_distill_response_mask,
            }
            if student_old_lp is not None and student_outcome_advantages is not None:
                on_policy_tensors["on_student_old_log_probs"] = student_old_lp[on_indices]
                on_policy_tensors["on_student_outcome_advantages"] = student_outcome_advantages[on_indices]
                on_policy_tensors["on_student_outcome_response_mask"] = student_outcome_response_mask[on_indices]

            if num_valid_on_policy > 0:
                nonempty_on_mask = (
                    (student_attention_mask_rep.sum(dim=-1) > 0)
                    & (student_response_mask_rep.sum(dim=-1) > 0)
                    & (on_teacher_attention_mask.sum(dim=-1) > 0)
                    & (on_teacher_response_mask.sum(dim=-1) > 0)
                )
                if not torch.all(nonempty_on_mask):
                    on_indices = on_indices[nonempty_on_mask]
                    on_question_indices = on_question_indices[nonempty_on_mask]
                    on_positive_mask = on_positive_mask[on_indices]
                    num_valid_on_policy = int(nonempty_on_mask.sum().item())
                    on_policy_tensors = {
                        key: tensor[nonempty_on_mask] for key, tensor in on_policy_tensors.items()
                    }
                else:
                    on_positive_mask = on_positive_mask[on_indices]

            on_teacher_attention_mask = on_policy_tensors["on_teacher_attention_mask"]
            valid_indices = on_indices[on_positive_mask]
            valid_question_indices = on_question_indices[on_positive_mask]

            off_policy_tensors = {}
            num_valid = 0
            if not skip_off_policy:
                if self_distill_negative_off_policy:
                    valid_indices = torch.arange(
                        batch_size,
                        device=student_output.batch["responses"].device,
                        dtype=torch.long,
                    )
                    valid_question_indices = valid_indices
                num_valid = len(valid_indices)

                if num_valid > 0:
                    teacher_correct_responses = teacher_output.batch["responses"][valid_indices]
                    teacher_correct_resp_mask = teacher_output.batch["response_mask"][valid_indices]

                    student_prompts_valid = student_output.batch["prompts"][valid_question_indices]
                    student_prompt_len = student_prompts_valid.shape[1]
                    student_prompt_mask_valid = student_output.batch["attention_mask"][valid_question_indices, :student_prompt_len]
                    student_prompt_position_ids_valid = student_output.batch["position_ids"][
                        valid_question_indices, ..., :student_prompt_len
                    ]
                    off_student_input_ids, off_student_attention_mask, off_student_position_ids = _concat_prompt_and_response(
                        prompt_ids=student_prompts_valid,
                        prompt_attention_mask=student_prompt_mask_valid,
                        prompt_position_ids=student_prompt_position_ids_valid,
                        response_ids=teacher_correct_responses,
                        response_mask=teacher_correct_resp_mask,
                    )

                    off_policy_tensors = {
                        "off_student_input_ids": off_student_input_ids,
                        "off_student_attention_mask": off_student_attention_mask,
                        "off_student_position_ids": off_student_position_ids,
                        "off_student_responses": teacher_correct_responses,
                        "off_student_response_mask": teacher_correct_resp_mask,
                        "off_teacher_input_ids": teacher_output.batch["input_ids"][valid_indices],
                        "off_teacher_attention_mask": teacher_output.batch["attention_mask"][valid_indices],
                        "off_teacher_position_ids": teacher_output.batch["position_ids"][valid_indices],
                        "off_teacher_responses": teacher_correct_responses,
                        "off_teacher_response_mask": teacher_correct_resp_mask,
                    }
            else:
                print("[OPSD] Skipping off-policy batch assembly (alpha=1.0, on-policy only)")

            metrics["opsd/num_valid_hints"] = num_valid
            metrics["opsd/num_total_hints"] = batch_size * K

        # Pad all tensors to same batch size (nav: B*K, on: B*K, off: num_valid)
        # We'll handle different sizes in the actor by processing each group separately
        # For now, pack them all into a single DataProto with separate key prefixes

        # Find the max sequence length across all groups to pad
        all_tensors = {}
        all_tensors.update(nav_tensors)
        all_tensors.update(on_policy_tensors)
        all_tensors.update(off_policy_tensors)

        # We need all tensors to have the same batch size for DataProto
        # Strategy: pad smaller batches to B*K (the largest group)
        target_bs = batch_size * K

        def pad_batch_to_size(tensor_dict, target_size, prefix=""):
            """Pad a dict of tensors along batch dim to target_size."""
            result = {}
            for key, tensor in tensor_dict.items():
                if tensor.shape[0] < target_size:
                    pad_size = target_size - tensor.shape[0]
                    pad_shape = list(tensor.shape)
                    pad_shape[0] = pad_size
                    padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
                    result[key] = torch.cat([tensor, padding], dim=0)
                elif tensor.shape[0] > target_size:
                    result[key] = tensor[:target_size]
                else:
                    result[key] = tensor
            return result

        # Pad all smaller groups to B*K so DataProto sees a consistent batch
        # dimension. The actor uses num_valid_on_policy/num_valid_off_policy
        # to ignore the padded tail.
        if on_policy_tensors:
            on_policy_tensors = pad_batch_to_size(on_policy_tensors, target_bs)
        if off_policy_tensors:
            off_policy_tensors = pad_batch_to_size(off_policy_tensors, target_bs)

        # Rebuild all_tensors
        all_tensors = {}
        all_tensors.update(nav_tensors)
        all_tensors.update(on_policy_tensors)
        if off_policy_tensors:
            all_tensors.update(off_policy_tensors)

        # --- Multi-modal data for OPSD forward passes ---
        # All three roles (Nav, Student, Teacher) may need pixel_values because
        # the question text contains <image> tags. We pass multi_modal_data from
        # the original batch so that _process_multi_modal_inputs in fsdp_workers
        # can construct multi_modal_inputs on the worker side.
        #
        # Layout: on-policy Student/Teacher share the same image (repeated K times).
        # Nav and off-policy similarly reference the same original images.
        original_mmd = batch.non_tensor_batch.get("multi_modal_data")
        original_uid = batch.non_tensor_batch.get("uid")

        def _repeat_and_pad(arr, repeat_k, target_size, pad_value):
            """Repeat each element K times, then pad to target_size."""
            if arr is None:
                return np.array([pad_value] * target_size, dtype=object)
            repeated = np.repeat(arr, repeat_k)
            if len(repeated) < target_size:
                pad = np.array([pad_value] * (target_size - len(repeated)), dtype=object)
                repeated = np.concatenate([repeated, pad])
            return repeated[:target_size]

        nav_mmd = _repeat_and_pad(original_mmd, K, target_bs, {})
        nav_uid = _repeat_and_pad(original_uid, K, target_bs, "__nav_pad__")
        if num_valid_on_policy > 0:
            on_question_indices_np = on_question_indices.detach().cpu().numpy()
            on_mmd = original_mmd[on_question_indices_np] if original_mmd is not None else None
            on_uid = original_uid[on_question_indices_np] if original_uid is not None else None
        else:
            on_mmd = None
            on_uid = None
        on_student_mmd = _repeat_and_pad(on_mmd, 1, target_bs, {})
        on_student_uid = _repeat_and_pad(on_uid, 1, target_bs, "__on_pad__")
        if num_valid > 0:
            valid_question_indices_np = valid_question_indices.detach().cpu().numpy()
            off_mmd = original_mmd[valid_question_indices_np] if original_mmd is not None else None
            off_uid = original_uid[valid_question_indices_np] if original_uid is not None else None
        else:
            off_mmd = None
            off_uid = None
        off_mmd = _repeat_and_pad(off_mmd, 1, target_bs, {})
        off_uid = _repeat_and_pad(off_uid, 1, target_bs, "__off_pad__")

        non_tensors = {
            "nav_multi_modal_data": nav_mmd,
            "nav_uid": nav_uid,
            "on_multi_modal_data": on_student_mmd,
            "on_uid": on_student_uid,
            "off_multi_modal_data": off_mmd,
            "off_uid": off_uid,
        }

        combined_batch = DataProto.from_dict(
            tensors=all_tensors,
            non_tensors=non_tensors,
            meta_info={
                "temperature": self.config.worker.rollout.temperature,
                "loss_type": "opsd",
                "opsd_config": {
                    "alpha": self.opsd_config.alpha,
                    "lambda_nav": self.opsd_config.lambda_nav,
                    "distillation_loss_type": self.opsd_config.distillation_loss_type,
                    "opsd_kl_penalty": self.opsd_config.opsd_kl_penalty,
                    "distillation_topk": self.opsd_config.distillation_topk,
                    "distillation_topk_source": self.opsd_config.distillation_topk_source,
                    "distillation_add_tail": self.opsd_config.distillation_add_tail,
                    "distill_token_selection": self.opsd_config.distill_token_selection,
                    "distill_token_keep_ratio": self.opsd_config.distill_token_keep_ratio,
                    "dynamic_sample_hint": self.opsd_config.dynamic_sample_hint,
                    "teacher_rollout_n": self.opsd_config.teacher_rollout_n,
                    "outcome_ppo_coef": self.opsd_config.outcome_ppo_coef,
                    "outcome_ppo_incorrect_only": self.opsd_config.outcome_ppo_incorrect_only,
                    "nav_clip_ratio_low": self.opsd_config.nav_clip_ratio_low,
                    "nav_clip_ratio_high": self.opsd_config.nav_clip_ratio_high,
                    "nav_clip_ratio_dual": self.opsd_config.nav_clip_ratio_dual,
                    "self_distill_negative_off_policy": self.opsd_config.self_distill_negative_off_policy,
                    "self_distill_negative_off_policy_coef": self.opsd_config.self_distill_negative_off_policy_coef,
                    "self_distill_mask_prefix_tokens": self.opsd_config.self_distill_mask_prefix_tokens,
                },
                "num_valid_off_policy": num_valid,
                "num_valid_on_policy": num_valid_on_policy,
                "global_token_num": torch.sum(
                    on_teacher_attention_mask, dim=-1
                ).tolist(),
                # Pixel config for _process_multi_modal_inputs
                "image_min_pixels": self.config.data.image_min_pixels,
                "image_max_pixels": self.config.data.image_max_pixels,
                "video_min_pixels": self.config.data.video_min_pixels,
                "video_max_pixels": self.config.data.video_max_pixels,
                "video_fps": self.config.data.video_fps,
                "video_max_frames": self.config.data.video_max_frames,
            },
        )

        # Timing metrics
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

        return combined_batch, batch, metrics

    def fit(self):
        """The OPSD training loop.

        Per step:
          1. Student rollout, Navigator hint generation, Teacher rollout
          2. Evaluate Teacher trajectories → binary rewards
          3. Navigator GRPO advantage
          4. Joint update: Navigator GRPO + OPSD distillation
        """
        self.logger = self._create_logger()
        self.global_step = 0
        main_tqdm = tqdm(range(self.training_steps), desc="OPSD Training", position=0)
        val_metrics = None

        # Load checkpoint
        self._load_checkpoint()
        main_tqdm.update(self.global_step)

        # Validation before training
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)

        while self.global_step < self.training_steps:
            self.global_step += 1

            metrics, timing_raw = {}, {}
            with timer("step", timing_raw):
                # === GENERATION PHASE ===
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    combined_batch, original_batch, gen_metrics = self._make_opsd_batch(metrics)
                    self.actor_rollout_ref_wg.release_rollout_engine()
                    metrics.update(gen_metrics)

                # === JOINT UPDATE ===
                with timer("update_actor", timing_raw):
                    actor_output = self.actor_rollout_ref_wg.update_actor(combined_batch)

                actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                metrics.update(actor_metrics)

                # === VALIDATION ===
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()
                    metrics.update(val_metrics)

                # === CHECKPOINT ===
                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            # Collect metrics
            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_timing_metrics(batch=original_batch, timing_raw=timing_raw))

            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()

        # Final validation
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics:\n{convert_dict_to_str(unflatten_dict(val_metrics))}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()

    def _create_logger(self):
        """Create logger instance."""
        from ..utils.logger import Tracker
        return Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())


class RaySimpleGRPOOPSDTrainer(RayOPSDTrainer):
    """Separate minimal GRPO+OPSD trainer without Navigator logic."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        if self.config.algorithm.adv_estimator != AdvantageEstimator.GRPO:
            raise ValueError("Simple GRPO+OPSD mode requires algorithm.adv_estimator=grpo.")
        if self.config.worker.rollout.n <= 1:
            raise ValueError("Simple GRPO+OPSD mode requires worker.rollout.n > 1.")
        print("[OPSD] Mode: Simple GRPO+RLSD (no Navigator)")
        print(f"[OPSD]   - rollout.n: {self.config.worker.rollout.n}")
        print(f"[OPSD]   - stgca_lambda: {self.opsd_config.stgca_lambda}")
        print(f"[OPSD]   - stgca_lambda_warmup_steps: {self.opsd_config.stgca_lambda_warmup_steps}")
        print(f"[OPSD]   - stgca_thought_only: {self.opsd_config.stgca_thought_only}")
        print(f"[OPSD]   - stgca_teacher_sync_interval: {self.opsd_config.stgca_teacher_sync_interval}")
        print(f"[OPSD]   - stgca_negative_only: {self.opsd_config.stgca_negative_only}")
        print(f"[OPSD]   - stgca_lambda_decay_steps: {self.opsd_config.stgca_lambda_decay_steps}")
        print(
            f"[OPSD]   - stgca_reweight_clip_range: "
            f"{self.opsd_config.stgca_reweight_clip_range}"
        )

    def _build_thought_span_mask(
        self, responses: torch.Tensor, response_mask: torch.Tensor
    ) -> torch.Tensor:
        """Build a mask that is 1 only for tokens inside <thought>...</thought> spans."""
        thought_start_ids = self.tokenizer.encode("<thought>", add_special_tokens=False)
        thought_end_ids = self.tokenizer.encode("</thought>", add_special_tokens=False)
        start_len = len(thought_start_ids)
        end_len = len(thought_end_ids)

        batch_size, seq_len = responses.shape
        mask = torch.zeros_like(response_mask)

        for i in range(batch_size):
            tokens = responses[i].tolist()
            in_thought = False
            t = 0
            while t < seq_len:
                if not in_thought:
                    if t + start_len <= seq_len and tokens[t : t + start_len] == thought_start_ids:
                        in_thought = True
                        t += start_len
                        continue
                else:
                    if t + end_len <= seq_len and tokens[t : t + end_len] == thought_end_ids:
                        in_thought = False
                        t += end_len
                        continue
                    mask[i, t] = 1.0
                t += 1

        return mask * response_mask

    def _augment_batch_with_grpo_opsd(self, batch: DataProto, metrics: dict[str, Any], timing_raw: dict[str, Any]) -> DataProto:
        """Attach teacher on-policy tensors aligned with the current GRPO batch."""
        with timer("grpo_opsd_teacher_prompt", timing_raw):
            hints = self._get_self_distill_hints(batch)
            num_trajectory_hints = sum(1 for hint in hints if hint and str(hint[0]).strip() != str(batch.non_tensor_batch["ground_truth"][0]).strip())
            empty_gt_answers = np.array([""] * len(batch), dtype=object)
            teacher_gen = construct_teacher_prompts(
                questions=self._get_teacher_questions(batch),
                hints=hints,
                gt_answers=empty_gt_answers,
                p_drop=0.0,
                tokenizer=self.tokenizer,
                processor=self.processor,
                template_path=self._teacher_template_path,
                max_prompt_length=self.config.data.max_prompt_length,
                multi_modal_data=batch.non_tensor_batch.get("multi_modal_data"),
                image_min_pixels=self.config.data.image_min_pixels,
                image_max_pixels=self.config.data.image_max_pixels,
                video_min_pixels=self.config.data.video_min_pixels,
                video_max_pixels=self.config.data.video_max_pixels,
                video_fps=self.config.data.video_fps,
                video_max_frames=self.config.data.video_max_frames,
                apply_chat_template_kwargs=self.config.data.apply_chat_template_kwargs,
            )

        teacher_input_ids, teacher_attention_mask, teacher_position_ids = _concat_prompt_and_response(
            prompt_ids=teacher_gen.batch["input_ids"],
            prompt_attention_mask=teacher_gen.batch["attention_mask"],
            prompt_position_ids=teacher_gen.batch["position_ids"],
            response_ids=batch.batch["responses"],
            response_mask=batch.batch["response_mask"],
        )

        batch.batch["grpo_opsd_teacher_input_ids"] = teacher_input_ids
        batch.batch["grpo_opsd_teacher_attention_mask"] = teacher_attention_mask
        batch.batch["grpo_opsd_teacher_position_ids"] = teacher_position_ids

        if self.opsd_config.stgca_thought_only:
            batch.batch["stgca_thought_mask"] = self._build_thought_span_mask(
                batch.batch["responses"], batch.batch["response_mask"]
            )
        warmup_steps = self.opsd_config.stgca_lambda_warmup_steps
        decay_steps = self.opsd_config.stgca_lambda_decay_steps
        target_lambda = self.opsd_config.stgca_lambda
        if warmup_steps > 0 and self.global_step < warmup_steps:
            effective_lambda = target_lambda * (self.global_step / warmup_steps)
        elif decay_steps > 0 and self.global_step >= warmup_steps:
            decay_progress = (self.global_step - warmup_steps) / decay_steps
            effective_lambda = target_lambda * max(1.0 - decay_progress, 0.0)
        else:
            effective_lambda = target_lambda

        batch.meta_info["loss_type"] = "grpo_opsd"
        batch.meta_info["opsd_config"] = {
            "stgca_lambda": effective_lambda,
            "stgca_reweight_clip_range": self.opsd_config.stgca_reweight_clip_range,
            "stgca_negative_only": self.opsd_config.stgca_negative_only,
            "stgca_thought_only": self.opsd_config.stgca_thought_only,
            "stgca_teacher_sync_interval": self.opsd_config.stgca_teacher_sync_interval,
            "global_step": self.global_step,
        }
        metrics["grpo_opsd/stgca_effective_lambda"] = effective_lambda
        metrics["grpo_opsd/teacher_context_from_trajectory"] = float(
            sum(
                1
                for idx, hint in enumerate(hints)
                if hint and str(hint[0]).strip() and str(hint[0]).strip() != str(batch.non_tensor_batch["ground_truth"][idx]).strip()
            )
        )
        return batch

    def fit(self):
        self.logger = self._create_logger()
        self.global_step = 0
        main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)
        val_metrics: Optional[dict[str, Any]] = None

        self._load_checkpoint()
        main_tqdm.update(self.global_step)

        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)
        while self.global_step < self.training_steps:
            self.global_step += 1

            metrics, timing_raw = {}, {}
            with timer("step", timing_raw):
                reward_futures = [] if self.config.algorithm.pipeline_reward else None
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    batch = self._make_batch_data(metrics=metrics, reward_futures=reward_futures)
                    self.actor_rollout_ref_wg.release_rollout_engine()

                global_idx = self._balance_batch(batch, metrics=metrics)
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                if "token_level_scores" not in batch.batch and not (reward_futures and len(reward_futures) > 0):
                    with timer("reward", timing_raw):
                        reward_batch = batch.select(
                            batch_keys=["responses", "response_mask"],
                            non_tensor_batch_keys=[k for k in batch.non_tensor_batch if k != "multi_modal_data"],
                        )
                        reward_ref = self.reward_fn.compute_reward.remote(reward_batch)

                with timer("old", timing_raw):
                    old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                    batch = batch.union(old_log_probs)

                if self.use_reference_policy:
                    with timer("ref", timing_raw):
                        ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)
                        batch = batch.union(ref_log_probs)

                with timer("adv", timing_raw):
                    if reward_futures and len(reward_futures) > 0:
                        with timer("reward", timing_raw):
                            all_reward_tensors = []
                            all_reward_metrics = defaultdict(list)
                            for ref, _mini_size in reward_futures:
                                reward_tensor_i, reward_metrics_i = ray.get(ref)
                                all_reward_tensors.append(reward_tensor_i)
                                for k, v in reward_metrics_i.items():
                                    all_reward_metrics[k].extend(v)
                            reward_tensor_all = torch.cat(all_reward_tensors, dim=0)
                            target_size = self.config.data.rollout_batch_size * self.config.worker.rollout.n
                            reward_tensor_all = reward_tensor_all[:target_size]
                            reward_tensor_all = reward_tensor_all[global_idx]
                            batch.batch["token_level_scores"] = reward_tensor_all
                            metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_reward_metrics).items()})
                    elif "token_level_scores" not in batch.batch:
                        reward_tensor, reward_metrics = ray.get(reward_ref)
                        batch.batch["token_level_scores"] = reward_tensor
                        metrics.update({f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()})

                    if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                        batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                    )

                batch = self._augment_batch_with_grpo_opsd(batch, metrics, timing_raw)

                with timer("update_actor", timing_raw):
                    actor_output = self.actor_rollout_ref_wg.update_actor(batch)
                actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                metrics.update(actor_metrics)

                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=batch, use_critic=False))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()

        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
