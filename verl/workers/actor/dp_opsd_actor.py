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
OPSD Actor: Extends DataParallelPPOActor with joint Navigator GRPO + Reasoner OPSD training.

Key additions:
  - update_policy_opsd(): Joint training loop for Navigator GRPO + OPSD distillation
  - Teacher forward passes in torch.no_grad() — gradients only flow through Student
"""

from collections import defaultdict
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from ray.experimental.tqdm_ray import tqdm

from ...protocol import DataProto, batch_collate
from ...trainer.core_algos import average_loss, compute_kl, compute_policy_loss
from ...trainer.opsd_algos import (
    compute_opsd_jsd_loss,
    compute_opsd_topk_jsd_loss,
    compute_opsd_topk_kl_loss,
)
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.seqlen_balancing import prepare_dynamic_batch
from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from .dp_actor import DataParallelPPOActor

try:
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
except ImportError:
    pass


__all__ = ["DataParallelOPSDActor"]


class DataParallelOPSDActor(DataParallelPPOActor):
    """OPSD Actor with joint Navigator GRPO + Reasoner self-distillation training."""

    def __init__(
        self,
        config,
        actor_module,
        actor_optimizer=None,
        teacher_module: Optional[torch.nn.Module] = None,
    ):
        super().__init__(config=config, actor_module=actor_module, actor_optimizer=actor_optimizer)
        self.teacher_module = teacher_module

    def _get_teacher_module(self):
        return self.teacher_module if self.teacher_module is not None else self.actor_module

    def _sync_teacher_from_actor(self):
        """Copy actor weights to teacher module for periodic teacher refresh."""
        if self.teacher_module is None or self.teacher_module is self.actor_module:
            return
        with torch.no_grad():
            for t_param, a_param in zip(self.teacher_module.parameters(), self.actor_module.parameters()):
                t_param.data.copy_(a_param.data)

    @staticmethod
    def _masked_entropy_from_logits(
        logits: torch.Tensor,
        response_mask: torch.Tensor,
        loss_avg_mode: str,
    ) -> torch.Tensor:
        """Compute masked token entropy from full logits."""
        token_entropy = DataParallelOPSDActor._token_entropy_from_logits(logits)
        return average_loss(token_entropy, response_mask, mode=loss_avg_mode)

    @staticmethod
    def _token_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        return -(probs * log_probs).sum(dim=-1)

    @staticmethod
    def _token_jsd_from_logits(
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        eps: float = 1e-10,
    ) -> torch.Tensor:
        teacher_log_probs = F.log_softmax(teacher_logits.detach(), dim=-1)
        teacher_probs = teacher_log_probs.exp()
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        student_probs = student_log_probs.exp()
        mix_probs = 0.5 * (teacher_probs + student_probs)
        mix_log_probs = mix_probs.clamp_min(eps).log()
        return 0.5 * (
            (teacher_probs * (teacher_log_probs - mix_log_probs)).sum(dim=-1)
            + (student_probs * (student_log_probs - mix_log_probs)).sum(dim=-1)
        )

    @staticmethod
    def _select_token_mask_from_scores(
        scores: torch.Tensor,
        response_mask: torch.Tensor,
        keep_ratio: float,
        *,
        largest: bool,
    ) -> torch.Tensor:
        if keep_ratio >= 1.0:
            return response_mask

        selected_mask = torch.zeros_like(response_mask)
        batch_size = response_mask.shape[0]
        for i in range(batch_size):
            valid_idx = torch.nonzero(response_mask[i] > 0, as_tuple=False).squeeze(-1)
            if valid_idx.numel() == 0:
                continue
            keep_count = max(1, int(valid_idx.numel() * keep_ratio + 0.999999))
            valid_scores = scores[i, valid_idx]
            topk_local = torch.topk(valid_scores, k=keep_count, largest=largest).indices
            chosen_idx = valid_idx[topk_local]
            selected_mask[i, chosen_idx] = 1
        return selected_mask

    def _build_distill_token_mask(
        self,
        *,
        teacher_logits: Optional[torch.Tensor],
        student_logits: Optional[torch.Tensor],
        base_mask: torch.Tensor,
        selection_mode: str,
        keep_ratio: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if selection_mode == "all" or keep_ratio >= 1.0:
            return base_mask, {
                "selected_token_ratio": 1.0,
                "selected_token_count": float(base_mask.sum().item()),
            }

        if teacher_logits is None or student_logits is None:
            raise ValueError(
                "Token-selection distillation requires teacher/student logits. "
                "Use distillation_loss_type=jsd or distillation_topk with logits-based distillation."
            )

        if selection_mode == "low_teacher_entropy":
            scores = self._token_entropy_from_logits(teacher_logits.detach())
            selected_mask = self._select_token_mask_from_scores(
                scores=scores,
                response_mask=base_mask,
                keep_ratio=keep_ratio,
                largest=False,
            )
        elif selection_mode == "high_teacher_student_gap":
            scores = self._token_jsd_from_logits(teacher_logits, student_logits).detach()
            selected_mask = self._select_token_mask_from_scores(
                scores=scores,
                response_mask=base_mask,
                keep_ratio=keep_ratio,
                largest=True,
            )
        else:
            raise ValueError(f"Unsupported distill token selection mode: {selection_mode}")

        selected_count = selected_mask.sum().item()
        total_count = base_mask.sum().item()
        selected_ratio = 0.0 if total_count <= 0 else float(selected_count / total_count)
        return selected_mask, {
            "selected_token_ratio": selected_ratio,
            "selected_token_count": float(selected_count),
        }

    @staticmethod
    def _select_topk_support(
        *,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        k: int,
        topk_source: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if topk_source == "teacher":
            source_log_softmax = F.log_softmax(teacher_logits.detach(), dim=-1)
            topk_log_probs, topk_indices = torch.topk(source_log_softmax, k=k, dim=-1)
            student_log_softmax = F.log_softmax(student_logits, dim=-1)
            student_selected_log_probs = student_log_softmax.gather(-1, topk_indices)
            return topk_log_probs.detach(), student_selected_log_probs, topk_indices

        if topk_source == "student":
            student_log_softmax = F.log_softmax(student_logits, dim=-1)
            student_selected_log_probs, topk_indices = torch.topk(student_log_softmax, k=k, dim=-1)
            teacher_log_softmax = F.log_softmax(teacher_logits.detach(), dim=-1)
            teacher_selected_log_probs = teacher_log_softmax.gather(-1, topk_indices)
            return teacher_selected_log_probs.detach(), student_selected_log_probs, topk_indices

        raise ValueError(
            f"Unsupported opsd.distillation_topk_source={topk_source}. "
            "Expected one of: teacher, student."
        )

    @staticmethod
    def _build_stgca_advantages(
        *,
        teacher_log_probs: torch.Tensor,
        student_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        lam: float,
        clip_range: float,
        negative_only: bool = False,
        thought_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        sign_adv = torch.sign(advantages)
        delta = (teacher_log_probs.detach() - student_log_probs.detach()) * response_mask
        weights = torch.exp(sign_adv * delta) * response_mask

        low_clip_mask = ((weights < (1.0 - clip_range)).to(response_mask.dtype)) * response_mask
        high_clip_mask = ((weights > (1.0 + clip_range)).to(response_mask.dtype)) * response_mask
        clipped_weights = torch.clamp(weights, min=1.0 - clip_range, max=1.0 + clip_range)
        reweight = ((1.0 - lam) + lam * clipped_weights) * response_mask

        if negative_only:
            seq_negative = (advantages.sum(dim=-1, keepdim=True) < 0).float()
            reweight = seq_negative * reweight + (1.0 - seq_negative) * response_mask

        if thought_mask is not None:
            # Only reweight tokens inside <thought>...</thought> spans.
            # Tokens outside (format tags, answer, etc.) keep reweight=1 (pure GRPO).
            reweight = thought_mask * reweight + (1.0 - thought_mask) * response_mask

        stgca_advantages = advantages * reweight.detach()

        neg_mask = response_mask * (advantages < 0).float()
        neg_token_count = neg_mask.sum().clamp(min=1.0)
        metrics = {
            "grpo_opsd/stgca_delta_mean": VF.masked_mean(delta.detach(), response_mask).item(),
            "grpo_opsd/stgca_weight_mean": VF.masked_mean(weights.detach(), response_mask).item(),
            "grpo_opsd/stgca_clipped_weight_mean": VF.masked_mean(clipped_weights.detach(), response_mask).item(),
            "grpo_opsd/stgca_clip_low_ratio": VF.masked_mean(low_clip_mask.detach(), response_mask).item(),
            "grpo_opsd/stgca_clip_high_ratio": VF.masked_mean(high_clip_mask.detach(), response_mask).item(),
            "grpo_opsd/stgca_adv_mean": VF.masked_mean(stgca_advantages.detach(), response_mask).item(),
            "grpo_opsd/stgca_adv_abs_mean": VF.masked_mean(
                stgca_advantages.detach().abs(), response_mask
            ).item(),
        }
        if negative_only:
            metrics["grpo_opsd/stgca_negative_only"] = 1.0
            metrics["grpo_opsd/stgca_neg_seq_ratio"] = (
                (neg_mask.sum(dim=-1) > 0).float().mean().item()
            )
            metrics["grpo_opsd/stgca_neg_delta_mean"] = (
                (delta.detach() * neg_mask).sum() / neg_token_count
            ).item()
        if thought_mask is not None:
            thought_token_count = thought_mask.sum().item()
            total_token_count = response_mask.sum().item()
            metrics["grpo_opsd/stgca_thought_only"] = 1.0
            metrics["grpo_opsd/stgca_thought_token_ratio"] = (
                thought_token_count / max(total_token_count, 1.0)
            )
        return stgca_advantages, metrics

    def _forward_micro_batch_with_module(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        module: Optional[torch.nn.Module] = None,
    ) -> torch.Tensor:
        """Equivalent to _forward_micro_batch, but allows selecting a different module."""
        forward_module = module if module is not None else self.actor_module
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:
            position_ids = position_ids.transpose(0, 1)

        if "multi_modal_inputs" in micro_batch:
            multi_modal_inputs = batch_collate(micro_batch["multi_modal_inputs"])
            multi_modal_inputs = {key: torch.cat(value, dim=0) for key, value in multi_modal_inputs.items()}
        else:
            multi_modal_inputs = {}

        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)

            if self.config.ulysses_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_size
                )
            else:
                pad_size = 0

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)
            output = forward_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits_rmpad = output.logits.squeeze(0)
            logits_rmpad.div_(temperature)
            log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

            if self.config.ulysses_size > 1:
                log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
            return full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]

        output = forward_module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **multi_modal_inputs,
            use_cache=False,
        )
        logits = output.logits
        logits.div_(temperature)
        return self.log_probs_from_logits(logits=logits[:, -response_length - 1 : -1], labels=responses)

    def _forward_micro_batch_logits(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        module: Optional[torch.nn.Module] = None,
    ) -> torch.Tensor:
        """Return sliced logits aligned with `responses` for top-k distillation."""
        forward_module = module if module is not None else self.actor_module
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:
            position_ids = position_ids.transpose(0, 1)

        if "multi_modal_inputs" in micro_batch:
            multi_modal_inputs = batch_collate(micro_batch["multi_modal_inputs"])
            multi_modal_inputs = {key: torch.cat(value, dim=0) for key, value in multi_modal_inputs.items()}
        else:
            multi_modal_inputs = {}

        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            if self.config.ulysses_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_size
                )
            else:
                pad_size = 0

            output = forward_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits_rmpad = output.logits.squeeze(0)
            logits_rmpad.div_(temperature)

            if self.config.ulysses_size > 1:
                logits_rmpad = gather_outputs_and_unpad(
                    logits_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                )

            full_logits = pad_input(hidden_states=logits_rmpad, indices=indices, batch=batch_size, seqlen=seqlen)
            logits = full_logits[:, -response_length - 1 : -1, :]
        else:
            output = forward_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]

        return logits

    def update_policy_grpo_opsd(self, data: DataProto) -> dict[str, Any]:
        """Joint update for simple GRPO+RLSD mode.

        This path keeps the standard PPO/GRPO policy loss and injects Teacher
        guidance through token-level advantage reweighting on the same sampled
        responses.
        """
        self.actor_module.train()
        teacher_module = self._get_teacher_module()
        if teacher_module is not self.actor_module:
            teacher_module.eval()

        temperature = data.meta_info["temperature"]
        opsd_config = data.meta_info["opsd_config"]
        stgca_lambda = opsd_config.get("stgca_lambda", 0.5)
        stgca_clip_range = opsd_config.get("stgca_reweight_clip_range")
        if stgca_clip_range is None:
            stgca_clip_range = float(self.config.clip_ratio_low)
        stgca_negative_only = opsd_config.get("stgca_negative_only", False)
        stgca_thought_only = opsd_config.get("stgca_thought_only", False)

        select_keys = [
            "input_ids",
            "attention_mask",
            "position_ids",
            "responses",
            "response_mask",
            "old_log_probs",
            "advantages",
            "grpo_opsd_teacher_input_ids",
            "grpo_opsd_teacher_attention_mask",
            "grpo_opsd_teacher_position_ids",
        ]
        if "ref_log_probs" in data.batch:
            select_keys.append("ref_log_probs")
        if "stgca_thought_mask" in data.batch:
            select_keys.append("stgca_thought_mask")

        mini_batches = data.select(select_keys, ["multi_modal_inputs"]).split(self.config.global_batch_size_per_device)

        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            if self.rank == 0:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=1)

            for mini_batch in mini_batches:
                total_response_tokens = torch.sum(mini_batch.batch["response_mask"])
                dist.all_reduce(total_response_tokens, op=dist.ReduceOp.SUM)

                if self.config.dynamic_batching:
                    if self.config.max_token_len_per_gpu is not None:
                        max_token_len = self.config.max_token_len_per_gpu
                    else:
                        max_input_len = max(
                            mini_batch.batch["input_ids"].size(-1),
                            mini_batch.batch["grpo_opsd_teacher_input_ids"].size(-1),
                        )
                        max_token_len = self.config.micro_batch_size_per_device_for_update * max_input_len
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)

                if self.rank == 0:
                    micro_batches = tqdm(micro_batches, desc="Update policy", position=2)

                for micro_batch in micro_batches:
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_probs = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    teacher_mb = {
                        "input_ids": micro_batch.batch["grpo_opsd_teacher_input_ids"],
                        "attention_mask": micro_batch.batch["grpo_opsd_teacher_attention_mask"],
                        "position_ids": micro_batch.batch["grpo_opsd_teacher_position_ids"],
                        "responses": micro_batch.batch["responses"],
                    }
                    if "multi_modal_inputs" in micro_batch.non_tensor_batch:
                        teacher_mb["multi_modal_inputs"] = micro_batch.non_tensor_batch["multi_modal_inputs"]

                    # Memory-efficient RLSD path: only compute per-token log probs
                    # without materializing the full (batch, seq, vocab) log_softmax tensor.
                    log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
                    teacher_was_training = teacher_module.training
                    teacher_module.eval()
                    with torch.no_grad():
                        teacher_lp = self._forward_micro_batch_with_module(
                            teacher_mb,
                            temperature=temperature,
                            module=teacher_module,
                        )
                    if teacher_was_training:
                        teacher_module.train()
                    thought_mask = model_inputs.get("stgca_thought_mask") if stgca_thought_only else None
                    advantages_for_pg, stgca_metrics = self._build_stgca_advantages(
                        teacher_log_probs=teacher_lp,
                        student_log_probs=log_probs,
                        advantages=advantages,
                        response_mask=response_mask,
                        lam=stgca_lambda,
                        clip_range=stgca_clip_range,
                        negative_only=stgca_negative_only,
                        thought_mask=thought_mask,
                    )

                    pg_loss, pg_metrics = compute_policy_loss(
                        old_log_probs=old_log_probs,
                        log_probs=log_probs,
                        advantages=advantages_for_pg,
                        response_mask=response_mask,
                        clip_ratio_low=self.config.clip_ratio_low,
                        clip_ratio_high=self.config.clip_ratio_high,
                        clip_ratio_dual=self.config.clip_ratio_dual,
                        loss_type=self.config.loss_type,
                        loss_avg_mode=self.config.loss_avg_mode,
                    )

                    base_kl_loss = None
                    if self.config.use_kl_loss and "ref_log_probs" in model_inputs:
                        ref_log_probs = model_inputs["ref_log_probs"]
                        kld = compute_kl(
                            log_probs=log_probs,
                            ref_log_probs=ref_log_probs,
                            kl_penalty=self.config.kl_penalty,
                        )
                        base_kl_loss = average_loss(kld, response_mask, mode=self.config.loss_avg_mode)
                        loss = pg_loss + base_kl_loss * self.config.kl_coef
                    else:
                        loss = pg_loss
                    loss = loss * torch.sum(response_mask) * self.world_size / total_response_tokens
                    loss.backward()

                    batch_metrics = {f"actor/{k}": v for k, v in pg_metrics.items()}
                    batch_metrics["actor/pg_loss"] = pg_loss.detach().item()
                    if base_kl_loss is not None:
                        batch_metrics["actor/kl_loss"] = base_kl_loss.detach().item()
                        batch_metrics["actor/kl_coef"] = self.config.kl_coef
                    batch_metrics.update(stgca_metrics)
                    append_to_dict(metrics, batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        # Periodic teacher sync
        sync_interval = opsd_config.get("stgca_teacher_sync_interval", 0)
        if sync_interval > 0:
            global_step = opsd_config.get("global_step", 0)
            if global_step > 0 and global_step % sync_interval == 0:
                self._sync_teacher_from_actor()
                append_to_dict(metrics, {"grpo_opsd/teacher_synced": 1.0})
            else:
                append_to_dict(metrics, {"grpo_opsd/teacher_synced": 0.0})

        return metrics

    def update_policy_opsd(self, data: DataProto) -> dict[str, Any]:
        """Joint update: Navigator GRPO loss + OPSD distillation loss.

        Data layout in data.batch:
            Navigator keys (prefixed with "nav_"):
                nav_input_ids, nav_attention_mask, nav_position_ids, nav_responses,
                nav_response_mask, nav_old_log_probs, nav_advantages

            OPSD on-policy keys (prefixed with "on_"):
                on_student_input_ids, on_student_attention_mask, on_student_position_ids,
                on_student_responses, on_student_response_mask,
                on_teacher_input_ids, on_teacher_attention_mask, on_teacher_position_ids,
                on_teacher_responses, on_teacher_response_mask

            OPSD off-policy keys (prefixed with "off_"):
                off_student_input_ids, off_student_attention_mask, off_student_position_ids,
                off_student_responses, off_student_response_mask,
                off_teacher_input_ids, off_teacher_attention_mask, off_teacher_position_ids,
                off_teacher_responses, off_teacher_response_mask

        Args:
            data: DataProto with all keys for joint training.

        Returns:
            metrics: dict of training metrics.
        """
        self.actor_module.train()
        teacher_module = self._get_teacher_module()
        if teacher_module is not self.actor_module:
            teacher_module.eval()

        temperature = data.meta_info["temperature"]
        opsd_config = data.meta_info["opsd_config"]
        alpha = opsd_config["alpha"]
        lambda_nav = opsd_config["lambda_nav"]
        distillation_loss_type = opsd_config.get("distillation_loss_type", "kl")
        kl_penalty = opsd_config["opsd_kl_penalty"]
        distillation_topk = opsd_config.get("distillation_topk")
        distillation_topk_source = opsd_config.get("distillation_topk_source", "teacher")
        distillation_add_tail = opsd_config.get("distillation_add_tail", True)
        distill_token_selection = opsd_config.get("distill_token_selection", "all")
        distill_token_keep_ratio = opsd_config.get("distill_token_keep_ratio", 1.0)
        outcome_ppo_coef = opsd_config.get("outcome_ppo_coef", 0.0)
        self_distill_negative_off_policy = opsd_config.get("self_distill_negative_off_policy", False)
        self_distill_negative_off_policy_coef = opsd_config.get("self_distill_negative_off_policy_coef", 0.0)

        skip_nav_grpo = (lambda_nav == 0)
        skip_off_policy = (alpha >= 1.0) and (not self_distill_negative_off_policy)

        if self_distill_negative_off_policy and distillation_loss_type != "jsd":
            raise ValueError(
                "opsd.self_distill_negative_off_policy requires distillation_loss_type=jsd. "
                "Negative KL is intentionally disallowed because it is numerically unstable."
            )

        metrics = defaultdict(list)

        for epoch in range(self.config.ppo_epochs):
            # === Navigator GRPO Data (skip when lambda_nav=0) ===
            if not skip_nav_grpo:
                nav_data = data.select(
                    batch_keys=[
                        "nav_input_ids", "nav_attention_mask", "nav_position_ids",
                        "nav_responses", "nav_response_mask", "nav_old_log_probs", "nav_advantages",
                    ],
                    non_tensor_batch_keys=["nav_multi_modal_inputs"],
                )

            # === OPSD On-Policy Data ===
            on_keys = [
                "on_student_input_ids", "on_student_attention_mask", "on_student_position_ids",
                "on_student_responses", "on_student_response_mask",
                "on_teacher_input_ids", "on_teacher_attention_mask", "on_teacher_position_ids",
                "on_teacher_responses", "on_teacher_response_mask",
                "on_student_old_log_probs", "on_student_outcome_advantages",
                "on_student_outcome_response_mask", "on_student_distill_response_mask",
            ]
            on_data = data.select(
                batch_keys=[k for k in on_keys if k in data.batch],
                non_tensor_batch_keys=["on_multi_modal_inputs"],
            )

            # === OPSD Off-Policy Data (skip when alpha=1.0) ===
            has_off_policy = False
            if not skip_off_policy:
                off_keys = [
                    "off_student_input_ids", "off_student_attention_mask", "off_student_position_ids",
                    "off_student_responses", "off_student_response_mask",
                    "off_teacher_input_ids", "off_teacher_attention_mask", "off_teacher_position_ids",
                    "off_teacher_responses", "off_teacher_response_mask",
                ]
                off_data = data.select(
                    batch_keys=[k for k in off_keys if k in data.batch],
                    non_tensor_batch_keys=["off_multi_modal_inputs"],
                )
                has_off_policy = data.meta_info.get("num_valid_off_policy", 0) > 0

            # Compute token counts for loss scaling.
            # Distillation should normalize over valid distillation tokens (H+) rather than all B*K tokens.
            device = (
                on_data.batch["on_student_response_mask"].device
                if "on_student_response_mask" in on_data.batch
                else torch.device("cuda", torch.cuda.current_device())
            )
            zero_tokens = torch.zeros((), device=device, dtype=torch.float32)
            on_distill_total_tokens = (
                torch.sum(on_data.batch["on_student_distill_response_mask"]).float()
                if "on_student_distill_response_mask" in on_data.batch
                else torch.sum(on_data.batch["on_student_response_mask"]).float()
                if "on_student_response_mask" in on_data.batch
                else zero_tokens.clone()
            )
            off_distill_total_tokens = (
                torch.sum(off_data.batch["off_teacher_response_mask"]).float()
                if (not skip_off_policy and "off_teacher_response_mask" in off_data.batch)
                else zero_tokens.clone()
            )
            nav_total_tokens = (
                torch.sum(nav_data.batch["nav_response_mask"]).float()
                if not skip_nav_grpo
                else zero_tokens.clone()
            )

            dist.all_reduce(on_distill_total_tokens, op=dist.ReduceOp.SUM)
            dist.all_reduce(off_distill_total_tokens, op=dist.ReduceOp.SUM)
            dist.all_reduce(nav_total_tokens, op=dist.ReduceOp.SUM)

            distill_total_tokens = (on_distill_total_tokens + off_distill_total_tokens).clamp(min=1.0)
            policy_total_tokens = (nav_total_tokens + on_distill_total_tokens).clamp(min=1.0)

            # Batch sizes
            if "on_student_input_ids" in on_data.batch and "on_teacher_input_ids" in on_data.batch:
                on_batch_size = min(
                    data.meta_info.get("num_valid_on_policy", 0),
                    on_data.batch["on_student_input_ids"].shape[0],
                    on_data.batch["on_teacher_input_ids"].shape[0],
                )
            else:
                on_batch_size = 0
            if has_off_policy and "off_student_input_ids" in off_data.batch and "off_teacher_input_ids" in off_data.batch:
                off_batch_size = min(
                    data.meta_info.get("num_valid_off_policy", 0),
                    off_data.batch["off_student_input_ids"].shape[0],
                    off_data.batch["off_teacher_input_ids"].shape[0],
                )
            else:
                off_batch_size = 0

            # Keep loop counts consistent across ranks to avoid collective desync.
            on_batch_size_t = torch.tensor(on_batch_size, device=device, dtype=torch.long)
            dist.all_reduce(on_batch_size_t, op=dist.ReduceOp.MIN)
            on_batch_size = int(on_batch_size_t.item())
            if not skip_off_policy:
                off_batch_size_t = torch.tensor(off_batch_size, device=device, dtype=torch.long)
                dist.all_reduce(off_batch_size_t, op=dist.ReduceOp.MIN)
                off_batch_size = int(off_batch_size_t.item())

            micro_bs = self.config.micro_batch_size_per_device_for_update
            dummy_on_rows = None
            if on_batch_size > 0:
                dummy_on_mask = (
                    (on_data.batch["on_teacher_attention_mask"].sum(dim=-1) > 0)
                    & (on_data.batch["on_student_attention_mask"].sum(dim=-1) > 0)
                    & (on_data.batch["on_student_response_mask"].sum(dim=-1) > 0)
                )
                if torch.any(dummy_on_mask):
                    dummy_on_rows = dummy_on_mask.nonzero(as_tuple=True)[0][:1]

            def _build_sync_dummy_on_pair():
                """Build a always-valid dummy teacher/student pair for collective-sync backward."""
                if "on_teacher_input_ids" not in on_data.batch or on_data.batch["on_teacher_input_ids"].shape[0] == 0:
                    return None, None
                row = dummy_on_rows if dummy_on_rows is not None else torch.tensor(
                    [0], device=on_data.batch["on_teacher_input_ids"].device, dtype=torch.long
                )
                dummy_teacher_mb = {
                    "input_ids": on_data.batch["on_teacher_input_ids"][row],
                    "attention_mask": on_data.batch["on_teacher_attention_mask"][row].clone(),
                    "position_ids": on_data.batch["on_teacher_position_ids"][row],
                    "responses": on_data.batch["on_student_responses"][row].clone(),
                }
                dummy_student_mb = {
                    "input_ids": on_data.batch["on_student_input_ids"][row],
                    "attention_mask": on_data.batch["on_student_attention_mask"][row].clone(),
                    "position_ids": on_data.batch["on_student_position_ids"][row],
                    "responses": on_data.batch["on_student_responses"][row].clone(),
                }
                if "on_multi_modal_inputs" in on_data.non_tensor_batch:
                    dummy_mm = on_data.non_tensor_batch["on_multi_modal_inputs"][row.cpu().numpy()]
                    dummy_teacher_mb["multi_modal_inputs"] = dummy_mm
                    dummy_student_mb["multi_modal_inputs"] = dummy_mm

                # Prevent padding-free unpad_input() from seeing all-zero attention.
                if torch.sum(dummy_teacher_mb["attention_mask"]) <= 0:
                    dummy_teacher_mb["attention_mask"][:, -1] = 1
                if torch.sum(dummy_student_mb["attention_mask"]) <= 0:
                    dummy_student_mb["attention_mask"][:, -1] = 1
                return dummy_teacher_mb, dummy_student_mb

            # --- Navigator GRPO micro-batches (skipped when lambda_nav=0) ---
            if not skip_nav_grpo:
                nav_batch_size = nav_data.batch["nav_input_ids"].shape[0]
                for start in range(0, nav_batch_size, micro_bs):
                    end = min(start + micro_bs, nav_batch_size)
                    nav_mb = {
                        "input_ids": nav_data.batch["nav_input_ids"][start:end],
                        "attention_mask": nav_data.batch["nav_attention_mask"][start:end],
                        "position_ids": nav_data.batch["nav_position_ids"][start:end],
                        "responses": nav_data.batch["nav_responses"][start:end],
                    }
                    if "nav_multi_modal_inputs" in nav_data.non_tensor_batch:
                        nav_mb["multi_modal_inputs"] = nav_data.non_tensor_batch["nav_multi_modal_inputs"][start:end]
                    nav_response_mask = nav_data.batch["nav_response_mask"][start:end]
                    nav_old_log_probs = nav_data.batch["nav_old_log_probs"][start:end]
                    nav_advantages = nav_data.batch["nav_advantages"][start:end]

                    nav_log_probs = self._forward_micro_batch(nav_mb, temperature=temperature)

                    nav_loss, nav_pg_metrics = compute_policy_loss(
                        old_log_probs=nav_old_log_probs,
                        log_probs=nav_log_probs,
                        advantages=nav_advantages,
                        response_mask=nav_response_mask,
                        clip_ratio_low=opsd_config.get("nav_clip_ratio_low", self.config.clip_ratio_low),
                        clip_ratio_high=opsd_config.get("nav_clip_ratio_high", self.config.clip_ratio_high),
                        clip_ratio_dual=opsd_config.get("nav_clip_ratio_dual", self.config.clip_ratio_dual),
                        loss_type=self.config.loss_type,
                        loss_avg_mode=self.config.loss_avg_mode,
                    )

                    scaled_nav_loss = (
                        lambda_nav * nav_loss * torch.sum(nav_response_mask) * self.world_size / policy_total_tokens
                    )
                    scaled_nav_loss.backward()

                    batch_metrics = {f"nav/{k}": v for k, v in nav_pg_metrics.items()}
                    batch_metrics["nav/grpo_loss"] = nav_loss.detach().item()
                    append_to_dict(metrics, batch_metrics)

            # --- OPSD On-Policy micro-batches ---
            for start in range(0, max(on_batch_size, 1), micro_bs):
                if on_batch_size == 0:
                    break
                end = min(start + micro_bs, on_batch_size)
                if end <= start:
                    continue

                valid_rows = (
                    (on_data.batch["on_teacher_attention_mask"][start:end].sum(dim=-1) > 0)
                    & (on_data.batch["on_student_attention_mask"][start:end].sum(dim=-1) > 0)
                    & (on_data.batch["on_student_response_mask"][start:end].sum(dim=-1) > 0)
                )
                if not torch.any(valid_rows):
                    dummy_teacher_mb, dummy_student_mb = _build_sync_dummy_on_pair()
                    if dummy_teacher_mb is None:
                        continue
                    if distillation_topk is not None or distillation_loss_type == "jsd":
                        with torch.no_grad():
                            _ = self._forward_micro_batch_logits(
                                dummy_teacher_mb,
                                temperature=temperature,
                                module=teacher_module,
                            )
                        dummy_student_logits = self._forward_micro_batch_logits(
                            dummy_student_mb,
                            temperature=temperature,
                        )
                        (dummy_student_logits.sum() * 0.0).backward()
                    else:
                        with torch.no_grad():
                            _ = self._forward_micro_batch_with_module(
                                dummy_teacher_mb,
                                temperature=temperature,
                                module=teacher_module,
                            )
                        dummy_student_lp = self._forward_micro_batch(
                            dummy_student_mb,
                            temperature=temperature,
                        )
                        (dummy_student_lp.sum() * 0.0).backward()
                    continue
                valid_idx = valid_rows.nonzero(as_tuple=True)[0]

                # Teacher forward (no grad) — evaluates student's response
                # Teacher prompt is pure text (question+hint), but may contain <image>
                # placeholders from the original question, so it needs multi_modal_inputs.
                teacher_on_mb = {
                    "input_ids": on_data.batch["on_teacher_input_ids"][start:end][valid_idx],
                    "attention_mask": on_data.batch["on_teacher_attention_mask"][start:end][valid_idx],
                    "position_ids": on_data.batch["on_teacher_position_ids"][start:end][valid_idx],
                    "responses": on_data.batch["on_student_responses"][start:end][valid_idx],  # Teacher sees student response
                }
                if "on_multi_modal_inputs" in on_data.non_tensor_batch:
                    teacher_on_mb["multi_modal_inputs"] = on_data.non_tensor_batch["on_multi_modal_inputs"][start:end][
                        valid_idx.cpu().numpy()
                    ]
                if teacher_on_mb["input_ids"].shape[0] == 0:
                    continue

                # Student forward (with grad)
                student_on_mb = {
                    "input_ids": on_data.batch["on_student_input_ids"][start:end][valid_idx],
                    "attention_mask": on_data.batch["on_student_attention_mask"][start:end][valid_idx],
                    "position_ids": on_data.batch["on_student_position_ids"][start:end][valid_idx],
                    "responses": on_data.batch["on_student_responses"][start:end][valid_idx],
                }
                on_response_mask = on_data.batch["on_student_response_mask"][start:end][valid_idx]
                distill_response_mask = (
                    on_data.batch["on_student_distill_response_mask"][start:end][valid_idx]
                    if "on_student_distill_response_mask" in on_data.batch
                    else on_response_mask
                )
                if "on_multi_modal_inputs" in on_data.non_tensor_batch:
                    student_on_mb["multi_modal_inputs"] = on_data.non_tensor_batch["on_multi_modal_inputs"][start:end][
                        valid_idx.cpu().numpy()
                    ]
                if student_on_mb["input_ids"].shape[0] == 0:
                    continue

                if distillation_topk is not None or distillation_loss_type == "jsd":
                    with torch.no_grad():
                        teacher_on_logits = self._forward_micro_batch_logits(
                            teacher_on_mb,
                            temperature=temperature,
                            module=teacher_module,
                        )
                    student_on_logits = self._forward_micro_batch_logits(student_on_mb, temperature=temperature)
                    selected_on_mask, selection_metrics = self._build_distill_token_mask(
                        teacher_logits=teacher_on_logits,
                        student_logits=student_on_logits,
                        base_mask=distill_response_mask,
                        selection_mode=distill_token_selection,
                        keep_ratio=distill_token_keep_ratio,
                    )
                    distill_mask = selected_on_mask
                    teacher_on_token_entropy = self._token_entropy_from_logits(teacher_on_logits.detach())
                    on_entropy_all_tokens = average_loss(
                        self._token_entropy_from_logits(student_on_logits),
                        distill_response_mask,
                        mode=self.config.loss_avg_mode,
                    )
                    on_selected_teacher_entropy = average_loss(
                        teacher_on_token_entropy,
                        distill_mask,
                        mode=self.config.loss_avg_mode,
                    )
                    if distillation_topk is not None:
                        k = min(distillation_topk, teacher_on_logits.size(-1))
                        teacher_topk_log_probs, student_topk_log_probs, teacher_topk_indices = (
                            self._select_topk_support(
                                teacher_logits=teacher_on_logits,
                                student_logits=student_on_logits,
                                k=k,
                                topk_source=distillation_topk_source,
                            )
                        )
                        if distillation_loss_type == "jsd":
                            l_on, topk_metrics = compute_opsd_topk_jsd_loss(
                                teacher_topk_log_probs=teacher_topk_log_probs.detach(),
                                student_topk_log_probs=student_topk_log_probs,
                                teacher_topk_indices=teacher_topk_indices,
                                student_all_logits=student_on_logits,
                                response_mask=distill_mask,
                                add_tail=distillation_add_tail,
                                loss_avg_mode=self.config.loss_avg_mode,
                            )
                            kl_on_mean = topk_metrics["opsd/topk_jsd_mean"]
                        else:
                            l_on, topk_metrics = compute_opsd_topk_kl_loss(
                                teacher_topk_log_probs=teacher_topk_log_probs.detach(),
                                student_topk_log_probs=student_topk_log_probs,
                                teacher_topk_indices=teacher_topk_indices,
                                student_all_logits=student_on_logits,
                                response_mask=distill_mask,
                                add_tail=distillation_add_tail,
                                loss_avg_mode=self.config.loss_avg_mode,
                            )
                            kl_on_mean = topk_metrics["opsd/topk_kl_mean"]
                    else:
                        l_on, jsd_metrics = compute_opsd_jsd_loss(
                            teacher_logits=teacher_on_logits,
                            student_logits=student_on_logits,
                            response_mask=distill_mask,
                            loss_avg_mode=self.config.loss_avg_mode,
                        )
                        kl_on_mean = jsd_metrics["opsd/jsd_mean"]
                    on_entropy = self._masked_entropy_from_logits(
                        student_on_logits,
                        distill_mask,
                        self.config.loss_avg_mode,
                    )
                    on_entropy_estimate = average_loss(
                        -F.log_softmax(student_on_logits, dim=-1)
                        .gather(dim=-1, index=student_on_mb["responses"].unsqueeze(-1))
                        .squeeze(-1),
                        distill_mask,
                        mode=self.config.loss_avg_mode,
                    )
                else:
                    if distill_token_selection != "all":
                        raise ValueError(
                            "opsd.distill_token_selection requires logits-based distillation. "
                            "Set opsd.distillation_loss_type=jsd or opsd.distillation_topk."
                        )
                    with torch.no_grad():
                        teacher_on_lp = self._forward_micro_batch_with_module(
                            teacher_on_mb,
                            temperature=temperature,
                            module=teacher_module,
                        )
                    student_on_lp = self._forward_micro_batch(student_on_mb, temperature=temperature)
                    kl_on = compute_kl(
                        log_probs=teacher_on_lp.detach(),
                        ref_log_probs=student_on_lp,
                        kl_penalty=kl_penalty,
                    )
                    l_on = average_loss(kl_on, distill_response_mask, mode=self.config.loss_avg_mode)
                    kl_on_mean = (
                        (kl_on.detach() * distill_response_mask).sum()
                        / distill_response_mask.sum().clamp(min=1)
                    )
                    on_entropy = None
                    on_entropy_estimate = average_loss(
                        -student_on_lp,
                        distill_response_mask,
                        mode=self.config.loss_avg_mode,
                    )
                    distill_mask = distill_response_mask
                    selection_metrics = {
                        "selected_token_ratio": 1.0,
                        "selected_token_count": float(distill_response_mask.sum().item()),
                    }
                    on_entropy_all_tokens = None
                    on_selected_teacher_entropy = None

                scaled_on_loss = alpha * l_on * torch.sum(distill_mask) * self.world_size / distill_total_tokens
                scaled_on_loss.backward()

                append_to_dict(metrics, {
                    "opsd/on_policy_loss": l_on.detach().item(),
                    "opsd/on_policy_kl": kl_on_mean if isinstance(kl_on_mean, float) else kl_on_mean.item(),
                    "opsd/on_policy_entropy_estimate": on_entropy_estimate.detach().item(),
                    "opsd/on_policy_selected_token_ratio": selection_metrics["selected_token_ratio"],
                    "opsd/on_policy_selected_token_count": selection_metrics["selected_token_count"],
                })
                if on_entropy is not None:
                    append_to_dict(metrics, {
                        "opsd/on_policy_entropy": on_entropy.detach().item(),
                        "opsd/on_policy_entropy_all_tokens": on_entropy_all_tokens.detach().item(),
                        "opsd/on_policy_selected_teacher_entropy": on_selected_teacher_entropy.detach().item(),
                    })

                if (
                    outcome_ppo_coef > 0
                    and "on_student_old_log_probs" in on_data.batch
                    and "on_student_outcome_advantages" in on_data.batch
                    and "on_student_outcome_response_mask" in on_data.batch
                ):
                    student_on_lp = self._forward_micro_batch(student_on_mb, temperature=temperature)
                    outcome_old_log_probs = on_data.batch["on_student_old_log_probs"][start:end][valid_idx]
                    outcome_advantages = on_data.batch["on_student_outcome_advantages"][start:end][valid_idx]
                    outcome_response_mask = on_data.batch["on_student_outcome_response_mask"][start:end][valid_idx]
                    outcome_pg_loss, outcome_pg_metrics = compute_policy_loss(
                        old_log_probs=outcome_old_log_probs,
                        log_probs=student_on_lp,
                        advantages=outcome_advantages,
                        response_mask=outcome_response_mask,
                        clip_ratio_low=self.config.clip_ratio_low,
                        clip_ratio_high=self.config.clip_ratio_high,
                        clip_ratio_dual=self.config.clip_ratio_dual,
                        loss_type=self.config.loss_type,
                        loss_avg_mode=self.config.loss_avg_mode,
                    )
                    scaled_outcome_loss = (
                        outcome_ppo_coef
                        * outcome_pg_loss
                        * torch.sum(outcome_response_mask)
                        * self.world_size
                        / policy_total_tokens
                    )
                    scaled_outcome_loss.backward()

                    batch_metrics = {f"opsd/outcome_{k}": v for k, v in outcome_pg_metrics.items()}
                    batch_metrics["opsd/outcome_ppo_loss"] = outcome_pg_loss.detach().item()
                    append_to_dict(metrics, batch_metrics)

            # --- OPSD Off-Policy micro-batches (skipped when alpha=1.0) ---
            for start in range(0, max(off_batch_size, 1), micro_bs):
                if skip_off_policy or not has_off_policy or off_batch_size == 0:
                    break
                end = min(start + micro_bs, off_batch_size)
                if end <= start:
                    continue

                valid_rows = (
                    (off_data.batch["off_teacher_attention_mask"][start:end].sum(dim=-1) > 0)
                    & (off_data.batch["off_student_attention_mask"][start:end].sum(dim=-1) > 0)
                    & (off_data.batch["off_teacher_response_mask"][start:end].sum(dim=-1) > 0)
                )
                if not torch.any(valid_rows):
                    dummy_teacher_mb, dummy_student_mb = _build_sync_dummy_on_pair()
                    if dummy_teacher_mb is None:
                        continue

                    if distillation_topk is not None or distillation_loss_type == "jsd":
                        with torch.no_grad():
                            _ = self._forward_micro_batch_logits(
                                dummy_teacher_mb,
                                temperature=temperature,
                                module=teacher_module,
                            )
                        dummy_student_logits = self._forward_micro_batch_logits(
                            dummy_student_mb,
                            temperature=temperature,
                        )
                        (dummy_student_logits.sum() * 0.0).backward()
                    else:
                        with torch.no_grad():
                            _ = self._forward_micro_batch_with_module(
                                dummy_teacher_mb,
                                temperature=temperature,
                                module=teacher_module,
                            )
                        dummy_student_lp = self._forward_micro_batch(
                            dummy_student_mb,
                            temperature=temperature,
                        )
                        (dummy_student_lp.sum() * 0.0).backward()

                    append_to_dict(metrics, {
                        "opsd/off_policy_loss": 0.0,
                        "opsd/off_policy_kl": 0.0,
                    })
                    continue
                valid_idx = valid_rows.nonzero(as_tuple=True)[0]

                # Teacher forward (no grad) — evaluates correct teacher response
                teacher_off_mb = {
                    "input_ids": off_data.batch["off_teacher_input_ids"][start:end][valid_idx],
                    "attention_mask": off_data.batch["off_teacher_attention_mask"][start:end][valid_idx],
                    "position_ids": off_data.batch["off_teacher_position_ids"][start:end][valid_idx],
                    "responses": off_data.batch["off_teacher_responses"][start:end][valid_idx],
                }
                if "off_multi_modal_inputs" in off_data.non_tensor_batch:
                    teacher_off_mb["multi_modal_inputs"] = off_data.non_tensor_batch["off_multi_modal_inputs"][start:end][
                        valid_idx.cpu().numpy()
                    ]
                if teacher_off_mb["input_ids"].shape[0] == 0:
                    continue

                # Student forward (with grad) — student processes correct teacher response
                student_off_mb = {
                    "input_ids": off_data.batch["off_student_input_ids"][start:end][valid_idx],
                    "attention_mask": off_data.batch["off_student_attention_mask"][start:end][valid_idx],
                    "position_ids": off_data.batch["off_student_position_ids"][start:end][valid_idx],
                    "responses": off_data.batch["off_teacher_responses"][start:end][valid_idx],
                }
                off_response_mask = off_data.batch["off_teacher_response_mask"][start:end][valid_idx]
                if "off_multi_modal_inputs" in off_data.non_tensor_batch:
                    student_off_mb["multi_modal_inputs"] = off_data.non_tensor_batch["off_multi_modal_inputs"][start:end][
                        valid_idx.cpu().numpy()
                    ]
                if student_off_mb["input_ids"].shape[0] == 0:
                    continue

                if distillation_topk is not None or distillation_loss_type == "jsd":
                    with torch.no_grad():
                        teacher_off_logits = self._forward_micro_batch_logits(
                            teacher_off_mb,
                            temperature=temperature,
                            module=teacher_module,
                        )
                    student_off_logits = self._forward_micro_batch_logits(student_off_mb, temperature=temperature)
                    selected_off_mask, selection_metrics = self._build_distill_token_mask(
                        teacher_logits=teacher_off_logits,
                        student_logits=student_off_logits,
                        base_mask=off_response_mask,
                        selection_mode=distill_token_selection,
                        keep_ratio=distill_token_keep_ratio,
                    )
                    distill_mask = selected_off_mask
                    teacher_off_token_entropy = self._token_entropy_from_logits(teacher_off_logits.detach())
                    off_entropy_all_tokens = average_loss(
                        self._token_entropy_from_logits(student_off_logits),
                        off_response_mask,
                        mode=self.config.loss_avg_mode,
                    )
                    off_selected_teacher_entropy = average_loss(
                        teacher_off_token_entropy,
                        distill_mask,
                        mode=self.config.loss_avg_mode,
                    )
                    if distillation_topk is not None:
                        k = min(distillation_topk, teacher_off_logits.size(-1))
                        teacher_topk_log_probs, student_topk_log_probs, teacher_topk_indices = (
                            self._select_topk_support(
                                teacher_logits=teacher_off_logits,
                                student_logits=student_off_logits,
                                k=k,
                                topk_source=distillation_topk_source,
                            )
                        )
                        if distillation_loss_type == "jsd":
                            l_off, topk_metrics = compute_opsd_topk_jsd_loss(
                                teacher_topk_log_probs=teacher_topk_log_probs.detach(),
                                student_topk_log_probs=student_topk_log_probs,
                                teacher_topk_indices=teacher_topk_indices,
                                student_all_logits=student_off_logits,
                                response_mask=distill_mask,
                                add_tail=distillation_add_tail,
                                loss_avg_mode=self.config.loss_avg_mode,
                            )
                            kl_off_mean = topk_metrics["opsd/topk_jsd_mean"]
                        else:
                            l_off, topk_metrics = compute_opsd_topk_kl_loss(
                                teacher_topk_log_probs=teacher_topk_log_probs.detach(),
                                student_topk_log_probs=student_topk_log_probs,
                                teacher_topk_indices=teacher_topk_indices,
                                student_all_logits=student_off_logits,
                                response_mask=distill_mask,
                                add_tail=distillation_add_tail,
                                loss_avg_mode=self.config.loss_avg_mode,
                            )
                            kl_off_mean = topk_metrics["opsd/topk_kl_mean"]
                    else:
                        l_off, jsd_metrics = compute_opsd_jsd_loss(
                            teacher_logits=teacher_off_logits,
                            student_logits=student_off_logits,
                            response_mask=distill_mask,
                            loss_avg_mode=self.config.loss_avg_mode,
                        )
                        kl_off_mean = jsd_metrics["opsd/jsd_mean"]
                    off_entropy = self._masked_entropy_from_logits(
                        student_off_logits,
                        distill_mask,
                        self.config.loss_avg_mode,
                    )
                    off_entropy_estimate = average_loss(
                        -F.log_softmax(student_off_logits, dim=-1)
                        .gather(dim=-1, index=student_off_mb["responses"].unsqueeze(-1))
                        .squeeze(-1),
                        distill_mask,
                        mode=self.config.loss_avg_mode,
                    )
                else:
                    if distill_token_selection != "all":
                        raise ValueError(
                            "opsd.distill_token_selection requires logits-based distillation. "
                            "Set opsd.distillation_loss_type=jsd or opsd.distillation_topk."
                        )
                    with torch.no_grad():
                        teacher_off_lp = self._forward_micro_batch_with_module(
                            teacher_off_mb,
                            temperature=temperature,
                            module=teacher_module,
                        )
                    student_off_lp = self._forward_micro_batch(student_off_mb, temperature=temperature)
                    kl_off = compute_kl(
                        log_probs=teacher_off_lp.detach(),
                        ref_log_probs=student_off_lp,
                        kl_penalty=kl_penalty,
                    )
                    l_off = average_loss(kl_off, off_response_mask, mode=self.config.loss_avg_mode)
                    kl_off_mean = (kl_off.detach() * off_response_mask).sum() / off_response_mask.sum().clamp(min=1)
                    off_entropy = None
                    off_entropy_estimate = average_loss(
                        -student_off_lp,
                        off_response_mask,
                        mode=self.config.loss_avg_mode,
                    )
                    distill_mask = off_response_mask
                    selection_metrics = {
                        "selected_token_ratio": 1.0,
                        "selected_token_count": float(off_response_mask.sum().item()),
                    }
                    off_entropy_all_tokens = None
                    off_selected_teacher_entropy = None

                if self_distill_negative_off_policy:
                    # Treat negative off-policy as an auxiliary regularizer rather than
                    # another token-count-weighted branch of the main OPSD objective.
                    # This avoids shrinking it again by off_tokens / total_tokens.
                    off_policy_coef = -self_distill_negative_off_policy_coef
                    scaled_off_loss = off_policy_coef * l_off
                else:
                    off_policy_coef = (1 - alpha)
                    scaled_off_loss = (
                        off_policy_coef * l_off * torch.sum(distill_mask) * self.world_size / distill_total_tokens
                    )
                scaled_off_loss.backward()

                append_to_dict(metrics, {
                    "opsd/off_policy_loss": l_off.detach().item(),
                    "opsd/off_policy_signed_objective": (off_policy_coef * l_off.detach()).item(),
                    "opsd/off_policy_kl": kl_off_mean if isinstance(kl_off_mean, float) else kl_off_mean.item(),
                    "opsd/off_policy_entropy_estimate": off_entropy_estimate.detach().item(),
                    "opsd/off_policy_selected_token_ratio": selection_metrics["selected_token_ratio"],
                    "opsd/off_policy_selected_token_count": selection_metrics["selected_token_count"],
                })
                if off_entropy is not None:
                    append_to_dict(metrics, {
                        "opsd/off_policy_entropy": off_entropy.detach().item(),
                        "opsd/off_policy_entropy_all_tokens": off_entropy_all_tokens.detach().item(),
                        "opsd/off_policy_selected_teacher_entropy": off_selected_teacher_entropy.detach().item(),
                    })

            # Optimizer step
            grad_norm = self._optimizer_step()
            append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        return metrics
