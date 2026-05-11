from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from jinja2 import Template

from ..protocol import DataProto
from ..utils import torch_functional as VF
from ..utils.dataset import process_image, process_video
from .core_algos import average_loss, compute_kl


def compute_opsd_kl_loss(
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    kl_penalty: Literal["kl", "abs", "mse", "low_var_kl", "full"] = "kl",
    loss_avg_mode: Literal["token", "seq"] = "token",
) -> tuple[torch.Tensor, dict[str, float]]:
    kl_values = compute_kl(
        log_probs=teacher_log_probs.detach(),
        ref_log_probs=student_log_probs,
        kl_penalty=kl_penalty,
    )
    loss = average_loss(kl_values, response_mask, mode=loss_avg_mode)
    kl_mean = VF.masked_mean(kl_values.detach(), response_mask).item()
    return loss, {"opsd/kl_mean": kl_mean}


def compute_opsd_topk_kl_loss(
    teacher_topk_log_probs: torch.Tensor,
    student_topk_log_probs: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    student_all_logits: torch.Tensor,
    response_mask: torch.Tensor,
    add_tail: bool = True,
    loss_avg_mode: Literal["token", "seq"] = "token",
) -> tuple[torch.Tensor, dict[str, float]]:
    teacher_topk_probs = teacher_topk_log_probs.detach().exp()
    topk_kl = teacher_topk_probs * (teacher_topk_log_probs.detach() - student_topk_log_probs)
    topk_kl = topk_kl.sum(dim=-1)

    if add_tail:
        teacher_tail_prob = (1.0 - teacher_topk_probs.sum(dim=-1)).clamp(min=1e-10)
        teacher_tail_log_prob = teacher_tail_prob.log()

        student_log_softmax = F.log_softmax(student_all_logits, dim=-1)
        student_topk_log_probs_full = student_log_softmax.gather(dim=-1, index=teacher_topk_indices)
        student_topk_probs = student_topk_log_probs_full.exp()
        student_tail_prob = (1.0 - student_topk_probs.sum(dim=-1)).clamp(min=1e-10)
        student_tail_log_prob = student_tail_prob.log()

        tail_kl = teacher_tail_prob * (teacher_tail_log_prob - student_tail_log_prob)
        topk_kl = topk_kl + tail_kl

    topk_kl = topk_kl.clamp(min=-10.0, max=10.0)
    loss = average_loss(topk_kl, response_mask, mode=loss_avg_mode)
    kl_mean = VF.masked_mean(topk_kl.detach(), response_mask).item()
    return loss, {"opsd/topk_kl_mean": kl_mean}


def compute_opsd_jsd_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    response_mask: torch.Tensor,
    loss_avg_mode: Literal["token", "seq"] = "token",
    eps: float = 1e-10,
) -> tuple[torch.Tensor, dict[str, float]]:
    teacher_log_probs = F.log_softmax(teacher_logits.detach(), dim=-1)
    teacher_probs = teacher_log_probs.exp()
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    student_probs = student_log_probs.exp()

    mix_probs = 0.5 * (teacher_probs + student_probs)
    mix_log_probs = mix_probs.clamp_min(eps).log()

    jsd = 0.5 * (
        (teacher_probs * (teacher_log_probs - mix_log_probs)).sum(dim=-1)
        + (student_probs * (student_log_probs - mix_log_probs)).sum(dim=-1)
    )
    jsd = jsd.clamp(min=0.0, max=10.0)

    loss = average_loss(jsd, response_mask, mode=loss_avg_mode)
    jsd_mean = VF.masked_mean(jsd.detach(), response_mask).item()
    return loss, {"opsd/jsd_mean": jsd_mean}


def compute_opsd_topk_jsd_loss(
    teacher_topk_log_probs: torch.Tensor,
    student_topk_log_probs: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    student_all_logits: torch.Tensor,
    response_mask: torch.Tensor,
    add_tail: bool = True,
    loss_avg_mode: Literal["token", "seq"] = "token",
    eps: float = 1e-10,
) -> tuple[torch.Tensor, dict[str, float]]:
    teacher_topk_probs = teacher_topk_log_probs.detach().exp()

    student_log_softmax = F.log_softmax(student_all_logits, dim=-1)
    student_topk_log_probs_full = student_log_softmax.gather(dim=-1, index=teacher_topk_indices)
    student_topk_probs = student_topk_log_probs_full.exp()

    if add_tail:
        teacher_tail_prob = (1.0 - teacher_topk_probs.sum(dim=-1, keepdim=True)).clamp(min=eps)
        student_tail_prob = (1.0 - student_topk_probs.sum(dim=-1, keepdim=True)).clamp(min=eps)

        teacher_probs = torch.cat([teacher_topk_probs, teacher_tail_prob], dim=-1)
        student_probs = torch.cat([student_topk_probs, student_tail_prob], dim=-1)
    else:
        teacher_probs = teacher_topk_probs / teacher_topk_probs.sum(dim=-1, keepdim=True).clamp(min=eps)
        student_probs = student_topk_probs / student_topk_probs.sum(dim=-1, keepdim=True).clamp(min=eps)

    teacher_log_probs = teacher_probs.clamp_min(eps).log()
    student_log_probs = student_probs.clamp_min(eps).log()
    mix_probs = 0.5 * (teacher_probs + student_probs)
    mix_log_probs = mix_probs.clamp_min(eps).log()

    jsd = 0.5 * (
        (teacher_probs * (teacher_log_probs - mix_log_probs)).sum(dim=-1)
        + (student_probs * (student_log_probs - mix_log_probs)).sum(dim=-1)
    )
    jsd = jsd.clamp(min=0.0, max=10.0)

    loss = average_loss(jsd, response_mask, mode=loss_avg_mode)
    jsd_mean = VF.masked_mean(jsd.detach(), response_mask).item()
    return loss, {"opsd/topk_jsd_mean": jsd_mean}


def compute_navigator_advantage(
    hint_rewards: torch.Tensor,
    hint_uids: np.ndarray | list[Any],
    K: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    del K
    scores = hint_rewards.float()
    id2score: defaultdict[str, list[torch.Tensor]] = defaultdict(list)
    id2mean: dict[str, torch.Tensor] = {}
    id2std: dict[str, torch.Tensor] = {}

    bsz = scores.shape[0]
    for i in range(bsz):
        uid = hint_uids[i] if isinstance(hint_uids[i], str) else str(hint_uids[i])
        id2score[uid].append(scores[i])

    for uid in id2score:
        group_scores = torch.tensor(id2score[uid], device=scores.device, dtype=scores.dtype)
        id2mean[uid] = group_scores.mean()
        id2std[uid] = group_scores.std()

    advantages = torch.zeros_like(scores)
    for i in range(bsz):
        uid = hint_uids[i] if isinstance(hint_uids[i], str) else str(hint_uids[i])
        std = id2std[uid]
        if std < eps:
            advantages[i] = 0.0
            continue
        advantages[i] = (scores[i] - id2mean[uid]) / (std + eps)

    return advantages


def compute_curriculum_p_drop(global_step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min(global_step / warmup_steps, 1.0)


def _compute_position_ids(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    model_inputs: dict[str, Any],
    processor: Any,
) -> torch.Tensor:
    if processor is not None and "Qwen2VLImageProcessor" in processor.image_processor.__class__.__name__:
        if "Qwen3VLProcessor" in processor.__class__.__name__:
            from ..models.transformers.qwen3_vl import get_rope_index
        else:
            from ..models.transformers.qwen2_vl import get_rope_index

        vision_position_ids = get_rope_index(
            processor,
            input_ids=input_ids,
            image_grid_thw=model_inputs.get("image_grid_thw", None),
            video_grid_thw=model_inputs.get("video_grid_thw", None),
            second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
            attention_mask=attention_mask,
        )
        text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)
        position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)
    else:
        position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)
    return position_ids


def _build_messages_from_prompt_text(
    prompt_text: str,
    multi_modal_item: Optional[dict[str, Any]],
) -> list[dict[str, Any]]:
    multi_modal_item = multi_modal_item or {}

    has_images = "images" in multi_modal_item and len(multi_modal_item["images"]) > 0
    has_videos = any(
        key in multi_modal_item and multi_modal_item[key] is not None and len(multi_modal_item[key]) > 0
        for key in ("video", "videos")
    ) or "preprocessed_video_path" in multi_modal_item

    if has_images:
        content_list: list[dict[str, Any]] = []
        for i, content in enumerate(prompt_text.split("<image>")):
            if i != 0:
                content_list.append({"type": "image"})
            if content:
                content_list.append({"type": "text", "text": content})
        return [{"role": "user", "content": content_list}]

    if has_videos:
        content_list = []
        for i, content in enumerate(prompt_text.split("<video>")):
            if i != 0:
                content_list.append({"type": "video"})
            if content:
                content_list.append({"type": "text", "text": content})
        return [{"role": "user", "content": content_list}]

    return [{"role": "user", "content": prompt_text}]


def truncate_text_by_tokens(text: Any, tokenizer: Any, max_tokens: Optional[int]) -> str:
    """Truncate a text string by tokenizer token count instead of character count."""
    if text is None:
        return ""

    text = text if isinstance(text, str) else str(text)
    if max_tokens is None:
        return text
    if max_tokens <= 0:
        return ""

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text

    truncated_ids = token_ids[:max_tokens]
    return tokenizer.decode(truncated_ids, skip_special_tokens=True).strip()


def _tokenize_prompt(
    prompt: str,
    tokenizer: Any,
    processor: Any,
    multi_modal_item: Optional[dict[str, Any]],
    max_prompt_length: int,
    image_min_pixels: Optional[int] = None,
    image_max_pixels: Optional[int] = None,
    video_min_pixels: Optional[int] = None,
    video_max_pixels: Optional[int] = None,
    video_fps: Optional[float] = None,
    video_max_frames: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], str]:
    multi_modal_item = multi_modal_item or {}

    if processor is not None and "images" in multi_modal_item and len(multi_modal_item["images"]) > 0:
        processed_images = [
            process_image(image, min_pixels=image_min_pixels, max_pixels=image_max_pixels)
            for image in multi_modal_item["images"]
        ]
        model_inputs = processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]
    elif processor is not None and "preprocessed_video_path" in multi_modal_item:
        preprocessed_data = torch.load(
            multi_modal_item["preprocessed_video_path"],
            map_location="cpu",
            weights_only=False,
        )
        model_inputs = processor(
            [prompt],
            [preprocessed_data["frames"]],
            add_special_tokens=False,
            video_metadata=[preprocessed_data["metadata"]],
            return_tensors="pt",
            do_resize=False,
            do_sample_frames=False,
        )
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]
    elif processor is not None and "video" in multi_modal_item and len(multi_modal_item["video"]) > 0:
        processor_kwargs: dict[str, Any] = {
            "text": [prompt],
            "videos": multi_modal_item["video"],
            "add_special_tokens": False,
            "return_tensors": "pt",
            "do_resize": False,
            "do_sample_frames": False,
        }
        video_metadatas = multi_modal_item.get("video_metadatas", None)
        if video_metadatas is not None and len(video_metadatas) > 0:
            processor_kwargs["video_metadata"] = video_metadatas
        model_inputs = processor(**processor_kwargs)
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]
    elif processor is not None and "videos" in multi_modal_item and len(multi_modal_item["videos"]) > 0:
        processed_video_frames = []
        video_metadatas = []
        for video in multi_modal_item["videos"]:
            result = process_video(
                video,
                video_min_pixels if video_min_pixels else 4096,
                video_max_pixels if video_max_pixels else 65536,
                video_max_frames,
                video_fps,
                True,
            )
            if isinstance(result, tuple) and len(result) == 2:
                video_data, _ = result
                if isinstance(video_data, tuple) and len(video_data) == 2:
                    frames, metadata = video_data
                    processed_video_frames.append(frames)
                    video_metadatas.append(metadata)
                else:
                    processed_video_frames.append(video_data)
                    video_metadatas = None
                    break
            else:
                processed_video_frames.append(result)
                video_metadatas = None

        processor_kwargs = {
            "text": [prompt],
            "videos": processed_video_frames,
            "add_special_tokens": False,
            "return_tensors": "pt",
            "do_sample_frames": False,
        }
        if video_metadatas is not None and len(video_metadatas) > 0:
            processor_kwargs["video_metadata"] = video_metadatas
            processor_kwargs["do_resize"] = False
        model_inputs = processor(**processor_kwargs)
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]
    else:
        model_inputs = {}
        model_inputs = tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
        input_ids = model_inputs["input_ids"][0]
        attention_mask = model_inputs["attention_mask"][0]

    position_ids = _compute_position_ids(input_ids, attention_mask, model_inputs, processor)
    input_ids, attention_mask, position_ids = VF.postprocess_data(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        max_length=max_prompt_length,
        pad_token_id=tokenizer.pad_token_id,
        left_pad=True,
        truncation="left",
    )

    raw_prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(raw_prompt_ids) > max_prompt_length:
        raw_prompt_ids = raw_prompt_ids[-max_prompt_length:]

    return input_ids, attention_mask, position_ids, raw_prompt_ids, prompt


def construct_navigator_prompts(
    questions: list[Any] | np.ndarray,
    trajectories: list[Any] | np.ndarray,
    tokenizer: Any,
    processor: Any,
    student_trajectories: Optional[list[Any] | np.ndarray] = None,
    template_path: Optional[str] = None,
    max_traj_len: int = 2048,
    max_prompt_length: int = 4096,
    multi_modal_data: Optional[list[dict[str, Any]] | np.ndarray] = None,
    image_min_pixels: Optional[int] = None,
    image_max_pixels: Optional[int] = None,
    video_min_pixels: Optional[int] = None,
    video_max_pixels: Optional[int] = None,
    video_fps: Optional[float] = None,
    video_max_frames: Optional[int] = None,
    apply_chat_template_kwargs: Optional[dict] = None,
) -> DataProto:
    if template_path:
        with open(template_path, encoding="utf-8") as f:
            template = Template(f.read().strip())
    else:
        template = Template(
            "You are a helpful navigator. Given a question and a reasoning trajectory from another model, "
            "extract a concise hint that captures the key insight for solving the problem.\n\n"
            "Question: {{ question }}\n\nTrajectory:\n{{ trajectory }}\n\n"
            "Provide a concise hint that captures the key reasoning insight. "
            "Output your hint within <hint> </hint> tags."
        )

    batch_size = len(questions)
    all_input_ids = []
    all_attention_mask = []
    all_position_ids = []
    all_raw_prompt_ids = []
    all_raw_prompt = []
    all_multi_modal = []

    for i in range(batch_size):
        question = questions[i] if isinstance(questions[i], str) else str(questions[i])
        trajectory = truncate_text_by_tokens(trajectories[i], tokenizer, max_traj_len)
        student_trajectory = ""
        if student_trajectories is not None:
            student_trajectory = truncate_text_by_tokens(student_trajectories[i], tokenizer, max_traj_len)

        prompt_text = template.render(
            question=question,
            trajectory=trajectory,
            student_trajectory=student_trajectory,
        )
        mm_item = multi_modal_data[i] if multi_modal_data is not None else None
        messages = _build_messages_from_prompt_text(prompt_text, mm_item)
        if processor is not None:
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **(apply_chat_template_kwargs or {}))
        else:
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **(apply_chat_template_kwargs or {}))

        input_ids, attention_mask, position_ids, raw_prompt_ids, raw_prompt = _tokenize_prompt(
            prompt,
            tokenizer,
            processor,
            mm_item,
            max_prompt_length,
            image_min_pixels,
            image_max_pixels,
            video_min_pixels,
            video_max_pixels,
            video_fps,
            video_max_frames,
        )

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_position_ids.append(position_ids)
        all_raw_prompt_ids.append(raw_prompt_ids)
        all_raw_prompt.append(raw_prompt)
        all_multi_modal.append(mm_item if mm_item is not None else {})

    tensors = {
        "input_ids": torch.stack(all_input_ids),
        "attention_mask": torch.stack(all_attention_mask),
        "position_ids": torch.stack(all_position_ids),
    }
    non_tensors = {
        "raw_prompt_ids": np.array(all_raw_prompt_ids, dtype=object),
        "raw_prompt": np.array(all_raw_prompt, dtype=object),
        "multi_modal_data": np.array(all_multi_modal, dtype=object),
    }
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


def construct_teacher_prompts(
    questions: list[Any] | np.ndarray,
    hints: list[list[str]] | np.ndarray,
    gt_answers: list[Any] | np.ndarray,
    p_drop: float,
    tokenizer: Any,
    processor: Any,
    reference_trajectories: Optional[list[Any] | np.ndarray] = None,
    student_trajectories: Optional[list[Any] | np.ndarray] = None,
    template_path: Optional[str] = None,
    template_text: Optional[str] = None,
    format_prompt_path: Optional[str] = None,
    max_prompt_length: int = 4096,
    multi_modal_data: Optional[list[dict[str, Any]] | np.ndarray] = None,
    image_min_pixels: Optional[int] = None,
    image_max_pixels: Optional[int] = None,
    video_min_pixels: Optional[int] = None,
    video_max_pixels: Optional[int] = None,
    video_fps: Optional[float] = None,
    video_max_frames: Optional[int] = None,
    apply_chat_template_kwargs: Optional[dict] = None,
) -> DataProto:
    del format_prompt_path
    if template_text is not None:
        hint_template = Template(template_text.strip())
    elif template_path:
        with open(template_path, encoding="utf-8") as f:
            hint_template = Template(f.read().strip())
    else:
        hint_template = Template(
            "{{ question }}\n\n[Hint from a reasoning navigator]\n{{ hint }}\n\n"
            "{% if gt_answer %}[Reference Answer]\n{{ gt_answer }}\n\n{% endif %}"
            "Using the above hint{% if gt_answer %} and reference answer{% endif %}, "
            "solve the problem step by step."
        )

    batch_size = len(questions)
    K = len(hints[0]) if batch_size > 0 else 0

    all_input_ids = []
    all_attention_mask = []
    all_position_ids = []
    all_raw_prompt_ids = []
    all_raw_prompt = []
    all_multi_modal = []

    for i in range(batch_size):
        question = questions[i] if isinstance(questions[i], str) else str(questions[i])
        gt_answer = gt_answers[i] if isinstance(gt_answers[i], str) else str(gt_answers[i])
        reference_trajectory = ""
        if reference_trajectories is not None:
            reference_trajectory = (
                reference_trajectories[i]
                if isinstance(reference_trajectories[i], str)
                else str(reference_trajectories[i])
            )
        student_trajectory = ""
        if student_trajectories is not None:
            student_trajectory = (
                student_trajectories[i]
                if isinstance(student_trajectories[i], str)
                else str(student_trajectories[i])
            )

        include_gt = random.random() >= p_drop
        gt_for_prompt = gt_answer if include_gt else ""

        for k in range(K):
            hint = hints[i][k]
            prompt_text = hint_template.render(
                question=question,
                hint=hint,
                gt_answer=gt_for_prompt,
                reference_trajectory=reference_trajectory,
                student_trajectory=student_trajectory,
            )
            mm_item = multi_modal_data[i] if multi_modal_data is not None else None
            messages = _build_messages_from_prompt_text(prompt_text, mm_item)
            if processor is not None:
                prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **(apply_chat_template_kwargs or {}))
            else:
                prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **(apply_chat_template_kwargs or {}))

            input_ids, attention_mask, position_ids, raw_prompt_ids, raw_prompt = _tokenize_prompt(
                prompt,
                tokenizer,
                processor,
                mm_item,
                max_prompt_length,
                image_min_pixels,
                image_max_pixels,
                video_min_pixels,
                video_max_pixels,
                video_fps,
                video_max_frames,
            )

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_position_ids.append(position_ids)
            all_raw_prompt_ids.append(raw_prompt_ids)
            all_raw_prompt.append(raw_prompt)
            all_multi_modal.append(mm_item if mm_item is not None else {})

    tensors = {
        "input_ids": torch.stack(all_input_ids),
        "attention_mask": torch.stack(all_attention_mask),
        "position_ids": torch.stack(all_position_ids),
    }
    non_tensors = {
        "raw_prompt_ids": np.array(all_raw_prompt_ids, dtype=object),
        "raw_prompt": np.array(all_raw_prompt, dtype=object),
        "multi_modal_data": np.array(all_multi_modal, dtype=object),
    }
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


def construct_sdpo_teacher_prompts(
    questions: list[Any] | np.ndarray,
    correct_solutions: list[str] | np.ndarray,
    incorrect_solutions: list[str] | np.ndarray,
    tokenizer: Any,
    processor: Any,
    template_path: Optional[str] = None,
    template_text: Optional[str] = None,
    max_prompt_length: int = 4096,
    multi_modal_data: Optional[list[dict[str, Any]] | np.ndarray] = None,
    image_min_pixels: Optional[int] = None,
    image_max_pixels: Optional[int] = None,
    video_min_pixels: Optional[int] = None,
    video_max_pixels: Optional[int] = None,
    video_fps: Optional[float] = None,
    video_max_frames: Optional[int] = None,
    apply_chat_template_kwargs: Optional[dict] = None,
) -> DataProto:
    if template_text is not None:
        teacher_template = Template(template_text.strip())
    elif template_path:
        with open(template_path, encoding="utf-8") as f:
            teacher_template = Template(f.read().strip())
    else:
        teacher_template = Template(
            "{{ question | trim }}\n\n"
            "{% if correct_solution %}Correct solution:\n{{ correct_solution | trim }}\n\n{% endif %}"
            "{% if incorrect_solution %}"
            "The following is feedback from your unsuccessful earlier attempt:\n"
            "{{ incorrect_solution | trim }}\n\n"
            "{% endif %}"
            "Correctly solve the original question."
        )

    all_input_ids = []
    all_attention_mask = []
    all_position_ids = []
    all_raw_prompt_ids = []
    all_raw_prompt = []
    all_multi_modal = []

    for i in range(len(questions)):
        question = questions[i] if isinstance(questions[i], str) else str(questions[i])
        correct_solution = (
            correct_solutions[i] if isinstance(correct_solutions[i], str) else str(correct_solutions[i])
        )
        incorrect_solution = (
            incorrect_solutions[i] if isinstance(incorrect_solutions[i], str) else str(incorrect_solutions[i])
        )

        prompt_text = teacher_template.render(
            question=question,
            correct_solution=correct_solution,
            incorrect_solution=incorrect_solution,
        )
        mm_item = multi_modal_data[i] if multi_modal_data is not None else None
        messages = _build_messages_from_prompt_text(prompt_text, mm_item)
        if processor is not None:
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **(apply_chat_template_kwargs or {}))
        else:
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **(apply_chat_template_kwargs or {}))

        input_ids, attention_mask, position_ids, raw_prompt_ids, raw_prompt = _tokenize_prompt(
            prompt,
            tokenizer,
            processor,
            mm_item,
            max_prompt_length,
            image_min_pixels,
            image_max_pixels,
            video_min_pixels,
            video_max_pixels,
            video_fps,
            video_max_frames,
        )

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_position_ids.append(position_ids)
        all_raw_prompt_ids.append(raw_prompt_ids)
        all_raw_prompt.append(raw_prompt)
        all_multi_modal.append(mm_item if mm_item is not None else {})

    return DataProto.from_dict(
        tensors={
            "input_ids": torch.stack(all_input_ids),
            "attention_mask": torch.stack(all_attention_mask),
            "position_ids": torch.stack(all_position_ids),
        },
        non_tensors={
            "raw_prompt_ids": np.array(all_raw_prompt_ids, dtype=object),
            "raw_prompt": np.array(all_raw_prompt, dtype=object),
            "multi_modal_data": np.array(all_multi_modal, dtype=object),
        },
    )


def decode_navigator_hints(nav_output: DataProto, tokenizer: Any) -> list[list[str]]:
    responses = nav_output.batch["responses"]
    total = responses.shape[0]

    decoded = []
    for i in range(total):
        text = tokenizer.decode(responses[i], skip_special_tokens=True).strip()
        decoded.append(text)

    K = nav_output.meta_info.get("n", 1)
    B = total // K

    hints = []
    for b in range(B):
        hints_for_question = []
        for k in range(K):
            hints_for_question.append(decoded[b * K + k])
        hints.append(hints_for_question)

    return hints


def construct_opsd_forward_batch(
    student_input_ids: torch.Tensor,
    student_attention_mask: torch.Tensor,
    student_position_ids: torch.Tensor,
    student_responses: torch.Tensor,
    student_response_mask: torch.Tensor,
    teacher_input_ids: torch.Tensor,
    teacher_attention_mask: torch.Tensor,
    teacher_position_ids: torch.Tensor,
    teacher_responses: torch.Tensor,
    teacher_response_mask: torch.Tensor,
    teacher_rewards: torch.Tensor,
    K: int,
    tokenizer: Any,
    max_seq_length: int,
) -> dict[str, Any]:
    del student_input_ids, student_attention_mask, student_position_ids, tokenizer, max_seq_length
    B = student_responses.shape[0]
    student_resp_len = student_responses.shape[1]
    teacher_resp_len = teacher_responses.shape[1]
    del B, student_resp_len, teacher_resp_len

    student_responses_rep = student_responses.repeat_interleave(K, dim=0)
    student_response_mask_rep = student_response_mask.repeat_interleave(K, dim=0)

    valid_mask = teacher_rewards > 0
    valid_indices = valid_mask.nonzero(as_tuple=True)[0]

    result = {
        "on_policy_student_responses": student_responses_rep,
        "on_policy_student_response_mask": student_response_mask_rep,
        "on_policy_teacher_input_ids": teacher_input_ids,
        "on_policy_teacher_attention_mask": teacher_attention_mask,
        "on_policy_teacher_position_ids": teacher_position_ids,
        "off_policy_teacher_responses": teacher_responses[valid_indices] if len(valid_indices) > 0 else teacher_responses[:0],
        "off_policy_teacher_response_mask": (
            teacher_response_mask[valid_indices] if len(valid_indices) > 0 else teacher_response_mask[:0]
        ),
        "off_policy_teacher_input_ids": teacher_input_ids[valid_indices] if len(valid_indices) > 0 else teacher_input_ids[:0],
        "off_policy_teacher_attention_mask": (
            teacher_attention_mask[valid_indices] if len(valid_indices) > 0 else teacher_attention_mask[:0]
        ),
        "off_policy_teacher_position_ids": (
            teacher_position_ids[valid_indices] if len(valid_indices) > 0 else teacher_position_ids[:0]
        ),
        "valid_hint_indices": valid_indices,
        "valid_hint_mask": valid_mask,
        "teacher_rewards": teacher_rewards,
        "K": K,
        "B": student_responses.shape[0],
    }
    return result
