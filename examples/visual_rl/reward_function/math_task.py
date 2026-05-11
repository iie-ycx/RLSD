# -*- coding: utf-8 -*-
"""
Math Task Reward Function
"""

import multiprocessing as mp
import re
from typing import Any, Dict, List

try:
    import resource
except ImportError:  # pragma: no cover - Unix only safeguard
    resource = None

from utils import preprocess_ground_truth, strip_math_string
from utils import extract_answer_math as extract_answer

REWARD_NAME = "math"
REWARD_TYPE = "batch"

_MB = 1024 * 1024
_VM_SIZE_PATTERN = re.compile(r"^VmSize:\s+(\d+)\s+kB$", re.MULTILINE)


def _read_vms_bytes() -> int:
    """Read the current process virtual memory size on Linux."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            match = _VM_SIZE_PATTERN.search(f.read())
        if match:
            return int(match.group(1)) * 1024
    except OSError:
        pass
    return 0


def _set_child_memory_limit(memory_headroom_mb: int) -> None:
    """Cap the child process address space growth to protect the trainer."""
    if resource is None or memory_headroom_mb <= 0:
        return

    current_vms = _read_vms_bytes()
    if current_vms <= 0:
        return

    limit_bytes = current_vms + memory_headroom_mb * _MB
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard not in (resource.RLIM_INFINITY, -1):
            limit_bytes = min(limit_bytes, hard)
        if limit_bytes <= current_vms:
            return
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    except (OSError, ValueError):
        pass


def _grade_answer_worker(pred: str, gt: str, conn, memory_headroom_mb: int) -> None:
    """Run the expensive symbolic grader in a disposable child process."""
    try:
        _set_child_memory_limit(memory_headroom_mb)
        from mathruler.grader import grade_answer

        result = bool(grade_answer(pred, gt))
    except BaseException:
        result = False

    try:
        conn.send(result)
    except OSError:
        pass
    finally:
        conn.close()


def grade_answer_safe(
    pred: str,
    gt: str,
    timeout: int = 10,
    memory_headroom_mb: int = 1_048_576,
) -> bool:
    """
    Run grade_answer in an isolated process so timeouts or memory spikes
    do not crash the training worker.
    """
    if not pred or not gt:
        return False

    start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    ctx = mp.get_context(start_method)
    recv_conn, send_conn = ctx.Pipe(duplex=False)
    proc = None

    try:
        proc = ctx.Process(
            target=_grade_answer_worker,
            args=(pred, gt, send_conn, memory_headroom_mb),
            daemon=True,
        )
        proc.start()
        send_conn.close()
        proc.join(timeout)

        if proc.is_alive():
            proc.terminate()
            proc.join(0.2)
            if proc.is_alive() and hasattr(proc, "kill"):
                proc.kill()
                proc.join(0.2)
            return False

        if recv_conn.poll():
            return bool(recv_conn.recv())
        return False
    except Exception:
        if proc is not None and proc.is_alive():
            proc.terminate()
            proc.join(0.2)
        return False
    finally:
        recv_conn.close()
        send_conn.close()


def _is_safe_for_symbolic_grading(
    text: str,
    max_chars: int,
    max_lines: int,
    max_tokens: int,
) -> bool:
    """
    Skip obviously bad inputs before calling the symbolic grader.
    Long natural-language answers are the common OOM trigger.
    """
    if not text:
        return False

    stripped = text.strip()
    if not stripped:
        return False
    if len(stripped) > max_chars:
        return False
    if stripped.count("\n") > max_lines:
        return False
    if len(stripped.split()) > max_tokens:
        return False
    return True


def format_reward(response: str) -> float:
    """检查格式: <thought>...</thought>...<answer>...</answer>"""
    pattern = re.compile(r"<thought>.*</thought>.*<answer>.*</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def soft_overlong_punishment(response_length: int, max_response_length: int, overlong_buffer_length: int = 3072) -> float:
    """
    长度惩罚：
    如果生成长度超过 max_response_length，给予线性惩罚。
    防止模型通过生成大量废话来骗取奖励。
    """
    expected_len = max_response_length - overlong_buffer_length
    if response_length <= expected_len:
        return 0.0
    elif response_length <= max_response_length:
        return (expected_len - response_length) / overlong_buffer_length
    else:
        return -1.0


def math_equivalent(
    gt: str,
    pred: str,
    grader_timeout: int = 10,
    grader_memory_headroom_mb: int = 1_048_576,
    grader_max_chars: int = 512,
    grader_max_lines: int = 8,
    grader_max_tokens: int = 128,
) -> bool:
    """数学等价验证（增强版）"""
    if not gt or not pred:
        return False

    gt = gt.strip()
    pred = pred.strip()
    if pred == gt:
        return True

    gt_norm = strip_math_string(gt)
    pred_norm = strip_math_string(pred)
    if gt_norm and pred_norm and pred_norm == gt_norm:
        return True

    if _is_safe_for_symbolic_grading(pred, grader_max_chars, grader_max_lines, grader_max_tokens) and _is_safe_for_symbolic_grading(gt, grader_max_chars, grader_max_lines, grader_max_tokens):
        if grade_answer_safe(
            pred,
            gt,
            timeout=grader_timeout,
            memory_headroom_mb=grader_memory_headroom_mb,
        ):
            return True

    if (
        gt_norm
        and pred_norm
        and (pred_norm != pred or gt_norm != gt)
        and _is_safe_for_symbolic_grading(
            pred_norm, grader_max_chars, grader_max_lines, grader_max_tokens
        )
        and _is_safe_for_symbolic_grading(
            gt_norm, grader_max_chars, grader_max_lines, grader_max_tokens
        )
        and grade_answer_safe(
            pred_norm,
            gt_norm,
            timeout=grader_timeout,
            memory_headroom_mb=grader_memory_headroom_mb,
        )
    ):
        return True

    return False


def accuracy_reward(response: str, ground_truth: str, **kwargs) -> float:
    ans = extract_answer(response)
    gt_ans = extract_answer(ground_truth) or (ground_truth or "")
    return 1.0 if math_equivalent(gt_ans, ans, **kwargs) else 0.0


def compute_score(
    reward_inputs: List[Dict[str, Any]],
    max_response_length: int = 16384,
    format_weight: float = 0.1,
    overlong_penalty_factor: float = 0.1,
    grader_timeout: int = 10,
    grader_memory_headroom_mb: int = 1_048_576,
    grader_max_chars: int = 512,
    grader_max_lines: int = 8,
    grader_max_tokens: int = 128,
    **kwargs
) -> List[Dict[str, float]]:
    scores = []
    for inp in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", inp.get("response", ""))
        ground_truth = preprocess_ground_truth(inp.get("ground_truth", ""))
        response_length = inp.get("response_length", len(response))

        format_score = format_reward(response)
        accuracy_score = accuracy_reward(
            response,
            ground_truth,
            grader_timeout=grader_timeout,
            grader_memory_headroom_mb=grader_memory_headroom_mb,
            grader_max_chars=grader_max_chars,
            grader_max_lines=grader_max_lines,
            grader_max_tokens=grader_max_tokens,
        )
        len_penalty = soft_overlong_punishment(response_length, max_response_length)

        scores.append({
            "overall": (1 - format_weight) * accuracy_score + format_weight * format_score + overlong_penalty_factor * len_penalty,
            "format": format_score,
            "accuracy": accuracy_score,
            "length_penalty": len_penalty,
        })

    return scores
