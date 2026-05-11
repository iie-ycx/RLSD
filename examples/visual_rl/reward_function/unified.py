# -*- coding: utf-8 -*-
"""
Unified Reward Function - 统一路由
根据 problem_type 自动调用对应的 reward 模块
"""

import os
import sys
import logging
from typing import Any, Dict, List

# 动态添加当前目录到 sys.path 以支持动态加载
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from utils import normalize_response, preprocess_ground_truth
import math_task as math_reward
import multiple_choice as mc_reward
import numerical as num_reward
import open_ended as oe_reward
import ocr as ocr_reward
import boolean as bool_reward
import code_task as code_reward
import llava as llava_reward
import grounding as grounding_reward

logger = logging.getLogger(__name__)

REWARD_NAME = "unified"
REWARD_TYPE = "batch"

# 任务类型到 reward 模块的映射
REWARD_MAPPING = {
    # 数学类
    "math": math_reward,
    "mathematics": math_reward,

    # 选择题类
    "multiple choice": mc_reward,
    "multiple_choice": mc_reward,

    # 数值类
    "numerical": num_reward,
    "number": num_reward,
    "regression": num_reward,

    # 开放式
    "open-ended": oe_reward,
    "open_ended": oe_reward,
    "video qa": oe_reward,
    "video description": oe_reward,
    "free-form": oe_reward,

    # OCR
    "ocr": ocr_reward,

    # 布尔
    "boolean": bool_reward,
    "binary classification": bool_reward,

    # LLaVA Critic
    "llava": llava_reward,
    "critic": llava_reward,

    # 代码类
    "code": code_reward,
    "coding": code_reward,
    "svg-code": code_reward,
    "html-code": code_reward,

    # 定位类（特殊处理）
    "spatial grounding": "spatial",
    "temporal grounding": "temporal",
    "spatial-temporal grounding": "spatial-temporal",
}


def compute_score(reward_inputs: List[Dict[str, Any]], **kwargs) -> List[Dict[str, float]]:
    """
    统一的 reward 计算入口
    根据每个样本的 problem_type 自动路由到对应的 reward 函数
    """
    results = []

    for idx, inp in enumerate(reward_inputs):
        try:
            # 预处理
            response = normalize_response(inp.get("response", ""))
            ground_truth = preprocess_ground_truth(inp.get("ground_truth", ""))
            problem_type = (inp.get("problem_type", "") or "").lower().strip()

            # 更新 inp
            inp_processed = {**inp, "response": response, "ground_truth": ground_truth}

            # 获取对应的 reward 模块/类型
            reward_target = REWARD_MAPPING.get(problem_type, oe_reward)

            # 特殊处理定位类任务
            if reward_target == "spatial":
                score_list = grounding_reward.spatial_compute_score([inp_processed], **kwargs)
            elif reward_target == "temporal":
                score_list = grounding_reward.temporal_compute_score([inp_processed], **kwargs)
            elif reward_target == "spatial-temporal":
                score_list = grounding_reward.spatial_temporal_compute_score([inp_processed], **kwargs)
            else:
                # 调用对应模块的 compute_score
                score_list = reward_target.compute_score([inp_processed], **kwargs)

            results.append(score_list[0])

        except Exception as e:
            logger.error(f"Error computing reward for sample {idx}: {e}")
            results.append({
                "overall": 0.0,
                "accuracy": 0.0,
                "format": 0.0,
                "length_penalty": 0.0,
            })

    return results
