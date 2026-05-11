# Self-Distilled RLVR (RLSD)

[English](README.md)

## 🧩 方法介绍

RLSD（Self-Distilled RLVR）将自蒸馏从“分布匹配目标”改造成 RLVR 中的 token 级 credit assignment 信号。教师模型可以利用 privileged information，但这部分信息不直接决定模型应该强化或惩罚哪些 token，也不改变参数更新方向；它只调节同一条轨迹中每个 token 获得的 credit 大小。

给定学生采样轨迹 $y=(y_1,\dots,y_T)$，RLSD 分别在学生上下文 $x$ 和教师上下文 $(x,r)$ 下计算每个 token 的 log-probability，并定义 privileged information gain：

$$
\Delta_t=\texttt{sg}\left(\log P_T(y_t)-\log P_S(y_t)\right)
$$

其中 $\texttt{sg}$ 表示 stop-gradient。$\Delta_t$ 衡量 privileged information $r$ 对当前 token 的边际贡献：正值表示教师信息更支持该 token，负值表示教师信息不支持该 token。由于 $\Delta_t$ 被停止梯度，它只作为权重信号使用，不引入额外的优化路径。

随后，RLSD 根据序列级 advantage 的符号构造方向感知的 evidence weight：

$$
w_t=\exp(\mathrm{sign}(A)\cdot\Delta_t)=\left(\frac{P_T(y_t)}{P_S(y_t)}\right)^{\mathrm{sign}(A)}
$$

当 $A>0$ 时，教师更支持的 token 获得更大的正向 credit；当 $A<0$ 时，权重反向使用，使教师更不支持的 token 承担更大的 blame。由于 $w_t$ 始终为正，RLSD 不会翻转 advantage 的符号：环境 reward 仍然决定整条轨迹被强化还是惩罚，教师只负责在轨迹内部重新分配 credit。

最后，RLSD 对 evidence weight 做 clipping，限制单个 token 对训练的影响：

$$
\hat{A}_t=A\cdot\mathrm{clip}(w_t,1-\epsilon_w,1+\epsilon_w)
$$

这一设计与 PPO/GRPO 的 clipped surrogate objective 类似：GRPO 裁剪 policy ratio 来约束更新步长，RLSD 裁剪 evidence ratio 来约束 credit redistribution 幅度，从而在利用教师信息的同时保持训练稳定。

## 🏆 性能

| Method | MMMU | MathVista | MathVision | ZeroBench | Wemath | Avg. |
|--------|-----:|----------:|-----------:|----------:|-------:|-----:|
| Base LLM | 62.44 | 73.80 | 47.37 | 19.76 | 54.10 | 51.49 |
| GRPO | 65.11 | 76.20 | 48.82 | 22.60 | 56.57 | 53.86 |
| OPSD | 63.82 | 75.10 | 47.53 | 21.06 | 54.95 | 52.49 |
| SDPO | 65.11 | 74.00 | 47.27 | **25.15** | 52.19 | 52.74 |
| GRPO+OPSD | 63.22 | 75.90 | 48.52 | 22.16 | 54.76 | 52.91 |
| **RLSD** *(Ours)* | **67.22** | **78.10** | **52.73** | 24.85 | **58.00** | **56.18** |

## 📐 安装

本项目的环境依赖于 [EasyVideoR1: Easier RL for Video Understanding](https://github.com/cyuQ1n/EasyVideoR1)。请先按照 EasyVideoR1 的方式创建基础环境，再在同一环境中安装本项目。

### 第一步：创建 Conda 环境

```bash
conda create -n rlsd python=3.11
conda activate rlsd
```

### 第二步：安装 EasyVideoR1 基础环境

```bash
git clone https://github.com/cyuQ1n/EasyVideoR1.git
cd EasyVideoR1
pip install -e .
pip install flash-attn==2.8.3 --no-build-isolation
```

### 第三步：安装本项目

```bash
git clone <this-repo-url>
cd rlsd_verl
pip install -e .
```

如果已经在本地准备好了本项目代码，可以直接进入当前目录安装：

```bash
cd /pfs/qcy/video_rl/rlsd_verl
pip install -e .
```

### 依赖说明

本项目沿用 EasyVideoR1 的 veRL、Ray、vLLM、Qwen-VL 相关依赖，并在当前仓库的 `requirements.txt` 中维护额外依赖。核心版本包括：

```text
qwen-vl-utils[decord]==0.0.14
transformers==4.57.3
vllm==0.11.0
flash-attn==2.8.3
```

## 🚀 快速开始

以下是最简启动训练流程。

### 第一步：准备数据

训练和验证数据可从 Hugging Face 数据集 [iieycx/rlsd-train-MMFineReason-123K](https://huggingface.co/datasets/iieycx/rlsd-train-MMFineReason-123K) 下载获得。

训练数据应为 JSON/JSONL 格式，每条数据至少包含问题、答案和图像字段.

### 第二步：使用已提供的配置和脚本

本仓库已在 `examples/visual_rl/` 下提供 RLSD 和 GRPO baseline 的配置文件与训练脚本，可直接使用：

| 方法 | 配置文件 | 训练脚本 |
|------|----------|----------|
| RLSD | `examples/visual_rl/rlsd_config.yaml` | `examples/visual_rl/rlsd_train.sh` |
| GRPO Baseline | `examples/visual_rl/grpo_baseline_config.yaml` | `examples/visual_rl/grpo_baseline_train_MMFineReason.sh` |

训练脚本中已经绑定对应的 YAML 配置、prompt 模板和 reward 函数。默认数据、模型和输出路径可在脚本中查看，也可以通过环境变量覆盖。

### 第三步：启动训练

运行 RLSD：

```bash
bash examples/visual_rl/rlsd_train.sh
```

运行 GRPO baseline：

```bash
bash examples/visual_rl/grpo_baseline_train_MMFineReason.sh
```

多节点训练时，在每个节点上设置 `WORLD_SIZE`、`RANK` 和 `MASTER_ADDR`，并运行对应脚本：

```bash
# RLSD 主节点
WORLD_SIZE=2 RANK=0 MASTER_ADDR=<主节点IP> bash examples/visual_rl/rlsd_train.sh

# RLSD 工作节点
WORLD_SIZE=2 RANK=1 MASTER_ADDR=<主节点IP> bash examples/visual_rl/rlsd_train.sh
```


## 📂 项目结构

```text
rlsd_verl/
├── verl/                       # 核心 RL 训练框架
│   ├── trainer/                # GRPO / OPSD / RLSD 训练入口与训练循环
│   ├── workers/                # Actor、rollout、reward、critic workers
│   ├── models/                 # Qwen-VL 模型适配
│   └── utils/                  # 数据集、分词、FSDP、日志与检查点工具
├── examples/
│   └── visual_rl/              # 视觉 RL 示例配置、训练脚本、prompt 与 reward
├── requirements.txt            # Python 依赖
├── setup.py                    # 可编辑安装入口
└── README_zh.md
```

## 🔧 示例管线

### RLSD (`examples/visual_rl/`)

RLSD 训练脚本默认使用 `verl.trainer.opsd_main`，并通过 `examples/visual_rl/reward_function/unified.py` 处理不同视觉推理任务的 reward。

```bash
bash examples/visual_rl/rlsd_train.sh
```

### GRPO Baseline (`examples/visual_rl/`)

标准 GRPO baseline 使用同一套数据、prompt 和 reward 接口，便于与 RLSD 进行对比。

```bash
bash examples/visual_rl/grpo_baseline_train_MMFineReason.sh
```

## 🙏 致谢

本项目基于以下优秀工作构建：

- [EasyVideoR1](https://github.com/cyuQ1n/EasyVideoR1) - Easier RL for Video Understanding
- [EasyR1](https://github.com/hiyouga/EasyR1) - 高效可扩展的多模态 RL 训练框架
- [veRL](https://github.com/volcengine/verl) - 高性能 RL 与 HybridEngine

## 📄 引用

如果使用本项目，请引用 Self-Distilled RLVR、EasyVideoR1、EasyR1 和 veRL：

```bibtex
@misc{yang2026selfdistilledrlvr,
      title={Self-Distilled RLVR},
      author={Chenxu Yang and Chuanyu Qin and Qingyi Si and Minghui Chen and Naibin Gu and Dingyu Yao and Zheng Lin and Weiping Wang and Jiaqi Wang and Nan Duan},
      year={2026},
      eprint={2604.03128},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.03128},
}

@misc{qin2026easyvideor1easierrlvideo,
      title={EasyVideoR1: Easier RL for Video Understanding},
      author={Chuanyu Qin and Chenxu Yang and Qingyi Si and Naibin Gu and Dingyu Yao and Zheng Lin and Peng Fu and Nan Duan and Jiaqi Wang},
      year={2026},
      eprint={2604.16893},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.16893},
}

```

## 📜 许可证

本项目遵循与 [EasyR1](https://github.com/hiyouga/EasyR1) 相同的许可证。
