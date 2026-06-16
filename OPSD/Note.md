# On-Policy Self-Distillation (OPSD) 原理与核心代码

**论文**: [Self-Distilled Reasoner](https://arxiv.org/pdf/2601.18734v3) | **代码**: [github.com/siyan-zhao/OPSD](https://github.com/siyan-zhao/OPSD) | **基础模型**: Qwen3

---

## 一、核心思想

### 1. 自蒸馏框架

同一个模型通过**上下文条件**的差异同时扮演学生和教师：

```
学生: π_S(· | x)                          → 仅看到问题
教师: π_T(· | x, y*)                      → 问题 + 标准答案（特权信息）
```

两个策略共享完全相同的一组参数 $\theta$，唯一区别是 conditioning 不同。**核心思路**：训练时，教师有条件看到标准答案 $y^*$，能给出更准确的 token 级分布作为监督信号；学生则模仿这种分布。推理时，模型（学生策略）不再有标准答案可用，但已通过训练学到了教师的推理模式，因此能独立产生更好的推理结果。

> 类比：做题时先看答案再理解思路（教师），然后合上答案自己重做（学生）。考试时没有答案，但已经内化了之前的解题方法。

### 2. On-Policy 范式

区别于 off-policy（教师预生成 → 学生模仿），OPSD 是 **学生自己生成轨迹，教师在同一段文本上逐 token 评分**。这解决了 off-policy 中的分布不匹配问题（exposure bias）。

### 3. 教师策略：固定初始策略

**论文明确以 fixed teacher（固定初始策略）为默认设置**：

> "Importantly, we fix the teacher policy to be the initial policy, rather than the currently updating learning policy, as we find this helps stabilize training and implicitly acts as regularization."

即教师权重冻结在训练开始时的初始策略（step 0），学生通过 LoRA 更新。这与"动态"（师生同权重）不同——固定教师能保持稳定的监督信号，防止师生分布过快坍缩。

### 4. 形式化目标

学生采样一条轨迹 $\hat{y} \sim \pi_S(\cdot | x)$，师生在该轨迹的每个 token 位置计算分布差异：

$$
\mathcal{L}_{\text{OPSD}} = \mathbb{E}_{(x, y^*) \sim \mathcal{D}} \left[ \mathbb{E}_{\hat{y} \sim \pi_\theta(\cdot | x)} \left[ \frac{1}{|\hat{y}|} \sum_{t=1}^{|\hat{y}|} \mathcal{D}\big( \pi_T(\cdot | x, y^*, \hat{y}_{<t}) \;\big\|\; \pi_S(\cdot | x, \hat{y}_{<t}) \big) \right] \right]
$$

其中 $\mathcal{D}$ 默认使用 $\beta=0$ 的前向 KL 散度（即 $\text{KL}(\pi_S \| \pi_T)$）。梯度**仅流经学生**，教师作为固定的监督目标。

---

## 二、数据组织

**核心逻辑**: 同一 batch 的 (problem, solution) 对构造出两种不同的输入序列。

- **学生 prompt**: 仅问题（chat template + 思考模式可选）
- **教师 prompt**: 问题 + 标准答案 + 过渡提示（chat template + 思考模式可选）

有两种模式：

| 模式 | 条件 | 教师 prompt 构造方式 |
|:----:|:----:|:----:|
| **直接蒸馏** | `reason_first=False` | 问题 + 标准答案 + transition_prompt 直接拼入 user message |
| **Reason First** | `reason_first=True` | 分两阶段：① 教师先分析答案（reasoning prompt）② 将分析文本 + transition_tokens 拼入 |

```
学生序列: [STUDENT_PROMPT | student_generated_tokens]
教师序列: [TEACHER_PROMPT | student_generated_tokens]  ← 同一段生成
                                   ↑
                            仅在此区域计算 loss
```

**核心代码** (`data_collator.py`):

```python
class SelfDistillationDataCollator:
    def __init__(self, tokenizer, max_length=2048, reason_first=True,
                 student_thinking=False, teacher_thinking=True):
        self.tokenizer = tokenizer
        self.reason_first = reason_first
        # 过渡提示: 告诉教师用自己的话重新推导
        self.transition_prompt = (
            "\n\nAfter reading the reference solution above, make sure you truly understand "
            "the reasoning behind each step — do not copy or paraphrase it. Now, using your "
            "own words and independent reasoning, derive the same final answer to the problem above. "
            "Think step by step, explore different approaches, and don't be afraid to backtrack "
            "or reconsider if something doesn't work out:\n"
        )

    def __call__(self, features):
        student_prompts, teacher_prompts = [], []
        for feature in features:
            problem, solution = feature["problem"], feature["solution"]

            # 学生: 仅问题（默认 student_thinking=False）
            student_msg = [{"role": "user",
                "content": f"Problem: {problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."}]
            student_prompts.append(
                tokenizer.apply_chat_template(
                    student_msg, tokenize=False, add_generation_prompt=True,
                    enable_thinking=self.student_thinking))

            if self.reason_first:
                # reason_first=True: 教师先分析答案，transition 在 training_step 中拼接
                teacher_prompts.append("")  # 占位
            else:
                # 直接蒸馏: 问题 + 标准答案 + 过渡提示
                teacher_msg = [{"role": "user",
                    "content": f"Problem: {problem}\n\n"
                    f"Here is a reference solution to this problem:\n"
                    f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
                    f"{self.transition_prompt}\n"
                    f"Please reason step by step, and put your final answer within \\boxed{{}}."}]
                teacher_prompts.append(
                    tokenizer.apply_chat_template(
                        teacher_msg, tokenize=False, add_generation_prompt=True,
                        enable_thinking=self.teacher_thinking))

        # tokenize + padding, 记录真实 prompt 长度用于 loss masking
        return {"student_prompts": ..., "teacher_prompts": ...,
                "student_prompt_lengths_per_example": ...}
```

**重要细节**:
- 学生默认 `student_thinking=False`（不启用 Qwen3 思考模式），教师默认 `teacher_thinking=True`（启用思考模式），这样教师分布更能提供数学相关 token 的监督信号
- Prompt 以 batch 内最长为标准做 padding，每个样本的真实长度记录在 `student_prompt_lengths_per_example` 中，用于 loss masking

---

## 三、训练流程

OPSD 的训练流程分为两个阶段：

### Phase 1: On-Policy Rollout（`training_step`）

从学生 prompt 采样生成轨迹，构造师生完整序列：

```python
def training_step(self, model, inputs):
    # === 1. 学生生成（on-policy rollout）===
    # 两种方式: transformers generate / vLLM
    if self.use_vllm:
        generated_ids, generated_attention_mask, _, _, _ = \
            self._generate_on_policy_outputs_vllm(inputs, ...)
    else:
        generated_ids, generated_attention_mask, _ = \
            self.generate_on_policy_outputs(model, inputs, ...)

    # 提取生成部分（去除 prompt）
    student_prompt_len = inputs["student_prompt_length"]
    generation_ids = generated_ids[:, student_prompt_len:]

    # 构造学生完整序列
    inputs["student_input_ids"] = generated_ids
    inputs["student_attention_mask"] = generated_attention_mask

    # 构造教师完整序列: [teacher_prompt | generation]
    teacher_prompts = inputs["teacher_prompts"]
    teacher_full_ids = torch.cat([teacher_prompts, generation_ids], dim=1)
    inputs["teacher_input_ids"] = teacher_full_ids

    # 构造 labels（-100 屏蔽 prompt 位置）
    labels = generated_ids.clone()
    for i in range(labels.shape[0]):
        actual_prompt_len = inputs["student_prompt_lengths_per_example"][i].item()
        labels[i, :actual_prompt_len] = -100
    inputs["labels"] = labels

    # === 2. 调用 compute_loss ===
    loss = super().training_step(model, inputs, num_items_in_batch)
    return loss
```

### Phase 2: Loss 计算（`compute_loss`）

```python
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    student_prompt_len = inputs["student_prompt_length"]
    teacher_prompt_len = inputs["teacher_prompt_length"]
    sampled_token_ids = inputs["student_input_ids"][:, student_prompt_len:]

    # === 学生前向（有梯度）===
    outputs_s = model(input_ids=inputs["student_input_ids"],
                      attention_mask=inputs["student_attention_mask"])
    student_logits = outputs_s.logits[:, student_prompt_len - 1 : -1, :]

    # === 教师前向（no_grad, 按策略选择 adapter_context）===
    adapter_context = self._get_teacher_context(model)
    with torch.no_grad(), adapter_context:
        outputs_t = model(input_ids=inputs["teacher_input_ids"],
                          attention_mask=inputs["teacher_attention_mask"])
        teacher_logits = outputs_t.logits[:, teacher_prompt_len - 1 : -1, :]

    # === 计算损失 ===
    if self.use_thinking_machines_loss:
        # Thinking Machines: policy gradient 变体（内存 O(1)）
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits / T, dim=-1)
        # 只取采样到的 token 的 log prob
        student_lp_sampled = torch.gather(student_log_probs, -1, sampled_token_ids.unsqueeze(-1))
        teacher_lp_sampled = torch.gather(teacher_log_probs, -1, sampled_token_ids.unsqueeze(-1))
        advantage = (teacher_lp_sampled - student_lp_sampled).detach()
        loss = -(advantage * student_lp_sampled).mean()
    else:
        # GKD 风格: full-vocabulary JSD
        loss = self.generalized_jsd_loss(
            student_logits, teacher_logits,
            labels=shifted_labels,
            beta=self.beta, temperature=self.temperature,
            top_k=self.top_k_loss, token_clip=self.jsd_token_clip)

    return loss
```

**关键细节**:
- `student_logits[:, prompt_len-1:-1, :]`：取每个 prompt token 之后的位置，第 `t` 个位置的 logits 预测第 `t` 个 generation token
- 学生默认 `do_sample=True`（非确定性采样）
- 前向完成后立即 `del` 大张量并 `empty_cache()` 以节省显存

---

## 四、损失函数

OPSD 支持两种损失函数，通过 `use_tinker_loss` 切换（默认使用 JSD 损失）。

### 1. 广义 JSD 散度（默认）

来自 [GKD](https://huggingface.co/papers/2306.13649)，通过 $\beta$ 控制不对称程度。混合分布定义为：

$$
m = \beta \cdot P_T + (1 - \beta) \cdot P_S
$$

广义 JSD 散度为：

$$
\text{JSD}_\beta(P_S \| P_T) = \beta \cdot \text{KL}(m \| P_T) + (1 - \beta) \cdot \text{KL}(m \| P_S)
$$

| $\beta$ | 行为 | OPSD 默认 |
|:-------:|------|:---------:|
| 0 | **前向 KL** $\text{KL}(P_S \| P_T)$，student 模式逼近 teacher | ✅ |
| 0.5 | 对称 JSD | |
| 1 | **反向 KL** $\text{KL}(P_T \| P_S)$，student 覆盖 teacher 模式 | |

**核心代码**:

```python
@staticmethod
def generalized_jsd_loss(student_logits, teacher_logits, labels=None,
                         beta=0.5, temperature=1.0, reduction="batchmean",
                         top_k=None, token_clip=None):
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature

    # 可选: top-K (只保留 teacher 概率最高的 K 个 token，节省显存)
    if top_k:
        _, indices = torch.topk(teacher_logits, k=top_k, dim=-1)
        student_logits = torch.gather(student_logits, -1, indices)
        teacher_logits = torch.gather(teacher_logits, -1, indices)

    log_p_s = F.log_softmax(student_logits, dim=-1)
    log_p_t = F.log_softmax(teacher_logits, dim=-1)

    if beta == 0:
        # 前向 KL: 直接用 F.kl_div(student || teacher)
        jsd = F.kl_div(log_p_s, log_p_t, reduction="none", log_target=True)
    elif beta == 1:
        # 反向 KL: 直接用 F.kl_div(teacher || student)
        jsd = F.kl_div(log_p_t, log_p_s, reduction="none", log_target=True)
    else:
        # 混合分布 log 概率
        beta_t = torch.tensor(beta, dtype=log_p_s.dtype, device=log_p_s.device)
        mixture = torch.logsumexp(
            torch.stack([log_p_s + torch.log1p(-beta_t), log_p_t + torch.log(beta_t)]), dim=0)

        kl_t = F.kl_div(mixture, log_p_t, reduction="none", log_target=True)
        kl_s = F.kl_div(mixture, log_p_s, reduction="none", log_target=True)
        jsd = beta * kl_t + (1 - beta) * kl_s

    # per-token 裁剪: 防止风格 token 主导
    if token_clip:
        jsd = jsd.clamp(max=token_clip)

    # 屏蔽 prompt 位置
    if labels is not None:
        mask = labels != -100
        jsd = jsd[mask]
        return jsd.sum() / mask.sum()
    return jsd.sum() / jsd.size(0)
```

**注意**：PyTorch 的 `F.kl_div` 的输入顺序与数学定义相反，代码中 `F.kl_div(log_q, log_p)` 实际计算 $\sum p \cdot (\log p - \log q) = \text{KL}(p \| q)$。

### 2. Per-Token JSD 裁剪（Pointwise KL Clipping）

**动机**: 风格 token（如 "wait", "therefore", "alternatively"）的 KL 散度是数学 token 的 **6-15 倍**。若不裁剪，这些风格 token 会主导梯度，导致训练不稳定甚至性能崩溃。

**解决**: `jsd = jsd.clamp(max=clip_value)`，论文默认值 `0.05`，将每个 token 的散度贡献限制在上限内。

### 3. Thinking Machines Loss（可选，`use_tinker_loss=True`）

内存更优的替代方案（$O(1)$ 每 token  vs $O(V)$ 全词表）。将蒸馏视为策略梯度，每个 token 的 advantage 为：

$$
A_t = \underbrace{[\log \pi_{\text{teacher}}(x_t) - \log \pi_{\text{student}}(x_t)]}_{\text{detached}}, \quad
\mathcal{L} = -\mathbb{E}[A_t \cdot \log \pi_{\text{student}}(x_t)]
$$

```python
# 只取采样到的 token 的 log prob（而非全词表）
student_lp = torch.gather(F.log_softmax(student_logits / T, dim=-1), -1, token_ids.unsqueeze(-1))
teacher_lp = torch.gather(F.log_softmax(teacher_logits / T, dim=-1), -1, token_ids.unsqueeze(-1))

advantage = (teacher_lp - student_lp).detach()  # 关键: detach!
loss = -(advantage * student_lp).mean()
```

**论文解读视角**: OPSD 的 token-wise 目标可以解释为**密集奖励策略梯度**（Dense-Reward Policy Gradient），每个 token 的 reward 是 teacher 对该 token 的相对偏好程度。这与 STaR 的序列级二元奖励形成对比。

---

## 五、教师策略

**论文默认**使用 **fixed teacher（固定初始策略）**，认为这能稳定训练并隐式正则化：

> "Importantly, we fix the teacher policy to be the initial policy, rather than the currently updating learning policy."

代码中通过 `adapter_context` 控制教师前向时的权重状态：

```python
# 获取教师前向的 adapter context
if self.use_ema_teacher:
    adapter_context = self._ema_teacher_context(model)
elif self.fixed_teacher and is_peft_model(model):
    adapter_context = self.accelerator.unwrap_model(model).disable_adapter()
else:
    adapter_context = nullcontext()  # 动态: 师生同权重

# 在 training_step 中:
with torch.no_grad(), adapter_context:
    outputs_teacher = model(...)
```

| 策略 | 触发条件 | 教师权重 | 实现方式 |
|:----:|:--------:|:---------|:---------|
| **固定教师 ✅ 默认** | `fixed_teacher=True` + PEFT | 初始策略（step 0 的基座权重） | `model.disable_adapter()` 禁用 LoRA |
| **动态教师** | 以上均未启用 | 当前学生权重（梯度更新中） | `nullcontext()` 无操作 |
| **EMA 教师** | `use_ema_teacher=True` | 权重的指数滑动平均 | `_ema_teacher_context()` 临时替换权重 |

**三种策略对比**:

| 策略 | 优点 | 缺点 | 依赖 |
|:----:|:----|:----|:----:|
| **固定教师** 🏆 | 教师稳定不变，师生分布差异大，监督信号持续有效；论文默认 | 需要 PEFT（LoRA）；教师不会随着学生进步而改进 | `use_peft=True` |
| **动态教师** | 实现简单，无额外存储 | 师生差距小，可能坍缩；论文不推荐 | 无 |
| **EMA 教师** | 平滑稳定，教师渐进改善 | 需额外存储 EMA 参数副本；训练初期无 EMA（首次 optimizer step 才初始化） | `ema_decay` 默认 0.999 |

**注意**: `fixed_teacher` 和 `use_ema_teacher` 互斥。

**EMA 更新逻辑**（在 `EMAUpdateCallback.on_step_end` 中调用）：

```python
def _update_ema(self):
    decay = self.ema_decay
    unwrapped = self.accelerator.unwrap_model(self.model)

    if self._ema_params is None:  # 首次: 快照当前权重
        self._ema_params = {n: p.data.clone() for n, p in
                            unwrapped.named_parameters() if p.requires_grad}
        return

    for name, param in unwrapped.named_parameters():
        if param.requires_grad and name in self._ema_params:
            ema = self._ema_params[name]
            ema.mul_(decay).add_(param.data, alpha=1.0 - decay)
```

---

## 六、Reason First 模式（`reason_first=True`）

**动机**: 教师直接看到标准答案时，其分布与学生分布差异过大，教师可能无法有效评估学生的生成。通过让教师先显式分析标准答案的推理过程，再对学生的生成进行评分，可以提高蒸馏质量。

**Prompt 构造**（`data_collator.py`）:

```
reason_first=False 时（直接蒸馏）:
  教师: [问题 | 标准答案 | transition_prompt]

reason_first=True 时（分两阶段）:
  阶段1-分析: [问题 | 标准答案 | reason_first_prompt]  → 教师生成分析文本
  阶段2-评分: [阶段1完整内容 | transition_tokens]  → 作为新的教师 prompt
```

其中：
- `reason_first_prompt`: 要求教师分析标准答案的推理步骤和解题策略，**不生成 <think> tag**
- `transition_prompt`: 告诉教师用自己的话重新推导答案

**流程**:

```
training_step 中:
1. model.generate(teacher_reasoning_prompts)        → 教师生成分析
2. 拼接: [reasoning_prompt | reasoning | transition] → 新的 teacher_prompts
3. model.generate(student_prompts)                  → 学生生成轨迹
4. 构造完整序列 → 计算 loss（同标准流程）
```

**核心代码**（`opsd_trainer.py`）:

```python
def training_step(self, model, inputs):
    # === Reason First: 教师先分析答案 ===
    if self.reason_first:
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped:
            teacher_reasoning_ids = self.generate_teacher_reasoning(
                unwrapped,
                inputs["teacher_reasoning_prompts"],
                inputs.get("teacher_reasoning_attention_mask"),
            )

        # 提取 reasoning 文本
        reasoning_completions = teacher_reasoning_ids[:, reasoning_prompt_len:]

        # 拼接: [reasoning_prompt | reasoning | transition_tokens]
        teacher_prompts_with_reasoning = torch.cat([
            inputs["teacher_reasoning_prompts"],
            reasoning_completions,
            inputs["teacher_transition_tokens"],
        ], dim=1)

        inputs["teacher_prompts"] = teacher_prompts_with_reasoning
        inputs["teacher_prompt_length"] = teacher_prompts_with_reasoning.shape[1]

    # === 后续: 学生生成 + loss 计算，与标准模式相同 ===
    ...
```

**重要**: Reason First 模式下，教师仍然**不生成**对学生生成的评分——教师只做分析（generation），评分阶段仍只需一次前向传播（one forward pass）。这与 STaR 不同，STaR 需要教师显式生成完整推理链。

---

## 七、实验结果

**训练配置**: 使用 OpenThoughts 数据集，OPSD 仅需 **1024 生成长度 × 1 rollout/问题**，而 GRPO 需 **16k 生成长度 × 8 rollouts/问题**。OPSD 比 GRPO 显著更 token 高效。

### Qwen3-1.7B (Thinking Mode, 100步峰值)

| 指标 | AIME24 | AIME25 | HMMT25 |
|:----:|:------:|:------:|:------:|
| Base (SFT) | 51.5% | 36.7% | 23.1% |
| OPSD | **57.2%** | **41.1%** | **29.2%** |

### Qwen3-8B (Non-Thinking, 50步峰值)

| 指标 | AIME24 | AIME25 | HMMT25 |
|:----:|:------:|:------:|:------:|
| Base (SFT) | 26.4% | 19.7% | 10.8% |
| OPSD | **49.7%** | **35.0%** | **18.3%** |

非思考模式提升更大（+23.3% AIME24），因为思考模式本身已有较强推理能力。

### 关键对比

| 对比项 | SFT | GRPO | OPSD |
|:------:|:---:|:----:|:----:|
| 反馈类型 | 序列级（teacher forcing） | 序列级（二元 outcome） | **Token 级（密集分布匹配）** |
| Token 效率 | 低（exposure bias） | 低（需多 rollout） | **高（1 rollout/问题）** |
| 处理"全对/全错"batch | 依赖数据质量 | **梯度消失**（reward 无差异） | **仍可学习**（分布差异>0） |
| 外部教师 | 需要 | 不需要 | **不需要**（自蒸馏） |

OPSD 的核心优势：即使当 GRPO 因 reward diversity collapse（>50% batch 的 reward 标准差为 0）而梯度消失时，OPSD 的密集蒸馏损失仍能提供学习信号。

---

## 八、参考资料

- [Self-Distilled Reasoner](https://arxiv.org/pdf/2601.18734v3) - OPSD 论文
- [项目主页](https://siyan-zhao.github.io/blog/2026/opsd/)
- [GKD: Generalized Knowledge Distillation](https://huggingface.co/papers/2306.13649)
- [Qwen3](https://github.com/QwenLM/Qwen3)
