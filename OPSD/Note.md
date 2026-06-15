# On-Policy Self-Distillation (OPSD) 原理与核心代码

**论文**: [Self-Distilled Reasoner](https://arxiv.org/pdf/2601.18734v3) | **代码**: [github.com/siyan-zhao/OPSD](https://github.com/siyan-zhao/OPSD) | **基础模型**: Qwen3

---

## 一、核心思想

### 1. 自蒸馏框架

同一个模型通过**上下文条件**的差异同时扮演学生和教师：

```
学生: π(· | problem)                        → 学生分布
教师: π(· | problem + solution + prompt)    → 教师分布
```

目标是让学生分布向教师分布靠拢，使模型没有标准答案时也能产生高质量推理。

### 2. On-Policy 范式

区别于 off-policy（教师生成 → 学生模仿），OPSD 是 **学生自己生成轨迹，教师在同一段文本上评分**。

### 3. 形式化目标

$$
\mathcal{L}_{\text{OPSD}} = \mathbb{E}_{\hat{y} \sim \pi_\theta(\cdot | c_s)} \left[ \sum_{t} \mathcal{D}\big(\pi_\theta(\cdot | c_t, \hat{y}_{<t}) \;\big\|\; \pi_\theta(\cdot | c_s, \hat{y}_{<t})\big) \right]
$$

---

## 二、数据组织

**核心逻辑**: 同一 batch 的 (problem, solution) 对构造出两种不同的输入序列。

- **学生 prompt**: 仅问题（chat template）
- **教师 prompt**: 问题 + 标准答案 + 过渡提示（chat template）

```
学生序列: [STUDENT_PROMPT | student_generated_tokens]
教师序列: [TEACHER_PROMPT | student_generated_tokens]  ← 同一段生成
                                ↑
                         仅在此区域计算 loss
```

**核心代码** (`data_collator.py`):

```python
class SelfDistillationDataCollator:
    def __call__(self, features):
        student_prompts, teacher_prompts = [], []
        for feature in features:
            problem, solution = feature["problem"], feature["solution"]

            # 学生: 仅问题
            student_msg = [{"role": "user",
                "content": f"Problem: {problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."}]
            student_prompts.append(
                tokenizer.apply_chat_template(student_msg, tokenize=False, add_generation_prompt=True))

            # 教师: 问题 + 标准答案 + 过渡提示
            teacher_msg = [{"role": "user",
                "content": f"Problem: {problem}\n\n=== Reference Solution ===\n{solution}\n=== End ===\n{transition_prompt}"}]
            teacher_prompts.append(
                tokenizer.apply_chat_template(teacher_msg, tokenize=False, add_generation_prompt=True))

        # tokenize + padding, 记录真实 prompt 长度用于 loss masking
        return {"student_prompts": ..., "teacher_prompts": ...,
                "student_prompt_lengths_per_example": ...}
```

---

## 三、训练流程 (training_step)

**核心代码** (`opsd_trainer.py`):

```python
def training_step(self, model, inputs):
    # === 1. 学生生成（on-policy rollout）===
    generated_ids = model.generate(
        inputs["student_prompts"],
        generation_config=GenerationConfig(
            max_new_tokens=max_completion_length,
            temperature=temperature, do_sample=True))

    generation_ids = generated_ids[:, student_prompt_len:]

    # 构造师生完整序列
    inputs["student_input_ids"] = torch.cat([student_prompt, generation_ids], dim=1)
    inputs["teacher_input_ids"] = torch.cat([teacher_prompt, generation_ids], dim=1)

    # === 2. 学生前向 ===
    out_s = model(inputs["student_input_ids"])
    student_logits = out_s.logits[:, prompt_len-1:-1, :]
    del out_s

    # === 3. 教师前向（no_grad）===
    with torch.no_grad(), adapter_context:   # 教师策略决定 adapter_context
        out_t = model(inputs["teacher_input_ids"])
        teacher_logits = out_t.logits[:, prompt_len-1:-1, :]
        del out_t

    # === 4. 计算 JSD 损失 ===
    loss = generalized_jsd_loss(
        student_logits, teacher_logits,
        labels=shifted_labels,  # -100 屏蔽 prompt 位置
        beta=beta, temperature=temperature,
        top_k=top_k_loss, token_clip=jsd_token_clip)

    return loss
```

---

## 四、损失函数

### 1. 广义 JSD 散度

来自 [GKD](https://huggingface.co/papers/2306.13649)，通过 $\beta$ 控制不对称程度：

$$
M = \beta \cdot P_t + (1 - \beta) \cdot P_s
$$

$$
\text{JSD}_\beta(P_s \| P_t) = \beta \cdot \text{KL}(M \| P_t) + (1 - \beta) \cdot \text{KL}(M \| P_s)
$$

| $\beta$ | 行为 | OPSD 默认 |
|:-------:|------|:---------:|
| 0 | 前向 KL, student 对齐 teacher | ✅ |
| 0.5 | 对称 JSD | |
| 1 | 反向 KL, student 覆盖 teacher | |

**核心代码**:

```python
@staticmethod
def generalized_jsd_loss(student_logits, teacher_logits, labels=None,
                         beta=0.5, temperature=1.0, top_k=None, token_clip=None):
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature

    # 可选: top-K (只保留 teacher 概率最高的 K 个 token)
    if top_k:
        _, indices = torch.topk(teacher_logits, k=top_k, dim=-1)
        student_logits = torch.gather(student_logits, -1, indices)
        teacher_logits = torch.gather(teacher_logits, -1, indices)

    log_p_s = F.log_softmax(student_logits, dim=-1)
    log_p_t = F.log_softmax(teacher_logits, dim=-1)

    # 混合分布 log 概率
    mixture = torch.logsumexp(
        torch.stack([log_p_s + torch.log1p(-beta), log_p_t + torch.log(beta)]), dim=0)

    kl_t = F.kl_div(mixture, log_p_t, reduction="none", log_target=True)
    kl_s = F.kl_div(mixture, log_p_s, reduction="none", log_target=True)
    jsd = beta * kl_t + (1 - beta) * kl_s

    # per-token 裁剪: 防止风格 token 主导
    if token_clip:
        jsd = jsd.clamp(max=token_clip)

    if labels is not None:
        jsd = jsd[labels != -100]  # 屏蔽 prompt
    return jsd.sum() / (labels != -100).sum()
```

### 2. Per-Token JSD 裁剪

**动机**: 风格 token（"wait", "alternatively"）的 KL 散度是数学 token 的 **6-15 倍**，会主导梯度。

**解决**: `jsd = jsd.clamp(max=clip_value)`，思考模式典型值 `0.05~0.06`。

### 3. Thinking Machines Loss（可选）

内存更优的替代方案（$O(1)$ vs $O(V)$）：

$$
A_t = \underbrace{[\log \pi_{\text{teacher}}(x_t) - \log \pi_{\text{student}}(x_t)]}_{\text{detached}}, \quad
\mathcal{L} = -\mathbb{E}[A_t \cdot \log \pi_{\text{student}}(x_t)]
$$

```python
advantage = (teacher_log_probs_sampled - student_log_probs_sampled).detach()
loss = -(advantage * student_log_probs_sampled).mean()
```

---

## 五、教师策略

```python
# 动态: 无操作，师生同权重
adapter_context = nullcontext()

# 固定: 禁用 LoRA adapter，用基座模型（初始化权重）
adapter_context = model.disable_adapter()   # 需 use_peft=True

# EMA: 指数滑动平均
adapter_context = self._ema_teacher_context(model)  # α=0.999
```

**EMA 更新**:

```python
def _update_ema(self):
    if self._ema_params is None:  # 首次: 快照当前权重
        self._ema_params = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        return
    for name, param in model.named_parameters():
        if param.requires_grad:
            self._ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
```

| 策略 | 教师权重 | 优缺点 |
|------|---------|--------|
| 动态 | 当前学生权重 | 简单，但师生差距小 |
| 固定 | 初始策略（LoRA 基座） | 教师稳定，师生差异大，需 PEFT |
| EMA | 权重滑动平均 | 平滑稳定，需额外存储 |

---

## 六、Reason First 模式

**动机**: 教师直接"看到答案"与学生"没看到答案"分布差异过大。

**改进**: 教师先分析标准答案的推理过程，再评分。

**流程**:
1. 教师分析答案 → 2. 拼接分析文本到提示词 → 3. 学生生成 → 4. 教师评分

```python
# 教师分析生成
teacher_reasoning = model.generate(inputs["teacher_reasoning_prompts"], ...)
# 拼接 → 完整教师上下文
inputs["teacher_prompts"] = concat(reasoning_prompt, reasoning, transition)
```

---

## 七、实验结果

### Qwen3-1.7B (Thinking, 100步峰值)

| 指标 | AIME24 | AIME25 | HMMT25 |
|:----:|:------:|:------:|:------:|
| Base | 51.5% | 36.7% | 23.1% |
| OPSD | **57.2%** | **41.1%** | **29.2%** |

### Qwen3-8B (Non-Thinking, 50步峰值)

| 指标 | AIME24 | AIME25 | HMMT25 |
|:----:|:------:|:------:|:------:|
| Base | 26.4% | 19.7% | 10.8% |
| OPSD | **49.7%** | **35.0%** | **18.3%** |

非思考模式提升更大，因为思考模式本身已有较强推理能力。

---

## 八、参考资料

- [Self-Distilled Reasoner](https://arxiv.org/pdf/2601.18734v3) - OPSD 论文
- [项目主页](https://siyan-zhao.github.io/blog/2026/opsd/)
- [GKD: Generalized Knowledge Distillation](https://huggingface.co/papers/2306.13649)
- [Qwen3](https://github.com/QwenLM/Qwen3)
