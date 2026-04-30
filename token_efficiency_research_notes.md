# 精简回答与 Token 效率研究笔记

更新时间：2026-04-30

目标：在数学 / 代码 RLVR 场景中，让模型保留长 CoT 带来的推理能力，同时减少无效显式输出 token。本文档用于持续记录调研、假设、实验设计、结果和下一步规划。

## 结论先行

这是一个有研究价值、也有机会做出有分量创新的方向。

原因有三点：

1. 长 CoT / long thinking 已经成为强推理模型的重要能力来源，但推理 token 成本、延迟和上下文占用都线性甚至更糟地增长。
2. 现有方法还没有统一解决“短而不降能力”的问题。静态长度惩罚容易压掉必要推理；纯 prompt budget 不稳定；压缩蒸馏和 latent reasoning 又各有工程成本。
3. 代码 RL 场景尤其值得做。相比数学，代码任务有可验证 reward、单测粒度诊断、执行失败类型，天然适合研究“哪些推理 token 真正有用、哪些只是冗余自检或迷路”。

我当前判断：直接要求模型短答不现实。更可行路线是“先学会长推理，再从成功长轨迹里压缩、内化、动态早停”，并用 correctness-gated 的长度目标防止能力上限下降。

## 问题定义

我们不只是追求短输出，而是追求 token efficiency：

```text
在保持 pass@1 / pass@k / accuracy 不下降或少下降的前提下，
最小化 response_len、reasoning_len、wall-clock latency 和 rollout cost。
```

推荐主指标：

- `accuracy` / `pass@1` / `pass@k`
- `response_len.avg`
- `correct_response_len.avg`：只统计答对样本的长度
- `tokens_per_correct = total_response_tokens / num_correct`
- `accuracy_at_budget(B)`：固定 token budget 下的准确率
- Pareto frontier：`accuracy` vs `response_len`
- `finish_reason/length`：是否被截断
- `no_eos_ratios`
- `raw_task_reward`、`overlong_penalty`、`task_reward`

## 研究现状

### 1. 长 CoT 是强推理的重要来源，但会自然变长

DeepSeek-R1 系列展示了 RLVR 可以诱导 self-reflection、verification、strategy adaptation 等长推理行为。其 Nature 版本明确指出，模型在解决推理问题时会倾向生成更长 responses，并包含验证、反思、替代路径探索等行为。

启示：

- 长思考确实能提升上限，不能简单视为坏事。
- 长度本身不是目标；关键是区分“有用探索”和“无效重复 / 迷路 / 过度自检”。

来源：

- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- [DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning, Nature 2025](https://www.nature.com/articles/s41586-025-09422-z)

### 2. DAPO / overlong reward shaping 是稳定手段，不是精简能力方案

DAPO 提出四个关键组件：Clip-Higher、Dynamic Sampling、Token-Level Policy Gradient Loss、Overlong Reward Shaping。Overlong shaping 的定位是降低超长样本带来的 reward noise、稳定训练，而不是让模型学会“最短正确推理”。

当前我们使用的 DAPO 风格长度惩罚属于安全阈值惩罚：接近 `max_new_tokens` 时线性扣分。它适合防止 runaway long CoT，但不适合作为唯一的精简策略。

风险：

- 阈值太低会惩罚必要推理，降低困难题上限。
- 对错题也惩罚长度时，模型可能学会“短错”。
- 总 reward 上升可能只是少扣长度分，不代表 correctness 提升。

来源：

- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
- [DAPO project page](https://dapo-sia.github.io/)
- [AReaL DAPO documentation](https://inclusionai.github.io/AReaL/algorithms/dapo.html)

### 3. Correctness-gated / shortest-correct RL 是更贴近目标的方向

ShorterBetter 提出 Sample Optimal Length（SOL）：对同一问题采样多个输出，找到最短正确响应长度，并用它作为动态长度信号。其报告在 in-domain / out-of-domain reasoning 上减少 50%-80% 输出长度，同时维持准确率。

这个思路和我们的 GRPO 训练形态很契合：

- 每个 prompt 本来就有 group rollouts。
- 只有 group 内存在正确样本时，长度信号才有意义。
- 可以把“最短正确样本”作为同组 baseline，而不是使用全局固定长度阈值。

来源：

- [ShorterBetter: Guiding Reasoning Models to Find Optimal Inference Length for Efficient Reasoning](https://arxiv.org/abs/2504.21370)

相关工作 Concise Reasoning via RL 从 PPO / GRPO 分析角度指出，错误答案会推动 verbosity，提出在小规模 solvable problems 上做第二阶段 RL，可显著减少长度并保持或提升准确率。

来源：

- [Concise Reasoning via Reinforcement Learning](https://arxiv.org/abs/2504.05185)

### 4. Iterative pruning / token limit curriculum 可以作为短化阶段

ThinkPrune 使用逐轮收紧 token limit 的 RL：超过限制且未完成的输出给零 reward，然后逐步降低上限。论文报告在 AIME24 上将 DeepSeek-R1-Distill-Qwen-1.5B 的推理长度减半，性能仅下降约 2%。

这个方向的优点是工程简单，和现有 reward 兼容；缺点是限制过硬，容易在早期破坏困难题探索。因此更适合作为“模型已经会做题之后”的短化阶段，而不是从头训练。

来源：

- [ThinkPrune: Pruning Long Chain-of-Thought of LLMs via Reinforcement Learning](https://arxiv.org/abs/2504.01296)

### 5. CoT 压缩蒸馏：长轨迹作为 teacher，短轨迹作为 student 行为

C3oT 的核心是先压缩长 CoT，再用 conditioned training 同时学习长 CoT 和短 CoT 的对应关系，推理时生成短 CoT。论文报告可压缩超过 50% 长度而不损害效果。

TokenSkip 从 token 重要性角度压缩 CoT，报告在 Qwen2.5-14B-Instruct + GSM8K 上从 313 tokens 降到 181 tokens，性能下降不到 0.4%。

这类方法对我们尤其重要：它更接近“把长思考能力内化为短表达”。但需要构造高质量压缩数据，不能简单摘要，否则容易丢关键推理。

来源：

- [C3oT: Generating Shorter Chain-of-Thought without Compromising Effectiveness](https://arxiv.org/abs/2412.11664)
- [TokenSkip: Controllable Chain-of-Thought Compression in LLMs](https://arxiv.org/abs/2502.12067)

### 6. Budget-conditioned reasoning：按题目难度动态分配 token

s1 的 budget forcing 展示了测试时控制思考长度的简单机制：模型想停时追加 "Wait" 可延长思考，或强制终止来减少思考。这不是最终训练方案，但说明 token budget 可以作为控制变量。

Token-Budget-Aware LLM Reasoning 进一步指出，合理 token budget 能压缩推理，但 budget 选择本身很关键，因此需要按题目复杂度动态调整。

对我们而言，这意味着不应该追求一个全局固定短度。更合理的是：

- 简单题：短答 / 少 CoT。
- 中等题：精简 CoT。
- 难题：允许长 CoT 或 fallback long mode。

来源：

- [s1: Simple test-time scaling](https://arxiv.org/abs/2501.19393)
- [Token-Budget-Aware LLM Reasoning](https://arxiv.org/abs/2412.18547)

### 7. Internal / latent reasoning 是长期方向

这类工作试图减少显式语言 CoT，而把推理迁移到 hidden states、pause tokens 或 continuous thoughts 中。

代表：

- Implicit CoT：把显式 CoT teacher 的推理蒸馏到 hidden states，使推理“纵向”发生在层间，而不是“横向”生成文本 token。
- Pause tokens：给模型额外 hidden computation，再开始输出答案。
- Quiet-STaR：学习在每个 token 前生成内部 rationale，提高困难 token 预测和问答能力。
- Coconut：用连续 hidden state 作为 thought 输入，而不是解码成自然语言 token；报告在需要搜索 / backtracking 的逻辑任务上有更好 accuracy-efficiency tradeoff。

这些方向更有研究创新性，但工程侵入性更强。短期不建议作为第一优先级；中期可以从 pause token / hidden scratchpad 的轻量版本做起。

来源：

- [Implicit Chain of Thought Reasoning via Knowledge Distillation](https://arxiv.org/abs/2311.01460)
- [Think before you speak: Training Language Models With Pause Tokens](https://proceedings.iclr.cc/paper_files/paper/2024/hash/76917808731dae9e6d62c2a7a6afb542-Abstract-Conference.html)
- [Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629)
- [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769)

### 8. Overthinking 评测正在形成，但还没有成为统一标准

已有工作专门研究 o1-like / R1-like 模型在简单题上的 overthinking，提出 outcome 和 process 角度的 efficiency metrics，并观察到许多长推理模型会在简单题上浪费 token。

这说明我们的方向不是局部工程优化，而是大模型推理研究中的明确开放问题。

来源：

- [Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs](https://arxiv.org/abs/2412.21187)
- [Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models](https://arxiv.org/abs/2503.16419)

## 当前空白与可创新点

### 空白 1：代码 RL 的 token efficiency 研究不足

多数已有工作集中在 GSM8K、MATH、AIME、commonsense。代码任务有更丰富的错误类型：

- compile error
- runtime error
- timeout
- wrong answer
- partial pass
- pass all tests

这允许我们研究不同错误类型和 CoT 长度之间的关系，例如：

- 长 CoT 是否降低 compile error 但增加 timeout？
- 反复自检是否真的提高 hidden tests pass rate？
- 简短解法是否更容易产生边界条件错误？

### 空白 2：静态长度惩罚和能力保持之间缺少机制性解释

目前常见做法是调 penalty coefficient。但更重要的问题是：

```text
什么时候长度是必要推理？
什么时候长度是无效重复？
```

我们可以用 group 内正确样本、失败类型、step-level IG reward、token-level logprob / entropy 变化来做更细粒度分析。

### 空白 3：长到短的课程设计仍不成熟

已有路线分散：

- 先长 RL，再短 RL。
- 压缩长 CoT 后 SFT。
- budget-conditioned multi-mode。
- latent / hidden reasoning。

有机会提出一个更系统的 pipeline：

```text
long exploration -> success trajectory filtering -> compression / pruning ->
short-mode SFT -> correctness-gated efficient RL -> adaptive fallback
```

## 建议研究假设

### H1：只对正确样本施加长度优化，能比 DAPO 静态惩罚更好保持能力

长度奖励只在 `raw_task_reward > 0` 或 group 内有正确样本时生效。错误样本不奖励短，避免“短错”。

可能实现：

```text
length_reward_i =
  0, if sample i incorrect
  -alpha * max(0, len_i - SOL_group) / max(SOL_group, 1), if sample i correct
```

其中 `SOL_group = min(len_j | sample j correct)`。

### H2：先长后短优于从头短训

阶段 A 用较宽松长度限制学能力；阶段 B 使用成功长轨迹做压缩 SFT / RL。预期直接短训会降低困难题探索和正确率。

### H3：长度目标应按题目难度自适应

简单题可以强短；难题允许长。可用信号：

- base model pass rate
- group reward variance
- prompt length / test case complexity
- early rollout entropy
- 单测失败类型

### H4：短化主要应删除冗余结构，而不是删除关键推理步骤

可重点统计和压缩：

- 重复 restatement
- 多次 "wait/check again" 但没有新信息
- 反复枚举已经排除的路径
- 结论前的过度自我确认

### H5：代码任务中“简洁代码 + 足够边界分析”比“长自然语言 CoT”更重要

代码 reward 最终看可执行程序。长 CoT 可能帮助探索算法，但过长解释未必提升代码质量。可以研究：

- CoT tokens vs code tokens 的比例
- 正确样本中的 code length / reasoning length
- 是否存在“短 reasoning + robust code”的高效模式

## 建议实验路线

### 实验 0：观测基线

目的：确认当前 DAPO overlong penalty 影响的是 correctness 还是少扣分。

需要曲线：

- `rollout/reward`
- `ppo_actor/raw_task_reward`
- `ppo_actor/overlong_penalty`
- `ppo_actor/task_reward`
- `ppo_actor/response_len`
- `ppo_actor/correct_seq_len`
- `ppo_actor/incorrect_seq_len`
- `ppo_actor/update/approx_kl`
- `ppo_actor/update/entropy`
- `ppo_actor/update/clip_ratio`
- `rollout/finish_reason/length`

### 实验 1：DAPO penalty sweep

目的：找“只防 runaway、不伤能力”的 penalty 区间。

变量：

- `overlong_tokens`
- `overlong_penalty_factor`
- `max_new_tokens`

判断：

- response_len 是否下降
- raw_task_reward 是否不降
- finish_reason/length 是否下降
- AIME / MATH / code validation 是否保持

### 实验 2：Group-relative shortest-correct reward

目的：实现 H1。

在 GRPO group 内计算：

- correct mask
- shortest correct length
- correct-only length penalty / bonus

关键对照：

- DAPO static overlong
- DAPO + shortest-correct
- shortest-correct only

### 实验 3：成功轨迹压缩 SFT

流程：

1. 用长 RL 模型收集答对 rollouts。
2. 对长 CoT 做压缩，保留必要推理和最终答案。
3. 训练 short-mode SFT。
4. 再接 correctness-gated efficient RL。

压缩方式：

- 规则压缩：删除重复检查和无信息语句。
- teacher 压缩：要求保留关键推理、边界条件、最终答案格式。
- self-compression：让模型自己把长解改写为短解，再用 verifier 过滤。

### 实验 4：Budget-conditioned 多模式

训练数据加入控制 token：

```text
<think_long>
<think_short>
<answer_only>
```

推理策略：

- 默认 `<think_short>`
- 若 confidence 低或 validation 失败，fallback `<think_long>`

对代码任务，可用 public tests / sampled tests 失败作为 fallback 信号。

### 实验 5：代码任务专门分析

统计：

- reasoning tokens
- code block tokens
- compile/runtime/wrong answer/timeout 分布
- pass rate by response length bucket
- pass rate by code length bucket
- 长 CoT 是否更容易出现 no code / malformed code

目标是找到代码任务特有的短化策略，而不是照搬数学 CoT。

## 近期最小可行计划

1. 保持当前 DAPO overlong penalty，只用于防止接近 `max_new_tokens` 的 runaway。
2. 补齐并检查指标：`raw_task_reward`、`overlong_penalty`、`response_len`、`finish_reason/length`。
3. 在当前训练日志上做长度分桶分析：
   - reward vs response_len
   - correct rate vs response_len bucket
   - code failure type vs response_len bucket
4. 实现 group-relative shortest-correct reward，先只在数学任务上 A/B，再迁移到 code。
5. 收集一批成功长轨迹，做压缩 SFT 的数据构造试验。

## 开放问题

- 短化是否应该优化 total response_len，还是只优化 reasoning_len，不压 code_len？
- 对代码任务，是否应该奖励更短自然语言，但不惩罚代码长度？
- group 内 shortest correct 是否会偏向偶然短对的样本，导致鲁棒性下降？
- 长 CoT 的能力是否能通过 SFT 压缩保留，还是必须继续 RL？
- 是否需要引入 hidden / pause tokens 来真正内化推理，而不只是压缩文本？

## 阅读清单

核心必读：

- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [DAPO](https://arxiv.org/abs/2503.14476)
- [ShorterBetter](https://arxiv.org/abs/2504.21370)
- [Concise Reasoning via RL](https://arxiv.org/abs/2504.05185)
- [ThinkPrune](https://arxiv.org/abs/2504.01296)
- [C3oT](https://arxiv.org/abs/2412.11664)
- [TokenSkip](https://arxiv.org/abs/2502.12067)

长期方向：

- [s1: Simple test-time scaling](https://arxiv.org/abs/2501.19393)
- [Token-Budget-Aware LLM Reasoning](https://arxiv.org/abs/2412.18547)
- [Implicit Chain of Thought via KD](https://arxiv.org/abs/2311.01460)
- [Pause Tokens](https://proceedings.iclr.cc/paper_files/paper/2024/hash/76917808731dae9e6d62c2a7a6afb542-Abstract-Conference.html)
- [Quiet-STaR](https://arxiv.org/abs/2403.09629)
- [Coconut](https://arxiv.org/abs/2412.06769)

综述 / 评测：

- [Do NOT Think That Much for 2+3=?](https://arxiv.org/abs/2412.21187)
- [Stop Overthinking: A Survey on Efficient Reasoning for LLMs](https://arxiv.org/abs/2503.16419)

## 实验记录

### 2026-04-30

- 已加入训练指标：
  - `ppo_actor/raw_task_reward`
  - `ppo_actor/overlong_penalty`
  - `rollout/finish_reason/{stop,length,abort}`
- 当前观察：
  - 加入超长惩罚后，训练 200-300 step 开始 reward 提升，response_len 下降。
- 待分析：
  - reward 提升来自 correctness 还是少扣长度分。
  - response_len 下降是否伴随 code / math validation 下降。
