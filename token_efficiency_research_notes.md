# 精简推理与 Token 效率优化技术报告

更新时间：2026-06-01

状态：工作稿

适用场景：数学 / 代码 RLVR 训练，尤其是基于 GRPO / PPO 的可验证任务。

目标：在保留长 CoT 推理能力的前提下，系统性降低无效显式输出
token。本文档面向内部技术报告和后续论文草稿沉淀，重点记录问题定义、相关工作、方法假设、实验协议、当前证据和风险边界。

## 摘要与主要结论

本方向具备明确研究价值，也有机会形成有分量的技术创新。

原因有三点：

1. 长 CoT / long thinking 已经成为强推理模型的重要能力来源，但推理 token 成本、延迟和上下文占用都线性甚至更糟地增长。
1. 现有方法还没有统一解决“短而不降能力”的问题。静态长度惩罚容易压掉必要推理；纯 prompt budget 不稳定；压缩蒸馏和 latent reasoning
   又各有工程成本。
1. 代码 RL 场景尤其值得做。相比数学，代码任务有可验证 reward、单测粒度诊断、执行失败类型，天然适合研究“哪些推理 token 真正有用、哪些只是冗余自检或迷路”。

当前技术判断：直接要求模型短答并不稳健。更可行的路线是“先学会长推理，再从成功长轨迹里压缩、内化、动态早停”，并用 correctness-gated
的长度目标防止能力上限下降。

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

## 相关工作与技术脉络

### 1. 长 CoT 是强推理的重要来源，但会自然变长

DeepSeek-R1 系列展示了 RLVR 可以诱导 self-reflection、verification、strategy adaptation 等长推理行为。其
Nature 版本明确指出，模型在解决推理问题时会倾向生成更长 responses，并包含验证、反思、替代路径探索等行为。

启示：

- 长思考确实能提升上限，不能简单视为坏事。
- 长度本身不是目标；关键是区分“有用探索”和“无效重复 / 迷路 / 过度自检”。

来源：

- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- [DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning, Nature 2025](https://www.nature.com/articles/s41586-025-09422-z)

### 2. DAPO / overlong reward shaping 是稳定手段，不是精简能力方案

DAPO 提出四个关键组件：Clip-Higher、Dynamic Sampling、Token-Level Policy Gradient Loss、Overlong
Reward Shaping。Overlong shaping 的定位是降低超长样本带来的 reward noise、稳定训练，而不是让模型学会“最短正确推理”。

当前我们使用的 DAPO 风格长度惩罚属于安全阈值惩罚：接近 `max_new_tokens` 时线性扣分。它适合防止 runaway long
CoT，但不适合作为唯一的精简策略。

风险：

- 阈值太低会惩罚必要推理，降低困难题上限。
- 对错题也惩罚长度时，模型可能学会“短错”。
- 总 reward 上升可能只是少扣长度分，不代表 correctness 提升。

来源：

- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
- [DAPO project page](https://dapo-sia.github.io/)
- [AReaL DAPO documentation](https://inclusionai.github.io/AReaL/algorithms/dapo.html)

### 3. Correctness-gated / shortest-correct RL 是更贴近目标的方向

ShorterBetter 提出 Sample Optimal Length（SOL）：对同一问题采样多个输出，找到最短正确响应长度，并用它作为动态长度信号。其报告在
in-domain / out-of-domain reasoning 上减少 50%-80% 输出长度，同时维持准确率。

这个思路和我们的 GRPO 训练形态很契合：

- 每个 prompt 本来就有 group rollouts。
- 只有 group 内存在正确样本时，长度信号才有意义。
- 可以把“最短正确样本”作为同组 baseline，而不是使用全局固定长度阈值。

来源：

- [ShorterBetter: Guiding Reasoning Models to Find Optimal Inference Length for Efficient Reasoning](https://arxiv.org/abs/2504.21370)

相关工作 Concise Reasoning via RL 从 PPO / GRPO 分析角度指出，错误答案会推动 verbosity，提出在小规模 solvable
problems 上做第二阶段 RL，可显著减少长度并保持或提升准确率。

来源：

- [Concise Reasoning via Reinforcement Learning](https://arxiv.org/abs/2504.05185)

### 4. Iterative pruning / token limit curriculum 可以作为短化阶段

ThinkPrune 使用逐轮收紧 token limit 的 RL：超过限制且未完成的输出给零 reward，然后逐步降低上限。论文报告在 AIME24 上将
DeepSeek-R1-Distill-Qwen-1.5B 的推理长度减半，性能仅下降约 2%。

这个方向的优点是工程简单，和现有 reward 兼容；缺点是限制过硬，容易在早期破坏困难题探索。因此更适合作为“模型已经会做题之后”的短化阶段，而不是从头训练。

来源：

- [ThinkPrune: Pruning Long Chain-of-Thought of LLMs via Reinforcement Learning](https://arxiv.org/abs/2504.01296)

### 5. CoT 压缩蒸馏：长轨迹作为 teacher，短轨迹作为 student 行为

C3oT 的核心是先压缩长 CoT，再用 conditioned training 同时学习长 CoT 和短 CoT 的对应关系，推理时生成短 CoT。论文报告可压缩超过
50% 长度而不损害效果。

TokenSkip 从 token 重要性角度压缩 CoT，报告在 Qwen2.5-14B-Instruct + GSM8K 上从 313 tokens 降到 181
tokens，性能下降不到 0.4%。

这类方法对我们尤其重要：它更接近“把长思考能力内化为短表达”。但需要构造高质量压缩数据，不能简单摘要，否则容易丢关键推理。

来源：

- [C3oT: Generating Shorter Chain-of-Thought without Compromising Effectiveness](https://arxiv.org/abs/2412.11664)
- [TokenSkip: Controllable Chain-of-Thought Compression in LLMs](https://arxiv.org/abs/2502.12067)

### 6. Budget-conditioned reasoning：按题目难度动态分配 token

s1 的 budget forcing 展示了测试时控制思考长度的简单机制：模型想停时追加 "Wait" 可延长思考，或强制终止来减少思考。这不是最终训练方案，但说明
token budget 可以作为控制变量。

Token-Budget-Aware LLM Reasoning 进一步指出，合理 token budget 能压缩推理，但 budget
选择本身很关键，因此需要按题目复杂度动态调整。

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
- Coconut：用连续 hidden state 作为 thought 输入，而不是解码成自然语言 token；报告在需要搜索 / backtracking
  的逻辑任务上有更好 accuracy-efficiency tradeoff。

这些方向更有研究创新性，但工程侵入性更强。短期不建议作为第一优先级；中期可以从 pause token / hidden scratchpad 的轻量版本做起。

来源：

- [Implicit Chain of Thought Reasoning via Knowledge Distillation](https://arxiv.org/abs/2311.01460)
- [Think before you speak: Training Language Models With Pause Tokens](https://proceedings.iclr.cc/paper_files/paper/2024/hash/76917808731dae9e6d62c2a7a6afb542-Abstract-Conference.html)
- [Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629)
- [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769)

### 8. Overthinking 评测正在形成，但还没有成为统一标准

已有工作专门研究 o1-like / R1-like 模型在简单题上的 overthinking，提出 outcome 和 process 角度的 efficiency
metrics，并观察到许多长推理模型会在简单题上浪费 token。

这说明我们的方向不是局部工程优化，而是大模型推理研究中的明确开放问题。

来源：

- [Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs](https://arxiv.org/abs/2412.21187)
- [Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models](https://arxiv.org/abs/2503.16419)

## 研究空白与技术机会

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

## 研究假设

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

## 技术路线与实验设计

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
1. 对长 CoT 做压缩，保留必要推理和最终答案。
1. 训练 short-mode SFT。
1. 再接 correctness-gated efficient RL。

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

## 阶段计划

1. 保持当前 DAPO overlong penalty，只用于防止接近 `max_new_tokens` 的 runaway。
1. 补齐并检查指标：`raw_task_reward`、`overlong_penalty`、`response_len`、`finish_reason/length`。
1. 在当前训练日志上做长度分桶分析：
   - reward vs response_len
   - correct rate vs response_len bucket
   - code failure type vs response_len bucket
1. 实现 group-relative shortest-correct reward，先只在数学任务上 A/B，再迁移到 code。
1. 收集一批成功长轨迹，做压缩 SFT 的数据构造试验。

## 风险与开放问题

- 短化是否应该优化 total response_len，还是只优化 reasoning_len，不压 code_len？
- 对代码任务，是否应该奖励更短自然语言，但不惩罚代码长度？
- group 内 shortest correct 是否会偏向偶然短对的样本，导致鲁棒性下降？
- 长 CoT 的能力是否能通过 SFT 压缩保留，还是必须继续 RL？
- 是否需要引入 hidden / pause tokens 来真正内化推理，而不只是压缩文本？

## 参考文献与阅读清单

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

## 当前实现与实验进展

本节采用技术报告格式沉淀当前进展，不按时间顺序堆叠记录。每个实验单元必须绑定研究问题、可复现实验配置、核心指标、证据等级和未决风险。尚未完成完整训练 A/B
的内容只标记为工程验证或待验证，不写成效果结论。

证据等级定义：

| 等级 | 含义                    | 可用于支撑的结论                 |
| ---- | ----------------------- | -------------------------------- |
| L0   | 方案设计                | 只能说明方法可实现，不能说明有效 |
| L1   | 单元测试 / 局部函数验证 | 可以说明机制符合预期             |
| L2   | 单次训练观察            | 可以说明存在现象，不能排除偶然性 |
| L3   | 控制变量 A/B            | 可以比较方法优劣                 |
| L4   | 多模型 / 多数据集复现   | 可以作为稳定结论或论文主结果     |

### 实现资产总览

| 模块                      | 状态                   | 作用                                           | 代码 / 配置位置                                                     | 证据等级 |
| ------------------------- | ---------------------- | ---------------------------------------------- | ------------------------------------------------------------------- | -------- |
| 原始任务 reward 保留      | 已落地                 | 区分 correctness gain 与 penalty gain          | `areal/trainer/ppo/actor.py`, `areal/trainer/ppo/actor_qun_team.py` | L1       |
| Overlong penalty 分解指标 | 已落地                 | 观测 DAPO 长度惩罚对总 reward 的贡献           | `areal/trainer/ppo/actor.py`, `areal/trainer/ppo/actor_qun_team.py` | L1       |
| 正确 / 错误样本长度统计   | 已落地                 | 判断长度下降是否发生在正确样本上               | `correct_seq_len`, `incorrect_seq_len`                              | L1       |
| rollout 停止原因统计      | 已落地                 | 识别是否仍存在 max length 截断                 | `areal/workflow/rlvr.py`, `areal/workflow/rlvr_qun_team.py`         | L1       |
| shortest-correct reward   | 已实现，待完整训练验证 | 只对正确样本做 group-relative 长度优化         | `areal/utils/functional/functional.py`                              | L1       |
| shortest-correct 运行参数 | 已接入                 | 支持训练脚本直接配置 alpha / group size 等变量 | `examples/*/grpo_template*.yaml`, `run_trainer_mtp.sh`              | L1       |

### 实验矩阵

| 编号 | 研究问题                                                     | 当前状态                       | 关键变量                                                       | 主指标                                                                                       | 成功判据                                              | 证据等级 |
| ---- | ------------------------------------------------------------ | ------------------------------ | -------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ----------------------------------------------------- | -------- |
| E0   | DAPO overlong penalty 能否抑制 runaway long CoT 且不伤正确率 | 已有单次训练观察，待补完整曲线 | `overlong_tokens`, `overlong_penalty_factor`, `max_new_tokens` | `raw_task_reward`, `overlong_penalty`, `response_len`, `finish_reason/length`                | `finish_reason/length` 下降，`raw_task_reward` 不下降 | L2       |
| E1   | 当前观测面能否解释 reward 上升来源                           | 已落地                         | 指标完整性                                                     | `raw_task_reward`, `task_reward`, `overlong_penalty`, `correct_seq_len`, `incorrect_seq_len` | 能区分 correctness gain 与 penalty gain               | L1       |
| E2   | shortest-correct reward 是否按预期只惩罚正确长样本           | 单元测试通过，待训练验证       | `alpha`, `reward_threshold`, `min_correct`, `max_penalty`      | `shortest_correct_penalty`, `shortest_correct_active`, `shortest_correct_target_len`         | 错误样本 penalty 为 0，正确长样本 penalty 符合公式    | L1       |
| E3   | shortest-correct 的有效 alpha 区间是多少                     | 待执行                         | `alpha=0.03/0.05/0.1`                                          | `raw_task_reward`, `correct_seq_len`, `incorrect_seq_len`, `response_len`                    | 正确样本长度下降，`raw_task_reward` 不下降            | L0       |
| E4   | 代码任务中应压缩 reasoning tokens 还是 total response tokens | 待执行                         | reasoning / code token 拆分方式                                | pass rate by length bucket, failure type by length bucket                                    | 找到不损害代码鲁棒性的压缩目标                        | L0       |

### E0：DAPO Overlong Penalty 初步观察

实验目的：确认 DAPO 风格 overlong penalty 是否主要解决 max length 截断与 runaway long CoT，而不是把模型推向短错。

参考配置：

| 参数                            | 取值   |
| ------------------------------- | ------ |
| `actor.overlong_reward_penalty` | `true` |
| `actor.overlong_tokens`         | `512`  |
| `actor.overlong_penalty_factor` | `1.0`  |
| `actor.reward_scaling`          | `10.0` |
| `actor.reward_bias`             | `-0.5` |
| `actor.kl_ctl`                  | `0.0`  |

当前观察：训练约 200-300 step 后，`rollout/reward` 开始提升，`response_len` 同期下降。该现象只能说明 overlong
penalty 对长度有抑制作用，尚不能证明任务正确率提升。

需要补齐的证据：

| 待补项                                                  | 目的                           |
| ------------------------------------------------------- | ------------------------------ |
| run id / 模型 / 数据集 / `max_new_tokens` / `n_samples` | 保证实验可复现                 |
| `raw_task_reward` 与 `overlong_penalty` 同图曲线        | 判断 reward 提升来源           |
| `correct_seq_len` 与 `incorrect_seq_len` 曲线           | 判断长度下降是否集中在正确样本 |
| validation pass rate / accuracy                         | 排除训练 reward 虚高           |

### E2：Shortest-Correct Reward 方法规格

方法目标：把长度优化限制在正确样本内部，使 reward shaping 指向“更短的正确解”，而不是“更短的任意输出”。

形式化定义：

```text
correct_i = raw_task_reward_i >= reward_threshold
SOL_group = min(response_len_j | correct_j)

penalty_i =
  0, if sample i incorrect
  0, if group correct count < min_correct
  -alpha * max(0, response_len_i - SOL_group) / max(SOL_group, min_shortest_len),
    if sample i correct
```

默认配置：

| 参数                    | 默认值                 | 作用                             |
| ----------------------- | ---------------------- | -------------------------------- |
| `enabled`               | `false`                | 默认关闭，避免影响既有实验       |
| `alpha`                 | `0.1`                  | 长度惩罚强度                     |
| `reward_threshold`      | `1.0`                  | 判定正确样本的 raw reward 阈值   |
| `min_correct`           | `2`                    | group 内至少多少个正确样本才激活 |
| `group_size`            | `${gconfig.n_samples}` | 与 GRPO group rollout 对齐       |
| `normalize_by_shortest` | `true`                 | 按最短正确长度归一化 penalty     |
| `max_penalty`           | `1.0`                  | 限制长度项最大负贡献             |
| `min_shortest_len`      | `1`                    | 避免极短分母不稳定               |

机制保护：

| 风险                       | 保护设计                     |
| -------------------------- | ---------------------------- |
| 错误样本被奖励短输出       | 错误样本 penalty 固定为 0    |
| 单个偶然短对样本主导 group | `min_correct >= 2` 后才激活  |
| 长度项压过任务 reward      | `max_penalty` 截断           |
| 不同题目长度尺度不可比     | `normalize_by_shortest=true` |

已验证内容：

| 测试文件                                | 验证点                                     |
| --------------------------------------- | ------------------------------------------ |
| `tests/test_shortest_correct_reward.py` | group 内 shortest correct 目标长度计算正确 |
| `tests/test_shortest_correct_reward.py` | partial reward 不会被误判为正确样本        |
| `tests/test_shortest_correct_reward.py` | 可与 overlong penalty 叠加                 |

当前边界：该方法目前只有机制级验证，尚未完成完整训练 A/B，不能声称优于 DAPO baseline。

### 下一轮实验协议

优先实验：在其他参数不变的条件下启用 shortest-correct reward，并先取 `alpha=0.05`。`0.05` 是相对默认 `0.1`
的保守减半设置，强于 `0.03`，更容易在短训练中观察趋势，同时降低长度项过强的风险。

固定条件：

| 类别                                       | 要求                                   |
| ------------------------------------------ | -------------------------------------- |
| 模型 / 数据集 / batch / `n_samples`        | 与 E0 baseline 保持一致                |
| `max_new_tokens` / temperature / optimizer | 与 E0 baseline 保持一致                |
| overlong penalty                           | 保持当前设置，或明确记录是否关闭       |
| 训练步数                                   | 至少覆盖 E0 中出现长度变化的 step 区间 |

观察指标：

| 指标                       | 解释                             |
| -------------------------- | -------------------------------- |
| `raw_task_reward`          | 判断 correctness 是否保持        |
| `task_reward`              | 观察最终进入 advantage 的 reward |
| `response_len`             | 总体长度变化                     |
| `correct_seq_len`          | 正确样本长度变化，主指标         |
| `incorrect_seq_len`        | 检查是否出现短错倾向             |
| `shortest_correct_active`  | 判断 group 内长度信号覆盖率      |
| `shortest_correct_penalty` | 判断长度项量级是否合理           |
| `finish_reason/length`     | 检查 max length 截断是否下降     |

通过标准：

| 标准                                                  | 判定             |
| ----------------------------------------------------- | ---------------- |
| `raw_task_reward` 不低于 E0 baseline                  | 能力未明显受损   |
| `correct_seq_len` 下降                                | 正确样本变短     |
| `incorrect_seq_len` 不异常下降                        | 未明显诱导短错   |
| `shortest_correct_penalty` 量级小于任务 reward 主信号 | 长度项未主导训练 |
| validation accuracy / pass rate 不下降                | 训练内收益可外推 |

若 `alpha=0.05` 导致 `raw_task_reward` 或 validation 明显下降，则回退到 `alpha=0.03`；若长度指标几乎无变化且
correctness 稳定，再考虑 `alpha=0.1` 或延长训练步数。

### 报告化待补材料

| 材料                         | 用途                             |
| ---------------------------- | -------------------------------- |
| E0 与 E2 的完整 run metadata | 支撑可复现性                     |
| 指标曲线截图或导出表         | 支撑实验结论                     |
| length bucket 分析           | 解释长度与正确率关系             |
| 代码任务 failure type 分析   | 判断短化是否伤害边界条件和鲁棒性 |
| 成功与失败样例对比           | 支撑机制解释和论文案例           |
