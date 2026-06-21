# 精简推理与 Token 效率优化技术报告

更新时间：2026-06-21

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

当前技术判断：直接要求模型短答并不稳健。更可行的路线是“先学会长推理，再从成功长轨迹里压缩、内化、动态早停”。correctness-gated 长度目标仍有研究价值，但当前
shortest-correct 实现没有达到预期：它显著压短 response_len，同时拉低 eval score，不能作为下一阶段的默认优先路线。

2026-06-02 追加结论：

- BigMath 0.5B 16k 系列中，`shortest_correct_reward` 的 `alpha=0.05` 已经把 eval 平均长度从约 7.3k 压到约
  3.8k，但 macro eval reward 从约 0.445 降到约 0.379。
- `alpha=0.2` 呈现更强剂量效应，eval 平均长度降到约 1.5k，macro eval reward 降到约 0.299。
- 分数下降主要发生在更吃推理预算的 AIME / HMMT 集合上，说明该方案更像是在压掉必要推理，而不是删除冗余 token。
- 下一步不应继续沿用当前 shortest 配置做简单 alpha sweep，而应先做 overlong baseline
  完整分析、长度分桶诊断，再重设计更弱、更晚激活的长度信号。

2026-06-11 追加结论：

- 父目录下 5 组 BigMath 0.5B 对照实验已完成横向分析：
  - `30k_overlong_8k` 仍是当前质量上限，best macro eval reward `0.521`，hard-set reward `0.420`，但
    eval 平均长度约 `13.7k`，hard-set 平均长度约 `15.3k`。
  - `16k_overlong_4k` 是当前 16k 内更稳的质量 / 长度折中，best macro eval reward `0.480`，平均长度约 `7.4k`。
  - `16k_overlong_8k` 进一步把平均长度降到约 `5.2k`，但 best macro eval reward 只有 `0.450`，hard-set
    reward 降到 `0.335` 左右。
  - `shortest_correct alpha=0.05/0.2` 都出现明显过短化。`alpha=0.05` final eval 平均长度约
    `3.4k`，macro reward `0.377`；`alpha=0.2` final eval 平均长度约 `1.6k`，macro reward
    `0.316`。
- `overlong_tokens` 的含义需要按 `max_new_tokens - overlong_tokens` 理解：16k + 8k penalty 从 8k
  以后开始扣，16k + 4k penalty 从 12k 以后开始扣，30k + 8k penalty 从 22k 以后才开始扣。因此 `16k_overlong_8k`
  不是“更宽松”，而是对 16k 训练更早施压。
- 当前证据说明：固定 overlong 可以做安全阈值和 token 预算控制，但会随阈值提前而伤害困难题；group shortest
  会被组内偶然短正确样本牵引，导致推理预算坍缩。两者都没有实现按题目难度自适应平衡。
- 下一步最值得优先做的不是继续 shortest alpha sweep，而是做“质量约束下的自适应长度目标”：先锁定 16k budget，在不显著低于
  `16k_overlong_4k` 的 hard-set reward 前提下，优化 tokens per correct。

2026-06-17 追加结论：

- `adaptive_length_reward` 首个 16k 完整训练已完成。当前 run 使用的是我们自己的 `mode=length_quantile`，不是原版
  `mode=alp`。
- 相比已完成的 `16k_overlong_8k`，adaptive final macro eval reward 从 `0.449` 提升到 `0.470`，AIME
  avg 从 `0.378` 提升到 `0.403`，HMMT25 从 `0.204` 提升到 `0.235`；训练末段 response_len 从 `4.1k` 降到
  `2.9k`。
- adaptive 的 best eval 出现在 step `3599`，macro reward `0.488`，AIME avg `0.431`，hard reward
  `0.383`；final step `6639` 有回落，但仍优于 16k overlong completed baseline。
- checkpoint 保存间隔为 500 step，没有正好 `3599` 的 checkpoint；已保存候选中优先复评 `globalstep3999`，其 macro
  reward `0.475`，高于 final `0.470`。
- `30k_overlong_8k` 仍是质量上限，best macro reward `0.521`，但 eval 平均长度约 `13.6k`，训练末段
  response_len 约 `9.7k`，且日志中有大量 rollout timeout；它不应作为 16k token-efficiency 主线。
- 当前主线建议：保留 `length_quantile` adaptive 作为 16k 方向，先做 checkpoint 复评和小矩阵稳健性验证；不再继续
  shortest-correct 简单 alpha sweep。

2026-06-18 追加结论：

- ALP / APL 术语说明：论文方法名是 ALP（Adaptive Length Penalty），实验目录中使用 `alp_reward`；本文优先写 ALP，引用历史
  run 名或旧结论时保留 `alp` / APL。
- ALP ablation 已有一次完整 16k 训练负结果：
  `mtp_grpo_muon_16k_groupsize_16_lr_4e-5_alp_reward_alpha005_min4096_20260617`。
- 该 run 配置文件中 `length_normalizer=null`，但 actor 调用时会用 `self.config.max_new_tokens` 作为
  fallback，因此 实际 normalizer 是 `16384`。该旧实现下 `alpha=0.05` 等效 per-token `beta≈3.05e-6`，约为
  ALP 论文报告 `beta=1e-7` 的 30 倍。因此这个 run 是“过强 ALP 长度成本”的负结果，不能直接坐实 paper-faithful ALP 失败。
- 原文 ALP 使用 group solve rate 缩放 per-token 长度成本，不使用最短正确样本长度；这和我们之前的 shortest-correct /
  SOL_group 方案不同。当前 APL 代码路径也不使用 `correct_only`、`min_solve_rate`、
  `target_quantile`、`min_correct` 或 `min_target_len` 这些 length-quantile 保护条件，因此 run 名中的
  `min4096` 对 APL 实际不起保护作用。
- ALP 训练快速坍缩：step 0 macro reward `0.415`、eval len `11.1k`；step 399 已降到 macro
  `0.296`、eval len `2.4k`；step 499 仅剩 macro `0.148`、eval len `416`；step 999 起 eval
  reward 为 `0`，final step `6639` 的 eval response_len 为 `1`。
- 训练侧也同步坍缩，`ppo_actor/response_len` 在 step `2266` 首次稳定到 `1`，last100 train raw reward 为
  `0`。这比 shortest-correct 的过短化更严重，属于 reward objective 方向失控，而不是 checkpoint selection 问题。
- 因此，旧参数 ALP 当前不应作为 16k token-efficiency 主线，也不应阻塞 `length_quantile` adaptive 的
  checkpoint 复评和小矩阵验证。代码已补 `alp_beta`；2026-06-21 已完成 paper-scale `alp_beta=1e-7`
  复跑，standalone ALP 仍为负，详见后续追加结论。

2026-06-21 追加结论：

- Paper-scale ALP `alp_beta=1e-7` 完整 run 已完成：
  `mtp_grpo_muon_16k_groupsize_16_lr_4e-5_alp_beta_1e-7_20260618`。日志开头有一个重复 step 0
  的旧尺度记录；从第二条 step 0 起，平均等效 beta 约为 `0.8e-7` 到 `1.4e-7`，可视为 paper-scale ALP run。
- 该 run 不再出现旧参数的 `response_len=1` 完全坍缩，但仍发生严重 under-thinking：final macro / hard reward 为
  `0.254 / 0.137`，final eval len / hard len 为 `656 / 739`。最好的有效非初始点是 step `99`，macro /
  hard reward `0.400 / 0.283`，平均 eval len 约 `9.0k`；之后随着长度继续下降，质量同步掉落。
- 训练侧后期 raw reward 会回升到约 `0.57`，但 eval hard reward 只有约 `0.14`，说明 ALP 训练 reward
  可以被短答案或训练分布 shortcut 欺骗，不能作为 token efficiency 成功信号。
- 结论：paper-scale ALP standalone 在 BigMath 0.5B 16k 设置下也不适合作为主线；它是比旧参数更温和但仍为负的
  ablation。该结果不能证明任何 gated / scheduled ALP 变体必然失败，但已经足以把当前主线固定在 `length_quantile`
  adaptive。
- 下一步先讨论是否需要做更弱 beta、late schedule 或 correctness gate 的 ALP 变体；在讨论前不建议继续占用主线训练资源。

2026-06-21 查重与路线图补充：

- 这个方向不是无人区。`adaptive length penalty`、`concise reasoning`、`shortest-correct / SOL`、budget
  control、 Lagrangian target length、difficulty-aware reward shaping
  都已经有近邻工作。论文写法不能宣称“首次提出 adaptive length penalty”。
- 目前没有检索到和我们完全一致的组合：`solve-rate gating + correct-only + correct-length quantile target + lower-bound protection`。我们的可防守创新点应聚焦在
  **safe adaptive target construction**，而不是泛泛的“长度惩罚”。
- 只证明 ALP 失败不够。ALP 是最近邻和一个重要失败例子，但顶会级 claim 至少还需要同协议实测 3-5 个高风险相关方法： R1-Alpha /
  correct-only mean-std penalty、LASER-D / difficulty-aware length shaping、Leash /
  dynamic target-length penalty、LAPO-style successful-length distribution，以及必要时的 ARLCP /
  reflection-aware variant。
- 路线图调整：先把相关方法拆成可复现 baselines，统一在 BigMath 0.5B 16k 协议下比较 macro reward、hard reward、eval
  length、tokens-per-correct 和 Pareto frontier；只有我们在这些强基线下仍保持稳定优势，才推进顶会主叙事。

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

### 3. Correctness-gated / shortest-correct RL 有价值，但初版实现失败

ShorterBetter 提出 Sample Optimal Length（SOL）：对同一问题采样多个输出，找到最短正确响应长度，并用它作为动态长度信号。其报告在
in-domain / out-of-domain reasoning 上减少 50%-80% 输出长度，同时维持准确率。

这个思路和我们的 GRPO 训练形态很契合：

- 每个 prompt 本来就有 group rollouts。
- 只有 group 内存在正确样本时，长度信号才有意义。
- 可以把“最短正确样本”作为同组 baseline，而不是使用全局固定长度阈值。

但最新实验显示，直接把 group 内最短正确样本作为强相对目标并不稳健。我们的初版 shortest-correct 会持续奖励同题中最短正确解，使模型把 target
length 推到几百 token 量级；数学难题 eval 上，这会伤害必要推理并降低正确率。因此，该方向需要重设计保护条件，而不是继续作为首选方案。

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

### 9. 近邻查重：当前方法不能只和 ALP 对比

2026-06-21 检索结论：efficient reasoning + length-aware RL 正在快速拥挤。ALP
是最近邻之一，但不是唯一需要击败的对照。下表按“撞车风险”和“是否必须同协议实测”整理。

| 方法线                            | 代表工作                                                               | 核心机制                                                        | 与我们最接近处                                 | 关键差异                                                                                  | 实测优先级          |
| --------------------------------- | ---------------------------------------------------------------------- | --------------------------------------------------------------- | ---------------------------------------------- | ----------------------------------------------------------------------------------------- | ------------------- |
| solve-rate absolute penalty       | [Just Enough Thinking / ALP](https://arxiv.org/abs/2506.05256)         | 用 group solve rate 缩放 per-token 长度成本                     | 同样使用 group solve rate 估计难度             | 不使用正确样本长度分位数 target；不 correct-only；无下界保护                              | 已测，standalone 负 |
| correct-only length penalty       | [Training LMs to Reason Efficiently](https://arxiv.org/abs/2502.04463) | 只惩罚正确响应，并用同 prompt rollout 长度均值 / 方差归一化     | correct-gated，group-wise，容易实现            | target 是 mean/std normalization，不是 correct-length quantile；没有显式 hard lower bound | P0                  |
| shortest-correct / SOL            | [ShorterBetter](https://arxiv.org/abs/2504.21370)                      | 多采样中最短正确响应作为 sample optimal length                  | 使用正确样本长度作为动态目标                   | target 是 min，过激；我们用 quantile + floor                                              | 已测近似版，负      |
| difficulty-aware dynamic reward   | LASER-D / length-aware efficient reasoning 系列                        | 根据题目难度动态调节长度惩罚                                    | difficulty-aware length shaping                | 需核对难度估计和 target 构造；不一定 correct-only quantile                                | P0                  |
| successful-length distribution    | [LAPO](https://arxiv.org/abs/2507.15758)                               | 两阶段学习 successful solution length distributions             | 从成功轨迹长度分布学习合适预算                 | 更偏两阶段 internalization / distribution guidance；不是单步 quantile reward              | P1                  |
| target-length dual control        | [Leash](https://arxiv.org/abs/2512.21540)                              | Lagrangian / primal-dual 动态调长度 penalty，使输出接近目标长度 | 动态调 penalty 强度，优化长度-质量 tradeoff    | 需要外部 target length；不是 group correct quantile target                                | P1                  |
| reflection-aware compression      | ARLCP / related concise-reasoning RL                                   | 同时惩罚 reflection 和 length，按复杂度调节                     | correct-response statistics + complexity-aware | 依赖 reflection token 定义；更侧重冗余反思结构                                            | P1                  |
| segment-wise / step-wise shaping  | DSS-GRPO, SAS 等                                                       | think/answer 分段或 step-level advantage selection              | 防止压掉答案段，关注过程 token                 | 改动 RL objective 或 token 分段，不只是 reward shaping                                    | P2                  |
| budget-conditioned / mode control | s1, Token-Budget-Aware, AdaptThink / DAST                              | 显式 budget 或思考模式控制                                      | 动态分配推理预算                               | 需要推理时控制 token 或多模式训练；不是纯 RL reward                                       | P2                  |
| compression distillation          | C3oT, TokenSkip, CLORE 等                                              | 压缩长 CoT，再 SFT / 蒸馏                                       | 使用成功长轨迹压缩表达                         | 数据管线不同，可作为后续 stage，不是直接 baseline                                         | P2                  |

查重后的定位：

```text
不要写：我们首次提出 adaptive length penalty。

应该写：现有方法多使用 absolute penalty、shortest-correct target、全局 target length、
mean/std normalization 或显式 budget control；这些方法容易在困难题上 under-think，
或需要手动预算。我们提出 correctness-gated quantile target：用 solve rate 判断何时压缩，
用同组正确样本长度分位数定义软 target，并用 lower bound 防止困难题推理预算坍缩。
```

因此，ALP 负结果只能支撑“absolute solve-rate penalty 在本设置下不稳”。要支撑顶会级主张，必须把上表 P0/P1
中的若干方法在同一训练协议下重跑或实现近似复现。

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

### H1：只对正确样本施加长度优化，未必比 DAPO 静态惩罚更稳

初始假设是：长度奖励只在 `raw_task_reward > 0` 或 group 内有正确样本时生效，错误样本不奖励短，从而避免“短错”。

可能实现：

```text
length_reward_i =
  0, if sample i incorrect
  -alpha * max(0, len_i - SOL_group) / max(SOL_group, 1), if sample i correct
```

其中 `SOL_group = min(len_j | sample j correct)`。

当前证据推翻了“初版 SOL_group
一定更稳”的乐观判断。虽然错误样本不直接被奖励短，但正确样本内部的相对长度竞争会把模型推向越来越短的正确轨迹；当训练题存在短解或偶然短对样本时，困难 eval
所需的长推理会被一起压掉。

修正后的假设：

```text
长度优化必须同时满足 correctness-gated、difficulty-aware、late-stage、lower-bound
约束，才能比 DAPO 式安全阈值惩罚更稳。
```

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

目的：验证 H1。当前初版结果为负，不再作为默认优先路线。

在 GRPO group 内计算：

- correct mask
- shortest correct length
- correct-only length penalty / bonus

关键对照：

- DAPO static overlong
- DAPO + shortest-correct
- shortest-correct only

当前结论：`shortest-correct only` 在 BigMath 0.5B 16k 上显著降低长度，但同步降低 eval score。后续只应尝试带更强保护的变体。

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

### 实验 5：ALP 消融

目的：确认原版绝对长度成本是否能作为 adaptive 的简单对照。

当前结果：两个 ALP standalone ablation 都为负。

- 旧 normalized-alpha APL 参数已失败，训练坍缩到 1-token 输出。该 run 配置文件中
  `length_normalizer=null`，但实际经由 actor fallback 使用 `max_new_tokens=16384`。`alpha=0.05`
  等效 per-token `beta≈3.05e-6`，明显强于论文报告的 `beta=1e-7`，只能作为过强参数失败参考。
- paper-scale `alp_beta=1e-7` 完整训练不再 1-token 坍缩，但 final macro / hard reward 只有
  `0.254 / 0.137`，final eval len / hard len 只有 `656 / 739`。step `99` 可在几乎不降质量的情况下把 eval
  len 从约 `11.1k` 降到 `9.0k`，但后续训练会继续压短并显著降质。
- 因此，ALP standalone 不再作为当前 16k token-efficiency 主线。若后续仍需要 ALP 相关 ablation，应明确加入
  correctness / difficulty gate、late schedule 或更弱 beta，并与 `length_quantile` 分开报告。

### 实验 6：代码任务专门分析

统计：

- reasoning tokens
- code block tokens
- compile/runtime/wrong answer/timeout 分布
- pass rate by response length bucket
- pass rate by code length bucket
- 长 CoT 是否更容易出现 no code / malformed code

目标是找到代码任务特有的短化策略，而不是照搬数学 CoT。

## 阶段计划

1. 保持 DAPO overlong penalty 作为安全阈值和 baseline，不把它作为主要短化机制。
1. 将 `length_quantile` adaptive 作为当前 16k 主线方案，优先复评 `globalstep3999` 和 final checkpoint。
1. 暂停当前 shortest-correct 配置，不再把 `alpha=0.05` 作为优先实验。
1. 在当前训练日志上做长度分桶分析：
   - reward vs response_len
   - correct rate vs response_len bucket
   - code failure type vs response_len bucket
1. 对 adaptive 做小矩阵稳健性验证，重点围绕 `alpha`、`min_solve_rate`、`target_quantile`、
   `min_target_len` 和 `max_penalty`。
1. `mode=alp` 已完成过强参数与 paper-scale `alp_beta=1e-7` 两次负结果。standalone ALP
   不再占用当前主线资源；若论文需要继续讨论 ALP，应先决定是否值得补 gating / schedule / 更弱 beta 作为独立 ablation。
1. 增加强相关方法同协议实测，优先级高于继续扩展我们自己的大矩阵。P0 baselines 包括 correct-only mean/std length
   penalty、LASER-D / difficulty-aware shaping；P1 baselines 包括 Leash-style target-length
   dual control、 LAPO-style successful-length distribution 和 ARLCP-style
   reflection-aware penalty。
1. 收集一批成功长轨迹，优先做压缩 SFT 的数据构造试验。

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
- [Just Enough Thinking / ALP](https://arxiv.org/abs/2506.05256)
- [Learn to Reason Efficiently with Adaptive Length-based Reward Shaping / LASER-D](https://arxiv.org/abs/2505.15612)
- [Training Language Models to Reason Efficiently](https://arxiv.org/abs/2502.04463)
- [ShorterBetter](https://arxiv.org/abs/2504.21370)
- [Concise Reasoning via RL](https://arxiv.org/abs/2504.05185)
- [LAPO: Internalizing Reasoning Efficiency via Length-Adaptive Policy Optimization](https://arxiv.org/abs/2507.15758)
- [Leash: Adaptive Length Penalty and Reward Shaping for Efficient Large Reasoning Model](https://arxiv.org/abs/2512.21540)
- [Stop Unnecessary Reflection / ARLCP](https://arxiv.org/abs/2602.12113)
- [Shorter Thoughts, Same Answers / DSS-GRPO](https://arxiv.org/abs/2603.07598)
- [Stabilizing Efficient Reasoning with Step-Level Advantage Selection / SAS](https://arxiv.org/abs/2604.24003)
- [Apriel-1.5-OpenReasoner](https://arxiv.org/abs/2604.02007)
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

| 模块                      | 状态                           | 作用                                           | 代码 / 配置位置                                                     | 证据等级 |
| ------------------------- | ------------------------------ | ---------------------------------------------- | ------------------------------------------------------------------- | -------- |
| 原始任务 reward 保留      | 已落地                         | 区分 correctness gain 与 penalty gain          | `areal/trainer/ppo/actor.py`, `areal/trainer/ppo/actor_qun_team.py` | L1       |
| Overlong penalty 分解指标 | 已落地                         | 观测 DAPO 长度惩罚对总 reward 的贡献           | `areal/trainer/ppo/actor.py`, `areal/trainer/ppo/actor_qun_team.py` | L1       |
| 正确 / 错误样本长度统计   | 已落地                         | 判断长度下降是否发生在正确样本上               | `correct_seq_len`, `incorrect_seq_len`                              | L1       |
| rollout 停止原因统计      | 已落地                         | 识别是否仍存在 max length 截断                 | `areal/workflow/rlvr.py`, `areal/workflow/rlvr_qun_team.py`         | L1       |
| shortest-correct reward   | 已实现，初版训练负结果         | 只对正确样本做 group-relative 长度优化         | `areal/utils/functional/functional.py`                              | L3       |
| shortest-correct 运行参数 | 已接入，需重调保护参数         | 支持训练脚本直接配置 alpha / group size 等变量 | `examples/*/grpo_template*.yaml`, `run_trainer_mtp.sh`              | L3       |
| adaptive length reward    | 已实现，首个 16k 正结果        | solve-rate 自适应正确样本分位数长度目标        | `areal/utils/functional/functional.py`                              | L3       |
| ALP mode                  | 已实现；paper-scale 训练负结果 | 对所有有效样本按 solve rate 与长度线性扣分     | `mode=alp`, `alp_beta`, `length_normalizer`                         | L2       |

### 实验矩阵

| 编号 | 研究问题                                                                   | 当前状态                                                                      | 关键变量                                                             | 主指标                                                                                       | 成功判据                                                         | 证据等级 |
| ---- | -------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- | -------- |
| E0   | DAPO overlong penalty 能否抑制 runaway long CoT 且不伤正确率               | 已有 3 组 overlong 对照，待补分桶曲线                                         | `overlong_tokens`, `overlong_penalty_factor`, `max_new_tokens`       | `raw_task_reward`, `overlong_penalty`, `response_len`, `finish_reason/length`                | `finish_reason/length` 下降，`raw_task_reward` 不下降            | L2       |
| E1   | 当前观测面能否解释 reward 上升来源                                         | 已落地                                                                        | 指标完整性                                                           | `raw_task_reward`, `task_reward`, `overlong_penalty`, `correct_seq_len`, `incorrect_seq_len` | 能区分 correctness gain 与 penalty gain                          | L1       |
| E2   | shortest-correct reward 是否按预期只惩罚正确长样本                         | 机制通过，训练效果未达预期                                                    | `alpha`, `reward_threshold`, `min_correct`, `max_penalty`            | `shortest_correct_penalty`, `shortest_correct_active`, `shortest_correct_target_len`         | 错误样本 penalty 为 0；但 eval 不降才可继续                      | L3       |
| E3   | shortest-correct 的有效 alpha 区间是多少                                   | `0.05/0.2` 已失败，暂停简单扫描                                               | `alpha=0.005/0.01`，加强保护条件                                     | `raw_task_reward`, `correct_seq_len`, `incorrect_seq_len`, `response_len`, eval reward       | 长度下降但 eval 不下降；否则判为压掉必要推理                     | L3       |
| E4   | solve-rate length-quantile adaptive penalty 能否按题目难度平衡长度与正确率 | 16k 完整训练完成，优于 16k overlong completed baseline                        | `min_solve_rate`, `target_quantile`, `min_target_len`, `max_penalty` | `adaptive_length_penalty`, `adaptive_length_solve_rate`, `response_len`, eval reward         | hard-set reward 基本不降且 eval len 有实质下降                   | L3       |
| E5   | ALP mode 是否能作为 adaptive 对照                                          | 旧 normalized-alpha ALP 坍缩；paper-scale `alp_beta=1e-7` 严重 under-thinking | `mode=alp`, `alp_beta`, gating / schedule                            | eval reward, response_len, raw_task_reward, collapse step                                    | standalone 已判负；若继续变体，hard reward 不能低于 16k baseline | L2       |
| E6   | 代码任务中应压缩 reasoning tokens 还是 total response tokens               | 待执行                                                                        | reasoning / code token 拆分方式                                      | pass rate by length bucket, failure type by length bucket                                    | 找到不损害代码鲁棒性的压缩目标                                   | L0       |
| E7   | 高风险相关方法是否已能超过 `length_quantile` adaptive                      | 待执行；ALP 和 shortest 已有负结果                                            | mean/std correct penalty, LASER-D, Leash, LAPO, ARLCP                | macro reward, hard reward, eval len, tokens-per-correct, Pareto frontier                     | 至少 3 个高风险 baseline 同协议不优于我们，或明确边界条件        | L0       |

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

2026-06-11 完整对照补充：

日志来源：`dataset/ag_data/logs/areal/experiments/logs/root/ailab_slm_0_5b_think_bigmath/`。

| run               | 有效长度方案                                   | penalty 起点 | final macro / hard reward | best macro / hard reward | final eval len / hard len | last100 train len |
| ----------------- | ---------------------------------------------- | ------------ | ------------------------- | ------------------------ | ------------------------- | ----------------- |
| `16k_overlong_4k` | `max_new_tokens=16384`, `overlong_tokens=4096` | 12k          | 0.457 / 0.342             | 0.480 / 0.370            | 7.3k / 8.1k               | 5.7k              |
| `16k_overlong_8k` | `max_new_tokens=16384`, `overlong_tokens=8192` | 8k           | 0.449 / 0.334             | 0.450 / 0.335            | 5.2k / 5.7k               | 4.0k              |
| `30k_overlong_8k` | `max_new_tokens=30720`, `overlong_tokens=8192` | 22k          | 0.512 / 0.408             | 0.521 / 0.420            | 13.6k / 15.1k             | 9.6k              |

解释：

- `30k_overlong_8k` 是当前质量上限，但平均 eval 长度超过 13k，不能作为 token efficiency 方案本身。
- `16k_overlong_8k` 相比 `16k_overlong_4k` 约省 29% eval token，但 best macro reward 下降约 3.0
  个点，hard-set reward 下降约 3.5 个点。它是有损压缩 baseline，而不是自适应平衡方案。
- 三组 overlong 训练后 `finish_reason/length` 基本接近 0，说明当前差异主要不是“是否解决截断”，而是 penalty
  起点改变了模型愿意保留的推理预算。
- 后续 overlong 评估应使用 Pareto
  口径：`hard_reward`、`avg_response_len`、`tokens_per_correct`，而不是只看 rollout reward。

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

当前边界：该方法机制级验证通过，但训练 A/B 显示初版配置不优于 DAPO baseline。它可以作为研究方向保留，但不能继续作为默认优先方案。

### E3：Shortest-Correct 负结果

实验问题：当前 shortest-correct 是否能在降低 response_len 的同时保持 eval score。

参考实验：

| trial                                                                        | 长度方案              | 关键参数                                                                  | 训练状态                            |
| ---------------------------------------------------------------------------- | --------------------- | ------------------------------------------------------------------------- | ----------------------------------- |
| `mtp_grpo_muon_16k_groupsize_16_lr_4e-5_overlong_penalty_4k_20260530`        | DAPO overlong         | `overlong_tokens=4096`                                                    | 运行到 step 4771，55 次 eval        |
| `mtp_grpo_muon_16k_groupsize_16_lr_4e-5_shortest_correct_alpha_005_20260601` | shortest-correct only | `alpha=0.05`, `min_correct=2`, `max_penalty=0.75`, `min_shortest_len=512` | 运行到 step 5310，61 次 eval        |
| `mtp_grpo_muon_16k_groupsize_16_lr_4e-5_shortest_correct_alpha_02_20260530`  | shortest-correct only | `alpha=0.2`, `min_correct=2`, `max_penalty=0.75`, `min_shortest_len=512`  | 运行到 step 6002，training complete |

同 step 关键对比：

| 实验                  | eval step | macro eval reward | macro eval len | train response_len |
| --------------------- | --------- | ----------------- | -------------- | ------------------ |
| 16k overlong baseline | 1327      | 0.445             | 7.3k           | 4.9k               |
| shortest `alpha=0.05` | 1327      | 0.379             | 3.8k           | 1.4k               |
| 16k overlong baseline | 1399      | 0.441             | 7.4k           | 5.1k               |
| shortest `alpha=0.2`  | 1399      | 0.299             | 1.5k           | 0.82k              |

alpha=0.05 在 step 1327 的分数据集对比：

| dataset | 16k overlong reward / len | shortest `alpha=0.05` reward / len |
| ------- | ------------------------- | ---------------------------------- |
| MATH500 | 0.904 / 3.7k              | 0.851 / 1.1k                       |
| AIME24  | 0.419 / 8.4k              | 0.379 / 4.9k                       |
| AIME25  | 0.329 / 8.0k              | 0.242 / 4.1k                       |
| AIME26  | 0.385 / 8.5k              | 0.275 / 4.5k                       |
| HMMT25  | 0.190 / 7.9k              | 0.148 / 4.2k                       |

机制解释：

- `shortest_correct` 只惩罚正确样本，但在 group 内形成“最短正确解”相对优势。
- 惩罚在 `reward_bias=-0.5`、`reward_scaling=10` 和 group reward norm 前加入，长度差异会进入 advantage。
- `min_shortest_len=512` 只限制分母，不保证生成长度下界；target length 仍会被推到几百 token。
- 数学难题需要较长推理预算，AIME / HMMT 下降更明显，说明该方法压掉了必要推理。

结论：当前 shortest-correct 达到了“降长度”，但没有达到“短而不降能力”。后续不应继续简单尝试 `alpha=0.03/0.05/0.1`
这类扫描，而应改成带强保护的弱约束。

完整日志补充结果：

| 实验                  | final macro / hard reward | best macro / hard reward | final eval len / hard len | last100 train len | shortest target / active |
| --------------------- | ------------------------- | ------------------------ | ------------------------- | ----------------- | ------------------------ |
| `shortest_alpha_0.05` | 0.377 / 0.259             | 0.422 / 0.307            | 3.4k / 4.0k               | 1.05k             | target 528, active 0.594 |
| `shortest_alpha_0.2`  | 0.316 / 0.192             | 0.413 / 0.301            | 1.6k / 1.8k               | 0.62k             | target 431, active 0.586 |

补充机制解释：

- `shortest_correct_target_len` 后期稳定落到约 400-530 tokens，说明 `min_shortest_len=512`
  只限制了归一化分母，不能阻止目标长度被组内短正确样本拉低。
- `shortest_correct_active` 长期约 0.58-0.59，长度信号覆盖面很高；即使平均 penalty 绝对值不大，也会通过 group
  advantage 持续塑造“越短越好”的偏好。
- `alpha=0.05` 的 best eval 出现在 step 299，随后长度继续下降但 reward 不再恢复；`alpha=0.2` 在 step 99
  后迅速坍缩。这说明问题不是训练不够，而是目标函数方向本身过强地偏向短正确样本。

### 下一轮实验协议

优先级调整：

| 优先级 | 实验                                   | 目的                                                 | 建议配置 / 判据                                                                                                  |
| ------ | -------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| P0     | adaptive checkpoint 复评               | 确认 best 附近 saved checkpoint 是否优于 final       | 优先评估 `globalstep3999`，同时保留 final；若 3999 稳定更好，将其作为当前 16k adaptive 候选模型                  |
| P0     | 长度分桶与 tokens-per-correct 离线分析 | 解释 adaptive 为什么提升 hard reward，同时控制 token | 对 overlong8k、overlong4k、adaptive 计算 `correct_rate_by_len_bucket`、`tokens_per_correct`、`correct_len` 曲线  |
| P0     | correct-only mean/std baseline         | 实测 R1-Alpha / efficient reasoning 近邻方法是否更强 | 同 BigMath 0.5B 16k 协议；只惩罚正确响应，用同 prompt rollout 长度均值 / 方差归一化；必须报告 hard reward 和 TPC |
| P0     | LASER-D / difficulty-aware baseline    | 实测 difficulty-aware length shaping 是否优于我们    | 先核对原文和开源公式；若可复现，保持同 group size / max_new_tokens / eval protocol；若不可复现，做最小忠实 proxy |
| P1     | `length_quantile` adaptive 小矩阵      | 验证当前正结果是否稳健                               | 围绕 `alpha=0.03/0.05`、`min_solve_rate=0.5/0.75`、`target_quantile=0.25/0.5`、`min_target_len=4096/8192`        |
| P1     | Leash-style target-length dual control | 实测动态 penalty 强度能否更稳                        | 目标长度取 overlong8k / adaptive 的 eval len 区间；用 Lagrangian 更新 penalty；判断是否牺牲 hard reward          |
| P1     | LAPO-style successful-length baseline  | 检查 successful length distribution 叙事是否撞车     | 先从已有成功 rollouts 估计 prompt / difficulty length prior；短跑验证是否比 quantile target 稳                   |
| P1     | ARLCP / reflection-aware proxy         | 检查反思冗余惩罚是否解释我们收益                     | 若数据有 `<think>` 或可识别 reflection pattern，则单独统计 / 惩罚重复反思；否则只作为分析项                      |
| P2     | ALP 变体是否继续讨论                   | 判断是否需要 standalone 失败之外的 ablation          | paper-scale `alp_beta=1e-7` 已完整训练且为负；若论文需要，再讨论更弱 beta、late schedule 或 correctness gate     |
| P1     | incorrect-only long penalty            | 优先剪掉“长错”而不是惩罚正确推理                     | 对错误样本超过动态阈值扣分；正确样本只保留很弱 tail penalty；观察 incorrect_len 是否下降且 hard reward 不降      |
| P2     | 成功长轨迹压缩 SFT                     | 从成功轨迹中删除冗余表达，而不是用 RL 强行追最短     | teacher compression + verifier filtering；作为后续长到短 pipeline 的数据阶段                                     |
| P2     | budget-conditioned 多模式              | 简单题短答，难题保留长推理 fallback                  | `<think_short>` / `<think_long>` / adaptive fallback                                                             |

当前最优先推荐：不要再继续插值式 overlong sweep，也不要继续 shortest alpha sweep。先把 adaptive 的 saved
checkpoint 复评和长度分桶补齐，同时启动 P0 相关方法 baseline。`mode=alp` standalone 已有过强参数与 paper-scale
两次负结果，除非要写失败 ablation 或讨论 gated / scheduled 变体，否则不占用主线资源。

强相关 baseline 的统一协议：

| 要求               | 说明                                                                                        |
| ------------------ | ------------------------------------------------------------------------------------------- |
| 同训练基础         | `ailab_slm_0_5b_think_bigmath`、16k max_new_tokens、group size 16、相同 eval sets           |
| 同报告口径         | macro reward、hard reward、avg eval len、hard len、tokens-per-correct、finish_reason/length |
| 同 checkpoint 规则 | 报告 final 和 best eval step；避免只挑早停最优点而忽略后期 under-thinking                   |
| 同失败判据         | hard reward 显著下降、eval len 坍到 1k 以下、train reward 与 eval reward 背离，都算失败信号 |
| 实现忠实度         | 优先复现原文公式；公式不完整时标注为 proxy，不能当作正式击败相关工作的证据                  |

### E7：高风险相关方法实测路线图

目的：回答“是不是已有方法已经能比我们好”。ALP 只是一个失败例子，不能代表整个相关工作空间。E7 的目标不是无限扩实验，而是用最少的强对照把论文 claim 边界钉牢。

阶段 A：公式核对与最小实现设计。

| 方法                               | 需要核对的问题                                                    | 预期实现方式                                              |
| ---------------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------- |
| Training LMs to Reason Efficiently | correct-only penalty 的长度归一化、超参范围、reward 注入位置      | 复用现有 adaptive reward hook，加 `mode=correct_mean_std` |
| LASER-D                            | step reward、dynamic schedule、difficulty-aware signal 的精确定义 | 若原文 / 代码完整，忠实复现；否则写明 proxy               |
| Leash                              | target length、dual variable 更新频率、penalty 上下界             | 加一个轻量 dual controller，先跑短程稳定性                |
| LAPO                               | successful-length distribution 如何估计、是否需要两阶段训练       | 先做 offline length prior，再决定是否训练                 |
| ARLCP                              | reflection token 的识别方式、是否适配当前输出格式                 | 先做日志分析；没有稳定 reflection parser 时不直接训练     |

阶段 B：P0 训练。

| baseline                      | 为什么先跑                                                | 成功 / 失败判据                                                             |
| ----------------------------- | --------------------------------------------------------- | --------------------------------------------------------------------------- |
| correct-only mean/std penalty | 和我们的 correct-gated quantile target 最近，且实现成本低 | 若 hard reward 与我们接近且 tokens 更低，说明 quantile target 优势不足      |
| LASER-D                       | 同样主打 dynamic + difficulty-aware，是最高风险撞车方向   | 若能复现其 Pareto 优势，需要重新定位 novelty 为 correct-quantile robustness |
| adaptive checkpoint 复评      | 先确认我们自己的 best checkpoint 上限                     | 若 3999 / final 不稳定，不能急着做大矩阵                                    |

阶段 C：P1 训练与分析。

| baseline / analysis                    | 目的                                                      |
| -------------------------------------- | --------------------------------------------------------- |
| Leash-style target-length dual control | 判断动态调 penalty 强度是否足以替代 group quantile target |
| LAPO-style successful-length prior     | 判断成功长度分布建模是否已经覆盖我们的核心想法            |
| ARLCP-style reflection analysis        | 判断收益是否主要来自减少反思冗余，而不是 adaptive target  |
| length bucket + tokens-per-correct     | 解释每个方法是在省冗余 token，还是压掉困难题必要推理      |

阶段 D：论文级证据门槛。

| 门槛            | 通过条件                                                                                             |
| --------------- | ---------------------------------------------------------------------------------------------------- |
| 强相关 baseline | ALP、shortest/SOL、correct mean/std、LASER-D 至少 4 类中，我们在 hard reward / TPC Pareto 上不被支配 |
| 稳健性          | `length_quantile` 至少 2-3 个 seed 或相邻超参保持优势                                                |
| 消融            | 去掉 `correct_only`、`min_target_len`、`target_quantile` 或 solve-rate gate 会明显变差               |
| 泛化            | 至少再覆盖代码 RLVR 或另一个 math 模型 / 数据集                                                      |
| 机制            | 能解释为什么 shortest 和 ALP under-think，而 quantile + floor 保留 hard-set 推理预算                 |

如果阶段 B 的 P0 baseline 已经明显强于我们，路线应转为吸收其机制并重新设计；如果 P0 不强或不稳，再继续推进 `length_quantile` 的
seed、消融和泛化。

弱 shortest-correct 的重启标准：

| 标准                                               | 判定     |
| -------------------------------------------------- | -------- |
| eval macro reward 不低于 overlong baseline         | 必须满足 |
| AIME / HMMT 不出现明显掉点                         | 必须满足 |
| `correct_seq_len` 温和下降，而不是快速塌到 1k 以下 | 必须满足 |
| target len 不持续低于 4k                           | 必须满足 |
| 长度信号只在训练后期或高 solve-rate group 激活     | 必须满足 |

### E4：Solve-Rate Adaptive Length-Quantile Penalty

研究问题：固定 overlong 和 group-min shortest 都缺少题目难度自适应。当前主线 `mode=length_quantile` 用 group
solve rate 近似 prompt 难度，并用正确样本长度分位数作为动态 target：

```text
solve_rate_g = correct_count_g / group_size
lambda_g = clip((solve_rate_g - min_solve_rate) / (max_solve_rate - min_solve_rate), 0, 1)
target_len_g = quantile(correct_lengths_g, q), with lower bound min_target_len
penalty_i = -alpha * lambda_g * max(0, len_i - target_len_g) / target_len_g
```

默认只惩罚正确样本：`correct_only=true`。这样 hard group 或低 solve-rate group 不会被强行压短，easy group
中明显长于同组正确分位数的样本才收到长度压力。

当前实现：

| 资产            | 位置                                                                   | 作用                                                        |
| --------------- | ---------------------------------------------------------------------- | ----------------------------------------------------------- |
| reward 函数     | `areal/utils/functional/functional.py::reward_adaptive_length_penalty` | 计算 solve-rate-scaled length penalty                       |
| actor hook      | `areal/trainer/ppo/actor.py`, `areal/trainer/ppo/actor_qun_team.py`    | 在 reward scaling / norm 前叠加 penalty 并记录指标          |
| example config  | `examples/math/bigmath_rl.py`, `examples/code/nemotron_configs.py`     | 暴露 `actor.adaptive_length_reward`，默认关闭               |
| YAML / launcher | `examples/*/grpo_template*.yaml`, `run_trainer_mtp.sh`                 | 支持模板与环境变量 override                                 |
| 单元测试        | `tests/test_adaptive_length_reward.py`                                 | 验证 hard group 不激活、solve-rate scaling、quantile target |

实现中还保留了 `mode=alp`：它不使用正确样本分位数 target，而是对所有有效样本按 solve rate 与长度线性扣分。 新增 `alp_beta` 后，ALP
可以走论文 per-token beta 路径；不设置 `alp_beta` 时仍保留旧的 `alpha * len / length_normalizer`
normalized-alpha 兼容路径。`mode=alp` 当前不读取 `correct_only`、
`min_solve_rate`、`target_quantile` 或 `min_target_len`。因此 `mode=alp` 与 `length_quantile`
是两类不同目标， 下面先报告 `length_quantile` 正结果，再单列 ALP 参数尺度修正和 paper-scale 负结果。

首个完成 run 配置：

| 参数                  | 取值              |
| --------------------- | ----------------- |
| `mode`                | `length_quantile` |
| `alpha`               | `0.05`            |
| `group_size`          | `16`              |
| `min_solve_rate`      | `0.75`            |
| `max_solve_rate`      | `1.0`             |
| `target_quantile`     | `0.5`             |
| `min_target_len`      | `4096`            |
| `normalize_by_target` | `true`            |
| `max_penalty`         | `0.05`            |
| `correct_only`        | `true`            |

完整训练结果：

| run                            | 状态                 | final macro / hard reward | best macro / hard reward  | final eval len / hard len | last100 train len |
| ------------------------------ | -------------------- | ------------------------- | ------------------------- | ------------------------- | ----------------- |
| `16k_adaptive_length_quantile` | training complete    | 0.470 / 0.361             | 0.488 / 0.383 @ step 3599 | 5.7k / 6.7k               | 2.9k              |
| `16k_overlong_8k`              | training complete    | 0.449 / 0.334             | 0.450 / 0.335 @ step 6299 | 5.2k / 5.7k               | 4.1k              |
| `16k_overlong_4k`              | 未见完成标记         | 0.457 / 0.342             | 0.480 / 0.370 @ step 3699 | 7.3k / 8.1k               | 5.7k              |
| `30k_overlong_8k`              | complete，有 timeout | 0.512 / 0.408             | 0.521 / 0.420 @ step 3983 | 13.6k / 15.1k             | 9.7k              |

adaptive 分数据集 final 指标：

| dataset | reward | response_len | finish_reason/length |
| ------- | ------ | ------------ | -------------------- |
| MATH500 | 0.907  | 2.0k         | 0.0028               |
| AIME24  | 0.465  | 6.5k         | 0.0292               |
| AIME25  | 0.352  | 6.6k         | 0.0250               |
| AIME26  | 0.392  | 6.7k         | 0.0271               |
| HMMT25  | 0.235  | 6.9k         | 0.0167               |

曲线形态：

| step | macro reward | AIME avg | HMMT25 | avg eval len |
| ---- | ------------ | -------- | ------ | ------------ |
| 2999 | 0.473        | 0.413    | 0.217  | 6.0k         |
| 3499 | 0.471        | 0.403    | 0.240  | 6.1k         |
| 3599 | 0.488        | 0.431    | 0.240  | 6.0k         |
| 3999 | 0.475        | 0.415    | 0.223  | 6.0k         |
| 6639 | 0.470        | 0.403    | 0.235  | 5.7k         |

机制解释：

- adaptive 相比 `16k_overlong_8k` 不是简单全局变短。MATH500 eval length 从 `3.1k` 降到 `2.0k`，但 AIME /
  HMMT hard-set length 从约 `5.7k` 提高到约 `6.7k`。这符合 difficulty-aware 预期：简单题压短，困难题保留更多推理预算。
- 训练末段 `adaptive_length_penalty/avg` 约 `-0.0003`，明显小于 shortest-correct 的长度项；但
  `adaptive_length_active/avg` 约 `0.59`，说明它是弱但覆盖稳定的偏好塑造，而不是强行压缩。
- `adaptive_length_target_len/avg` 日志把 inactive 样本记为 0；final 记录为 `2707`，结合 active 率约
  `0.634`，活跃样本实际 target 约 `4.3k`，符合 `min_target_len=4096`。
- final eval 相比 best 有回落，说明需要 checkpoint selection 或后期衰减。保存点每 500 step 一次，已有 checkpoint
  `globalstep3499`、`globalstep3999`、`globalstep4499` 等；当前优先复评 `globalstep3999`。
- adaptive hard-set `finish_reason/length` 平均约 `2%`，高于 overlong completed baseline 的接近
  0；这不是当前主要瓶颈，但后续需要监控，避免困难题重新接近 16k 截断。

后续小矩阵建议：

| 参数              | 候选                  | 目的                                                         |
| ----------------- | --------------------- | ------------------------------------------------------------ |
| `alpha`           | `0.03`, `0.05`        | 控制后期回落和长度压力                                       |
| `min_solve_rate`  | `0.5`, `0.75`         | 调整 easy group 激活范围                                     |
| `target_quantile` | `0.25`, `0.5`         | 控制 target 激进程度                                         |
| `min_target_len`  | `4096`, `8192`        | 保护 hard-set 推理预算                                       |
| `max_penalty`     | `0.03`, `0.05`, `0.1` | 控制长度项对 advantage 影响                                  |
| `mode`            | `length_quantile`     | 当前主线只继续验证分位数 adaptive；ALP standalone 已单独判负 |

### E5：ALP Mode 参数尺度修正与 paper-scale 负结果

研究问题：ALP 风格的 solve-rate-scaled 长度成本，能否作为 `length_quantile` adaptive 的简单对照，在降低 token
的同时保持 BigMath 0.5B 16k 质量。

原文核对：

- ALP 原文定义的是 `p_solved(q)=correct_count/K`，并用 solve rate 缩放 per-token 长度成本；easy prompts
  高 penalty，hard prompts 低 penalty。
- ALP 不使用“答对采样中的最短长度”作为 target。最短正确长度更接近我们此前的 shortest-correct / SOL_group 方案，已经在
  BigMath 16k 上出现过短化负结果。
- 论文实验报告使用 `beta=1e-7`，训练 context window 为 16k。

日志来源：

| run                     | 路径                                                                                                                                                        |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 旧 normalized-alpha ALP | `dataset/ag_data/logs/areal/experiments/logs/root/ailab_slm_0_5b_think_bigmath/mtp_grpo_muon_16k_groupsize_16_lr_4e-5_alp_reward_alpha005_min4096_20260617` |
| paper-scale ALP         | `dataset/ag_data/logs/areal/experiments/logs/root/ailab_slm_0_5b_think_bigmath/mtp_grpo_muon_16k_groupsize_16_lr_4e-5_alp_beta_1e-7_20260618`               |

当前结论：旧 normalized-alpha ALP 是过强参数失败，paper-scale `alp_beta=1e-7` 则排除了“只是 beta 过强导致
1-token collapse”的解释，但 standalone ALP 仍会把模型推向 under-thinking，不能作为当前主线。

两组 ALP ablation 对比：

| run                     | 长度系数                          | final macro / hard reward | best valid macro / hard reward                  | final eval len / hard len | 结论                                |
| ----------------------- | --------------------------------- | ------------------------- | ----------------------------------------------- | ------------------------- | ----------------------------------- |
| 旧 normalized-alpha ALP | `alpha=0.05`，等效 `beta≈3.05e-6` | 0.000 / 0.000             | 0.415 / 0.301 @ step 0                          | 1 / 1                     | 过强绝对长度成本，完全坍缩          |
| paper-scale ALP         | `alp_beta=1e-7`                   | 0.254 / 0.137             | 0.401 / 0.283 @ step 0；0.400 / 0.283 @ step 99 | 656 / 739                 | 不再完全坍缩，但严重 under-thinking |

paper-scale run 配置核对：

| 参数             | 取值      | 备注                                                                |
| ---------------- | --------- | ------------------------------------------------------------------- |
| `mode`           | `alp`     | 绝对长度成本，不使用正确样本分位数 target                           |
| `alp_beta`       | `1.0e-07` | 使用 paper-scale per-token beta 路径；`alpha=0.05` 不再决定长度系数 |
| `group_size`     | `16`      | 与 GRPO group rollout 对齐                                          |
| `max_new_tokens` | `16384`   | 与论文 16k context setting 对齐                                     |
| `max_penalty`    | `0.05`    | 保留单样本长度项上限配置                                            |
| `correct_only`   | `true`    | 在 ALP 代码路径中未生效                                             |
| `min_solve_rate` | `0.75`    | 在 ALP 代码路径中未生效                                             |
| `min_target_len` | `4096`    | 在 ALP 代码路径中未生效；run 名中的 `min4096` 不能提供下界保护      |

指标文件开头有两个 `global_step=0` 记录，其中第一条仍带旧尺度长度项，第二条起才是连续 paper-scale run。按第二条 step 0
之后统计，平均等效 beta 约为 `0.8e-7` 到 `1.4e-7`，和目标 `alp_beta=1e-7` 一致。

paper-scale ALP eval 曲线：

| step | macro reward | hard reward | avg eval len | hard len | 现象                             |
| ---- | ------------ | ----------- | ------------ | -------- | -------------------------------- |
| 0    | 0.401        | 0.283       | 11.1k        | 12.8k    | 有效初始点                       |
| 99   | 0.400        | 0.283       | 9.0k         | 10.5k    | 早期省约 19% token，质量几乎不变 |
| 199  | 0.376        | 0.256       | 6.6k         | 7.8k     | 质量开始明显下降                 |
| 299  | 0.348        | 0.226       | 4.7k         | 5.5k     | hard-set 受损                    |
| 499  | 0.291        | 0.167       | 1.8k         | 2.1k     | 过短化加速                       |
| 999  | 0.144        | 0.026       | 126          | 106      | hard-set 基本失效                |
| 3999 | 0.233        | 0.113       | 475          | 516      | 训练侧恢复不等于 eval 恢复       |
| 6639 | 0.254        | 0.137       | 656          | 739      | final 仍显著低于 baseline        |

paper-scale ALP 训练侧窗口：

| step window | avg train len | raw task reward | rollout len | rollout reward | 解释                                       |
| ----------- | ------------- | --------------- | ----------- | -------------- | ------------------------------------------ |
| 0-99        | 5.3k          | 0.458           | 5.3k        | 0.458          | 初期仍保留长推理                           |
| 100-499     | 2.1k          | 0.423           | 2.3k        | 0.424          | 长度快速下降，reward 已受影响              |
| 500-999     | 285           | 0.283           | 315         | 0.293          | 进入短输出吸引子                           |
| 2000-3999   | 254           | 0.446           | 309         | 0.470          | train reward 开始恢复，但 eval hard 仍低   |
| 4000-6639   | 263           | 0.508           | 323         | 0.534          | 训练分布 shortcut 明显                     |
| 6540-6639   | 246           | 0.524           | 305         | 0.550          | final train reward 高，eval quality 不匹配 |

机制解释：

- 旧 normalized-alpha run 的失败主要来自长度系数过强：`alpha=0.05 / 16384≈3.05e-6`，约为论文 `1e-7` 的 30
  倍，因此不能直接代表 paper-faithful ALP。
- paper-scale run 说明把 beta 降回 `1e-7` 后确实不会再迅速归零，但 ALP standalone 仍缺少目标长度、正确样本下界和停止机制。
- ALP 对所有有效样本按长度扣分，错误样本也会得到“更短更好”的方向；`correct_only`、`min_solve_rate`、 `min_target_len` 等
  protection 在 ALP path 中不参与计算。
- 训练 raw reward 后期在 250-token 左右短输出上恢复，而 eval hard reward 只有 `0.137`，说明 rollout
  训练分布上的短答案 shortcut 不能外推到评测集。
- 与 `length_quantile` 不同，ALP 不区分“超过正确样本分位数的冗余长度”和“困难题需要的推理预算”。即便单步 beta 很小，经过数千步 GRPO
  更新后也会持续把策略推向 under-thinking。

后续处理：

| 选项                               | 建议                                                                                        |
| ---------------------------------- | ------------------------------------------------------------------------------------------- |
| 作为当前 16k 主线继续调参          | 不建议；主线仍应放在 `length_quantile` adaptive                                             |
| 作为论文失败 ablation              | 可以保留，但要分开报告“过强 normalized-alpha 失败”和“paper-scale standalone under-thinking” |
| 继续 standalone ALP beta sweep     | 暂不建议；`1e-7` 已经表现为训练后期 under-thinking，继续扫更强 beta 价值低                  |
| 讨论 ALP 变体                      | 若有论文叙事需要，可讨论更弱 beta、late schedule、correctness gate 或 lower-bound target    |
| 与 `length_quantile` adaptive 对比 | paper-scale standalone 已可比较，结论是不竞争；后续只比较 gated / scheduled 变体            |

### 报告化待补材料

| 材料                         | 用途                             |
| ---------------------------- | -------------------------------- |
| E0 与 E2 的完整 run metadata | 支撑可复现性                     |
| 指标曲线截图或导出表         | 支撑实验结论                     |
| length bucket 分析           | 解释长度与正确率关系             |
| 代码任务 failure type 分析   | 判断短化是否伤害边界条件和鲁棒性 |
| 成功与失败样例对比           | 支撑机制解释和论文案例           |
