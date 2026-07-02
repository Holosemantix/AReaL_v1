# Token Efficiency 顶会路线评估与下一步规划

更新时间：2026-07-02

关联工作稿：`token_efficiency_research_notes.md`

适用范围：BigMath / Code RLVR，GRPO / PPO，可验证 reward 任务。

---

## 0. 一句话结论

这个方向有顶会潜力，但当前还不是顶会完成度。

当前最有希望的主线不是泛泛的 `adaptive length penalty`，而是：

> **Safe Adaptive Length Targeting for Token-Efficient RLVR**：用 solve-rate gate 判断题目是否适合压缩，用 correct-only 避免奖励短错，用 correct-length quantile target 避免 shortest-correct 过激，用 lower-bound protection 防止困难题推理预算坍缩。

当前证据已经足以说明：

1. naive length penalty 容易 under-think；
2. shortest-correct / SOL-style min target 会压掉必要推理；
3. standalone ALP 在当前 BigMath 0.5B 16k 设置下不稳；
4. correct-only mean/std 虽然比 shortest 和 ALP 温和，但仍会持续压短正确样本；
5. `length_quantile` adaptive 已经出现有竞争力的 reward / token tradeoff。

但要冲顶会，还需要补齐强相关 baseline、机制分析、多 seed / 多 setting 稳健性，以及最好加入代码 RLVR 泛化。

---

## 1. 我们现在到底在做什么

研究问题不是“让模型输出更短”，而是：

```text
在保持 pass@1 / accuracy / hard-set reward 尽量不下降的前提下，
最小化 response_len、correct_response_len、tokens_per_correct、rollout cost 和推理延迟。
```

更具体地说，当前研究聚焦在 RLVR 训练中：

- 长 CoT 能提高困难题上限，但显式 token 成本高；
- 固定长度惩罚容易把必要推理也压掉；
- 直接用 prompt budget 不稳定；
- shortest-correct 会被偶然短对样本牵引；
- 代码 RLVR 有 compile/runtime/timeout/wrong answer/partial pass/pass all tests 等更丰富可验证信号，适合研究哪些 token 真有用。

因此，当前方法应被定位为一种 **安全的自适应长度目标构造**，而不是又一个普通长度惩罚项。

---

## 2. 当前进展总结

### 2.1 已有主结果与负结果

#### Shortest-correct / SOL-style：负结果

已有 BigMath 0.5B 16k 实验显示：

- `shortest_correct alpha=0.05`：eval 平均长度从约 `7.3k` 压到约 `3.8k`，但 macro reward 从约 `0.445` 降到约 `0.379`；
- `shortest_correct alpha=0.2`：eval 平均长度约 `1.5k`，macro reward 约 `0.299`；
- 主要掉分发生在 AIME / HMMT 等更吃推理预算的数据集上。

解释：min target 过激。即使只惩罚正确样本，group 内“最短正确”也会形成持续的 moving target，把正确轨迹越推越短，最终压掉困难题必要推理。

#### Fixed overlong penalty：可做安全阈值，但不是核心短化能力

已有对照：

| run | 定位 | 主要现象 |
| --- | --- | --- |
| `30k_overlong_8k` | 当前质量上限 | best macro / hard 约 `0.521 / 0.420`，但 eval 平均长度约 `13.6k+` |
| `16k_overlong_4k` | 16k 内较稳折中 | best macro / hard 约 `0.480 / 0.370`，平均长度约 `7.4k` |
| `16k_overlong_8k` | 更早施压的短 baseline | 长度约 `5.2k`，但 hard reward 明显下降 |

解释：overlong penalty 主要控制 runaway long CoT 和接近 max length 的样本，不会自动学到按题目难度分配推理预算。

#### ALP standalone：负结果

已有两组 ALP ablation：

| run | 结论 |
| --- | --- |
| old normalized-alpha ALP | 等效 beta 过强，训练坍到 1-token 输出 |
| paper-scale `alp_beta=1e-7` | 不再完全坍缩，但 final macro / hard 只有约 `0.254 / 0.137`，eval len / hard len 只有约 `656 / 739` |

解释：ALP 对有效样本按 solve rate 和绝对长度持续扣分，不使用 correct-length quantile target，也没有 lower-bound target。它可以早期省 token，但长训练后会持续推向 under-thinking。

#### Correct-only mean/std：弱负结果

已有 `alpha=0.05` 和 `alpha=0.02` 两组：

- 比 shortest / ALP 温和；
- 但仍会持续压短正确样本；
- `alpha=0.02` 虽然更弱，但没有形成相对 `length_quantile` 的 Pareto 优势；
- 降低 alpha 不能修复目标偏差。

解释：mean/std 目标会跟着策略整体变短而继续下移，也缺少 solve-rate gate 和 lower-bound protection。

### 2.2 当前主线：length_quantile adaptive

当前主线方法：

```text
solve_rate_g = correct_count_g / group_size
lambda_g = clip((solve_rate_g - min_solve_rate) / (max_solve_rate - min_solve_rate), 0, 1)
target_len_g = quantile(correct_lengths_g, q), with lower bound min_target_len
penalty_i = -alpha * lambda_g * max(0, len_i - target_len_g) / target_len_g
```

默认只惩罚正确样本：`correct_only=true`。

首个完成 run 配置：

| 参数 | 值 |
| --- | --- |
| `mode` | `length_quantile` |
| `alpha` | `0.05` |
| `group_size` | `16` |
| `min_solve_rate` | `0.75` |
| `max_solve_rate` | `1.0` |
| `target_quantile` | `0.5` |
| `min_target_len` | `4096` |
| `max_penalty` | `0.05` |
| `correct_only` | `true` |

结果摘要：

| run | final macro / hard | best macro / hard | final eval len / hard len |
| --- | --- | --- | --- |
| `16k_adaptive_length_quantile` | `0.470 / 0.361` | `0.488 / 0.383 @ step 3599` | `5.7k / 6.7k` |
| `16k_overlong_8k` | `0.449 / 0.334` | `0.450 / 0.335` | `5.2k / 5.7k` |
| `16k_overlong_4k` | `0.457 / 0.342` | `0.480 / 0.370` | `7.3k / 8.1k` |
| `16k_no_length_reward` first 1k | step 999: `0.474 / 0.364` | 暂未完整 | `11.4k / 13.0k` |

关键解释：

- `length_quantile` 不是无损超过 no-length 质量上界；
- 它的价值是用小幅 hard reward 代价显著降低 token；
- 相比 `16k_overlong_8k`，它不是全局更短，而是简单题更短、hard-set 保留更多推理预算；
- 这正符合 difficulty-aware compression 的论文叙事。

---

## 3. 顶会希望判断

### 3.1 有希望的原因

1. 问题重要：long CoT / long thinking 已经成为强推理模型能力来源，但 token 成本、延迟和上下文占用明显增长。
2. 当前方法不是简单 reward shaping，而是有明确失败案例驱动的 safe target design。
3. 已有负结果很有价值：可以系统展示 naive efficient-reasoning rewards 如何 under-think。
4. `length_quantile` 的机制有可解释性：easy group 压缩，hard group 保护。
5. 如果扩展到代码 RLVR，会比纯数学 efficient reasoning 更有差异化。

### 3.2 当前不足

如果现在直接写论文，风险较高：

- 相关工作太拥挤，不能宣称“首次 adaptive length penalty”；
- 当前主结果主要集中在 BigMath 0.5B 16k 单协议；
- 强相关 baselines 还没补全，尤其 LASER-D / difficulty-aware；
- no-length baseline 质量略高，需要用 Pareto / tokens-per-correct 重新组织叙事；
- 代码 RLVR 目前还是机会点，不是完成结果；
- seed / checkpoint selection / ablation 还不足以支撑主会级稳定性。

### 3.3 建议目标定位

不要写：

```text
We propose the first adaptive length penalty for reasoning LLMs.
```

应该写：

```text
Existing length-aware RL methods often rely on absolute length penalties,
shortest-correct targets, mean/std normalized correct-only penalties,
or manually specified token budgets. These objectives can induce
under-thinking on hard prompts. We propose a safe adaptive target
construction that combines solve-rate gating, correct-only quantile targets,
and lower-bound protection to improve the accuracy-token Pareto frontier
in RLVR.
```

---

## 4. 下一步执行计划

### P0：马上做，决定论文是否成立

#### P0.1 复评 `length_quantile` saved checkpoints

目的：确认 best 附近 saved checkpoint 是否优于 final。

优先复评：

- `globalstep3499`
- `globalstep3999`
- `globalstep4499`
- final checkpoint

输出统一表：

| checkpoint | macro reward | hard reward | avg eval len | hard len | tokens_per_correct | finish_reason/length |
| --- | --- | --- | --- | --- | --- | --- |

成功判据：

- 至少一个 saved checkpoint 在 hard reward / TPC Pareto 上优于 final；
- checkpoint selection 不应只报 best，要同时报告 final。

#### P0.2 补齐 no-length baseline

目的：明确质量上界和 token 上界。

当前 first 1k 已显示 no-length 质量略高但 token 多约 50%+。后续需要：

- 跑完整或至少跑到与主线可比的 step；
- 报告 step-aligned 对比；
- 把 no-length 作为 upper-bound baseline，不回避。

论文表述：

```text
No-length RL remains a strong quality upper bound, but it pays substantially
higher token cost and shows higher truncation on hard sets. Our goal is not
to dominate no-length in absolute accuracy, but to improve the reward-token
Pareto frontier.
```

#### P0.3 离线 length bucket / Pareto / TPC 分析

对以下 runs 做统一离线分析：

- no-length 16k
- `16k_overlong_4k`
- `16k_overlong_8k`
- `16k_adaptive_length_quantile`
- shortest `alpha=0.05 / 0.2`
- ALP paper-scale
- correct mean/std `alpha=0.02 / 0.05`

必要图表：

1. `macro reward` vs `avg response_len`
2. `hard reward` vs `hard response_len`
3. `tokens_per_correct` by method
4. `accuracy_at_budget(B)`，例如 B = 2k / 4k / 8k / 12k / 16k
5. correct rate by response length bucket
6. hard-set correct rate by response length bucket
7. `correct_response_len` vs `incorrect_response_len`
8. `finish_reason/length` by dataset

重点解释：

- LQ 是否主要压缩简单题；
- LQ 是否给 AIME/HMMT 保留更多长度；
- ALP / shortest / meanstd 是如何进入 under-thinking；
- no-length 是否更多靠 hard-set 长推理换质量。

#### P0.4 补 LASER-D / difficulty-aware baseline

这是最高风险相关 baseline。

执行顺序：

1. 核对原文公式与代码；
2. 能 faithful reproduce 就忠实复现；
3. 不能 faithful reproduce 就实现最小 proxy，并明确标注为 proxy；
4. 使用同一 BigMath 0.5B 16k 协议；
5. 报告 final 和 best，不只报早停点。

判据：

- 若 LASER-D 明显支配 LQ，主线需要吸收其机制并重新定位；
- 若 LASER-D 不稳或 hard-set under-think，则 LQ 的 safe target design 更有说服力。

---

### P1：形成顶会级证据

#### P1.1 `length_quantile` 核心消融

只做能支撑机制 claim 的消融，不做无意义大扫参。

| 消融 | 目的 |
| --- | --- |
| 去掉 solve-rate gate | 证明不是所有 prompt 都该压短 |
| `correct_only=false` | 证明惩罚错误样本会诱导 short wrong / shortcut |
| `target_quantile=min / 0.25 / 0.5` | 证明 shortest/min target 过激，quantile 更稳 |
| `min_target_len=0 / 4096 / 8192` | 证明 lower-bound protection 对 hard-set 重要 |
| `alpha=0.03 / 0.05` | 控制后期回落和长度压力 |
| late schedule / decay | 处理 step 3599 后 final 回落 |

优先组合建议：

| run | 目的 |
| --- | --- |
| LQ default | 主线 |
| LQ no floor | 验证 floor 必要性 |
| LQ no solve gate | 验证 difficulty gate 必要性 |
| LQ all samples | 验证 correct-only 必要性 |
| LQ q=0.25 | 检查更激进 quantile 是否伤 hard set |
| LQ alpha=0.03 | 检查弱化 penalty 是否减少后期回落 |

#### P1.2 其他强相关 baseline

优先级：

| baseline | 优先级 | 目的 |
| --- | --- | --- |
| Leash-style dual target control | P1 | 判断动态调 penalty 强度是否能替代 group quantile target |
| LAPO-style successful length prior | P1 | 判断成功长度分布建模是否覆盖我们的核心想法 |
| AALC / late accuracy-aware penalty | P1 | 检查 late-stage penalty 是否足以避免 under-thinking |
| ARLCP reflection-aware proxy | P1/P2 | 判断收益是否主要来自减少 reflection 冗余 |
| DSS-GRPO / SAS proxy | P2 | 检查 step-wise / segment-wise shaping 是否更强 |

#### P1.3 多 seed / 多 setting

顶会最少需要：

- 当前 BigMath 0.5B 16k 至少 2-3 seed 或相邻超参稳定；
- 至少一个额外 setting：
  - 另一个 math 模型大小；或
  - 另一个数学数据集；或
  - 代码 RLVR；
- 每个方法报告 final + best，避免只靠 checkpoint selection。

---

### P2：代码 RLVR 差异化方向

代码任务是顶会增量的关键。

#### P2.1 先做离线诊断

对代码 RLVR rollouts 拆分：

- reasoning / think tokens
- code tokens
- answer wrapper tokens
- total response tokens

按 failure type 统计：

- compile error
- runtime error
- timeout
- wrong answer
- partial pass
- pass all tests
- no code / malformed code

分析表：

| bucket | pass rate | compile error | runtime error | timeout | wrong answer | avg think len | avg code len |
| --- | --- | --- | --- | --- | --- | --- | --- |

要回答的问题：

1. 长 CoT 是否真的降低 compile error？
2. 长 CoT 是否增加 timeout 或代码复杂度？
3. 正确代码更依赖 code length 还是 reasoning length？
4. 简短 reasoning 是否更容易漏边界条件？
5. 是否存在“短 reasoning + robust code”的高效模式？

#### P2.2 方法改造：只压 reasoning，不压 code

代码任务中不应无脑优化 total response_len。

建议实现：

```text
response_len = think_len + code_len + wrapper_len
length_penalty applies to think_len only
code_len is either unpenalized or weakly regularized only when timeout risk is high
```

这会比数学任务更有 novelty：不是让答案短，而是让自然语言推理更精简，同时保留必要代码表达。

#### P2.3 fallback long mode

代码任务可以利用 public tests / sampled tests 做 fallback：

```text
default: short reasoning mode
if public tests fail or confidence low:
    retry long reasoning mode
```

这可以形成 practical system story：训练时学 token efficiency，推理时按验证信号自适应分配预算。

---

## 5. 论文主线建议

### Title 候选

- Safe Adaptive Length Targeting for Token-Efficient RLVR
- Learning When to Think Less: Safe Length Targets for Efficient Reasoning RL
- Avoiding Under-Thinking in Length-Aware RL for Reasoning Models
- Token-Efficient RLVR via Correctness-Gated Quantile Length Targets

### Abstract 叙事骨架

```text
Long chain-of-thought improves reasoning models but incurs high inference and
rollout cost. Existing length-aware RL objectives often reduce tokens by
inducing under-thinking, especially on hard prompts. We first provide a
controlled diagnosis of several common objectives, including fixed overlong
penalties, shortest-correct targets, solve-rate absolute penalties, and
correct-only mean/std penalties. We show that these methods can collapse
necessary reasoning budgets or create moving targets that continuously shorten
correct trajectories. We then propose Safe Adaptive Length Targeting, a simple
RLVR reward shaping method that activates length pressure only for sufficiently
solvable prompts, penalizes only correct trajectories, uses a quantile of
successful lengths as a soft target, and enforces a lower-bound budget. On
BigMath 0.5B 16k RLVR, our method improves the hard-reward/token Pareto frontier
relative to strong length-aware baselines. Further analyses show that it
compresses easy prompts while preserving reasoning budget on hard sets.
```

### 论文贡献点

1. **Diagnosis**：统一协议下系统证明 naive length objectives 如何导致 under-thinking。
2. **Method**：提出 solve-rate gated、correct-only、quantile target、floor-protected 的 safe adaptive target。
3. **Evidence**：在 BigMath 0.5B 16k 上获得更好的 reward-token Pareto frontier。
4. **Mechanism**：展示 simple/hard prompt 的长度分配差异，解释为什么 shortest / ALP / meanstd 失败。
5. **Generalization**：最好补代码 RLVR 或第二个 math setting。

---

## 6. 成功判据

论文主结果至少应满足：

1. 在同一协议下，LQ 不被以下方法在 `hard_reward` × `tokens_per_correct` Pareto 上支配：
   - no-length baseline
   - overlong 4k / 8k
   - shortest / SOL-style
   - ALP paper-scale
   - correct mean/std
   - LASER-D faithful/proxy
2. 相对 no-length baseline：
   - token 明显下降，例如 `>=30%`；
   - hard reward 下降可解释且较小；
   - TPC 明显改善。
3. 相对 overlong baseline：
   - hard reward 不低或更高；
   - TPC 不差；
   - hard-set 不发生 truncation 或 under-thinking。
4. 消融证明：
   - 去掉 floor 会伤 hard set；
   - 去掉 solve gate 会过度压缩；
   - target=min 会接近 shortest failure；
   - correct_only=false 会引入 short wrong 风险。
5. 至少一个额外 setting 复现：代码 RLVR 或另一个 math setting。

---

## 7. 暂时不要做的事情

不建议继续：

- shortest alpha 简单 sweep；
- correct mean/std alpha 简单 sweep；
- standalone ALP beta 大量 sweep；
- 只看 rollout reward，不看 raw reward / eval hard reward / TPC；
- 只报 best checkpoint，不报 final；
- 把 no-length baseline 藏起来；
- 声称“首次 adaptive length penalty”。

---

## 8. 给 Codex 的可执行任务清单

### Task A：评测与表格脚本

新增或整理脚本，输入多个 run 的 eval logs，输出：

- `macro_reward`
- `hard_reward`
- `avg_eval_len`
- `hard_len`
- `tokens_per_correct`
- `finish_reason_length`
- `correct_response_len`
- `incorrect_response_len`
- `accuracy_at_budget`

输出格式：

```text
reports/token_efficiency/main_results.csv
reports/token_efficiency/pareto_points.csv
reports/token_efficiency/length_buckets.csv
```

### Task B：checkpoint 复评配置

生成 `length_quantile` checkpoint 复评配置，至少覆盖：

- `globalstep3499`
- `globalstep3999`
- `globalstep4499`
- final

要求复用相同 eval sets 和相同 decoding config。

### Task C：LASER-D / difficulty-aware baseline

先阅读原文公式和可用实现，添加：

```text
mode=laser_d
```

如果不能忠实复现，在 config 和 report 中标注：

```text
laser_d_proxy=true
```

不能把 proxy 写成正式复现。

### Task D：LQ ablation configs

新增配置矩阵：

| config name | 变化 |
| --- | --- |
| `lq_default` | 当前主线 |
| `lq_no_floor` | `min_target_len=1` 或 0 |
| `lq_no_solve_gate` | `min_solve_rate=0` |
| `lq_all_samples` | `correct_only=false` |
| `lq_q025` | `target_quantile=0.25` |
| `lq_alpha003` | `alpha=0.03` |
| `lq_floor8192` | `min_target_len=8192` |

### Task E：代码 RLVR 离线分析

实现 response parser：

- 提取 `<think>` / reasoning 区间；
- 提取 code block；
- 计算 think_len / code_len / total_len；
- join verifier failure type；
- 输出 failure type × length bucket 表。

### Task F：报告自动生成

从 CSV 自动生成 Markdown 表格：

```text
reports/token_efficiency/topconf_results.md
```

包含：

- main results table
- Pareto table
- checkpoint selection table
- baseline failure table
- ablation table
- code RLVR diagnostics table

---

## 9. 当前最推荐的三步

如果只做三件事，按这个顺序：

1. **复评 LQ checkpoint，并补 TPC / Pareto / length bucket。**
2. **跑 LASER-D faithful/proxy，并把 ALP、SOL、mean/std、overlong、no-length 统一成同协议表。**
3. **启动代码 RLVR 的 reasoning/code token 拆分和 failure-type 分析。**

做到这三点后，这个方向才从“有趣的 reward shaping 实验”升级为“可以写顶会主线的系统研究”。
