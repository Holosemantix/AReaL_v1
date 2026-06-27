# 单页阶段性汇报 PPT 生成提示

请生成一页 16:9 学术汇报 slide，风格简洁、技术感、适合组会/阶段性成果汇报。页面只做一页，不要做封面。

标题：

安全自适应长度目标：降低推理 token，同时保留困难题能力

副标题：

BigMath 0.5B / GRPO / 16k max tokens / group size 16

版式：

- 左侧 35%-40% 放方法示意图。
- 右侧 60%-65% 放实验对比图，上下两个 panel：上方 AIME24/25/26 + HMMT25 reward，下方 average eval response
  length。
- 底部放一句结论 callout。

方法示意图内容：

1. Prompt 进入同题 group rollouts，K=16。
1. Verifier 标出 correct mask，并保留原始 task reward。
1. 从 correct mask 得到两个信号：
   - solve-rate gate：只在组内正确率足够高时激活，避免困难题被过早压短。
   - correct-length quantile + minimum floor：用正确样本长度分位数构造目标长度，并设置下界保护。
1. 合并成 per-prompt adaptive target length。
1. 只对 correct 且 len > target 的样本施加长度 penalty。
1. 进入 GRPO update。

图中的核心公式可简写为：

```text
target_g = max(quantile(correct_lengths_g, q), min_target_len)
penalty_i = -alpha * gate(solve_rate_g) * max(0, len_i - target_g) / target_g
```

实验图请使用或复刻这些数据结论：

- 曲线范围：training step 0-999。
- Score panel：AIME24/25/26 + HMMT25 reward。
- Length panel：average eval response length，单位 k tokens。
- 方法曲线：
  - Ours: length-quantile
  - No length constraint
  - Overlong 4k
  - Overlong 8k
  - Shortest alpha=0.05
  - ALP beta=1e-7

关键数值 callout：

- Step 999: ours hard reward 0.344 @ 7.4k tokens。
- No length constraint: 0.364 @ 11.4k avg eval tokens，AIME/HMMT length 13.0k，质量高但 token
  成本最高。
- ALP beta=1e-7: 0.026 @ 0.1k tokens，出现 under-thinking。
- Overlong 4k: 0.329 @ 7.3k tokens，接近我们的长度但 hard reward 更低。

底部结论：

相比无长度约束、ALP、shortest 和 correct mean/std，length-quantile 的优势不在于追求最高 raw
reward，而在于选择性更好：只压缩已解决/容易题的冗余长度，同时给困难题保留推理预算。

可上传素材：

- `method_schematic.png`：方法示意图。
- `first1000_metrics_dashboard.png`：前 1000 step reward/length 对比图。
- `one_slide_mockup_v2.png`：完整一页 mockup，可作为版式参考。
