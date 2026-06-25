# 单页阶段性汇报 PPT 生成提示（v2）

请生成一页 16:9 学术汇报 slide，风格简洁、技术感、适合组会/阶段性成果汇报。页面只做一页，不要做封面。

标题：

安全自适应长度目标：降低推理 token，同时保留困难题能力

副标题：

BigMath 0.5B / GRPO / 16k max tokens / group size 16 / first 1k training steps

核心解释：

mean/std baseline 可以从主图中剔除。它和我们的 length-quantile 不同：mean/std 是组内相对长度惩罚，会随着组内均值变短而继续追着正确样本压短；length-quantile 是有保护的目标长度构造，只在 solve-rate 足够高的 easy/solved group 上，对超过正确样本长度分位数和下界保护的冗余部分施压。

版式：

- 左侧 30%-35% 放方法示意图。
- 右侧 65%-70% 放 2x3 指标 dashboard。
- 图例放在 dashboard 顶部。
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

实验图请展示 6 个 panel：

- Train raw reward：训练 raw task reward，25-step rolling average。
- Train length：训练 response length，25-step rolling average，单位 k tokens。
- MATH500 reward：简单集合准确率/奖励。
- Hard-set reward：AIME24/25/26 + HMMT25 平均 reward。
- MATH500 length：MATH500 eval response length，单位 k tokens。
- Hard-set length：AIME24/25/26 + HMMT25 eval response length，单位 k tokens。

方法曲线保留：

- Ours: length-quantile
- Overlong 4k
- Overlong 8k
- Shortest alpha=0.05
- ALP beta=1e-7

不要把 mean/std 画进主图；它只在文字中作为负结果说明。

关键数值 callout：

- Step 999: ours hard reward 0.344 @ 7.4k avg eval tokens。
- Overlong 4k: 0.329 @ 7.3k，长度接近但 hard reward 更低。
- ALP beta=1e-7: 0.026 @ 0.1k，明显 under-thinking。
- MATH500 多数方法接近饱和，因此必须同时展示 hard-set；只看 MATH500 会掩盖困难题能力差异。

底部结论：

length-quantile 的优势不是更强惩罚，而是更安全的目标构造：MATH500 保持高分，hard-set 不被压垮，训练 reward 没有像 ALP/shortest 那样和短输出吸引子绑定。

可上传素材：

- `method_schematic.png`：方法示意图。
- `first1000_metrics_dashboard.png`：2x3 指标 dashboard。
- `one_slide_mockup_v2.png`：完整一页 mockup，可作为版式参考。
- `first1000_train_eval_metrics_v2.csv`：绘图数据。
