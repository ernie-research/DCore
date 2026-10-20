# When Does Better Acceptance Pay? A Controlled Study of Speculative Decoding

**English** | [中文介绍](README.zh-CN.md)

This repository accompanies **When Does Better Acceptance Pay? A Controlled Study of Speculative Decoding**.

Code and model weights will be released after the paper is formally published.

## More accepted tokens. Faster generation?

Speculative decoding lets a lightweight **drafter** propose tokens for a **target model** to verify in parallel. Accepting more tokens per step sounds like a win. But proposing and verifying them takes time.

**The goal is throughput, not acceptance length.** Our question is: *when is additional acceptance worth the cost of obtaining it?*

We compare six recipes across three families—autoregressive, block-parallel, and semi-autoregressive—using a common training, inference, and evaluation protocol, ten benchmarks, and two target models.

**Why these six drafters?** EAGLE-3, DFlash, DFlare, D-PACE, Domino, and DSpark represent learned drafting families actively explored by mainstream LLMs, making the comparison relevant to practical inference design. The story connects three decisions: **how to compare methods, how to improve them, and when their improvements pay off**.

## 1. Before trusting the ranking, align the conditions

Published rankings mix differences in training data and evaluation protocols. Re-running released checkpoints aligns evaluation, but leaves training conditions different.

**Figure 1** separates results **as reported**, released checkpoints **re-measured** in our runtime, and recipes **re-trained** on shared target-generated data under a common evaluation protocol.

[![Figure 1: Acceptance and speedup across reported, re-measured, and re-trained comparisons.](assets/figure-1.png)](assets/figure-1.png)

*Pale bars show acceptance length; solid bars show speedup. Read from left to right to see the effect of alignment.*

On MT-Bench, the highest-to-lowest speedup gap falls from **1.36 to 0.85 to 0.23**—differences in speedup multiples, not percentages. The apparent separation becomes much smaller.

**Table 8** expands this comparison to seven benchmarks and both targets. Acceptance changes differently across recipes; retraining is not simply a boost for everyone.

[![Table 8: Acceptance length as reported, re-measured, and re-trained across seven benchmarks and both target models.](assets/table-8.png)](assets/table-8.png)

*Each benchmark has three columns: reported, re-measured, and re-trained. Blue/yellow shading marks an increase/decrease from the preceding stage, not a cross-method ranking. Compare methods at matched budgets and temperatures. Click to enlarge.*

For example, on MT-Bench with Qwen3-8B, budget 16, and target|draft temperatures of 1|0, EAGLE-3's acceptance changes **2.70 → 2.53 → 3.59**, while DFlare's changes **4.37 → 3.37 → 3.13**. Their acceptance ordering reverses after retraining.

**A checkpoint comparison alone is not a clean comparison of drafter recipes.** Alignment gives us a clearer starting point for investigating the differences that remain.

## 2. Follow the costs, and a strong baseline emerges

**Figure 2** examines component and training-objective changes together with their execution costs.

[![Figure 2: Design ablation roadmap showing speedup, acceptance length, and proposal latency.](assets/figure-2.png)](assets/figure-2.png)

*Colored bars show speedup, pale bars acceptance length, and stars proposal latency. The gold star marks DCore.*

There are two routes to better acceptance: improve supervision or target context with little additional measured inference overhead, or add computation through deeper networks and stronger token-to-token conditioning. The second route pays only when acceptance gains cover the added cost. Useful components also need testing in combination.

This motivates **DCore**: DFlash's efficient backbone, DFlare's target-layer context mixture, and the D-PACE objective. In the roadmap configuration, speedup rises from **3.004× to 3.261×** at unchanged **1.79 ms** measured proposal latency.

Does that recipe hold up beyond the design search? **Table 1** compares DCore with all six re-trained drafters. It achieves the highest mean throughput in **18 of 24 single-request settings**, defined by target, budget, temperature pair, and task-entropy group.

[![Table 1: Mean throughput and acceptance length for the re-trained drafters and DCore.](assets/table-1.png)](assets/table-1.png)

*Cells show tokens per second, with acceptance length in the subscript. Headers give target|draft temperatures. Throughput and acceptance are ranked separately; bold marks the best result. Click to enlarge.*

DCore is a strong baseline distilled from the analysis. Other recipes lead in some settings, motivating a deeper question: **what makes a setting favorable?**

## 3. Look at uncertainty, not just the task label

**Entropy** measures next-token uncertainty. Even two coding tasks can differ greatly, so domain labels do not fully describe drafting difficulty.

**The full entropy heatmaps** look inside decoding paths across three drafter families, both targets, two budgets, and the temperature settings shown. Accepted prefixes tend to have lower target entropy than the rejected position ending the path.

[![Full target-entropy heatmaps across EAGLE-3 chain decoding, DFlash, and DSpark, with both targets and budgets.](assets/figure-12.png)](assets/figure-12.png)

*Brighter cells indicate greater target uncertainty; black borders mark rejected endpoints. Each panel contains a budget-16 grid and a budget-8 inset. Top rows show Qwen3-4B at 0|0 and 1|1; bottom rows show Qwen3-8B at 0|0 and 1|0. Gray bars show step frequencies. Click to enlarge.*

This recurring pattern makes uncertainty a useful lens for understanding acceptance. But acceptance is only one side of the payoff.

## 4. A larger budget can help—or slow you down

**Figure 4** compares verification budgets of 8 and 16 on Qwen3-8B with sampled target output and greedy drafts.

[![Figure 4: Acceptance gains versus throughput gains when the verification budget increases from 8 to 16.](assets/figure-4.png)](assets/figure-4.png)

*Right means more acceptance; up means more throughput. Solid/hollow markers represent low/high-entropy tasks. Log changes separate throughput gains into acceptance gains and efficiency changes; diagonal guides show the latter.*

Block-parallel recipes best preserve efficiency as the budget grows. Both block-parallel and semi-autoregressive recipes gain up to about **20%** throughput on low-entropy tasks, but at most **2.8%** on high-entropy tasks. EAGLE-3's wider tree gains about **12%** in both groups. Extending its sequential chain instead slows both groups: acceptance cannot repay the extra drafting time.

The block-based comparisons use budget-specific recipes; EAGLE-3's tree and chain comparisons use the same checkpoint. These single-request results lead to the deployment question: **what happens when requests arrive together?**

## 5. The server load changes the answer again

**Figure 9** evaluates a shuffled mixture of requests from all ten benchmarks, called **SpecMix**, across increasing concurrency.

[![Figure 9: Speculative-decoding speedup across concurrency, both target models, two budgets, and three temperature pairs.](assets/figure-9.png)](assets/figure-9.png)

*Rows vary target and budget; columns vary target|draft temperatures. The horizontal axis is concurrent requests per replica. Speedup is relative to the target-only baseline at the same load; the dashed line is 1×. EAGLE-3 uses chain decoding here. Insets clarify nearby curves. Click to enlarge.*

A larger budget can help at low load yet lose to a smaller budget as concurrency grows. DCore remains competitive under greedy drafting, while semi-autoregressive methods become more competitive under sampled drafting. The preferred recipe and budget therefore depend on how the model is served, as well as what it generates.

**The practical lesson:** first establish a common comparison, then seek acceptance improvements with little inference overhead, and finally select the drafter and budget together using end-to-end throughput under the intended workload and serving conditions.

## Acknowledgments

Our implementation is built on [SpecForge](https://github.com/sgl-project/SpecForge) and [SGLang](https://github.com/sgl-project/sglang). We thank their authors and maintainers for the training and serving foundations of this work.

We also thank the following projects, whose open-source implementations informed our training and evaluation code:

- [EAGLE](https://github.com/SafeAILab/EAGLE) for EAGLE-3.
- [DFlash](https://github.com/z-lab/dflash).
- [D-PACE](https://github.com/Lucas-TY/D-PACE).
- [AngelSlim](https://github.com/Tencent/AngelSlim) for DFlare.
- [Domino](https://github.com/jianuo-huang/Domino).
- [DeepSpec](https://github.com/deepseek-ai/DeepSpec) for DSpark.
