# 示例评测报告（Sample Reports）

> 本目录存放真实评测的**输出示例**（HTML / Markdown / CSV），
> 供快速了解报告形态，不作为基准数据。结果均可通过 `./run_all_50.sh 50` 重新复现。

## 说明（备注）

- 硬件环境：AMD Ryzen AI MAX+ 395（128GB 统一内存）· llama.cpp · Vulkan
- 评测口径：5 任务 × 50 题 · seed 42 · temperature 0
- 全部结果可复现：`./run_all_50.sh 50 && python generate_report.py`

## muse-glimmer-30B（Q4_K_M · 启用 DFlash 投机解码）

| 项 | 值 |
|---|---|
| 报告生成时间 | 2026-08-13 15:23 |
| 整体准确率 | 88.40%（221/250，微/宏平均一致） |
| 生成速度 | 22.91 tok/s |
| DFlash 接受率 | 59.64% |
| 备注 | DFlash 投机解码已启用（draft 模型配对）；GSM8K/MMLU 94%、ARC 96% |

- `benchmark_report.html`（含交互图表的完整报告）
- `benchmark_report.md` / `benchmark_summary.csv`

## Qwen3.5-35B-A3B（Q4_K_M · 未启用投机解码）

| 项 | 值 |
|---|---|
| 报告生成时间 | 2026-08-14 09:15 |
| 整体准确率 | 89.20%（223/250） |
| 生成速度 | 63.30 tok/s |
| DFlash 接受率 | 0.00% |
| 备注 | **未配置 draft 模型**，投机解码未启用（DFlash 为 0）；MoE 结构（3B 激活）端侧速度约 2.8× muse-glimmer-30B；C-Eval 88%、ARC 98% 表现突出，TruthfulQA 72% 弱于 muse |

- `benchmark_report.html`（交互式 HTML 报告，标题已动态化为模型名）

## 对比观察（同一评测体系下的模型选型参考）

| 维度 | muse-glimmer-30B | Qwen3.5-35B-A3B |
|---|---|---|
| 整体准确率 | 88.40% | 89.20% |
| 中文能力（C-Eval） | 78% | 88% |
| 真实性（TruthfulQA） | 80% | 72% |
| 端侧速度 | 22.91 tok/s | 63.30 tok/s |
| 推理优化 | DFlash 已启用 | 未启用（可配 draft 后提升） |

> 注意：两份报告的 DFlash 配置不同（muse 开、Qwen 未开），速度对比不代表模型能力差异，
> 而是"部署配置 × 模型结构"的端到端差异——这正是本评测体系的定位：测硬件上的真实表现。
