# Muse Glimmer 30B Benchmark Report

> Generated: 2026-08-13 15:23:06
> Model: **muse-glimmer-30B**
> Random Seed: **42**

## Overall

| Metric | Result |
|---|---:|
| Benchmark | 5 |
| Samples | 250 |
| Correct | 221 |
| Invalid | 4 |
| Micro Accuracy | 88.40% |
| Macro Accuracy | 88.40% |
| Generation Speed | 22.91 tok/s |
| DFlash Acceptance | 59.64% |

## Benchmark Results

| Benchmark | Ability | Samples | Correct | Accuracy | Avg Latency | Generation | DFlash |
|---|---|---:|---:|---:|---:|---:|---:|
| GSM8K | 数学推理 | 50 | 47 | 94.00% | 22.42 s | 26.29 tok/s | 76.58% |
| MMLU | 综合知识 | 50 | 47 | 94.00% | 45.88 s | 21.67 tok/s | 54.29% |
| TruthfulQA | 真实性 / 幻觉控制 | 50 | 40 | 80.00% | 31.69 s | 22.89 tok/s | 59.03% |
| C-Eval | 中文综合知识 | 50 | 39 | 78.00% | 55.77 s | 23.03 tok/s | 59.53% |
| ARC | 科学推理 | 50 | 48 | 96.00% | 22.17 s | 21.75 tok/s | 55.26% |

## Token Statistics

| Benchmark | Prompt Tokens | Completion Tokens | Total Tokens |
|---|---:|---:|---:|
| GSM8K | 12,013 | 29,478 | 41,491 |
| MMLU | 10,251 | 49,702 | 59,953 |
| TruthfulQA | 8,040 | 36,282 | 44,322 |
| C-Eval | 9,506 | 64,221 | 73,727 |
| ARC | 7,297 | 24,106 | 31,403 |

## Notes

- Accuracy uses the benchmark result files directly.
- Overall Accuracy is calculated as micro-average.
- Macro Accuracy is the arithmetic mean of benchmark accuracies.
- Generation speed uses total completion tokens divided by total generation latency.
- DFlash acceptance uses total accepted draft tokens divided by total draft tokens.
- HumanEval is intentionally not included yet.
