# Qwen3.5-35B-A3B-Q4_K_M Benchmark Report

> Generated: 2026-08-14 09:15:16
> Model: **Qwen3.5-35B-A3B-Q4_K_M**
> Random Seed: **42**

## Overall

| Metric | Result |
|---|---:|
| Benchmark | 5 |
| Samples | 250 |
| Correct | 223 |
| Invalid | 1 |
| Micro Accuracy | 89.20% |
| Macro Accuracy | 89.20% |
| Generation Speed | 63.30 tok/s |
| DFlash Acceptance | 0.00% |

## Benchmark Results

| Benchmark | Ability | Samples | Correct | Accuracy | Avg Latency | Generation | DFlash |
|---|---|---:|---:|---:|---:|---:|---:|
| GSM8K | 数学推理 | 50 | 48 | 96.00% | 6.50 s | 59.68 tok/s | 0.00% |
| MMLU | 综合知识 | 50 | 46 | 92.00% | 11.64 s | 64.58 tok/s | 0.00% |
| TruthfulQA | 真实性 / 幻觉控制 | 50 | 36 | 72.00% | 10.47 s | 63.82 tok/s | 0.00% |
| C-Eval | 中文综合知识 | 50 | 44 | 88.00% | 10.89 s | 64.27 tok/s | 0.00% |
| ARC | 科学推理 | 50 | 49 | 98.00% | 7.82 s | 62.36 tok/s | 0.00% |

## Token Statistics

| Benchmark | Prompt Tokens | Completion Tokens | Total Tokens |
|---|---:|---:|---:|
| GSM8K | 10,647 | 19,405 | 30,052 |
| MMLU | 8,477 | 37,600 | 46,077 |
| TruthfulQA | 6,238 | 33,410 | 39,648 |
| C-Eval | 7,394 | 35,011 | 42,405 |
| ARC | 5,410 | 24,392 | 29,802 |

## Notes

- Accuracy uses the benchmark result files directly.
- Overall Accuracy is calculated as micro-average.
- Macro Accuracy is the arithmetic mean of benchmark accuracies.
- Generation speed uses total completion tokens divided by total generation latency.
- DFlash acceptance uses total accepted draft tokens divided by total draft tokens.
- HumanEval is intentionally not included yet.
