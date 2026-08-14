#!/usr/bin/env python3
"""校验 GSM8K 本地缓存 vs 官方数据，并检测官方数据中的算术错误。

用法：
    python scripts/check_gsm8k.py             # 校验本地缓存 vs 官方 + 算术自检
    python scripts/check_gsm8k.py --fix       # 用官方数据覆盖本地缓存
    python scripts/check_gsm8k.py --report check_report.json   # 导出详细报告

背景说明：
    官方 GSM8K 数据集中存在极少数标注错误（业界已知）。
    例如某题官方答案写 "750+430+400+700=<<...=2280>>2280"，但实际
    Maryam 应算 700 而非 400，正确总额是 2180（模型算对了）。
    本脚本核心功能：
      1) 本地缓存与官方数据逐条对比（排查缓存是否被改写过）
      2) --fix 用官方数据重建本地缓存
    [REF] 算术自检只作怀疑提示（含大量误报：方程写法 <<x+30=110>>、
    百分比 3/6=50% 等都会被误判），不代表官方数据有错。
"""

import argparse
import json
import re
import sys
from pathlib import Path

LOCAL_DATA = Path("data/gsm8k_test.jsonl")
REPORT = Path("gsm8k_check_report.json")

ARITH_RE = re.compile(
    r"([-+]?\d[\d\s+\-*/().]*?)\s*=\s*"
    r"([-+]?\d[\d,]*(?:\.\d+)?)"
)

FINAL_RE = re.compile(
    r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)"
)


def safe_eval(expr):
    if not re.fullmatch(r"[\d\s+\-*/().]+", expr):
        return None
    try:
        return eval(expr, {"__builtins__": {}}, {})
    except Exception:
        return None


def load_local():
    rows = []
    if LOCAL_DATA.exists():
        with open(LOCAL_DATA, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def load_official():
    from datasets import load_dataset

    print("Loading official GSM8K (openai/gsm8k, main, test)...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    return [dict(row) for row in ds]


BOUNDARY_CHARS = set("<=( [{\t\r\n")


def _is_expr_start(text, pos):
    """表达式起点必须是边界字符（排除被截断的片段/方程中间）。"""
    if pos <= 0:
        return True
    return text[pos - 1] in BOUNDARY_CHARS


def _clean_expr(expr):
    expr_clean = expr.replace(" ", "")
    if not expr_clean:
        return None
    if not re.fullmatch(r"[\d+\-*/().]+", expr_clean):
        return None
    return expr_clean


def check_arithmetic(row):
    """检测答案文本内 `expr = N` 的算术矛盾，返回问题列表。

    只检测"从边界开始的纯数字表达式"，排除含未知数的方程写法
    （如 <<x+30=110>>）以及被截断的片段（如 +30=110）。
    """
    issues = []
    text = row.get("answer", "")
    for m in ARITH_RE.finditer(text):
        expr, claimed = m.group(1), m.group(2)

        if not _is_expr_start(text, m.start()):
            continue

        expr_clean = _clean_expr(expr)
        if expr_clean is None:
            continue

        value = safe_eval(expr_clean)
        if value is None:
            continue

        claimed_f = float(claimed.replace(",", ""))
        if abs(value - claimed_f) > 1e-6:
            issues.append(
                {
                    "expr": expr_clean,
                    "computed": value,
                    "claimed": claimed_f,
                }
            )
    return issues


def check_final_answer(row):
    """检测 `#### N` 最终答案与最后一个算式是否矛盾。

    从后往前找第一个"从边界开始的纯数字算式"进行比较。
    """
    text = row.get("answer", "")
    m = FINAL_RE.search(text)
    if not m:
        return None

    final_val = float(m.group(1).replace(",", ""))

    head = text[: m.start()]
    matches = list(ARITH_RE.finditer(head))
    if not matches:
        return None

    for last in reversed(matches):
        if not _is_expr_start(head, last.start()):
            continue
        expr_clean = _clean_expr(last.group(1))
        if expr_clean is None:
            continue
        val = safe_eval(expr_clean)
        if val is None:
            continue
        if abs(val - final_val) > 1e-6:
            return {
                "expr": expr_clean,
                "computed": val,
                "final": final_val,
            }
        return None

    return None


def main():

    parser = argparse.ArgumentParser(
        description="Check / rebuild the local GSM8K cache."
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Overwrite local cache with official data.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Write detailed JSON report to this path.",
    )
    args = parser.parse_args()

    local = load_local()
    print(f"Local cache  : {len(local)} questions")

    official = load_official()
    print(f"Official     : {len(official)} questions")

    # --------------------------------------------------------
    # 1) 本地 vs 官方一致性
    # --------------------------------------------------------

    if len(local) != len(official):
        print(
            f"  [MISMATCH] length differs: "
            f"{len(local)} vs {len(official)}"
        )

    mismatched = []
    n = min(len(local), len(official))
    for i in range(n):
        lq = local[i].get("question", "")
        oq = official[i].get("question", "")
        la = local[i].get("answer", "")
        oa = official[i].get("answer", "")
        if lq != oq or la != oa:
            mismatched.append(
                {
                    "index": i,
                    "local_question": lq[:100],
                    "official_question": oq[:100],
                    "local_answer_tail": la[-80:],
                    "official_answer_tail": oa[-80:],
                }
            )

    if mismatched:
        print(f"  [MISMATCH] {len(mismatched)} entries differ from official:")
        for m in mismatched[:10]:
            print(f"    #{m['index']}: {m['local_question'][:60]}...")
        if len(mismatched) > 10:
            print(f"    ... and {len(mismatched) - 10} more")
    else:
        print("  [OK] local cache matches official data exactly")

    # --------------------------------------------------------
    # 2) 官方数据内部算术自检
    # --------------------------------------------------------

    arith_issues = []
    for i, row in enumerate(official):
        issues = check_arithmetic(row)
        if issues:
            arith_issues.append(
                {
                    "index": i,
                    "question": row.get("question", "")[:120],
                    "answer_tail": row.get("answer", "")[-100:],
                    "issues": issues,
                }
            )

    if arith_issues:
        print(
            f"  [REF] {len(arith_issues)} entries flagged by the arithmetic "
            f"self-check (REFERENCE ONLY, high false-positive rate: "
            f"equation forms like <<x+30=110>> and percentages are miscaught)"
        )
        for a in arith_issues[:5]:
            print(f"    #{a['index']}: {a['question'][:60]}...")
    else:
        print("  [OK] no self-inconsistent arithmetic found")

    # --------------------------------------------------------
    # 2b) `#### N` 最终答案与最后算式矛盾
    # --------------------------------------------------------

    final_issues = []
    for i, row in enumerate(official):
        iss = check_final_answer(row)
        if iss:
            final_issues.append(
                {
                    "index": i,
                    "question": row.get("question", "")[:120],
                    "answer_tail": row.get("answer", "")[-100:],
                    "issue": iss,
                }
            )

    if final_issues:
        print(
            f"  [REF] {len(final_issues)} entries where the final answer "
            f"appears to contradict the last arithmetic line "
            f"(REFERENCE ONLY, same false-positive caveats)"
        )
        for a in final_issues[:5]:
            print(f"    #{a['index']}: {a['question'][:60]}...")
    else:
        print("  [OK] final answers are consistent with the arithmetic")

    # --------------------------------------------------------
    # 3) 重建缓存（可选）
    # --------------------------------------------------------

    if args.fix:
        LOCAL_DATA.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(LOCAL_DATA) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for row in official:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        import os

        os.replace(tmp, LOCAL_DATA)
        print(f"  [FIX] local cache overwritten with official data ({len(official)} rows)")

    # --------------------------------------------------------
    # 4) 导出报告（可选）
    # --------------------------------------------------------

    if args.report:
        report = {
            "local_count": len(local),
            "official_count": len(official),
            "mismatched": mismatched,
            "arithmetic_issues": arith_issues,
            "final_answer_issues": final_issues,
        }
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"  [REPORT] written to {args.report}")

    print()
    print("Note: the official GSM8K dataset contains a small number of")
    print("labeling errors (e.g. the carnival question: 2180 vs 2280).")
    print("These are interpretation-level mistakes that cannot be")
    print("auto-detected; review any question where the model's reasoning")
    print("looks right but is marked FAIL. The [REF] lines above are")
    print("suspicion hints only, not a verdict.")


if __name__ == "__main__":
    main()
