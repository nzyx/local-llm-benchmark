#!/usr/bin/env python3
"""一键生成本地数据集缓存（MMLU / ARC）。

用法：
    python scripts/prepare_data.py              # 生成全部缓存
    python scripts/prepare_data.py --only mmlu  # 只生成 MMLU
    python scripts/prepare_data.py --only arc   # 只生成 ARC

生成后 benchmarks/* 将完全离线运行（只在准备阶段联网一次）：

    data/mmlu_all.json    cais/mmlu "all" test split,   14,042 题
    data/arc_test.json    allenai/ai2_arc ARC-Challenge test, 1,172 题

原始字段原样保留（question / choices / answer 或 answerKey），
与 benchmarks/mmlu.py、benchmarks/arc.py 的判分逻辑直接兼容。
"""

import argparse
import json
import os
import sys
from pathlib import Path

DATA_DIR = Path("data")

MMLU_OUT = DATA_DIR / "mmlu_all.json"
ARC_OUT = DATA_DIR / "arc_test.json"


def save_json(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    os.replace(tmp, path)


def prepare_mmlu():

    from datasets import load_dataset

    print("Downloading MMLU (cais/mmlu, config=all, split=test)...")
    print("This is the full 57-subject test set, may take a few minutes.")

    ds = load_dataset("cais/mmlu", "all", split="test")

    rows = [dict(row) for row in ds]

    if not rows:
        sys.exit("ERROR: MMLU download returned an empty dataset.")

    save_json(MMLU_OUT, rows)

    print(f"MMLU cached: {len(rows)} questions -> {MMLU_OUT}")


def prepare_arc():

    from datasets import load_dataset

    print("Downloading ARC (allenai/ai2_arc, config=ARC-Challenge, split=test)...")

    ds = load_dataset(
        "allenai/ai2_arc",
        "ARC-Challenge",
        split="test",
    )

    rows = [dict(row) for row in ds]

    if not rows:
        sys.exit("ERROR: ARC download returned an empty dataset.")

    save_json(ARC_OUT, rows)

    print(f"ARC cached: {len(rows)} questions -> {ARC_OUT}")


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate local dataset caches for MMLU and ARC "
            "so benchmarks run fully offline."
        )
    )

    parser.add_argument(
        "--only",
        choices=["mmlu", "arc"],
        default=None,
        help="Only prepare the given dataset (default: both).",
    )

    args = parser.parse_args()

    if args.only in (None, "mmlu"):
        prepare_mmlu()

    if args.only in (None, "arc"):
        prepare_arc()

    print()
    print("All requested datasets are cached locally. Benchmarks can now run offline.")


if __name__ == "__main__":
    main()
