#!/usr/bin/env python3
"""数据集指纹与结构校验 CLI。

用法：
    python scripts/check_data.py                  # 校验全部数据集（指纹+结构）
    python scripts/check_data.py --bench gsm8k    # 只校验某个
    python scripts/check_data.py --fingerprint    # 重新生成指纹（重建数据后执行）
"""

import argparse
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from core import datacheck


def main():

    parser = argparse.ArgumentParser(
        description="Verify local dataset integrity (fingerprint + structure)."
    )
    parser.add_argument(
        "--bench",
        choices=list(datacheck.BENCH_DATA_MAP.keys()),
        default=None,
        help="Only check the given benchmark (default: all).",
    )
    parser.add_argument(
        "--fingerprint",
        action="store_true",
        help="Regenerate fingerprints from current files. "
        "Run this AFTER rebuilding caches from official data.",
    )
    args = parser.parse_args()

    if args.fingerprint:
        data = datacheck.generate_fingerprints()
        files = data["files"]
        print(f"Fingerprints written to {datacheck.FINGERPRINT_FILE}")
        print(f"  {len(files)} dataset files recorded:")
        for rel, entry in files.items():
            print(
                f"    {rel}  rows={entry['rows']}  "
                f"sha256={entry['sha256'][:12]}..."
            )
        print("Next: run this script without --fingerprint to verify.")
        return

    benches = (
        [args.bench]
        if args.bench
        else list(datacheck.BENCH_DATA_MAP.keys())
    )

    all_ok = True

    for bench in benches:
        ok, msgs = datacheck.verify_benchmark(bench)
        if ok:
            print(f"[OK]   {bench}")
        else:
            all_ok = False
            print(f"[FAIL] {bench}")
            for m in msgs:
                print(f"       {m}")

    print()
    if all_ok:
        print("All datasets verified. Safe to run benchmarks.")
    else:
        print(
            "Some datasets FAILED verification. "
            "Do NOT run benchmarks until data is restored/rebuilt, "
            "then regenerate fingerprints."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
