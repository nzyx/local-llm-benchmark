import argparse
import sys

from core.api import MuseGlimmerAPI
from core import datacheck

from benchmarks.gsm8k import run as run_gsm8k
from benchmarks.mmlu import run as run_mmlu
from benchmarks.arc import run as run_arc
from benchmarks.truthfulqa import run as run_truthfulqa
from benchmarks.ceval import run as run_ceval


def main():

    parser = argparse.ArgumentParser(
        description="Muse Glimmer 30B Benchmark Runner"
    )

    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model id (default: auto-detect "
            "from /v1/models)"
        ),
    )

    parser.add_argument(
        "--bench",
        default="gsm8k",
        choices=[
            "gsm8k",
            "mmlu",
            "arc",
            "truthfulqa",
            "ceval",
        ],
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    api = MuseGlimmerAPI(
        model=args.model,
        base_url=(
            "http://127.0.0.1:8080"
            "/v1/chat/completions"
        ),
        temperature=0.0,
        max_tokens=4096,
    )

    RESULTS_BASE = (
        f"results/{api.model_safe}"
    )

    print()
    print("=" * 70)
    print("Muse Glimmer 30B Benchmark")
    print("=" * 70)

    print(
        "Model :",
        api.model,
    )

    print(
        "API   :",
        api.base_url,
    )

    print(
        "Bench :",
        args.bench,
    )

    print(
        "Limit :",
        args.limit,
    )

    print(
        "Seed  :",
        args.seed,
    )

    # ========================================================
    # 数据完整性校验（指纹 + 结构）
    # ========================================================

    ok, msgs = datacheck.verify_benchmark(args.bench)

    if not ok:
        print()
        print("=" * 70)
        print("DATA VERIFICATION FAILED")
        print("=" * 70)
        for m in msgs:
            print("  " + m)
        print()
        print("Benchmark aborted: the dataset is missing, modified,")
        print("or not fingerprinted yet. Restore the official data, then:")
        print("  python scripts/check_data.py --fingerprint")
        sys.exit(2)

    # ========================================================
    # GSM8K
    # ========================================================

    if args.bench == "gsm8k":

        run_gsm8k(
            api,
            limit=args.limit,
            output_path=(
                f"{RESULTS_BASE}/gsm8k/results.json"
            ),
        )

    # ========================================================
    # MMLU
    # ========================================================

    elif args.bench == "mmlu":

        run_mmlu(
            api,
            limit=args.limit,
            seed=args.seed,
            output_path=(
                f"{RESULTS_BASE}/mmlu/results.json"
            ),
        )

    # ========================================================
    # ARC
    # ========================================================

    elif args.bench == "arc":

        run_arc(
            api,
            limit=args.limit,
            seed=args.seed,
            output_path=(
                f"{RESULTS_BASE}/arc/results.json"
            ),
        )

    # ========================================================
    # TruthfulQA
    # ========================================================

    elif args.bench == "truthfulqa":

        run_truthfulqa(
            api,
            limit=args.limit,
            seed=args.seed,
            output_path=(
                f"{RESULTS_BASE}/truthfulqa/results.json"
            ),
        )

    # ========================================================
    # C-Eval
    # ========================================================

    elif args.bench == "ceval":

        run_ceval(
            api,
            limit=args.limit,
            seed=args.seed,
            output_path=(
                f"{RESULTS_BASE}/ceval/results.json"
            ),
        )


if __name__ == "__main__":
    main()
