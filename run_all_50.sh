#!/bin/bash

# ============================================================
# Muse Glimmer 30B
# Benchmark suite (Seed: 42)
#
# 用法:
#   ./run_all_50.sh                  # 全部 5 个 benchmark, 每项 50 题
#   ./run_all_50.sh 100              # 每项 100 题
#   ./run_all_50.sh 100 gsm8k,mmlu   # 只跑 gsm8k 和 mmlu, 每项 100 题
#   ./run_all_50.sh 0                # 全量（不抽样）
# ============================================================

set +e

LIMIT=${1:-50}
SEED=42

BENCH_SELECT="${2:-}"

if [ -n "${BENCH_SELECT}" ]; then
    BENCHMARKS=(${BENCH_SELECT//,/ })
else
    BENCHMARKS=(
        gsm8k
        mmlu
        truthfulqa
        ceval
        arc
    )
fi

echo
echo "======================================================================"
echo " Benchmark Suite"
echo "======================================================================"
echo
echo "Model : auto-detect (from /v1/models)"
echo "Limit : ${LIMIT}"
echo "Seed  : ${SEED}"
echo
echo "Benchmarks:"
for BENCH in "${BENCHMARKS[@]}"
do
    echo "  - ${BENCH}"
done
echo
echo "======================================================================"
echo

START_TIME=$(date +%s)

for BENCH in "${BENCHMARKS[@]}"
do

    echo
    echo "######################################################################"
    echo " START: ${BENCH}"
    echo "######################################################################"
    echo

    BENCH_START=$(date +%s)

    python run_benchmark.py \
        --bench "${BENCH}" \
        --limit "${LIMIT}" \
        --seed "${SEED}"

    STATUS=$?

    BENCH_END=$(date +%s)
    ELAPSED=$((BENCH_END - BENCH_START))

    echo
    echo "----------------------------------------------------------------------"

    if [ ${STATUS} -eq 0 ]; then
        echo "${BENCH}: PASS"
    else
        echo "${BENCH}: FAILED"
        echo "Exit code: ${STATUS}"
    fi

    echo "Elapsed: ${ELAPSED}s"
    echo "----------------------------------------------------------------------"

done

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo
echo "======================================================================"
echo " Benchmark Suite Finished"
echo "======================================================================"
echo
echo "Limit : ${LIMIT}"
echo "Seed  : ${SEED}"
echo
echo "Total elapsed: ${TOTAL_TIME}s"
echo
echo "Results:"
echo
for BENCH in "${BENCHMARKS[@]}"
do
    echo "  results/${BENCH}/results.json"
done
echo
echo "======================================================================"
