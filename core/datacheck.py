"""本地数据集指纹校验：防止数据文件被悄悄替换/污染。

- 为每个数据集缓存文件记录 sha256 + 来源 + 行数（data/dataset_fingerprints.json）
- 跑评测前校验：文件 hash 与指纹不符即告警
- 附带结构级校验（字段是否齐全、取值是否合法）

用法（脚本入口见 scripts/check_data.py）：
    重建数据后运行 scripts/check_data.py --fingerprint  生成/更新指纹
    之后每次 run_benchmark.py 会自动校验对应数据集
"""

import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path

FINGERPRINT_FILE = Path("data/dataset_fingerprints.json")

BENCH_DATA_MAP = {
    "gsm8k": "data/gsm8k_test.jsonl",
    "mmlu": "data/mmlu_all.json",
    "arc": "data/arc_test.json",
    "ceval": "data/ceval_all.json",
    "truthfulqa": "data/truthfulqa/truthfulqa_mc.json",
}

SOURCES = {
    "gsm8k": "huggingface:openai/gsm8k:main:test",
    "mmlu": "huggingface:cais/mmlu:all:test",
    "arc": "huggingface:allenai/ai2_arc:ARC-Challenge:test",
    "ceval": "huggingface:ceval/ceval-exam:52-subjects:val",
    "truthfulqa": "huggingface:truthfulqa/truthful_qa:multiple_choice:validation",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_rows(path):
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        n = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
        return n
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return len(data)


def _load_rows(rel_path):
    path = Path(rel_path)
    suffix = path.suffix.lower()
    with open(path, "r", encoding="utf-8") as f:
        if suffix == ".jsonl":
            return [
                json.loads(line)
                for line in f
                if line.strip()
            ]
        return json.load(f)


def structure_checks(bench):
    """对数据集做结构级校验，返回问题列表。"""
    rel = BENCH_DATA_MAP[bench]
    if not Path(rel).exists():
        return [f"data file missing: {rel}"]

    rows = _load_rows(rel)
    problems = []

    if bench == "gsm8k":
        # 注意：官方 GSM8K 文本本身含弯引号/连字符等非 ASCII 字符，
        # 非 ASCII 不能作为污染信号（check_gsm8k.py 已验证本地=官方）。
        # 这里只做字段存在性检查。
        bad = 0
        for r in rows[:200]:
            if "question" not in r or "answer" not in r:
                bad += 1
                break
        if not rows:
            problems.append("empty dataset")
        if bad:
            problems.append("rows missing question/answer")

    elif bench == "mmlu":
        bad = 0
        for r in rows[:500]:
            if "question" not in r or "choices" not in r or "answer" not in r:
                bad += 1
                continue
            choices = r["choices"]
            answer = r["answer"]
            if not isinstance(choices, list) or len(choices) < 2:
                bad += 1
                continue
            if not isinstance(answer, int) or not (0 <= answer < len(choices)):
                bad += 1
        if bad:
            problems.append(f"{bad} malformed rows (out of {len(rows)})")

    elif bench == "arc":
        # 官方数据混用字母(A-E)与数字(1-5, 1-based)两种 answerKey 格式，
        # 且题目文本可能含特殊字符，故两种格式都视为合法。
        valid_keys = ("A", "B", "C", "D", "E", "1", "2", "3", "4", "5")
        bad = 0
        for r in rows[:500]:
            choices = r.get("choices")
            if "question" not in r or not isinstance(choices, dict):
                bad += 1
                continue
            if (
                "text" not in choices
                or "label" not in choices
                or not isinstance(choices["text"], list)
                or not isinstance(choices["label"], list)
            ):
                bad += 1
                continue
            if r.get("answerKey") not in valid_keys:
                bad += 1
        if bad:
            problems.append(f"{bad} malformed rows (out of {len(rows)})")

    elif bench == "ceval":
        # 本地缓存结构：{subject, question, choices: [A,B,C,D], answer: "A".."D"}
        bad = 0
        for r in rows[:300]:
            if not all(
                k in r
                for k in ("subject", "question", "choices", "answer")
            ):
                bad += 1
                continue
            choices = r["choices"]
            if not isinstance(choices, list) or len(choices) != 4:
                bad += 1
                continue
            if r["answer"] not in ("A", "B", "C", "D"):
                bad += 1
        if bad:
            problems.append(f"{bad} malformed rows (out of {len(rows)})")

    elif bench == "truthfulqa":
        # 本地缓存结构（build_truthfulqa_items 转换后）：
        # {index, question, choices: [...], labels: ["A"..], answer: "A"..}
        bad = 0
        for r in rows[:300]:
            if not all(
                k in r
                for k in ("question", "choices", "labels", "answer")
            ):
                bad += 1
                continue
            choices = r["choices"]
            if not isinstance(choices, list) or len(choices) < 2:
                bad += 1
                continue
            valid = tuple(chr(65 + i) for i in range(len(choices)))
            if r["answer"] not in valid:
                bad += 1
        if bad:
            problems.append(f"{bad} malformed rows (out of {len(rows)})")

    return problems


def generate_fingerprints():
    fingerprints = {}
    for bench, rel in BENCH_DATA_MAP.items():
        path = Path(rel)
        if path.exists():
            fingerprints[rel] = {
                "sha256": sha256_file(path),
                "rows": _count_rows(path),
                "source": SOURCES[bench],
                "benchmark": bench,
            }

    data = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "files": fingerprints,
    }

    FINGERPRINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(FINGERPRINT_FILE) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, FINGERPRINT_FILE)
    return data


def verify_benchmark(bench):
    """校验单个 benchmark 的数据完整性。

    返回 (ok: bool, messages: list[str])

    策略：
      - 文件缺失          -> 放行（ceval/truthfulqa 会在 run() 时自动从官方重建），仅提示
      - hash 与指纹不符   -> 拦截（数据被改动，防替换的核心防线）
      - 结构问题          -> 拦截
      - 无指纹文件        -> 放行但提示先 --fingerprint
    """
    messages = []

    rel = BENCH_DATA_MAP.get(bench)
    if rel is None:
        return True, []

    path = Path(rel)

    if not path.exists():
        return (
            True,
            [
                f"data file missing: {rel} "
                "(will be rebuilt automatically when the benchmark runs)"
            ],
        )

    if not FINGERPRINT_FILE.exists():
        return (
            True,
            [
                f"no fingerprint file ({FINGERPRINT_FILE}). "
                "Recommended: python scripts/check_data.py --fingerprint"
            ],
        )

    with open(FINGERPRINT_FILE, "r", encoding="utf-8") as f:
        fp = json.load(f)
    files = fp.get("files", {})

    entry = files.get(rel)
    if entry is None:
        return (
            True,
            [
                f"no fingerprint for {rel} (file is new?). "
                "Recommended: python scripts/check_data.py --fingerprint"
            ],
        )

    current = sha256_file(path)
    if current != entry.get("sha256"):
        return (
            False,
            [
                f"HASH MISMATCH for {rel}",
                f"  expected: {entry.get('sha256')}",
                f"  actual  : {current}",
                "The dataset file was modified since fingerprints were "
                "generated. Stop and restore the official data.",
            ],
        )

    problems = structure_checks(bench)
    if problems:
        messages.append(f"structure issues in {rel}:")
        messages.extend(f"  - {p}" for p in problems[:8])

    return (len(messages) == 0), messages


def verify_all():
    results = {}
    for bench in BENCH_DATA_MAP:
        ok, msgs = verify_benchmark(bench)
        results[bench] = {"ok": ok, "messages": msgs}
    return results
