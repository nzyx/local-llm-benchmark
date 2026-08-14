import json
import random
import re
from pathlib import Path

from datasets import load_dataset


# ============================================================
# 配置
# ============================================================

CACHE_DIR = Path("data/truthfulqa")
CACHE_FILE = CACHE_DIR / "truthfulqa_mc.json"

CHOICES = ["A", "B", "C", "D"]


# ============================================================
# 基础工具
# ============================================================

def normalize_text(text):
    if text is None:
        return ""

    text = str(text)

    text = text.replace("\u201c", '"')
    text = text.replace("\u201d", '"')
    text = text.replace("\u2018", "'")
    text = text.replace("\u2019", "'")

    return text.strip()


def extract_choice_from_text(text, choices):
    """
    从模型回答中提取 A/B/C/D。

    支持：

    A
    B.
    Final answer: C
    Answer: D
    The correct answer is A
    I choose B
    I would choose C
    答案：D
    正确答案是 A
    最终答案：B
    **C**

    如果没有明确字母，则尝试通过：
        模型回答中是否复述了某个选项文本

    返回：
        A / B / C / D
        None
    """

    text = normalize_text(text)

    if not text:
        return None

    # --------------------------------------------------------
    # 1. 明确的最终答案表达
    # --------------------------------------------------------

    patterns = [
        # English
        r"(?:final\s+answer|final\s+choice)"
        r"\s*(?:is|:)?\s*\**\s*([ABCD])\b",

        r"(?:the\s+)?(?:correct\s+answer|correct\s+choice)"
        r"\s*(?:is|:)?\s*\**\s*([ABCD])\b",

        r"(?:answer|choice)"
        r"\s*(?:is|:)?\s*\**\s*([ABCD])\b",

        r"(?:i\s+would\s+choose|i\s+choose|i\s+select)"
        r"\s*\**\s*([ABCD])\b",

        # Chinese
        r"(?:答案|正确答案|最终答案|选择)"
        r"\s*(?:是|为|：|:)?\s*\**\s*([ABCD])\b",
    ]

    for pattern in patterns:
        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if matches:
            return matches[-1].upper()

    # --------------------------------------------------------
    # 2. Markdown / 独立选项
    # --------------------------------------------------------

    lines = text.splitlines()

    for line in reversed(lines):
        line = line.strip()

        # **A**
        match = re.fullmatch(
            r"\**\s*([ABCD])\s*\**[\.\:\)]?",
            line,
            flags=re.IGNORECASE,
        )

        if match:
            return match.group(1).upper()

        # A. xxx
        match = re.match(
            r"^\s*\**\s*([ABCD])\s*[\.\:\)]\s*",
            line,
            flags=re.IGNORECASE,
        )

        if match:
            return match.group(1).upper()

    # --------------------------------------------------------
    # 3. 如果最后一段非常明确地以 A/B/C/D 结尾
    # --------------------------------------------------------

    tail = text[-300:]

    matches = re.findall(
        r"(?:^|[\s\(\[\{,:：])\**([ABCD])\**(?:[\.\)\]]|$)",
        tail,
        flags=re.IGNORECASE,
    )

    if matches:
        return matches[-1].upper()

    return None


# ============================================================
# 从 TruthfulQA 数据中提取正确答案
# ============================================================

def build_truthfulqa_items(raw_dataset):
    """
    TruthfulQA multiple-choice 数据格式通常包含：

        question
        mc1_targets

    mc1_targets:
        {
            "choices": [...],
            "labels": [...]
        }

    label=1 表示正确答案。
    """

    items = []

    for index, row in enumerate(raw_dataset):

        question = row.get("question", "")

        targets = row.get("mc1_targets")

        if not question or not targets:
            continue

        choices = targets.get("choices", [])
        labels = targets.get("labels", [])

        if not choices or not labels:
            continue

        if len(choices) != len(labels):
            continue

        correct_indices = [
            i
            for i, label in enumerate(labels)
            if int(label) == 1
        ]

        if not correct_indices:
            continue

        # TruthfulQA MC1 通常只有一个正确答案
        correct_index = correct_indices[0]

        if correct_index >= len(choices):
            continue

        # 这里保留所有原始选项。
        # 如果超过 4 个选项，后面使用 A/B/C/...。
        labels_text = [
            chr(65 + i)
            for i in range(len(choices))
        ]

        items.append(
            {
                "index": index,
                "question": question,
                "choices": choices,
                "labels": labels_text,
                "answer": labels_text[correct_index],
                "answer_index": correct_index,
            }
        )

    return items


# ============================================================
# 加载 TruthfulQA
# ============================================================

def load_truthfulqa(limit=None, seed=42):
    """
    加载 TruthfulQA multiple-choice。

    优先：
        data/truthfulqa/truthfulqa_mc.json

    如果本地缓存不存在：
        从 Hugging Face 下载一次
        然后保存到本地。

    limit:
        None = 全部
        N    = 随机 N 题

    seed:
        随机种子
    """

    print("Loading TruthfulQA dataset...")

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 本地缓存
    # --------------------------------------------------------

    if CACHE_FILE.exists():

        print(
            f"Using local TruthfulQA cache: "
            f"{CACHE_FILE}"
        )

        with open(
            CACHE_FILE,
            "r",
            encoding="utf-8",
        ) as f:
            items = json.load(f)

    else:

        print("Local TruthfulQA cache not found.")
        print("Downloading TruthfulQA MC dataset...")

        # 当前 Hugging Face datasets 版本下，
        # 使用 truthfulqa/truthful_qa / multiple_choice。
        # 注意：新版 datasets/huggingface_hub 要求 repo id 必须是
        # namespace/name 格式（写 "truthful_qa" 会报 HfUriError）。
        dataset = load_dataset(
            "truthfulqa/truthful_qa",
            "multiple_choice",
            split="validation",
        )

        items = build_truthfulqa_items(dataset)

        with open(
            CACHE_FILE,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                items,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(
            f"Saved local TruthfulQA cache: "
            f"{CACHE_FILE}"
        )

    print(
        f"Total TruthfulQA questions available: "
        f"{len(items)}"
    )

    # --------------------------------------------------------
    # 随机抽样
    # --------------------------------------------------------

    if limit is not None:

        limit = min(
            int(limit),
            len(items),
        )

        rng = random.Random(seed)

        indices = rng.sample(
            range(len(items)),
            limit,
        )

        selected = [
            items[i]
            for i in indices
        ]

        print(
            f"Random sample: {len(selected)} "
            f"(seed={seed})"
        )

        return selected

    return items


# ============================================================
# 构造 Prompt
# ============================================================

def build_prompt(item):
    question = item["question"]
    choices = item["choices"]

    lines = []

    lines.append(
        "Answer the following multiple-choice question."
    )

    lines.append("")
    lines.append(
        f"Question: {question}"
    )

    lines.append("")

    for i, choice in enumerate(choices):
        label = chr(65 + i)

        lines.append(
            f"{label}. {choice}"
        )

    lines.append("")

    lines.append(
        "Please reason carefully and give your final answer "
        "as one of the option letters (A, B, C, or D)."
    )

    lines.append(
        "Final answer:"
    )

    return "\n".join(lines)


# ============================================================
# 根据选项文本兜底判断
# ============================================================

def match_choice_by_content(text, choices):
    """
    如果模型没有输出 A/B/C/D，
    尝试判断它最后是在支持哪个选项。

    例如模型回答：

        "The correct statement is that water boils at 100°C."

    如果选项 C 正好是这个内容，
    就返回 C。
    """

    text_norm = normalize_text(text).lower()

    if not text_norm:
        return None

    # 优先检查回答末尾
    tail = text_norm[-500:]

    candidates = []

    for i, choice in enumerate(choices):

        choice_norm = normalize_text(choice).lower()

        if not choice_norm:
            continue

        # 去掉过短内容，避免误匹配
        if len(choice_norm) < 8:
            continue

        # 完整选项匹配
        if choice_norm in tail:
            candidates.append(
                (
                    len(choice_norm),
                    chr(65 + i),
                )
            )

    if candidates:

        candidates.sort(
            reverse=True
        )

        return candidates[0][1]

    return None


# ============================================================
# 提取 TruthfulQA 答案
# ============================================================

def extract_truthfulqa_answer(text, choices):
    """
    第一优先级：
        A/B/C/D

    第二优先级：
        选项文本匹配

    返回：
        A/B/C/D/...
        None
    """

    answer = extract_choice_from_text(
        text,
        choices,
    )

    if answer is not None:
        return answer

    return match_choice_by_content(
        text,
        choices,
    )


# ============================================================
# Benchmark Runner
# ============================================================

def run(
    api,
    limit=5,
    seed=42,
    output_path="results/truthfulqa/results.json",
):

    dataset = load_truthfulqa(
        limit=limit,
        seed=seed,
    )

    results = []

    correct = 0
    invalid = 0

    for i, item in enumerate(dataset):

        question = item["question"]
        choices = item["choices"]
        gold = item["answer"]

        prompt = build_prompt(item)

        print()
        print("=" * 70)
        print(
            f"TruthfulQA {i + 1}/{len(dataset)}"
        )
        print("=" * 70)

        print()
        print(question)

        print()

        for j, choice in enumerate(choices):

            print(
                f"{chr(65 + j)}. {choice}"
            )

        # ----------------------------------------------------
        # 调用模型
        # ----------------------------------------------------

        result = api.generate(
            prompt
        )

        content = result.get(
            "content",
            "",
        )

        # ----------------------------------------------------
        # 提取答案
        # ----------------------------------------------------

        predicted = extract_truthfulqa_answer(
            content,
            choices,
        )

        # ----------------------------------------------------
        # 判断
        # ----------------------------------------------------

        if predicted is None:

            invalid += 1

            is_correct = False

        else:

            is_correct = (
                predicted.upper()
                == gold.upper()
            )

            if is_correct:
                correct += 1

        # ----------------------------------------------------
        # 输出
        # ----------------------------------------------------

        print()
        print("MODEL:")
        print(content)

        print()

        print(
            "Predicted :",
            predicted,
        )

        print(
            "Gold      :",
            gold,
        )

        print(
            "Result    :",
            "PASS" if is_correct else "FAIL",
        )

        if result.get("generation_tps") is not None:

            print(
                "Generation tok/s : %.2f"
                % result["generation_tps"]
            )

        if result.get("draft_tokens"):

            print(
                "DFlash    : %.2f%%"
                % (
                    result.get(
                        "draft_acceptance_rate",
                        0.0,
                    )
                    * 100
                )
            )

        # ----------------------------------------------------
        # 保存单题结果
        # ----------------------------------------------------

        results.append(
            {
                "index": i,

                "question": question,

                "choices": {
                    "text": choices,
                    "label": [
                        chr(65 + j)
                        for j in range(len(choices))
                    ],
                },

                "gold": gold,

                "prediction": predicted,

                "correct": is_correct,

                "invalid": (
                    predicted is None
                ),

                "latency": result.get(
                    "latency",
                    0,
                ),

                "prompt_tokens": result.get(
                    "prompt_tokens",
                    0,
                ),

                "completion_tokens": result.get(
                    "completion_tokens",
                    0,
                ),

                "total_tokens": result.get(
                    "total_tokens",
                    0,
                ),

                "generation_tps": result.get(
                    "generation_tps",
                    0,
                ),

                "draft_tokens": result.get(
                    "draft_tokens",
                    0,
                ),

                "draft_accepted": result.get(
                    "draft_accepted",
                    0,
                ),

                "draft_acceptance_rate": result.get(
                    "draft_acceptance_rate",
                    0,
                ),
            }
        )

    # ========================================================
    # Summary
    # ========================================================

    samples = len(results)

    accuracy = (
        correct / samples
        if samples > 0
        else 0.0
    )

    # ========================================================
    # 保存 JSON
    # ========================================================

    output = {
        "benchmark": "truthfulqa",
        "model": api.model,

        "samples": samples,

        "correct": correct,

        "invalid": invalid,

        "accuracy": accuracy,

        "seed": seed,

        "results": results,
    }

    path = Path(
        output_path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 70)
    print("TruthfulQA SUMMARY")
    print("=" * 70)

    print(
        "Samples :",
        samples,
    )

    print(
        "Correct :",
        correct,
    )

    print(
        "Invalid :",
        invalid,
    )

    print(
        "Accuracy: %.2f%%"
        % (
            accuracy * 100
        )
    )

    print(
        "Seed    :",
        seed,
    )

    print()
    print(
        "Results saved to:",
        path,
    )

    return output
