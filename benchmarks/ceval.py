import json
import random
import re
from pathlib import Path

from datasets import load_dataset


# ============================================================
# C-Eval configuration
# ============================================================

DATA_CACHE = Path("data/ceval_all.json")

CHOICES = ["A", "B", "C", "D"]


# C-Eval 52 subjects
SUBJECTS = [
    "accountant",
    "advanced_mathematics",
    "art_studies",
    "basic_medicine",
    "business_administration",
    "chinese_language_and_literature",
    "civil_servant",
    "clinical_medicine",
    "college_chemistry",
    "college_economics",
    "college_physics",
    "college_programming",
    "computer_architecture",
    "computer_network",
    "discrete_mathematics",
    "education_science",
    "electrical_engineer",
    "environmental_impact_assessment_engineer",
    "fire_engineer",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_chinese",
    "high_school_geography",
    "high_school_history",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_politics",
    "ideological_and_moral_cultivation",
    "law",
    "legal_professional",
    "logic",
    "mao_zedong_thought",
    "marxism",
    "metrology_engineer",
    "middle_school_biology",
    "middle_school_chemistry",
    "middle_school_geography",
    "middle_school_history",
    "middle_school_mathematics",
    "middle_school_physics",
    "middle_school_politics",
    "modern_chinese_history",
    "operating_system",
    "physician",
    "plant_protection",
    "probability_and_statistics",
    "professional_tour_guide",
    "sports_science",
    "tax_accountant",
    "teacher_qualification",
    "urban_and_rural_planner",
    "veterinary_medicine",
]


# ============================================================
# Load C-Eval
# ============================================================

def _download_and_cache():
    """
    第一次运行：

    1. 从 Hugging Face 加载 C-Eval
    2. 读取所有 subject 的 validation split
    3. 转成统一格式
    4. 保存到 data/ceval_all.json

    后续运行不会再次访问 Hugging Face。
    """

    print()
    print("=" * 70)
    print("Downloading C-Eval dataset")
    print("=" * 70)

    all_questions = []

    for index, subject in enumerate(SUBJECTS, start=1):

        print(
            f"[{index:02d}/{len(SUBJECTS):02d}] "
            f"Loading {subject}..."
        )

        try:
            dataset = load_dataset(
                "ceval/ceval-exam",
                subject,
                split="val",
            )

        except Exception as e:

            print()
            print(
                f"WARNING: failed to load subject: {subject}"
            )
            print(e)

            continue

        for item in dataset:

            question = item.get("question", "")

            choices = [
                item.get("A", ""),
                item.get("B", ""),
                item.get("C", ""),
                item.get("D", ""),
            ]

            answer = item.get("answer", "")

            if not question:
                continue

            if answer not in CHOICES:
                continue

            all_questions.append(
                {
                    "subject": subject,
                    "question": question,
                    "choices": choices,
                    "answer": answer,
                }
            )

    print()
    print(
        f"Loaded {len(all_questions)} C-Eval questions "
        f"from {len(SUBJECTS)} subjects."
    )

    if not all_questions:
        raise RuntimeError(
            "C-Eval dataset is empty."
        )

    # --------------------------------------------------------
    # Save local cache
    # --------------------------------------------------------

    DATA_CACHE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        DATA_CACHE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            all_questions,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        "C-Eval local cache saved to:",
        DATA_CACHE,
    )

    return all_questions


def load_ceval(
    limit=None,
    seed=42,
):
    """
    加载 C-Eval。

    参数：

        limit:
            None = 全部
            N    = 随机抽 N 题

        seed:
            随机种子，默认 42

    数据策略：

        如果 data/ceval_all.json 存在：
            直接读取本地缓存

        如果不存在：
            第一次从 Hugging Face 下载并建立缓存
    """

    print()
    print("Loading C-Eval dataset...")

    # ========================================================
    # Local cache
    # ========================================================

    if DATA_CACHE.exists():

        print(
            "Using local C-Eval cache:",
            DATA_CACHE,
        )

        with open(
            DATA_CACHE,
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

    else:

        print(
            "Local C-Eval cache not found."
        )

        data = _download_and_cache()

    print(
        f"Total C-Eval questions available: {len(data)}"
    )

    # ========================================================
    # Random sampling
    # ========================================================

    if limit is not None:

        limit = min(
            int(limit),
            len(data),
        )

        rng = random.Random(seed)

        indices = rng.sample(
            range(len(data)),
            limit,
        )

        data = [
            data[i]
            for i in indices
        ]

        print(
            f"Random sample: {len(data)} "
            f"(seed={seed})"
        )

    else:

        print(
            f"Using all {len(data)} questions"
        )

    return data


# ============================================================
# Extract answer
# ============================================================

def extract_choice(text):
    """
    从模型输出中提取 A/B/C/D。

    支持：

        A
        B.
        Final answer: C
        The answer is D
        **B**
        答案：C
        最终答案：D
    """

    if not text:
        return None

    text = text.strip()

    # --------------------------------------------------------
    # 1. 明确的最终答案
    # --------------------------------------------------------

    patterns = [

        r"(?:final answer|final choice)"
        r"\s*(?:is|:)?\s*\**\s*([ABCD])\b",

        r"(?:the answer|answer|correct answer)"
        r"\s*(?:is|:)?\s*\**\s*([ABCD])\b",

        r"(?:答案|正确答案|最终答案)"
        r"\s*(?:是|为|：|:)?\s*\**\s*([ABCD])\b",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:

            return match.group(1).upper()

    # --------------------------------------------------------
    # 2. 最后一行
    # --------------------------------------------------------

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if lines:

        last_line = lines[-1]

        match = re.search(
            r"^\**\s*([ABCD])\s*[\.\:\)]?\s*\**$",
            last_line,
            flags=re.IGNORECASE,
        )

        if match:

            return match.group(1).upper()

    # --------------------------------------------------------
    # 3. 最后出现的独立 A/B/C/D
    # --------------------------------------------------------

    matches = re.findall(
        r"\b([ABCD])\b",
        text,
        flags=re.IGNORECASE,
    )

    if matches:

        return matches[-1].upper()

    return None


# ============================================================
# Build prompt
# ============================================================

def build_prompt(item):

    question = item["question"]

    choices = item["choices"]

    prompt = f"""
请回答下面的选择题。

题目：
{question}

选项：
A. {choices[0]}
B. {choices[1]}
C. {choices[2]}
D. {choices[3]}

请先进行分析，然后在最后明确给出最终选项。
最终答案格式必须为：

Final answer: A

只允许选择 A、B、C、D 其中一个。
"""

    return prompt.strip()


# ============================================================
# Benchmark runner
# ============================================================

def run(
    api,
    limit=5,
    seed=42,
    output_path="results/ceval/results.json",
):

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    test_set = load_ceval(
        limit=limit,
        seed=seed,
    )

    results = []

    correct = 0
    invalid = 0

    # --------------------------------------------------------
    # Run benchmark
    # --------------------------------------------------------

    for i, item in enumerate(test_set):

        subject = item["subject"]

        question = item["question"]

        choices = item["choices"]

        gold = item["answer"]

        prompt = build_prompt(item)

        print()
        print("=" * 70)
        print(
            f"C-Eval {i + 1}/{len(test_set)}"
        )
        print("=" * 70)

        print(
            f"Subject : {subject}"
        )

        print()

        print(question)

        print()

        for index, choice in enumerate(choices):

            print(
                f"{CHOICES[index]}. {choice}"
            )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        result = api.generate(prompt)

        content = result["content"]

        predicted = extract_choice(
            content
        )

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        if predicted is None:

            invalid += 1

            is_correct = False

        else:

            is_correct = (
                predicted == gold
            )

        if is_correct:

            correct += 1

        # ----------------------------------------------------
        # Print
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

        if result.get("draft_tokens"):

            print(
                "DFlash    : %.2f%%"
                % (
                    result[
                        "draft_acceptance_rate"
                    ]
                    * 100
                )
            )

        # ----------------------------------------------------
        # Save individual result
        # ----------------------------------------------------

        results.append(
            {
                "index": i,

                "subject": subject,

                "question": question,

                "choices": {
                    "text": choices,
                    "label": CHOICES,
                },

                "gold": gold,

                "prediction": predicted,

                "correct": is_correct,

                "invalid": predicted is None,

                "latency": result[
                    "latency"
                ],

                "prompt_tokens": result[
                    "prompt_tokens"
                ],

                "completion_tokens": result[
                    "completion_tokens"
                ],

                "total_tokens": result[
                    "total_tokens"
                ],

                "generation_tps": result[
                    "generation_tps"
                ],

                "draft_tokens": result[
                    "draft_tokens"
                ],

                "draft_accepted": result[
                    "draft_accepted"
                ],

                "draft_acceptance_rate": result[
                    "draft_acceptance_rate"
                ],
            }
        )

    # ========================================================
    # Summary
    # ========================================================

    total = len(results)

    if total > 0:

        accuracy = (
            correct / total
        )

    else:

        accuracy = 0.0

    # ========================================================
    # Output
    # ========================================================

    output = {

        "benchmark": "ceval",

        "model": api.model,

        "seed": seed,

        "samples": total,

        "correct": correct,

        "invalid": invalid,

        "accuracy": accuracy,

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
    # Print summary
    # ========================================================

    print()
    print("=" * 70)
    print("C-Eval SUMMARY")
    print("=" * 70)

    print(
        "Samples :",
        total,
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

    print()

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
