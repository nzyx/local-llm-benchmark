import json
import random
import re
from pathlib import Path


# ============================================================
# GSM8K 配置
# ============================================================

LOCAL_DATA = Path("data/gsm8k_test.jsonl")


# ============================================================
# 数据加载
# ============================================================

def load_gsm8k(limit=None, seed=42):
    """
    从本地 GSM8K JSONL 加载数据。

    limit:
        None = 全部
        N    = 随机 N 题

    seed:
        随机种子
    """

    print("Loading GSM8K dataset...")

    if not LOCAL_DATA.exists():
        raise FileNotFoundError(
            f"GSM8K local cache not found: {LOCAL_DATA}"
        )

    print(
        f"Using local GSM8K cache: {LOCAL_DATA}"
    )

    dataset = []

    with open(
        LOCAL_DATA,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            line = line.strip()

            if not line:
                continue

            dataset.append(
                json.loads(line)
            )

    print(
        f"Loaded {len(dataset)} GSM8K questions "
        f"from local cache."
    )

    if limit is None:
        return dataset

    limit = min(
        limit,
        len(dataset),
    )

    rng = random.Random(seed)

    selected_indices = rng.sample(
        range(len(dataset)),
        limit,
    )

    selected = [
        dataset[i]
        for i in selected_indices
    ]

    print(
        f"Random sample: {len(selected)} "
        f"(seed={seed})"
    )

    return selected


# ============================================================
# Gold 答案
# ============================================================

def extract_gold_answer(text):
    """
    从 GSM8K 官方答案中提取 #### 后面的最终数字。
    """

    if not text:
        return None

    match = re.search(
        r"####\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)",
        text,
    )

    if match:
        return normalize_answer(
            match.group(1)
        )

    return None


# ============================================================
# 答案标准化
# ============================================================

def normalize_answer(answer):
    """
    标准化数字答案。

    例如：

    1,000   -> 1000
    1000.0  -> 1000
    18      -> 18
    """

    if answer is None:
        return None

    answer = str(answer).strip()

    answer = answer.replace(",", "")
    answer = answer.replace("$", "")

    # 去掉常见单位
    answer = re.sub(
        r"(?:dollars?|eggs?|miles?|hours?|hrs?|people|days?)$",
        "",
        answer,
        flags=re.IGNORECASE,
    )

    answer = answer.strip()

    try:
        value = float(answer)

        if value.is_integer():
            return str(int(value))

        return str(value)

    except ValueError:
        return answer


# ============================================================
# 模型答案提取
# ============================================================

def extract_answer(text):
    """
    从模型输出中提取最终答案。

    优先级：

    1. FINAL: 18
    2. FINAL ANSWER: 18
    3. #### 18
    4. Final answer: 18
    5. The answer is 18
    6. 计算结果 = 5 hours
    7. 最后一个数字兜底

    核心目的：

    防止这种情况：

        Return time = 30 / 6 = 5 hours.
        Leaving at 4 PM he would get back at 9 PM.

    被错误识别成 9。
    """

    if not text:
        return None

    text = str(text).strip()

    # --------------------------------------------------------
    # 1. FINAL: 18
    # --------------------------------------------------------

    patterns = [

        r"(?im)^\s*FINAL\s*:\s*"
        r"([-+]?\d+(?:,\d{3})*(?:\.\d+)?)"
        r"\s*$",

        r"(?im)^\s*FINAL\s+ANSWER\s*:\s*"
        r"([-+]?\d+(?:,\d{3})*(?:\.\d+)?)"
        r"\s*$",

        r"(?im)^\s*FINAL\s*=\s*"
        r"([-+]?\d+(?:,\d{3})*(?:\.\d+)?)"
        r"\s*$",
    ]

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text,
        )

        if matches:
            return normalize_answer(
                matches[-1]
            )

    # --------------------------------------------------------
    # 2. GSM8K 官方格式 #### 18
    # --------------------------------------------------------

    matches = re.findall(
        r"####\s*"
        r"([-+]?\d+(?:,\d{3})*(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )

    if matches:
        return normalize_answer(
            matches[-1]
        )

    # --------------------------------------------------------
    # 3. Final answer: 18
    # --------------------------------------------------------

    patterns = [

        r"(?i)final\s+answer\s*"
        r"(?:is|:|=)?\s*"
        r"\$?\s*"
        r"([-+]?\d+(?:,\d{3})*(?:\.\d+)?)",

        r"(?i)answer\s*"
        r"(?:is|:|=)\s*"
        r"\$?\s*"
        r"([-+]?\d+(?:,\d{3})*(?:\.\d+)?)",

        r"(?i)the\s+answer\s+is\s*"
        r"\$?\s*"
        r"([-+]?\d+(?:,\d{3})*(?:\.\d+)?)",
    ]

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text,
        )

        if matches:
            return normalize_answer(
                matches[-1]
            )

    # --------------------------------------------------------
    # 4. 中文最终答案
    # --------------------------------------------------------

    patterns = [

        r"(?:最终答案|答案|结果)"
        r"\s*(?:是|为|：|:|=)\s*"
        r"\$?\s*"
        r"([-+]?\d+(?:,\d{3})*(?:\.\d+)?)",
    ]

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text,
        )

        if matches:
            return normalize_answer(
                matches[-1]
            )

    # --------------------------------------------------------
    # 5. 优先寻找带计算结果的表达式
    #
    # 例如：
    #
    # Return time = 30 / 6 = 5 hours
    #
    # 应该提取 5，而不是后面的 9 PM。
    # --------------------------------------------------------

    patterns = [

        r"=\s*"
        r"([-+]?\d+(?:,\d{3})*(?:\.\d+)?)"
        r"\s*(?:hours?|hrs?|h)\b",

        r"=\s*"
        r"\$?\s*"
        r"([-+]?\d+(?:,\d{3})*(?:\.\d+)?)"
        r"\s*(?:dollars?|USD)\b",

        r"(?:equals?|equal\s+to|comes\s+to)"
        r"\s*"
        r"\$?\s*"
        r"([-+]?\d+(?:,\d{3})*(?:\.\d+)?)",
    ]

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if matches:
            return normalize_answer(
                matches[-1]
            )

    # --------------------------------------------------------
    # 6. 最后数字兜底
    # --------------------------------------------------------

    numbers = re.findall(
        r"(?<![\w.])"
        r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?"
        r"(?![\w.])",
        text,
    )

    if numbers:
        return normalize_answer(
            numbers[-1]
        )

    return None


# ============================================================
# Prompt
# ============================================================

def build_prompt(question):
    """
    GSM8K 专用 Prompt。

    允许模型完整推理，但要求最终答案
    单独使用 FINAL: <number> 输出。
    """

    return f"""
Solve the following math word problem.

Think through the problem carefully and calculate the answer step by step.

IMPORTANT:
1. Make sure the final number directly answers the question.
2. Do not confuse intermediate numbers, times, years, quantities, or other numbers in the explanation with the final answer.
3. After completing your reasoning, output exactly one final line:
FINAL: <number>
4. The FINAL line must contain only the numerical answer.
5. Do not put units, explanations, or any other text on the FINAL line.
6. Do not write anything after the FINAL line.

Question:
{question}

Let's solve it step by step.
""".strip()


# ============================================================
# 单题运行
# ============================================================

def run_one(
    api,
    item,
    index,
    total,
):
    question = item["question"]

    gold = extract_gold_answer(
        item["answer"]
    )

    prompt = build_prompt(
        question
    )

    print()
    print("=" * 70)
    print(
        f"GSM8K {index + 1}/{total}"
    )
    print("=" * 70)

    print(question)

    result = api.generate(
        prompt
    )

    content = result.get(
        "content",
        "",
    )

    predicted = extract_answer(
        content
    )

    is_correct = (
        predicted == gold
    )

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
                result.get(
                    "draft_acceptance_rate",
                    0.0,
                ) * 100
            )
        )

    return {
        "index": index,

        "question": question,

        "gold": gold,

        "prediction": predicted,

        "correct": is_correct,

        "invalid": predicted is None,

        "latency": result.get(
            "latency",
            0.0,
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
            0.0,
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
            0.0,
        ),
    }


# ============================================================
# Benchmark
# ============================================================

def run(
    api,
    limit=5,
    seed=42,
    output_path="results/gsm8k/results.json",
):
    """
    运行 GSM8K。

    默认使用固定 seed 随机抽题，
    保证不同模型使用完全相同的题目。
    """

    dataset = load_gsm8k(
        limit=limit,
        seed=seed,
    )

    results = []

    correct = 0
    invalid = 0

    total = len(dataset)

    for i, item in enumerate(dataset):

        result = run_one(
            api,
            item,
            i,
            total,
        )

        results.append(
            result
        )

        if result["correct"]:
            correct += 1

        if result["invalid"]:
            invalid += 1

    accuracy = (
        correct / len(results)
        if results
        else 0.0
    )

    # ========================================================
    # 保存结果
    # ========================================================

    output = {
        "benchmark": "gsm8k",

        "model": api.model,

        "samples": len(results),

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
    print("GSM8K SUMMARY")
    print("=" * 70)

    print(
        "Samples :",
        len(results),
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
