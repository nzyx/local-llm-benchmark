import json
import random
import re
from pathlib import Path


CHOICES = ["A", "B", "C", "D"]

LOCAL_DATA = Path("data/mmlu_all.json")


def select_random(dataset, limit, seed):

    if limit is None:
        return dataset

    total = len(dataset)

    if limit >= total:
        return dataset

    rng = random.Random(seed)

    indices = list(range(total))
    rng.shuffle(indices)

    selected = indices[:limit]

    return [dataset[i] for i in selected]


def extract_choice(text):

    if not text:
        return None

    text = text.strip()

    patterns = [

        r"(?:final answer|final choice|answer|correct answer)"
        r"\s*(?:is|:)?\s*\**\s*([ABCD])\b",

        r"(?:答案|正确答案|最终答案)"
        r"\s*(?:是|为|：|:)?\s*\**\s*([ABCD])\b",

    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            return (
                match.group(1)
                .upper()
            )

    # 最后寻找独立的 A/B/C/D
    matches = re.findall(
        r"(?<![A-Za-z])"
        r"([ABCD])"
        r"(?![A-Za-z])",
        text.upper()
    )

    if matches:
        return matches[-1]

    return None


def load_cache():

    print("Loading MMLU from local cache...")

    if not LOCAL_DATA.exists():
        raise FileNotFoundError(
            "MMLU local cache not found. "
            "Run: python scripts/prepare_data.py --only mmlu"
        )

    with open(
        LOCAL_DATA,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def load_mmlu(limit=None, seed=42):

    dataset = load_cache()

    return select_random(
        dataset,
        limit,
        seed
    )


def run(
    api,
    limit=50,
    seed=42,
    output_path="results/mmlu/results.json",
):

    dataset = load_cache()

    original_size = len(dataset)

    test_set = select_random(
        dataset,
        limit,
        seed
    )

    print()
    print(
        "Dataset size :",
        original_size
    )

    print(
        "Random seed  :",
        seed
    )

    print(
        "Samples      :",
        len(test_set)
    )

    results = []

    correct = 0
    invalid = 0

    for i, item in enumerate(test_set):

        question = item["question"]

        choices = item["choices"]

        gold_index = int(
            item["answer"]
        )

        gold = CHOICES[
            gold_index
        ]

        choice_text = "\n".join(

            f"{CHOICES[j]}. {choices[j]}"

            for j in range(
                len(choices)
            )
        )

        prompt = f"""Question:
{question}

{choice_text}

Please solve the question step by step.
At the end, give your final answer as one of A, B, C, or D.
Final answer:"""

        print()
        print("=" * 70)

        print(
            f"MMLU {i + 1}/{len(test_set)}"
        )

        print("=" * 70)

        print(question)

        print()

        print(choice_text)

        result = api.generate(
            prompt
        )

        prediction = extract_choice(
            result["content"]
        )

        if prediction is None:

            invalid += 1
            is_correct = False

        else:

            is_correct = (
                prediction == gold
            )

            if is_correct:
                correct += 1

        print()
        print("MODEL:")
        print(result["content"])

        print()
        print(
            "Predicted :",
            prediction
        )

        print(
            "Gold      :",
            gold
        )

        print(
            "Result    :",
            "PASS"
            if is_correct
            else "FAIL"
        )

        if result.get("draft_tokens"):

            print(
                "DFlash    : %.2f%%"
                % (
                    result[
                        "draft_acceptance_rate"
                    ] * 100
                )
            )

        results.append({

            "index": i,

            "question": question,

            "choices": choices,

            "gold": gold,

            "prediction": prediction,

            "correct": is_correct,

            "invalid": prediction is None,

            "latency":
                result["latency"],

            "prompt_tokens":
                result["prompt_tokens"],

            "completion_tokens":
                result["completion_tokens"],

            "total_tokens":
                result["total_tokens"],

            "generation_tps":
                result["generation_tps"],

            "draft_tokens":
                result["draft_tokens"],

            "draft_accepted":
                result["draft_accepted"],

            "draft_acceptance_rate":
                result[
                    "draft_acceptance_rate"
                ],
        })

    samples = len(results)

    accuracy = (
        correct / samples
        if samples
        else 0
    )

    output = {

        "benchmark": "mmlu",

        "model": api.model,

        "seed": seed,

        "requested_samples": limit,

        "samples": samples,

        "correct": correct,

        "invalid": invalid,

        "accuracy": accuracy,

        "results": results,
    }

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print("=" * 70)
    print("MMLU SUMMARY")
    print("=" * 70)

    print(
        "Samples :",
        samples
    )

    print(
        "Correct :",
        correct
    )

    print(
        "Invalid :",
        invalid
    )

    print(
        "Accuracy: %.2f%%"
        % (accuracy * 100)
    )

    print(
        "Seed    :",
        seed
    )

    print()
    print(
        "Results saved to:",
        path
    )

    return output
