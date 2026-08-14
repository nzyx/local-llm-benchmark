import csv
import json
import html
from pathlib import Path
from datetime import datetime


# ============================================================
# 配置
# ============================================================

RESULTS_DIR = Path("results")

BENCHMARKS = [
    ("gsm8k", "GSM8K", "数学推理"),
    ("mmlu", "MMLU", "综合知识"),
    ("truthfulqa", "TruthfulQA", "真实性 / 幻觉控制"),
    ("ceval", "C-Eval", "中文综合知识"),
    ("arc", "ARC", "科学推理"),
]

REPORT_MD = RESULTS_DIR / "benchmark_report.md"
REPORT_CSV = RESULTS_DIR / "benchmark_summary.csv"
REPORT_HTML = RESULTS_DIR / "benchmark_report.html"


# ============================================================
# 基础工具
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def pct(value):
    return f"{value * 100:.2f}%"


def fmt(value, digits=2):
    return f"{value:.{digits}f}"


def esc(value):
    return html.escape(str(value))


# ============================================================
# Benchmark 数据
# ============================================================

def get_benchmark_data(name, model_dir=None):
    # 新结构：results/<model>/<bench>/results.json
    # 旧结构：results/<bench>/results.json（model_dir 为 None）
    if model_dir:
        path = RESULTS_DIR / model_dir / name / "results.json"
    else:
        path = RESULTS_DIR / name / "results.json"

    if not path.exists():
        return None

    try:
        return load_json(path)

    except Exception as e:
        print(
            f"WARNING: failed to load {path}: {e}"
        )
        return None


def discover_models():
    """扫描 results/ 下的评测目标。

    返回 (legacy_ok, models)：
      legacy_ok: 旧结构 results/<bench>/results.json 是否有数据
      models:    模型目录名列表（新结构 results/<model>/）
    """
    models = []
    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        if any(
            (d / b / "results.json").exists()
            for b, _, _ in BENCHMARKS
        ):
            models.append(d.name)

    legacy_ok = any(
        (RESULTS_DIR / b / "results.json").exists()
        for b, _, _ in BENCHMARKS
    )
    return legacy_ok, models


# ============================================================
# 单个 Benchmark 统计
# ============================================================

def summarize_benchmark(data, key):

    results = data.get("results", [])

    samples = len(results)

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    correct = sum(
        1
        for item in results
        if item.get("correct") is True
    )

    invalid = sum(
        1
        for item in results
        if item.get("invalid") is True
    )

    accuracy = (
        correct / samples
        if samples > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Token
    # --------------------------------------------------------

    prompt_tokens = sum(
        safe_int(
            item.get("prompt_tokens")
        )
        for item in results
    )

    completion_tokens = sum(
        safe_int(
            item.get("completion_tokens")
        )
        for item in results
    )

    total_tokens = sum(
        safe_int(
            item.get("total_tokens")
        )
        for item in results
    )

    # --------------------------------------------------------
    # Latency
    # --------------------------------------------------------

    total_latency = sum(
        safe_float(
            item.get("latency")
        )
        for item in results
    )

    avg_latency = (
        total_latency / samples
        if samples > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Generation TPS
    #
    # 加权计算：
    #
    # total completion tokens
    # -----------------------
    # total latency
    #
    # 比简单平均每题 tok/s 更合理
    # --------------------------------------------------------

    generation_tps = (
        completion_tokens / total_latency
        if total_latency > 0
        else 0.0
    )

    # --------------------------------------------------------
    # DFlash
    #
    # accepted / draft
    # --------------------------------------------------------

    draft_tokens = sum(
        safe_int(
            item.get("draft_tokens")
        )
        for item in results
    )

    draft_accepted = sum(
        safe_int(
            item.get("draft_accepted")
        )
        for item in results
    )

    dflash_acceptance = (
        draft_accepted / draft_tokens
        if draft_tokens > 0
        else 0.0
    )

    # --------------------------------------------------------
    # 平均 Token
    # --------------------------------------------------------

    avg_prompt_tokens = (
        prompt_tokens / samples
        if samples > 0
        else 0.0
    )

    avg_completion_tokens = (
        completion_tokens / samples
        if samples > 0
        else 0.0
    )

    # --------------------------------------------------------
    # 元信息
    # --------------------------------------------------------

    name_map = {
        item[0]: item[1]
        for item in BENCHMARKS
    }

    description_map = {
        item[0]: item[2]
        for item in BENCHMARKS
    }

    return {
        "key": key,

        "name": name_map.get(
            key,
            key.upper(),
        ),

        "description": description_map.get(
            key,
            "",
        ),

        "samples": samples,

        "correct": correct,

        "invalid": invalid,

        "accuracy": accuracy,

        "prompt_tokens": prompt_tokens,

        "completion_tokens": completion_tokens,

        "total_tokens": total_tokens,

        "avg_prompt_tokens": avg_prompt_tokens,

        "avg_completion_tokens": avg_completion_tokens,

        "total_latency": total_latency,

        "avg_latency": avg_latency,

        "generation_tps": generation_tps,

        "draft_tokens": draft_tokens,

        "draft_accepted": draft_accepted,

        "dflash_acceptance": dflash_acceptance,

        "results": results,
    }


# ============================================================
# 总体报告数据
# ============================================================

def build_report_data(model_dir=None):

    summaries = []

    model = "Unknown"

    seed = None

    # --------------------------------------------------------
    # 读取所有 Benchmark
    # --------------------------------------------------------

    for key, name, description in BENCHMARKS:

        data = get_benchmark_data(
            key,
            model_dir,
        )

        if data is None:
            print(
                f"Skipping {name}: "
                f"results not found."
            )
            continue

        if model == "Unknown":

            model = data.get(
                "model",
                "Unknown",
            )

        if seed is None:

            seed = data.get(
                "seed"
            )

        summary = summarize_benchmark(
            data,
            key,
        )

        summaries.append(summary)

    if not summaries:

        raise RuntimeError(
            "No benchmark result files found."
        )

    # --------------------------------------------------------
    # 总体 Accuracy
    #
    # Micro Average
    # --------------------------------------------------------

    total_samples = sum(
        x["samples"]
        for x in summaries
    )

    total_correct = sum(
        x["correct"]
        for x in summaries
    )

    total_invalid = sum(
        x["invalid"]
        for x in summaries
    )

    overall_accuracy = (
        total_correct / total_samples
        if total_samples > 0
        else 0.0
    )

    # --------------------------------------------------------
    # 平均 Benchmark Accuracy
    #
    # Macro Average
    # --------------------------------------------------------

    macro_accuracy = (
        sum(
            x["accuracy"]
            for x in summaries
        )
        / len(summaries)
    )

    # --------------------------------------------------------
    # 总 Token
    # --------------------------------------------------------

    total_prompt_tokens = sum(
        x["prompt_tokens"]
        for x in summaries
    )

    total_completion_tokens = sum(
        x["completion_tokens"]
        for x in summaries
    )

    total_tokens = sum(
        x["total_tokens"]
        for x in summaries
    )

    total_latency = sum(
        x["total_latency"]
        for x in summaries
    )

    # --------------------------------------------------------
    # 整体 Generation TPS
    # --------------------------------------------------------

    overall_generation_tps = (
        total_completion_tokens
        / total_latency
        if total_latency > 0
        else 0.0
    )

    # --------------------------------------------------------
    # 整体 DFlash
    # --------------------------------------------------------

    total_draft_tokens = sum(
        x["draft_tokens"]
        for x in summaries
    )

    total_draft_accepted = sum(
        x["draft_accepted"]
        for x in summaries
    )

    overall_dflash = (
        total_draft_accepted
        / total_draft_tokens
        if total_draft_tokens > 0
        else 0.0
    )

    # --------------------------------------------------------
    # 时间
    # --------------------------------------------------------

    generated_at = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    return {
        "model": model,

        "seed": seed,

        "generated_at": generated_at,

        "benchmarks": summaries,

        "total_samples": total_samples,

        "total_correct": total_correct,

        "total_invalid": total_invalid,

        "overall_accuracy": overall_accuracy,

        "macro_accuracy": macro_accuracy,

        "total_prompt_tokens": total_prompt_tokens,

        "total_completion_tokens": (
            total_completion_tokens
        ),

        "total_tokens": total_tokens,

        "total_latency": total_latency,

        "overall_generation_tps": (
            overall_generation_tps
        ),

        "total_draft_tokens": (
            total_draft_tokens
        ),

        "total_draft_accepted": (
            total_draft_accepted
        ),

        "overall_dflash": overall_dflash,
    }


# ============================================================
# Markdown 报告
# ============================================================

def generate_markdown(report):

    lines = []

    lines.append(
        f"# {report['model']} Benchmark Report"
    )

    lines.append("")

    lines.append(
        f"> Generated: {report['generated_at']}"
    )

    lines.append(
        f"> Model: **{report['model']}**"
    )

    if report["seed"] is not None:

        lines.append(
            f"> Random Seed: **{report['seed']}**"
        )

    lines.append("")

    lines.append("## Overall")

    lines.append("")

    lines.append(
        "| Metric | Result |"
    )

    lines.append(
        "|---|---:|"
    )

    lines.append(
        f"| Benchmark | "
        f"{len(report['benchmarks'])} |"
    )

    lines.append(
        f"| Samples | "
        f"{report['total_samples']} |"
    )

    lines.append(
        f"| Correct | "
        f"{report['total_correct']} |"
    )

    lines.append(
        f"| Invalid | "
        f"{report['total_invalid']} |"
    )

    lines.append(
        f"| Micro Accuracy | "
        f"{pct(report['overall_accuracy'])} |"
    )

    lines.append(
        f"| Macro Accuracy | "
        f"{pct(report['macro_accuracy'])} |"
    )

    lines.append(
        f"| Generation Speed | "
        f"{fmt(report['overall_generation_tps'])} tok/s |"
    )

    lines.append(
        f"| DFlash Acceptance | "
        f"{pct(report['overall_dflash'])} |"
    )

    lines.append("")

    lines.append("## Benchmark Results")

    lines.append("")

    lines.append(
        "| Benchmark | Ability | Samples | "
        "Correct | Accuracy | Avg Latency | "
        "Generation | DFlash |"
    )

    lines.append(
        "|---|---|---:|---:|---:|---:|---:|---:|"
    )

    for item in report["benchmarks"]:

        lines.append(
            f"| {item['name']} "
            f"| {item['description']} "
            f"| {item['samples']} "
            f"| {item['correct']} "
            f"| {pct(item['accuracy'])} "
            f"| {fmt(item['avg_latency'])} s "
            f"| {fmt(item['generation_tps'])} tok/s "
            f"| {pct(item['dflash_acceptance'])} |"
        )

    lines.append("")

    lines.append("## Token Statistics")

    lines.append("")

    lines.append(
        "| Benchmark | Prompt Tokens | "
        "Completion Tokens | Total Tokens |"
    )

    lines.append(
        "|---|---:|---:|---:|"
    )

    for item in report["benchmarks"]:

        lines.append(
            f"| {item['name']} "
            f"| {item['prompt_tokens']:,} "
            f"| {item['completion_tokens']:,} "
            f"| {item['total_tokens']:,} |"
        )

    lines.append("")

    lines.append("## Notes")

    lines.append("")

    lines.append(
        "- Accuracy uses the benchmark result files directly."
    )

    lines.append(
        "- Overall Accuracy is calculated as micro-average."
    )

    lines.append(
        "- Macro Accuracy is the arithmetic mean "
        "of benchmark accuracies."
    )

    lines.append(
        "- Generation speed uses total completion "
        "tokens divided by total generation latency."
    )

    lines.append(
        "- DFlash acceptance uses total accepted draft "
        "tokens divided by total draft tokens."
    )

    lines.append(
        "- HumanEval is intentionally not included yet."
    )

    lines.append("")

    return "\n".join(lines)


# ============================================================
# CSV
# ============================================================

def generate_csv(report, path=None):

    out_path = path or REPORT_CSV

    fieldnames = [
        "benchmark",
        "ability",
        "samples",
        "correct",
        "invalid",
        "accuracy",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "avg_prompt_tokens",
        "avg_completion_tokens",
        "total_latency",
        "avg_latency",
        "generation_tps",
        "draft_tokens",
        "draft_accepted",
        "dflash_acceptance",
    ]

    with open(
        out_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for item in report["benchmarks"]:

            writer.writerow(
                {
                    "benchmark": item["name"],

                    "ability": item["description"],

                    "samples": item["samples"],

                    "correct": item["correct"],

                    "invalid": item["invalid"],

                    "accuracy": (
                        item["accuracy"]
                    ),

                    "prompt_tokens": (
                        item["prompt_tokens"]
                    ),

                    "completion_tokens": (
                        item["completion_tokens"]
                    ),

                    "total_tokens": (
                        item["total_tokens"]
                    ),

                    "avg_prompt_tokens": (
                        item["avg_prompt_tokens"]
                    ),

                    "avg_completion_tokens": (
                        item["avg_completion_tokens"]
                    ),

                    "total_latency": (
                        item["total_latency"]
                    ),

                    "avg_latency": (
                        item["avg_latency"]
                    ),

                    "generation_tps": (
                        item["generation_tps"]
                    ),

                    "draft_tokens": (
                        item["draft_tokens"]
                    ),

                    "draft_accepted": (
                        item["draft_accepted"]
                    ),

                    "dflash_acceptance": (
                        item["dflash_acceptance"]
                    ),
                }
            )


# ============================================================
# HTML
# ============================================================

def accuracy_class(value):

    if value >= 0.90:
        return "excellent"

    if value >= 0.75:
        return "good"

    if value >= 0.60:
        return "medium"

    return "low"


def _charts_html(report):
    """生成图表区（Chart.js）：准确率 / 速度 / DFlash / 双维散点。"""
    import json as _json

    items = [
        {
            "name": b["name"],
            "accuracy": round(b["accuracy"] * 100, 1),
            "tps": round(b["generation_tps"], 1),
            "dflash": round(b["dflash_acceptance"] * 100, 1),
            "samples": b["samples"],
        }
        for b in report["benchmarks"]
    ]
    data_json = _json.dumps(items, ensure_ascii=False)

    section = """
<div class="section">
<h2>Charts</h2>
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px;">

  <div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:16px;">
    <h3 style="margin:0 0 8px;font-size:14px;">Accuracy (%)</h3>
    <div style="position:relative;height:220px;">
      <canvas id="accChart" role="img" aria-label="Accuracy by benchmark"></canvas>
    </div>
  </div>

  <div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:16px;">
    <h3 style="margin:0 0 8px;font-size:14px;">Generation speed (tok/s)</h3>
    <div style="position:relative;height:220px;">
      <canvas id="tpsChart" role="img" aria-label="Generation speed by benchmark"></canvas>
    </div>
  </div>

  <div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:16px;">
    <h3 style="margin:0 0 8px;font-size:14px;">DFlash acceptance (%)</h3>
    <div style="position:relative;height:220px;">
      <canvas id="dfChart" role="img" aria-label="DFlash acceptance by benchmark"></canvas>
    </div>
  </div>

  <div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:16px;">
    <h3 style="margin:0 0 8px;font-size:14px;">Accuracy vs speed (bubble size = DFlash)</h3>
    <div style="position:relative;height:220px;">
      <canvas id="scatterChart" role="img" aria-label="Accuracy vs speed scatter"></canvas>
    </div>
  </div>

</div>
</div>
"""

    script = """
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
const DATA = __DATA_JSON__;
const NAMES = DATA.map(d => d.name);

function mkBar(id, values, label, suffix, color) {
  new Chart(document.getElementById(id), {
    type: 'bar',
    data: {
      labels: NAMES,
      datasets: [{ label: label, data: values, backgroundColor: color, borderRadius: 4 }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        y: { beginAtZero: true, ticks: { callback: v => v + suffix } },
        x: { ticks: { autoSkip: false, maxRotation: 30 } }
      },
      plugins: { legend: { display: false } }
    }
  });
}

mkBar('accChart', DATA.map(d => d.accuracy), 'Accuracy', '%', '#185FA5');
mkBar('tpsChart', DATA.map(d => d.tps), 'Speed', ' tok/s', '#1D9E75');
mkBar('dfChart', DATA.map(d => d.dflash), 'DFlash', '%', '#BA7517');

new Chart(document.getElementById('scatterChart'), {
  type: 'scatter',
  data: { datasets: [{
    data: DATA.map(d => ({ x: d.accuracy, y: d.tps, r: Math.max(5, d.dflash / 6) })),
    backgroundColor: '#378ADD', pointBackgroundColor: '#185FA5'
  }]},
  options: {
    responsive: true, maintainAspectRatio: false,
    scales: {
      x: { title: { display: true, text: 'Accuracy (%)' }, min: 50, max: 105 },
      y: { title: { display: true, text: 'Speed (tok/s)' }, beginAtZero: false }
    },
    plugins: {
      legend: { display: false },
      tooltip: { callbacks: { label: ctx => {
        const d = DATA[ctx.dataIndex];
        return d.name + '  acc ' + d.accuracy + '%  ' + d.tps + ' tok/s  DFlash ' + d.dflash + '%';
      } } }
    }
  }
});
</script>
"""
    script = script.replace("__DATA_JSON__", data_json)
    return section, script


def generate_html(report, path=None):

    out_path = path or REPORT_HTML

    charts_section, charts_script = _charts_html(report)

    benchmark_rows = []

    for item in report["benchmarks"]:

        cls = accuracy_class(
            item["accuracy"]
        )

        benchmark_rows.append(
            f"""
            <tr>
                <td>
                    <strong>{esc(item['name'])}</strong>
                    <div class="sub">
                        {esc(item['description'])}
                    </div>
                </td>

                <td>{item['samples']}</td>

                <td>{item['correct']}</td>

                <td>
                    <span class="score {cls}">
                        {pct(item['accuracy'])}
                    </span>
                </td>

                <td>
                    {fmt(item['avg_latency'])} s
                </td>

                <td>
                    {fmt(item['generation_tps'])} tok/s
                </td>

                <td>
                    {pct(item['dflash_acceptance'])}
                </td>
            </tr>
            """
        )

    cards = []

    for item in report["benchmarks"]:

        cls = accuracy_class(
            item["accuracy"]
        )

        cards.append(
            f"""
            <div class="card">

                <div class="card-title">
                    {esc(item['name'])}
                </div>

                <div class="card-desc">
                    {esc(item['description'])}
                </div>

                <div class="big-score {cls}">
                    {pct(item['accuracy'])}
                </div>

                <div class="card-meta">
                    {item['correct']} / {item['samples']}
                    correct
                </div>

                <div class="mini-grid">

                    <div>
                        <span>Speed</span>
                        <strong>
                            {fmt(item['generation_tps'])}
                        </strong>
                        <small>tok/s</small>
                    </div>

                    <div>
                        <span>Latency</span>
                        <strong>
                            {fmt(item['avg_latency'])}
                        </strong>
                        <small>sec</small>
                    </div>

                    <div>
                        <span>DFlash</span>
                        <strong>
                            {pct(item['dflash_acceptance'])}
                        </strong>
                        <small>accept</small>
                    </div>

                </div>

            </div>
            """
        )

    html_content = f"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>
{esc(report['model'])} Benchmark Report
</title>

<style>

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    background: #f5f6f8;
    color: #1f2937;
    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        Arial,
        sans-serif;
}}

.container {{
    max-width: 1400px;
    margin: 0 auto;
    padding: 40px 28px 60px;
}}

.header {{
    margin-bottom: 28px;
}}

.header h1 {{
    margin: 0 0 10px;
    font-size: 32px;
}}

.header p {{
    margin: 5px 0;
    color: #6b7280;
}}

.overview {{
    display: grid;
    grid-template-columns:
        repeat(5, minmax(0, 1fr));
    gap: 16px;
    margin-bottom: 28px;
}}

.overview-card {{
    background: white;
    border-radius: 14px;
    padding: 22px;
    border: 1px solid #e5e7eb;
}}

.overview-card .label {{
    color: #6b7280;
    font-size: 13px;
    margin-bottom: 8px;
}}

.overview-card .value {{
    font-size: 28px;
    font-weight: 700;
}}

.section {{
    margin-top: 30px;
}}

.section h2 {{
    margin-bottom: 16px;
    font-size: 22px;
}}

.cards {{
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(230px, 1fr));
    gap: 16px;
}}

.card {{
    background: white;
    border-radius: 14px;
    padding: 22px;
    border: 1px solid #e5e7eb;
}}

.card-title {{
    font-size: 20px;
    font-weight: 700;
}}

.card-desc {{
    color: #6b7280;
    font-size: 13px;
    margin-top: 4px;
}}

.big-score {{
    font-size: 38px;
    font-weight: 800;
    margin-top: 22px;
}}

.card-meta {{
    color: #6b7280;
    margin-top: 4px;
}}

.mini-grid {{
    display: grid;
    grid-template-columns:
        repeat(3, 1fr);
    gap: 8px;
    margin-top: 22px;
}}

.mini-grid div {{
    background: #f8fafc;
    padding: 10px;
    border-radius: 8px;
}}

.mini-grid span {{
    display: block;
    color: #6b7280;
    font-size: 11px;
}}

.mini-grid strong {{
    display: block;
    font-size: 15px;
    margin-top: 3px;
}}

.mini-grid small {{
    color: #6b7280;
}}

.score.excellent,
.big-score.excellent {{
    color: #15803d;
}}

.score.good,
.big-score.good {{
    color: #2563eb;
}}

.score.medium,
.big-score.medium {{
    color: #d97706;
}}

.score.low,
.big-score.low {{
    color: #dc2626;
}}

.table-container {{
    background: white;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
    overflow-x: auto;
}}

table {{
    width: 100%;
    border-collapse: collapse;
}}

th {{
    text-align: left;
    background: #f8fafc;
    color: #6b7280;
    font-size: 12px;
    font-weight: 600;
    padding: 14px 16px;
    white-space: nowrap;
}}

td {{
    padding: 16px;
    border-top: 1px solid #eef0f2;
    white-space: nowrap;
}}

.sub {{
    color: #6b7280;
    font-size: 12px;
    margin-top: 3px;
}}

.footer {{
    margin-top: 40px;
    color: #9ca3af;
    font-size: 12px;
}}

@media (max-width: 900px) {{

    .overview {{
        grid-template-columns:
            repeat(2, 1fr);
    }}

}}

@media (max-width: 600px) {{

    .container {{
        padding: 20px 14px;
    }}

    .overview {{
        grid-template-columns:
            1fr;
    }}

}}

</style>

</head>

<body>

<div class="container">

<div class="header">

<h1>
{esc(report['model'])} Benchmark Report
</h1>

<p>
Model:
<strong>{esc(report['model'])}</strong>
</p>

<p>
Generated:
{esc(report['generated_at'])}
</p>

{
    f"<p>Random Seed: <strong>{report['seed']}</strong></p>"
    if report["seed"] is not None
    else ""
}

</div>


<!-- =====================================================
     Overall
     ===================================================== -->

<div class="overview">

<div class="overview-card">

<div class="label">
Overall Accuracy
</div>

<div class="value">
{pct(report['overall_accuracy'])}
</div>

</div>


<div class="overview-card">

<div class="label">
Correct
</div>

<div class="value">
{report['total_correct']}
/
{report['total_samples']}
</div>

</div>


<div class="overview-card">

<div class="label">
Benchmarks
</div>

<div class="value">
{len(report['benchmarks'])}
</div>

</div>


<div class="overview-card">

<div class="label">
Generation Speed
</div>

<div class="value">
{fmt(report['overall_generation_tps'])}
</div>

<div class="label">
tok/s
</div>

</div>


<div class="overview-card">

<div class="label">
DFlash Acceptance
</div>

<div class="value">
{pct(report['overall_dflash'])}
</div>

</div>

</div>


<!-- =====================================================
     Benchmark Cards
     ===================================================== -->

<div class="section">

<h2>
Benchmark Overview
</h2>

<div class="cards">

{''.join(cards)}

</div>

</div>


<!-- =====================================================
     Charts
     ===================================================== -->

{charts_section}


<!-- =====================================================
     Detailed Table
     ===================================================== -->

<div class="section">

<h2>
Detailed Results
</h2>

<div class="table-container">

<table>

<thead>

<tr>

<th>Benchmark</th>

<th>Samples</th>

<th>Correct</th>

<th>Accuracy</th>

<th>Avg Latency</th>

<th>Generation</th>

<th>DFlash</th>

</tr>

</thead>

<tbody>

{''.join(benchmark_rows)}

</tbody>

</table>

</div>

</div>


<!-- =====================================================
     Token Statistics
     ===================================================== -->

<div class="section">

<h2>
Token Statistics
</h2>

<div class="table-container">

<table>

<thead>

<tr>

<th>Benchmark</th>

<th>Prompt Tokens</th>

<th>Completion Tokens</th>

<th>Total Tokens</th>

<th>Avg Prompt</th>

<th>Avg Completion</th>

</tr>

</thead>

<tbody>

{
    ''.join(
        f'''
        <tr>

        <td>
        <strong>{esc(item["name"])}</strong>
        </td>

        <td>
        {item["prompt_tokens"]:,}
        </td>

        <td>
        {item["completion_tokens"]:,}
        </td>

        <td>
        {item["total_tokens"]:,}
        </td>

        <td>
        {fmt(item["avg_prompt_tokens"])}
        </td>

        <td>
        {fmt(item["avg_completion_tokens"])}
        </td>

        </tr>
        '''
        for item in report["benchmarks"]
    )
}

</tbody>

</table>

</div>

</div>


<!-- =====================================================
     Summary
     ===================================================== -->

<div class="section">

<h2>
Summary
</h2>

<div class="table-container">

<table>

<tbody>

<tr>

<td>
Total Prompt Tokens
</td>

<td>
<strong>
{report['total_prompt_tokens']:,}
</strong>
</td>

</tr>

<tr>

<td>
Total Completion Tokens
</td>

<td>
<strong>
{report['total_completion_tokens']:,}
</strong>
</td>

</tr>

<tr>

<td>
Total Tokens
</td>

<td>
<strong>
{report['total_tokens']:,}
</strong>
</td>

</tr>

<tr>

<td>
Total Generation Time
</td>

<td>
<strong>
{fmt(report['total_latency'])} sec
</strong>
</td>

</tr>

<tr>

<td>
Micro Accuracy
</td>

<td>
<strong>
{pct(report['overall_accuracy'])}
</strong>
</td>

</tr>

<tr>

<td>
Macro Accuracy
</td>

<td>
<strong>
{pct(report['macro_accuracy'])}
</strong>
</td>

</tr>

</tbody>

</table>

</div>

</div>


<div class="footer">

{esc(report['model'])} Benchmark Suite ·
GSM8K · MMLU · TruthfulQA · C-Eval · ARC

</div>

</div>

{charts_script}

</body>

</html>
"""

    with open(
        out_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(html_content)


# ============================================================
# 主程序
# ============================================================

def main():

    print()
    print("=" * 70)
    print("Generating Benchmark Report")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # 发现评测目标：旧结构 results/<bench>/ + 各模型目录
    # --------------------------------------------------------

    legacy_ok, models = discover_models()

    targets = []
    if legacy_ok:
        targets.append((None, RESULTS_DIR))
    for m in models:
        targets.append((m, RESULTS_DIR / m))

    if not targets:
        print("No benchmark result files found under results/.")
        return

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for model_dir, out_dir in targets:

        print()
        print("-" * 70)
        print(
            "Report for: "
            f"{model_dir or 'legacy results/'}"
        )
        print("-" * 70)

        report = build_report_data(model_dir)

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        md_path = out_dir / "benchmark_report.md"
        csv_path = out_dir / "benchmark_summary.csv"
        html_path = out_dir / "benchmark_report.html"

        markdown = generate_markdown(report)

        with open(
            md_path,
            "w",
            encoding="utf-8",
        ) as f:
            f.write(markdown)

        generate_csv(
            report,
            path=csv_path,
        )

        generate_html(
            report,
            path=html_path,
        )

        print(
            f"Model    : {report['model']}"
        )
        print(
            f"Overall  : {pct(report['overall_accuracy'])}"
            f"  Macro {pct(report['macro_accuracy'])}"
        )
        print(
            f"Gen      : {fmt(report['overall_generation_tps'])} tok/s"
            f"  DFlash {pct(report['overall_dflash'])}"
        )
        print(f"  MD     : {md_path}")
        print(f"  CSV    : {csv_path}")
        print(f"  HTML   : {html_path}")

    print()
    print("=" * 70)
    print("Report generation finished")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
