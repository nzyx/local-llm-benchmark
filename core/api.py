import re
import time
from urllib.parse import urlparse, urlunparse

import requests


DEFAULT_MODEL = "muse-glimmer-30B"


def _pretty_model_name(raw):
    """把模型标识转成友好名：
       /path/to/Qwen3.5-27B-Q4_K_M.gguf -> Qwen3.5-27B-Q4_K_M
       已设置的 alias 原样返回。
    """
    if not raw:
        return DEFAULT_MODEL

    name = str(raw).strip()

    if "/" in name or "\\" in name:
        name = name.replace("\\", "/").rsplit("/", 1)[-1]

    if name.lower().endswith(".gguf"):
        name = name[: -len(".gguf")]

    return name


class MuseGlimmerAPI:

    def __init__(
        self,
        model=None,
        base_url="http://127.0.0.1:8080/v1/chat/completions",
        temperature=0.0,
        max_tokens=4096,
        timeout=600,
    ):
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        self.session = requests.Session()

        if model:
            self.model = model
            self.model_raw = model
        else:
            self.model_raw = self._fetch_raw_model_id()
            self.model = _pretty_model_name(self.model_raw)

        self.model_safe = re.sub(
            r"[^a-zA-Z0-9_.-]+",
            "_",
            self.model,
        )

        if not model:
            print(
                f"[API] Auto-detected model: {self.model}"
            )

    def _fetch_raw_model_id(self):
        """从 OpenAI 兼容端点抓取当前加载的模型标识（原始 id）。

        请求 GET {host}/v1/models，取 data[0].id。
        失败时返回 None（调用方回退默认名）。
        """
        parsed = urlparse(self.base_url)

        path = parsed.path
        if path.endswith("/chat/completions"):
            path = path[: -len("/chat/completions")] + "/models"
        else:
            path = "/v1/models"

        models_url = urlunparse(
            (
                parsed.scheme or "http",
                parsed.netloc,
                path,
                "",
                "",
                "",
            )
        )

        try:
            resp = self.session.get(
                models_url,
                timeout=5,
            )
            resp.raise_for_status()

            models = resp.json().get("data", [])

            if models and models[0].get("id"):
                return models[0]["id"]

        except Exception as e:
            print(
                f"[API] WARNING: model auto-detection failed "
                f"({e}), fallback to {DEFAULT_MODEL}"
            )

        return None

    def generate(self, prompt):

        payload = {
            "model": self.model_raw or self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        start = time.perf_counter()

        response = self.session.post(
            self.base_url,
            json=payload,
            timeout=self.timeout,
        )

        latency = time.perf_counter() - start

        response.raise_for_status()

        data = response.json()

        message = data["choices"][0]["message"]

        content = message.get("content", "") or ""

        reasoning_content = message.get(
            "reasoning_content",
            ""
        ) or ""

        # 思考模型（如 Qwen3.5-A3B）可能把全部输出放进 reasoning_content，
        # content 为空（思考过长被 finish_reason=length 截断）。
        # 此时用 reasoning_content 兜底，保证提取器有内容可用。
        if not content.strip() and reasoning_content.strip():
            content = reasoning_content

        usage = data.get("usage", {})
        timings = data.get("timings", {})

        completion_tokens = usage.get(
            "completion_tokens",
            0
        )

        prompt_tokens = usage.get(
            "prompt_tokens",
            0
        )

        total_tokens = usage.get(
            "total_tokens",
            0
        )

        generation_tps = timings.get(
            "predicted_per_second"
        )

        draft_tokens = timings.get(
            "draft_n"
        )

        draft_accepted = timings.get(
            "draft_n_accepted"
        )

        acceptance_rate = None

        if draft_tokens:
            acceptance_rate = (
                draft_accepted / draft_tokens
            )

        return {
            "content": content,
            "reasoning_content": reasoning_content,

            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,

            "generation_tps": generation_tps,

            "draft_tokens": draft_tokens,
            "draft_accepted": draft_accepted,
            "draft_acceptance_rate": acceptance_rate,

            "latency": latency,

            "raw": data,
        }
