"""LLM judges for the deepeval metrics.

EVAL_JUDGE=nvidia (default when NVIDIA_API_KEY is set): an NVIDIA Nemotron model through NVIDIA's
OpenAI-compatible API. EVAL_JUDGE=ollama: a local Ollama model (slow on a small GPU).
"""

import json
import os
import re

from deepeval.models import DeepEvalBaseLLM, OllamaModel
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from analyst.agent import OLLAMA_HOST

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_JUDGE_MODEL = os.getenv("EVAL_NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
OLLAMA_JUDGE_MODEL = os.getenv("EVAL_OLLAMA_MODEL", "mistral")


class NemotronJudge(DeepEvalBaseLLM):
    """deepeval asks for output matching a pydantic schema; we ask for JSON and validate it."""

    def __init__(self, model: str = NVIDIA_JUDGE_MODEL):
        key = os.environ["NVIDIA_API_KEY"]
        self.model_name = model
        self.client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=key, timeout=180, max_retries=3)
        self.async_client = AsyncOpenAI(base_url=NVIDIA_BASE_URL, api_key=key, timeout=180, max_retries=3)

    def load_model(self):
        return self.client

    def get_model_name(self) -> str:
        return self.model_name

    def _messages(self, prompt: str, schema: type[BaseModel] | None) -> list[dict]:
        if schema is not None:
            prompt += ("\n\nReturn only a JSON object matching this JSON schema, with no other text:\n"
                       + json.dumps(schema.model_json_schema()))
        return [{"role": "user", "content": prompt}]

    def generate(self, prompt: str, schema: type[BaseModel] | None = None):
        r = self.client.chat.completions.create(model=self.model_name, messages=self._messages(prompt, schema),
                                                temperature=0, max_tokens=4096)
        return _parse(r.choices[0].message.content, schema)

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None):
        r = await self.async_client.chat.completions.create(
            model=self.model_name, messages=self._messages(prompt, schema), temperature=0, max_tokens=4096)
        return _parse(r.choices[0].message.content, schema)


def _parse(text: str, schema: type[BaseModel] | None):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    if schema is None:
        return text
    match = re.search(r"\{.*\}", text, re.DOTALL)  # drop ```json fences or stray prose
    return schema.model_validate_json(match.group(0) if match else text)


def make_judge() -> tuple[DeepEvalBaseLLM, str]:
    """Returns (judge, display name)."""
    kind = os.getenv("EVAL_JUDGE", "nvidia" if os.getenv("NVIDIA_API_KEY") else "ollama")
    if kind == "nvidia":
        return NemotronJudge(), NVIDIA_JUDGE_MODEL
    return OllamaModel(model=OLLAMA_JUDGE_MODEL, base_url=OLLAMA_HOST, temperature=0), OLLAMA_JUDGE_MODEL
