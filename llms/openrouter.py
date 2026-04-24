"""
OpenRouter API client compatible with CriSPO LargeLanguageModel interface.
Uses OpenAI SDK with custom base_url for OpenRouter.
"""

import asyncio
import logging
import os
import random
import time
from typing import Any, Dict, List, Tuple, Union

from tqdm.asyncio import tqdm_asyncio
from crispo.llms import LargeLanguageModel, TYPE_PROMPT

logger = logging.getLogger(__name__)


def _to_messages(prompt: TYPE_PROMPT) -> List[dict]:
    """Convert prompt to OpenAI chat format."""
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return prompt


class OpenRouterLLM(LargeLanguageModel):
    """OpenRouter LLM client with retry logic."""

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        api_key: str = None,
        base_url: str = "https://openrouter.ai/api/v1",
        max_new_tokens: int = 80000,
        temperature: float = 0.7,
        top_p: float = 0.0,
        top_k: int = 1,
        concurrency: int = 8,
        stop_sequences: Tuple[str, ...] = (),
        max_retries: int = 5,
        retry_delay: float = 2.0,
        retry_jitter: float = 0.5,
        request_timeout: float = 180.0,
        app_name: str = "MoRe-AST",
        http_referer: str = "https://cursor.local",
    ):
        super().__init__(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            concurrency=concurrency,
            stop_sequences=stop_sequences,
        )
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_jitter = retry_jitter
        self.request_timeout = request_timeout
        self.app_name = app_name
        self.http_referer = http_referer
        self._client = None
        self._batch_loop: asyncio.AbstractEventLoop | None = None
        self._stats: Dict[str, int] = {
            "calls_total": 0,
            "calls_success": 0,
            "calls_empty": 0,
            "calls_retried": 0,
            "calls_exhausted": 0,
            "calls_failed": 0,
        }

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key or "dummy",
                timeout=self.request_timeout,
                default_headers={
                    "HTTP-Referer": self.http_referer,
                    "X-Title": self.app_name,
                },
            )
        return self._client

    def stats(self) -> Dict[str, Any]:
        return {"model": self.model, **self._stats}

    def config_snapshot(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "concurrency": self.concurrency,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "retry_jitter": self.retry_jitter,
            "request_timeout": self.request_timeout,
            "stop_sequences": list(self.stop_sequences),
        }

    def load_stats(self, stats: Dict[str, Any]) -> None:
        for key in self._stats:
            if key in stats:
                self._stats[key] = int(stats[key])

    def log_stats(self, label: str = "") -> None:
        prefix = f"[{label}] " if label else ""
        logger.info("%sLLM stats: %s", prefix, self.stats())

    def _retry_sleep(self, attempt: int) -> None:
        delay = self.retry_delay * (2 ** attempt)
        jitter = random.uniform(0.0, self.retry_jitter)
        time.sleep(delay + jitter)

    def _is_retriable_error(self, exc: Exception) -> bool:
        name = exc.__class__.__name__.lower()
        text = str(exc).lower()
        retriable_tokens = [
            "rate",
            "429",
            "timeout",
            "timed out",
            "connection",
            "temporarily unavailable",
            "502",
            "503",
            "504",
            "529",
            "server error",
        ]
        return any(token in text for token in retriable_tokens) or any(
            token in name for token in ["timeout", "rate", "connection"]
        )

    def generate(self, prompt: TYPE_PROMPT) -> str:
        if not prompt:
            return ""
        messages = _to_messages(prompt)
        self._stats["calls_total"] += 1
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    stop=list(self.stop_sequences) if self.stop_sequences else None,
                )
                choice = response.choices[0]
                content = choice.message.content or ""
                finish_reason = getattr(choice, "finish_reason", None)
                if finish_reason == "length":
                    logger.warning(
                        "OpenRouter response for %s hit finish_reason=length (max_tokens=%s).",
                        self.model,
                        self.max_new_tokens,
                    )
                if not content.strip():
                    self._stats["calls_empty"] += 1
                    logger.warning(
                        "OpenRouter returned empty content for %s (finish_reason=%s).",
                        self.model,
                        finish_reason,
                    )
                self._stats["calls_success"] += 1
                return content
            except Exception as e:
                if self._is_retriable_error(e) and attempt < self.max_retries - 1:
                    self._stats["calls_retried"] += 1
                    logger.warning(
                        "Retrying OpenRouter call for %s after error (%s/%s): %s",
                        self.model,
                        attempt + 1,
                        self.max_retries,
                        e,
                    )
                    self._retry_sleep(attempt)
                    continue
                if self._is_retriable_error(e):
                    self._stats["calls_exhausted"] += 1
                    logger.warning("Exhausted retries for %s: %s", self.model, e)
                    return ""
                self._stats["calls_failed"] += 1
                logger.exception("Unrecoverable OpenRouter error for %s", self.model)
                return ""
        return ""

    def _get_batch_loop(self) -> asyncio.AbstractEventLoop:
        if self._batch_loop is None or self._batch_loop.is_closed():
            self._batch_loop = asyncio.new_event_loop()
            self.lock = asyncio.Semaphore(self.concurrency)
        return self._batch_loop

    def batch_generate(self, prompts: List[TYPE_PROMPT], desc: str = "Generating") -> List[str]:
        async def fire(tasks: List[asyncio.Future]) -> List[str]:
            return await tqdm_asyncio.gather(*tasks, desc=desc, disable=not desc)

        loop = self._get_batch_loop()
        tasks = [self.generate_async(prompt) for prompt in prompts]
        return loop.run_until_complete(fire(tasks))
