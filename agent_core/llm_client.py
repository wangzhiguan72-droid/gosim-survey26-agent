"""A tiny OpenAI-compatible chat client, configured through environment variables.
Defaults to the Kimi Coding Plan endpoint, since that is what this example ships with;
any other OpenAI-compatible `/chat/completions` endpoint works too by overriding the
base URL and model:

    OPENAI_BASE_URL   default https://api.kimi.com/coding/v1 (see README for the
                      api.kimi.ai alternative for accounts outside mainland China)
    OPENAI_MODEL      default "k3"
    OPENAI_API_KEY    bearer token (KIMI_API_KEY is also accepted)

See https://www.kimi.com/code/docs/en/ for the Kimi Coding Plan API. This client never
sets or overrides the HTTP User-Agent header -- whatever Python's standard library
sends by default is left alone.

Every call has a timeout (default 12 s) and the whole run has a total time budget
(default 300 s) and a call cap (default 100), so the two LLM-advised planning steps
stay well inside the 900 s wall clock. A call that fails or times out is retried up to
`max_retries` times; if all of those fail, that one planning step falls back to its
rule-based answer for this night -- the next night's calls still go through normally.

Standard library only (urllib) so the example has zero third-party dependencies.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Optional

DEFAULT_BASE_URL = "https://api.kimi.com/coding/v1"
DEFAULT_MODEL = "k3"
# Endpoint self-healing: the platform injects OPENAI_BASE_URL/MODEL, which can
# lag behind the team model actually saved (observed frozen at kimi/k3 while a
# GLM coding-plan key was live). On auth/model errors we retry once against
# this alternate endpoint with the SAME injected key. No keys are baked in.
FALLBACK_ENDPOINTS = [
    ("https://open.bigmodel.cn/api/coding/paas/v4", "glm-5.3-flash"),
    ("https://api.kimi.com/coding/v1", "k3"),
]
_JSON_OBJECT = re.compile(r"\{.*\}", re.S)


class MissingAPIKeyError(RuntimeError):
    pass


def require_api_key() -> None:
    """Raise MissingAPIKeyError if neither OPENAI_API_KEY nor KIMI_API_KEY is set.
    Called once at process startup, before reading anything from stdin."""
    if not (os.environ.get("OPENAI_API_KEY", "").strip() or os.environ.get("KIMI_API_KEY", "").strip()):
        raise MissingAPIKeyError("missing API key: set OPENAI_API_KEY")


class LLMClient:
    def __init__(self, log=lambda text: None, call_timeout_seconds: float = 12.0,
                 total_budget_seconds: float = 300.0, max_calls: int = 100, max_retries: int = 3):
        self.log = log
        self.base_url = os.environ.get("OPENAI_BASE_URL", "").strip().rstrip("/") or DEFAULT_BASE_URL
        self.api_key = os.environ.get("OPENAI_API_KEY", "").strip() or os.environ.get("KIMI_API_KEY", "").strip()
        self.model = os.environ.get("OPENAI_MODEL", "").strip() or DEFAULT_MODEL
        self.call_timeout_seconds = call_timeout_seconds
        self.total_budget_seconds = total_budget_seconds
        self.max_calls = max_calls
        self.max_retries = max_retries
        self.spent_seconds = 0.0
        self.calls_made = 0
        self.quota_dead = False
        self._fallbacks = [(b, m) for b, m in FALLBACK_ENDPOINTS
                           if b != self.base_url]

    def _budget_left(self, wallclock_remaining_seconds: float) -> float:
        # Never let a model call eat into the last minute of wall clock, and never
        # exceed this run's own LLM time allowance.
        return min(self.call_timeout_seconds, self.total_budget_seconds - self.spent_seconds,
                   max(0.0, wallclock_remaining_seconds - 60.0))

    def _attempt(self, system_prompt: str, user_payload: dict, timeout: float) -> dict:
        """One HTTP attempt. Raises on any problem; the caller retries or gives up."""
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
            "temperature": 1,
            "max_tokens": 250,
        }).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + "/chat/completions", data=body, method="POST",
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.api_key},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:200]
            except Exception:  # noqa: BLE001
                pass
            self.log(f"llm: HTTP {exc.code} from {self.base_url}: {detail}")
            raise
        text = data["choices"][0]["message"]["content"] or ""
        match = _JSON_OBJECT.search(text)
        if not match:
            raise ValueError("no JSON object in model reply")
        parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            raise ValueError("model reply was not a JSON object")
        return parsed

    def ask_json(self, system_prompt: str, user_payload: dict, wallclock_remaining_seconds: float) -> Optional[dict]:
        """One planning question, answered as exactly one JSON object. Retries up to
        `max_retries` times on failure; returns None once the budget/call cap/retries
        are exhausted, so the caller's rule-based answer can take over for this step."""
        if self.quota_dead:
            return None
        last_error: Optional[Exception] = None
        for _attempt_number in range(self.max_retries):
            if self.calls_made >= self.max_calls:
                self.log("llm: call cap reached for this run; using the rule-based path")
                return None
            timeout = self._budget_left(wallclock_remaining_seconds)
            if timeout < 1.5:
                self.log("llm: LLM time budget exhausted; using the rule-based path")
                return None
            started = time.monotonic()
            self.calls_made += 1
            try:
                return self._attempt(system_prompt, user_payload, timeout)
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code in (401, 402, 403, 404):
                    # Auth/model mismatch or exhausted quota on THIS endpoint:
                    # try the alternate endpoint with the same injected key
                    # before declaring the run LLM-dead.
                    if self._fallbacks:
                        self.base_url, self.model = self._fallbacks.pop(0)
                        self.log(f"llm: HTTP {exc.code}; switching endpoint to {self.base_url} model {self.model}")
                        continue
                    self.quota_dead = True
                    self.log(f"llm: HTTP {exc.code} (auth/quota) and no endpoint left; LLM off for this run")
                    return None
                if exc.code == 429:
                    self.quota_dead = True
                    self.log("llm: HTTP 429 (rate/quota); LLM off for this run")
                    return None
            except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError, TypeError) as exc:
                last_error = exc
            finally:
                self.spent_seconds += time.monotonic() - started
        self.log(f"llm: call failed after {self.max_retries} attempts ({type(last_error).__name__}); "
                 "this step falls back to its rule-based answer")
        return None
