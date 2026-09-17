"""In-process token-bucket rate limiting for the two write paths (P7.4):
`submit_underwriting_decision` over MCP, and `POST /applications/underwrite`.

Hand-rolled rather than adding `slowapi`/`limits` as a dependency -- what's actually
being protected is the OpenAI bill and the database behind these two write paths (each
call makes several LLM calls), not a DDoS, and the MCP stdio transport needs the same
gate a FastAPI-only library wouldn't give us anyway.

Known limitation, written down rather than hidden: this state is per-process. Running
more than one uvicorn worker gives each its own bucket -- correct for this project's
single-process demo scope, not for a real multi-worker deployment (that needs shared
state, e.g. Redis).
"""

from __future__ import annotations

import threading
import time


class RateLimitExceeded(Exception):
    """Raised by `TokenBucketLimiter.check` when a caller has no tokens left.
    `retry_after_seconds` is how long until one more token is available."""

    def __init__(self, retry_after_seconds: float):
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"Rate limit exceeded; retry after {retry_after_seconds:.1f}s")


class TokenBucketLimiter:
    """One bucket per key (caller identity or client IP). Refills continuously at
    `rate_per_minute` tokens/minute up to `burst` capacity."""

    def __init__(self, rate_per_minute: int, burst: int):
        self._rate_per_second = rate_per_minute / 60.0
        self._capacity = float(burst)
        self._lock = threading.Lock()
        self._tokens: dict[str, float] = {}
        self._last_check: dict[str, float] = {}

    def check(self, key: str) -> None:
        """Consume one token for `key`. Raises RateLimitExceeded if none are left."""
        now = time.monotonic()
        with self._lock:
            last = self._last_check.get(key, now)
            tokens = self._tokens.get(key, self._capacity)
            tokens = min(self._capacity, tokens + (now - last) * self._rate_per_second)

            if tokens < 1.0:
                self._tokens[key] = tokens
                self._last_check[key] = now
                deficit = 1.0 - tokens
                raise RateLimitExceeded(retry_after_seconds=deficit / self._rate_per_second)

            self._tokens[key] = tokens - 1.0
            self._last_check[key] = now
