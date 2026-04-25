"""Asynchronous LLM client wrapper.

Wraps any synchronous LLMClient (ZaiCodingClient, ClaudeCodeClient, etc.) with
bounded-concurrency asynchronous batching via asyncio. The underlying sync
client is invoked from a thread pool so callers do not need to be async.

Design notes:
  * Persists one event loop per AsyncLLMClient instance (avoids per-call setup
    cost). The loop is created lazily and closed in __del__.
  * Uses asyncio.Semaphore to bound concurrency.
  * Returns results in input order.
  * Tracks cumulative token usage across all calls.
  * Raises RateLimitError on HTTP 429 responses surfaced by the underlying
    client (best-effort string detection since we do not control client
    error types).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import re
from typing import Optional


DEFAULT_CONCURRENCY = 10
MAX_CONCURRENCY = 50


def make_progress_printer(label: str, width: int = 40) -> callable:
    """Return a callback ``fn(completed, total)`` that prints a progress bar.

    Example output::

        Eliciting preferences: [████████████░░░░░░░░]  612/1000  (61.2%)  ~2.1 calls/s
    """
    import time
    _start = time.monotonic()

    def _on_progress(completed: int, total: int):
        elapsed = time.monotonic() - _start
        pct = completed / total if total > 0 else 1.0
        filled = int(width * pct)
        bar = "\u2588" * filled + "\u2591" * (width - filled)
        rate = completed / elapsed if elapsed > 0 else 0.0
        print(
            f"\r    {label}: [{bar}]  {completed}/{total}  "
            f"({pct:.1%})  ~{rate:.1f} calls/s",
            end="", flush=True,
        )
        if completed >= total:
            print(flush=True)  # newline when done

    return _on_progress


class RateLimitError(RuntimeError):
    """Raised when the upstream LLM API returns HTTP 429."""

    def __init__(self, provider: str, concurrency: int, message: str = ""):
        self.provider = provider
        self.concurrency = concurrency
        super().__init__(
            f"Rate limited by provider '{provider}' at concurrency={concurrency}. "
            f"Lower --llm-concurrency or upgrade your plan. Original: {message}"
        )


def _looks_like_rate_limit(exc: BaseException) -> bool:
    """Best-effort detection of a 429 in arbitrary client exceptions."""
    text = str(exc)
    if "429" in text:
        return True
    if "rate limit" in text.lower():
        return True
    if "ratelimit" in text.lower():
        return True
    # Anthropic SDK exposes a `RateLimitError` class
    if exc.__class__.__name__ == "RateLimitError":
        return True
    return False


def _estimate_tokens(text: str) -> int:
    """Cheap token approximation: ~4 characters per token."""
    if not text:
        return 0
    return max(1, len(text) // 4)


class AsyncLLMClient:
    """Async batching wrapper around a synchronous LLM client.

    Each instance owns a dedicated ``ThreadPoolExecutor`` sized to its
    ``concurrency``. This is critical for multi-provider setups where
    concurrency may exceed the asyncio default executor's worker count
    (``min(32, cpu_count + 4)``) — without a sized pool, the thread cap
    silently becomes the real concurrency bound.

    Args:
        client: Any object with a `complete(prompt, system="") -> str` method.
        concurrency: Maximum simultaneous in-flight requests (default 10).
        provider_name: Human-readable provider label used in error messages.
    """

    def __init__(
        self,
        client,
        concurrency: int = DEFAULT_CONCURRENCY,
        provider_name: Optional[str] = None,
    ):
        if concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {concurrency}")
        if concurrency > MAX_CONCURRENCY:
            raise ValueError(
                f"concurrency must be <= {MAX_CONCURRENCY}, got {concurrency}"
            )
        self.client = client
        self.concurrency = concurrency
        self.provider_name = provider_name or client.__class__.__name__
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    # ------------------------------------------------------------------
    # Event loop + executor lifecycle
    # ------------------------------------------------------------------

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def _get_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        """Return a dedicated thread pool sized to match concurrency.

        Sized exactly to ``concurrency`` so the semaphore is the real
        bound, not the thread pool. A shared asyncio default executor
        would cap at ~28 workers, which silently serializes multi-provider
        runs whose nominal concurrency exceeds that.
        """
        if self._executor is None:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.concurrency,
                thread_name_prefix=f"async-llm-{self.provider_name}",
            )
        return self._executor

    def close(self):
        """Explicitly close the event loop + executor. Safe to call multiple times.

        Drains any pending tasks first, then closes. Any errors during
        close are swallowed since this is best-effort cleanup.
        """
        loop = getattr(self, "_loop", None)
        if loop is not None and not loop.is_closed():
            try:
                try:
                    pending = [
                        t for t in asyncio.all_tasks(loop=loop) if not t.done()
                    ]
                    for task in pending:
                        task.cancel()
                    if pending:
                        loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                except Exception:
                    pass
                loop.close()
            except Exception:
                pass
        self._loop = None

        executor = getattr(self, "_executor", None)
        if executor is not None:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
        self._executor = None

    def __del__(self):
        # Best-effort cleanup at GC time. Suppress all errors because the
        # interpreter may already be shutting down (loop torn down, asyncio
        # internals unavailable, etc.).
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def complete_many(
        self,
        prompts: list[str],
        systems: Optional[list[str]] = None,
        on_progress: Optional[callable] = None,
        on_result: Optional[callable] = None,
    ) -> list[str]:
        """Issue N concurrent completions and return results in input order.

        Args:
            prompts: List of user prompts.
            systems: Optional list of system messages, one per prompt. If a
                shorter list is given, it is broadcast (must match length or
                be a single string-equivalent list).
            on_progress: Optional callback ``fn(completed, total)`` called
                each time a request finishes. Useful for progress bars.
            on_result: Optional callback ``fn(idx, response, error)`` called
                per-completion with the input index, response string (or
                empty if error), and exception (or None). Fires before
                ``on_progress``. Lets callers persist results incrementally
                (e.g., write to disk cache as each preference parses).

        Returns:
            List of completion strings, same length and order as `prompts`.
        """
        if not prompts:
            return []
        if systems is None:
            systems = [""] * len(prompts)
        elif len(systems) == 1 and len(prompts) > 1:
            systems = systems * len(prompts)
        elif len(systems) != len(prompts):
            raise ValueError(
                f"systems length ({len(systems)}) must be 0, 1, or match prompts ({len(prompts)})"
            )

        loop = self._get_loop()
        return loop.run_until_complete(
            self._run_batch(prompts, systems, on_progress=on_progress,
                            on_result=on_result)
        )

    async def _run_batch(
        self,
        prompts: list[str],
        systems: list[str],
        on_progress: Optional[callable] = None,
        on_result: Optional[callable] = None,
    ) -> list[str]:
        sem = asyncio.Semaphore(self.concurrency)
        executor = self._get_executor()
        results: list[Optional[str]] = [None] * len(prompts)
        errors: list[Optional[Exception]] = [None] * len(prompts)
        completed_count = 0
        total = len(prompts)

        async def one(idx: int, prompt: str, system: str):
            nonlocal completed_count
            async with sem:
                try:
                    response = await asyncio.get_running_loop().run_in_executor(
                        executor,
                        lambda: self.client.complete(prompt, system=system),
                    )
                except Exception as exc:
                    errors[idx] = exc
                    completed_count += 1
                    if on_result:
                        try: on_result(idx, "", exc)
                        except Exception: pass
                    if on_progress:
                        on_progress(completed_count, total)
                    return

                self.total_input_tokens += _estimate_tokens(prompt) + _estimate_tokens(system)
                self.total_output_tokens += _estimate_tokens(response)
                results[idx] = response
                completed_count += 1
                if on_result:
                    try: on_result(idx, response, None)
                    except Exception: pass
                if on_progress:
                    on_progress(completed_count, total)

        # Materialize coroutines into a list so they're all referenced before
        # gather() runs. This avoids Python 3.9 garbage-collecting partially
        # constructed coroutines during interpreter shutdown.
        # Use return_exceptions=True to prevent one failure from cancelling
        # all other in-flight calls — important for multi-provider setups
        # where individual failures are expected and handled per-slot.
        coros = [one(i, p, s) for i, (p, s) in enumerate(zip(prompts, systems))]
        await asyncio.gather(*coros, return_exceptions=True)

        # If EVERY call failed, raise the first error so the caller knows.
        # Prioritize surfacing a RateLimitError (the common remediation case).
        if all(r is None for r in results):
            first_rate_limit = next(
                (e for e in errors if e is not None and _looks_like_rate_limit(e)),
                None,
            )
            if first_rate_limit is not None:
                raise RateLimitError(
                    self.provider_name, self.concurrency, str(first_rate_limit)
                ) from first_rate_limit
            first_exc = next((e for e in errors if e is not None), None)
            if first_exc is not None:
                raise first_exc

        return [r if r is not None else "" for r in results]
