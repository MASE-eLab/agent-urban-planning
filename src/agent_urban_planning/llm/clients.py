"""LLM-based decision engine and preference elicitation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Protocol

from agent_urban_planning.core.agents import Agent, PreferenceWeights
from agent_urban_planning.decisions.base import ZoneChoice
from agent_urban_planning.decisions._legacy_singapore_utility import UtilityEngine
from agent_urban_planning.core.environment import Environment


class LLMClient(Protocol):
    """Interface for LLM API clients."""

    def complete(self, prompt: str, system: str = "") -> str:
        """Send a prompt to the LLM and return the response text."""
        ...


class LLMCallFailedError(RuntimeError):
    """Raised when an LLM call fails after all retry attempts.

    In "pure LLM" mode, this aborts the simulation rather than silently
    falling back to a utility decision, because mixing utility-based and
    LLM-based decisions would produce scientifically meaningless results.

    Attributes:
        attempts: Total number of sub-client attempts made.
        last_exc: The final exception seen.
        context: Optional dict with agent_id, provider, etc.
    """

    def __init__(
        self,
        message: str,
        attempts: int,
        last_exc: Optional[BaseException] = None,
        context: Optional[dict] = None,
    ):
        super().__init__(message)
        self.attempts = attempts
        self.last_exc = last_exc
        self.context = context or {}


class ZaiCodingClient:
    """LLM client for Z.ai Coding Plan (Anthropic-compatible proxy)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.z.ai/api/anthropic",
        model: Optional[str] = None,
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Install anthropic SDK: pip install anthropic")

        self.model = model or os.environ.get("ZAI_MODEL", "glm-4.7")
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ZAI_API_KEY", ""),
            base_url=base_url,
        )

    def complete(self, prompt: str, system: str = "") -> str:
        kwargs = {
            "model": self.model,
            "max_tokens": 500,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        msg = self.client.messages.create(**kwargs)
        return msg.content[0].text


class CodexCliClient:
    """LLM client that wraps the OpenAI Codex CLI as a local subprocess.

    Designed for users on the OpenAI Codex Coding Plan whose access is via
    OAuth (no API key). Each call invokes ``codex exec`` non-interactively,
    feeds the prompt on stdin, and reads the model's final message from a
    temporary file specified via ``--output-last-message``.

    The Codex CLI handles OAuth refresh internally — this client never
    touches credentials directly.

    Notes:
        * Each call spawns a subprocess and incurs ~2-3 seconds of overhead
          even for trivial prompts (Codex injects a large system prompt).
        * `--ephemeral` skips session persistence; `--skip-git-repo-check`
          allows running outside a git repo.
        * Default model is ``gpt-5.4-mini``; reasoning is set to ``low`` to
          minimize per-call latency.
        * Stderr from Codex is suppressed unless an error occurs.
        * No rate limiting observed up to ~30 concurrent subprocesses on the
          Codex Coding Plan; the practical sweet spot is concurrency=15.

    Environment overrides:
        ``CODEX_MODEL``           - overrides default model
        ``CODEX_REASONING_EFFORT`` - overrides default reasoning effort

    Args:
        model: Override for ``codex exec --model``. Default ``gpt-5.4-mini``.
        reasoning_effort: Override for ``-c model_reasoning_effort=…``. Default ``low``.
        timeout: Per-call subprocess timeout in seconds (default 120).
        codex_path: Path to the Codex executable (default: looks up "codex" on PATH).
    """

    DEFAULT_MODEL = "gpt-5.4-mini"
    DEFAULT_REASONING_EFFORT = "low"

    def __init__(
        self,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        timeout: float = 120.0,
        codex_path: Optional[str] = None,
    ):
        import shutil

        self.codex_path = codex_path or shutil.which("codex")
        if not self.codex_path:
            raise FileNotFoundError(
                "Codex CLI not found on PATH. Install it from https://platform.openai.com/codex"
            )
        self.model = model or os.environ.get("CODEX_MODEL", self.DEFAULT_MODEL)
        self.reasoning_effort = (
            reasoning_effort
            or os.environ.get("CODEX_REASONING_EFFORT", self.DEFAULT_REASONING_EFFORT)
        )
        self.timeout = timeout

    def complete(self, prompt: str, system: str = "") -> str:
        import subprocess
        import tempfile

        full_prompt = (system + "\n\n" + prompt) if system else prompt

        with tempfile.NamedTemporaryFile(
            mode="r", suffix=".txt", delete=False
        ) as out_file:
            out_path = out_file.name

        try:
            args = [
                self.codex_path,
                "exec",
                "--skip-git-repo-check",
                "--ephemeral",
                "-o",
                out_path,
            ]
            if self.model:
                args.extend(["--model", self.model])
            if self.reasoning_effort:
                args.extend(["-c", f"model_reasoning_effort={json.dumps(self.reasoning_effort)}"])

            try:
                result = subprocess.run(
                    args,
                    input=full_prompt,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"Codex CLI timed out after {self.timeout}s"
                ) from exc

            if result.returncode != 0:
                stderr = (result.stderr or "").strip()
                if len(stderr) > 500:
                    stderr = stderr[-500:]
                raise RuntimeError(
                    f"Codex CLI exited {result.returncode}: {stderr}"
                )

            try:
                with open(out_path, "r") as f:
                    response = f.read().strip()
            except FileNotFoundError:
                response = ""

            return response
        finally:
            try:
                os.unlink(out_path)
            except OSError:
                pass


class ClaudeCodeClient:
    """LLM client that wraps the Anthropic Claude Code CLI as a local subprocess.

    Designed for users on a Claude subscription (Pro / Max / coding plans)
    whose access is via OAuth — no API key. Each call invokes ``claude -p``
    non-interactively, feeds the prompt on stdin, and captures the model's
    response from stdout.

    The Claude Code CLI handles OAuth refresh internally — this client
    never touches credentials directly.

    Notes:
        * Each call spawns a subprocess and incurs ~2-3 seconds of overhead
          even for trivial prompts (Claude Code bootstraps tools/MCP on
          every invocation).
        * Default model is ``haiku`` (fast and cheap — matches the simulator's
          latency-sensitive decision loop).
        * Effort is set to ``low`` by default so the CLI does not silently
          inherit a higher local default from the user's Claude configuration.
        * ``--append-system-prompt`` is used for the system message so the
          default Claude Code behavior is preserved.
        * The ``--bare`` flag is NOT used: it forces ``ANTHROPIC_API_KEY``
          authentication and bypasses OAuth, which breaks subscription
          access.
        * Stderr is suppressed unless an error occurs.

    Environment overrides:
        ``CLAUDE_MODEL``  - overrides the default model
        ``CLAUDE_EFFORT`` - overrides the default effort level

    Args:
        model: Override for ``claude --model``. Default ``haiku``.
        effort: Override for ``claude --effort``. Default ``low``.
        timeout: Per-call subprocess timeout in seconds (default 120).
        claude_path: Path to the claude executable (default: looks up
            "claude" on PATH).
    """

    DEFAULT_MODEL = "haiku"
    DEFAULT_EFFORT = "low"

    def __init__(
        self,
        model: Optional[str] = None,
        effort: Optional[str] = None,
        timeout: float = 120.0,
        claude_path: Optional[str] = None,
    ):
        import shutil

        self.claude_path = claude_path or shutil.which("claude")
        if not self.claude_path:
            raise FileNotFoundError(
                "Claude Code CLI not found on PATH. Install it from "
                "https://claude.com/product/claude-code"
            )
        self.model = model or os.environ.get("CLAUDE_MODEL", self.DEFAULT_MODEL)
        self.effort = effort or os.environ.get("CLAUDE_EFFORT", self.DEFAULT_EFFORT)
        self.timeout = timeout

    def complete(self, prompt: str, system: str = "") -> str:
        import subprocess

        args = [self.claude_path, "-p"]
        if self.model:
            args.extend(["--model", self.model])
        if self.effort:
            args.extend(["--effort", self.effort])
        if system:
            args.extend(["--append-system-prompt", system])

        try:
            result = subprocess.run(
                args,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Claude Code CLI timed out after {self.timeout}s"
            ) from exc

        if result.returncode != 0:
            raise RuntimeError(
                f"Claude Code CLI exited {result.returncode}: "
                f"{result.stderr.strip()[:500]}"
            )

        return (result.stdout or "").strip()


class MultiProviderClient:
    """Load-balancing LLM client that fans out across multiple sub-clients.

    Implements the same ``LLMClient`` interface (``complete(prompt, system)``)
    so it slots into ``LLMEngine`` exactly like a single client. Calls are
    dispatched across sub-clients using a round-robin strategy, with per-
    sub-client concurrency limits enforced via threading semaphores so that
    no individual provider is overwhelmed.

    Per-sub-client stats (call count, token usage) are tracked so the run
    metadata can report which provider did which share of the work.

    Retry behavior (critical for "pure LLM" mode):
        * On sub-client failure, immediately try the next sub-client in the
          round-robin order. This handles rate limits and transient errors
          without waiting.
        * When *all* sub-clients have failed in a single round, the call
          waits with exponential backoff (1s, 2s, 4s, 8s, 16s) and retries
          the full round. The backoff gives rate-limited providers time
          to recover.
        * After ``max_retries`` full rounds all fail, raises
          ``LLMCallFailedError``. The caller is expected to abort the
          simulation rather than silently fall back to utility decisions.

    Typical usage::

        client = MultiProviderClient([
            (ZaiCodingClient(), 10),
            (CodexCliClient(), 15),
        ], max_retries=5)
        engine = LLMEngine(client, concurrency=25)
    """

    DEFAULT_MAX_RETRIES = 5
    DEFAULT_BACKOFF_BASE = 1.0
    DEFAULT_BACKOFF_CAP = 16.0
    _QUOTA_EXHAUSTION_PATTERNS = (
        "usage limit reached",
        "limit will reset",
        "insufficient_quota",
        "reached your current quota",
        "quota exceeded",
        "billing hard limit",
        "credit balance is too low",
    )

    def __init__(
        self,
        providers: list[tuple[object, int]],
        strategy: str = "round_robin",
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        backoff_cap: float = DEFAULT_BACKOFF_CAP,
        verbose: bool = False,
    ):
        import threading

        if not providers:
            raise ValueError("At least one provider is required")
        if strategy not in ("round_robin",):
            raise ValueError(f"Unsupported dispatch strategy: {strategy}")
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")

        self.strategy = strategy
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self.verbose = verbose
        self._slots = []
        for client, max_concurrency in providers:
            if max_concurrency < 1:
                raise ValueError(
                    f"max_concurrency must be >= 1, got {max_concurrency}"
                )
            name = getattr(client, "__class__", type(client)).__name__
            self._slots.append({
                "client": client,
                "name": name,
                "max_concurrency": max_concurrency,
                "semaphore": threading.Semaphore(max_concurrency),
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "errors": 0,
                "retired": False,
                "retired_reason": None,
            })
        self._stats_lock = threading.Lock()
        self._rr_index = 0
        self._rr_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    @property
    def total_concurrency(self) -> int:
        """Sum of per-sub-client concurrency limits. Use this as the
        concurrency value for the outer AsyncLLMClient."""
        return sum(s["max_concurrency"] for s in self._slots)

    @property
    def provider_names(self) -> list[str]:
        return [s["name"] for s in self._slots]

    def _active_slot_indices(self) -> list[int]:
        with self._stats_lock:
            return [i for i, slot in enumerate(self._slots) if not slot["retired"]]

    def _should_retire_provider(self, exc: BaseException) -> bool:
        """Return True when an error looks like hard quota exhaustion.

        We deliberately distinguish quota exhaustion from ordinary transient
        failures: providers that are merely flaky or briefly rate limited
        should stay in rotation, but providers whose plan/quota is exhausted
        should be removed for the rest of the run so the remaining providers
        can continue making progress.
        """
        text = str(exc).lower()
        return any(pattern in text for pattern in self._QUOTA_EXHAUSTION_PATTERNS)

    def _retire_slot(self, slot: dict, exc: BaseException):
        should_log = False
        reason = str(exc)[:500]
        with self._stats_lock:
            if not slot["retired"]:
                slot["retired"] = True
                slot["retired_reason"] = reason
                should_log = True
        if should_log and self.verbose:
            print(
                f"  Retiring provider '{slot['name']}' due to quota exhaustion: {reason}",
                flush=True,
            )

    def _pick_slot(self) -> Optional[dict]:
        """Pick the next available slot. Round-robin with semaphore gating.

        Tries slots in round-robin order, acquiring the first one whose
        semaphore is free. If all are full, blocks on the next one in the
        round-robin sequence.
        """
        n = len(self._slots)
        active_indices = self._active_slot_indices()
        if not active_indices:
            return None
        active_set = set(active_indices)

        with self._rr_lock:
            start = self._rr_index
            self._rr_index = (self._rr_index + 1) % n

        # Try each slot once non-blocking in round-robin order
        for offset in range(n):
            idx = (start + offset) % n
            if idx not in active_set:
                continue
            slot = self._slots[idx]
            if slot["semaphore"].acquire(blocking=False):
                return slot

        # All active providers are busy — block on the first active slot in
        # round-robin order.
        for offset in range(n):
            idx = (start + offset) % n
            if idx not in active_set:
                continue
            slot = self._slots[idx]
            slot["semaphore"].acquire(blocking=True)
            if slot["retired"]:
                slot["semaphore"].release()
                continue
            return slot

        return None

    def complete(self, prompt: str, system: str = "") -> str:
        """Dispatch one call to an available sub-client, with retry + backoff.

        Strategy:
          1. Try each sub-client in round-robin order (one round).
          2. If any sub-client succeeds, return immediately.
          3. If all sub-clients fail in this round, wait with exponential
             backoff (1s, 2s, 4s, 8s, 16s cap) and retry the full round.
          4. After ``max_retries`` full rounds, raise ``LLMCallFailedError``.

        Total attempts = (1 + max_retries) * len(sub_clients). With 2
        sub-clients and max_retries=5, up to 12 sub-client calls before
        giving up.

        Raises:
            LLMCallFailedError: All retry attempts exhausted. Simulation
                should abort rather than fall back to utility decisions.
        """
        import time

        total_attempts = 0
        last_exc: Optional[Exception] = None

        for round_idx in range(self.max_retries + 1):
            # One full round: try every sub-client exactly once
            tried: set[int] = set()

            for _ in range(len(self._slots)):
                slot = self._pick_slot()
                if slot is None:
                    break
                slot_id = id(slot)
                if slot_id in tried:
                    slot["semaphore"].release()
                    continue
                tried.add(slot_id)
                total_attempts += 1

                try:
                    response = slot["client"].complete(prompt, system=system)
                    with self._stats_lock:
                        slot["calls"] += 1
                        slot["input_tokens"] += max(
                            1, (len(prompt) + len(system)) // 4
                        )
                        slot["output_tokens"] += (
                            max(1, len(response) // 4) if response else 0
                        )
                    return response
                except Exception as exc:
                    with self._stats_lock:
                        slot["errors"] += 1
                    last_exc = exc
                    if self._should_retire_provider(exc):
                        self._retire_slot(slot, exc)
                    # Try the next sub-client in this round
                    continue
                finally:
                    slot["semaphore"].release()

            if not self._active_slot_indices():
                break

            # All sub-clients failed in this round. Back off and retry.
            if round_idx < self.max_retries:
                wait = min(
                    self.backoff_base * (2 ** round_idx),
                    self.backoff_cap,
                )
                time.sleep(wait)

        # Exhausted all retries
        providers = ", ".join(s["name"] for s in self._slots)
        raise LLMCallFailedError(
            message=(
                f"All LLM providers ({providers}) failed after "
                f"{self.max_retries + 1} rounds ({total_attempts} total attempts). "
                f"Last error: {type(last_exc).__name__ if last_exc else 'unknown'}: "
                f"{str(last_exc)[:200] if last_exc else 'no exception recorded'}"
            ),
            attempts=total_attempts,
            last_exc=last_exc,
            context={"providers": [s["name"] for s in self._slots]},
        )

    # ------------------------------------------------------------------
    # Stats reporting
    # ------------------------------------------------------------------

    def get_stats(self) -> list[dict]:
        """Return per-sub-client stats snapshot for metadata."""
        with self._stats_lock:
            return [
                {
                    "provider": s["name"],
                    "max_concurrency": s["max_concurrency"],
                    "calls": s["calls"],
                    "input_tokens": s["input_tokens"],
                    "output_tokens": s["output_tokens"],
                    "errors": s["errors"],
                    "retired": s["retired"],
                    "retired_reason": s["retired_reason"],
                }
                for s in self._slots
            ]

    def total_calls(self) -> int:
        with self._stats_lock:
            return sum(s["calls"] for s in self._slots)


class RetryingClient:
    """Wraps a single LLMClient with bounded retry + exponential backoff.

    Used for single-provider "pure LLM" mode where the goal is to ensure
    every agent decision comes from a real LLM call, not a utility
    fallback. If all retries fail, raises ``LLMCallFailedError`` so the
    simulation can abort loudly rather than silently substitute utility
    decisions.

    When the underlying client is already a ``MultiProviderClient`` (which
    has its own retry logic), this wrapper is NOT applied — the factory
    detects that case and skips double-wrapping.

    Retry schedule: base * 2^attempt, capped at backoff_cap.
    Default: 1s, 2s, 4s, 8s, 16s for max_retries=5.
    """

    DEFAULT_MAX_RETRIES = 5
    DEFAULT_BACKOFF_BASE = 1.0
    DEFAULT_BACKOFF_CAP = 16.0

    def __init__(
        self,
        client: object,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        backoff_cap: float = DEFAULT_BACKOFF_CAP,
        provider_name: Optional[str] = None,
    ):
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")
        self.client = client
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self.provider_name = provider_name or client.__class__.__name__
        self.successful_calls = 0
        self.failed_calls = 0
        self.retry_count = 0

    def complete(self, prompt: str, system: str = "") -> str:
        """Call the underlying client, retrying with backoff on any error.

        Raises:
            LLMCallFailedError: All retry attempts exhausted.
        """
        import time

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.complete(prompt, system=system)
                self.successful_calls += 1
                return response
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    self.retry_count += 1
                    wait = min(
                        self.backoff_base * (2 ** attempt),
                        self.backoff_cap,
                    )
                    time.sleep(wait)
                    continue
                # Final attempt failed
                self.failed_calls += 1
                break

        raise LLMCallFailedError(
            message=(
                f"LLM provider '{self.provider_name}' failed after "
                f"{self.max_retries + 1} attempts. "
                f"Last error: {type(last_exc).__name__ if last_exc else 'unknown'}: "
                f"{str(last_exc)[:200] if last_exc else 'no exception recorded'}"
            ),
            attempts=self.max_retries + 1,
            last_exc=last_exc,
            context={"provider": self.provider_name},
        )


def _build_persona(agent: Agent) -> str:
    """Generate a persona description from agent demographics."""
    parts = []
    parts.append(f"You are a {agent.age_head}-year-old head of a household of {agent.household_size}.")
    parts.append(f"Your monthly household income is S${agent.income:.0f}.")
    parts.append(f"You work in the {agent.job_location} area.")

    if agent.has_children:
        parts.append("You have school-age children.")
    if agent.has_elderly:
        parts.append("You have elderly family members living with you who need regular healthcare.")
    if agent.car_owner:
        parts.append("You own a car.")
    else:
        parts.append("You do not own a car and rely on public transport.")

    return " ".join(parts)


def _build_elicitation_prompt(agent: Agent) -> str:
    """Build a prompt to elicit preference weights from an LLM."""
    traits = []
    traits.append(f"age {agent.age_head}")
    traits.append(f"household of {agent.household_size}")
    traits.append(f"income ${agent.income:.0f}/month")
    if agent.has_children:
        traits.append("has school-age children")
    if agent.has_elderly:
        traits.append("has elderly dependents")
    if not agent.car_owner:
        traits.append("relies on public transport")
    profile = ", ".join(traits)

    return (
        f"Rate housing preferences for this person on a scale of 1-10:\n"
        f"Person: {profile}\n"
        f"housing=?, commute=?, services=?, amenities=?\n"
        f"Reply as JSON with integer values."
    )


def _parse_preference_response(response: str) -> PreferenceWeights:
    """Parse LLM response into preference weights."""
    text = response.strip()
    # Strip markdown code blocks if present
    if "```" in text:
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    # Try to find JSON object in the response
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in response: {text}")

    data = json.loads(text[start:end])
    total = data["housing"] + data["commute"] + data["services"] + data["amenities"]
    if total == 0:
        total = 1
    return PreferenceWeights(
        alpha=data["housing"] / total,
        beta=data["commute"] / total,
        gamma=data["services"] / total,
        delta=data["amenities"] / total,
    )


class LLMPreferenceElicitor:
    """Elicit preference weights from an LLM for each agent type."""

    def __init__(self, client: LLMClient, cache_dir: Optional[str | Path] = None):
        self.client = client
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory: dict[str, PreferenceWeights] = {}

    def _cache_key(self, agent: Agent) -> str:
        """Generate a cache key from agent demographics."""
        data = f"{agent.age_head}_{agent.household_size}_{agent.has_children}_" \
               f"{agent.has_elderly}_{agent.income:.0f}_{agent.job_location}_{agent.car_owner}"
        return hashlib.md5(data.encode()).hexdigest()

    def _load_cached(self, key: str) -> Optional[PreferenceWeights]:
        return self._load_cached_from(key, self.cache_dir, self._memory)

    def _save_cache(self, key: str, pw: PreferenceWeights):
        self._save_cache_to(key, pw, self.cache_dir, self._memory)

    @staticmethod
    def _load_cached_from(
        key: str,
        cache_dir: Optional[Path],
        memory: dict[str, PreferenceWeights],
    ) -> Optional[PreferenceWeights]:
        if key in memory:
            return memory[key]
        if cache_dir:
            path = cache_dir / f"{key}.json"
            if path.exists():
                data = json.loads(path.read_text())
                pw = PreferenceWeights(**data)
                memory[key] = pw
                return pw
        return None

    @staticmethod
    def _save_cache_to(
        key: str,
        pw: PreferenceWeights,
        cache_dir: Optional[Path],
        memory: dict[str, PreferenceWeights],
    ):
        memory[key] = pw
        if cache_dir:
            path = cache_dir / f"{key}.json"
            data = {"alpha": pw.alpha, "beta": pw.beta, "gamma": pw.gamma, "delta": pw.delta}
            path.write_text(json.dumps(data))

    def elicit(self, agent: Agent, max_retries: int = 3) -> PreferenceWeights:
        """Get preference weights for an agent, using cache if available."""
        key = self._cache_key(agent)
        cached = self._load_cached(key)
        if cached is not None:
            return cached

        prompt = _build_elicitation_prompt(agent)
        system = "You are a JSON generator. Output only valid JSON, no other text."
        for attempt in range(max_retries):
            try:
                response = self.client.complete(prompt, system=system)
                pw = _parse_preference_response(response)
                self._save_cache(key, pw)
                return pw
            except (ValueError, json.JSONDecodeError, KeyError):
                continue

        # Fallback after 3 failed LLM parse attempts: return neutral preferences.
        # Only used in hybrid mode. NOT a model parameter for utility computation.
        pw = PreferenceWeights(0.25, 0.25, 0.25, 0.25)
        self._save_cache(key, pw)
        return pw

    def elicit_batch(
        self,
        agents: list[Agent],
        *,
        cache_dir: Optional[str | Path] = None,
        concurrency: int = 0,
        verbose: bool = False,
    ) -> list[PreferenceWeights]:
        """Batched elicitation returning weights aligned to input order.

        Dispatches all cache-miss prompts through ``AsyncLLMClient.complete_many()``
        in one pass. Parse failures go through a second async batch; persistent
        failures default to neutral weights (0.25, 0.25, 0.25, 0.25).

        Args:
            agents: Agents to elicit for. Returned list is aligned to this order.
            cache_dir: If provided, read/write cache entries exclusively within
                this directory for this call. Does not mutate ``self.cache_dir``.
                If omitted, uses ``self.cache_dir`` (the default set at construction).
            concurrency: Max parallel LLM calls. 0 = auto
                (``client.total_concurrency`` for multi-provider, or 10).
            verbose: Print progress.

        Returns:
            List of ``PreferenceWeights`` in the same order as ``agents``.
        """
        import logging
        from agent_urban_planning.llm.async_client import AsyncLLMClient, make_progress_printer

        # Resolve effective cache location for this call. A separate in-memory
        # dict for overridden cache_dir avoids leaking entries across namespaces.
        if cache_dir is not None:
            effective_cache = Path(cache_dir)
            effective_cache.mkdir(parents=True, exist_ok=True)
            memory: dict[str, PreferenceWeights] = {}
        else:
            effective_cache = self.cache_dir
            memory = self._memory

        n = len(agents)
        if n == 0:
            return []

        if concurrency <= 0:
            concurrency = getattr(self.client, "total_concurrency", None) or 10

        results: list[Optional[PreferenceWeights]] = [None] * n
        pending_idx: list[int] = []
        pending_prompts: list[str] = []
        pending_keys: list[str] = []
        system_msg = "You are a JSON generator. Output only valid JSON, no other text."

        # Phase 1 — cache lookup + DEMOGRAPHIC DEDUPLICATION.
        #
        # Agents with identical demographic features produce the same cache
        # key. Without deduplication we'd call the LLM once per agent even
        # when thousands share the same key — burning compute on identical
        # prompts. This pass collapses uncached agents to a unique-key set,
        # calls the LLM once per unique key, then fans the result back out
        # to all agents sharing that key.
        cached_count = 0
        key_to_first_agent_idx: dict[str, int] = {}
        key_to_sharing_agents: dict[str, list[int]] = {}
        agent_key: list[str] = []
        for i, agent in enumerate(agents):
            key = self._cache_key(agent)
            agent_key.append(key)
            cached = self._load_cached_from(key, effective_cache, memory)
            if cached is not None:
                results[i] = cached
                cached_count += 1
                continue
            if key in key_to_first_agent_idx:
                # A sibling agent with identical demographics is already queued.
                key_to_sharing_agents[key].append(i)
            else:
                key_to_first_agent_idx[key] = i
                key_to_sharing_agents[key] = [i]
                pending_idx.append(i)
                pending_prompts.append(_build_elicitation_prompt(agent))
                pending_keys.append(key)

        n_unique_pending = len(pending_prompts)
        n_deduped = len(agents) - cached_count - n_unique_pending
        if verbose:
            print(
                f"Eliciting preferences: {cached_count} cached, "
                f"{n_unique_pending} unique demographic profiles to fetch "
                f"(deduplicated {n_deduped} duplicate agents), "
                f"concurrency={concurrency}",
                flush=True,
            )

        if not pending_prompts:
            if verbose:
                print("  All preferences loaded from cache.\n", flush=True)
            return [r for r in results]  # type: ignore[return-value]

        # Phase 2 — first async batch with INCREMENTAL cache writes.
        # The on_result callback below writes each successfully-parsed
        # preference to disk as soon as its response returns. Crash-resilient
        # and user-visible: cache-file count grows live.
        async_client = AsyncLLMClient(
            self.client,
            concurrency=min(concurrency, 50),
            provider_name="preference-elicitor",
        )
        progress_cb = (
            make_progress_printer("Eliciting preferences") if verbose else None
        )
        # Incremental-save callback. Captures the local arrays via closure.
        # idx is the position in pending_prompts (one per UNIQUE demographic
        # profile); we fan the result out to all agents sharing that key.
        _local = self  # for staticmethod access
        def _on_result(idx: int, response: str, error):
            if error is not None or not response:
                return
            try:
                pw = _parse_preference_response(response)
            except (ValueError, json.JSONDecodeError, KeyError):
                return  # parse failure — retry phase handles it later
            key = pending_keys[idx]
            _local._save_cache_to(key, pw, effective_cache, memory)
            # Populate every agent sharing this demographic profile.
            for agent_idx in key_to_sharing_agents[key]:
                results[agent_idx] = pw

        try:
            systems = [system_msg] * len(pending_prompts)
            responses = async_client.complete_many(
                pending_prompts, systems=systems,
                on_progress=progress_cb,
                on_result=_on_result,
            )
        finally:
            async_client.close()

        # Phase 3 — parse; collect parse failures for retry.
        retry_j: list[int] = []
        success_count = 0
        fallback_count = 0
        for j, response in enumerate(responses):
            key = pending_keys[j]
            # Check if the FIRST agent sharing this key was populated by the
            # on_result callback (other sharers are too, via fan-out).
            if results[pending_idx[j]] is not None:
                success_count += 1
                continue
            try:
                pw = _parse_preference_response(response)
                self._save_cache_to(key, pw, effective_cache, memory)
                for agent_idx in key_to_sharing_agents[key]:
                    results[agent_idx] = pw
                success_count += 1
            except (ValueError, json.JSONDecodeError, KeyError):
                retry_j.append(j)

        if retry_j:
            if verbose:
                print(f"  Retrying {len(retry_j)} parse failures...", flush=True)
            retry_prompts = [pending_prompts[j] for j in retry_j]
            retry_systems = [system_msg] * len(retry_prompts)

            async_client2 = AsyncLLMClient(
                self.client,
                concurrency=min(concurrency, 50),
                provider_name="preference-elicitor-retry",
            )
            try:
                retry_responses = async_client2.complete_many(
                    retry_prompts, systems=retry_systems,
                )
            finally:
                async_client2.close()

            for k, response in enumerate(retry_responses):
                orig_j = retry_j[k]
                key = pending_keys[orig_j]
                try:
                    pw = _parse_preference_response(response)
                    self._save_cache_to(key, pw, effective_cache, memory)
                    for agent_idx in key_to_sharing_agents[key]:
                        results[agent_idx] = pw
                    success_count += 1
                except (ValueError, json.JSONDecodeError, KeyError):
                    pw = PreferenceWeights(0.25, 0.25, 0.25, 0.25)
                    self._save_cache_to(key, pw, effective_cache, memory)
                    for agent_idx in key_to_sharing_agents[key]:
                        results[agent_idx] = pw
                    fallback_count += 1
                    logging.getLogger(__name__).warning(
                        "Preference elicitation failed for key %s after 2 attempts "
                        "(covers %d agents). Using neutral weights.",
                        key[:8], len(key_to_sharing_agents[key]),
                    )

        if verbose:
            print(
                f"  Fetched {success_count}/{len(pending_prompts)} preferences, "
                f"{fallback_count} fallbacks. Elicitation complete.",
                flush=True,
            )

        assert all(r is not None for r in results)
        return [r for r in results]  # type: ignore[return-value]


def _build_zone_choice_prompt(
    agent: Agent,
    environment: Environment,
    zone_options: list[str],
    prices: dict[str, float],
) -> str:
    """Build the LLM prompt for one agent's zone choice."""
    persona = _build_persona(agent)
    zone_descs = []
    for zname in zone_options:
        zone = environment.get_zone(zname)
        price = prices.get(zname, zone.housing_base_price)
        route = environment.transport.get_best_route(zname, agent.job_location)
        commute = (
            f"{route.time_minutes:.0f} min by {route.mode}" if route else "no direct route"
        )
        facilities = ", ".join(f.type for f in zone.facilities) if zone.facilities else "none"
        zone_descs.append(
            f"- {zname}: rent S${price:.0f}/month, commute {commute}, "
            f"facilities: {facilities}, amenity score: {zone.amenity_score:.1f}"
        )
    return (
        f"{persona}\n\n"
        f"You are choosing where to live. Here are your options:\n\n"
        f"{chr(10).join(zone_descs)}\n\n"
        f"Choose the zone that best fits your needs. Respond with ONLY a JSON object:\n"
        f'{{"zone": "<zone_name>", "reason": "<brief reason>"}}'
    )


def _parse_zone_response(response: str, zone_options: list[str]) -> Optional[str]:
    """Extract a valid zone name from an LLM response, or None if malformed."""
    if not response:
        return None
    text = response.strip()
    if "```" in text:
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        data = json.loads(text[start:end])
    except (json.JSONDecodeError, ValueError):
        return None
    chosen = data.get("zone")
    if chosen in zone_options:
        return chosen
    return None


class LLMEngine:
    """Full LLM decision engine — the LLM makes the zone choice directly.

    Supports both sequential (`decide`) and batched (`decide_batch`) calls.
    The batched path uses an `AsyncLLMClient` to issue concurrent API calls
    bounded by `concurrency`. Both paths consult an injected `LLMCallCache`
    when available.
    """

    def __init__(
        self,
        client: LLMClient,
        max_retries: int = 3,
        concurrency: int = 10,
        provider_name: Optional[str] = None,
        budget_constraint: bool = True,
        verbose: bool = False,
    ):
        from agent_urban_planning.llm.async_client import AsyncLLMClient

        self.client = client
        self.max_retries = max_retries
        self.concurrency = concurrency
        self.budget_constraint = budget_constraint
        self.verbose = verbose
        self._async_client = AsyncLLMClient(
            client, concurrency=concurrency, provider_name=provider_name
        )
        self._fallback = UtilityEngine(budget_constraint=budget_constraint)
        self._cache = None

    def close(self):
        """Release the async client's event loop. Safe to call multiple times."""
        ac = getattr(self, "_async_client", None)
        if ac is not None:
            try:
                ac.close()
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Cache hook
    # ------------------------------------------------------------------

    def set_cache(self, cache) -> None:
        self._cache = cache

    @property
    def cache(self):
        return self._cache

    def _flush_cache(self) -> None:
        cache = self._cache
        if cache is None:
            return
        flush = getattr(cache, "flush", None)
        if callable(flush):
            flush()

    # ------------------------------------------------------------------
    # Token usage (delegated to async client)
    # ------------------------------------------------------------------

    @property
    def total_input_tokens(self) -> int:
        return self._async_client.total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self._async_client.total_output_tokens

    # ------------------------------------------------------------------
    # Single-agent decision (legacy path, used by tests and tools)
    # ------------------------------------------------------------------

    def decide(
        self,
        agent: Agent,
        environment: Environment,
        zone_options: list[str],
        prices: dict[str, float],
    ) -> ZoneChoice:
        """Make one LLM zone choice for a single agent.

        Raises:
            LLMCallFailedError: The LLM call failed after all retries.
                The underlying client is expected to have already retried
                (via RetryingClient or MultiProviderClient). If the LLM
                responds but its output is malformed, this method retries
                parsing up to ``max_retries`` times — those are parse
                retries, not network retries.
        """
        # Budget constraint: filter to affordable zones
        if self.budget_constraint:
            from agent_urban_planning.core.constraints import affordable_zones
            filtered = affordable_zones(
                agent.income, {z: prices.get(z, 0) for z in zone_options}
            )
            if not filtered:
                # Outside option
                home = getattr(agent, "home_zone", "") or (zone_options[0] if zone_options else "unknown")
                return ZoneChoice(
                    zone_name=home,
                    utility=0.0,
                    zone_utilities={z: 0.0 for z in zone_options},
                    workplace=agent.job_location,
                )
            effective_zones = filtered
        else:
            effective_zones = list(zone_options)

        # Cache lookup first
        if self._cache is not None:
            cached = self._cache.get(agent.agent_id, prices)
            if cached is not None:
                return cached

        prompt = _build_zone_choice_prompt(agent, environment, effective_zones, prices)
        chosen: Optional[str] = None
        last_raw_response: str = ""
        for _ in range(self.max_retries):
            response = self.client.complete(prompt)
            last_raw_response = response
            chosen = _parse_zone_response(response, effective_zones)
            if chosen is not None:
                break

        if chosen is None:
            # LLM returned parseable-garbage max_retries times in a row.
            # This is a content failure, not a network failure. Raise
            # LLMCallFailedError so the caller can see it.
            raise LLMCallFailedError(
                message=(
                    f"LLM returned unparseable response {self.max_retries} times "
                    f"for agent {agent.agent_id}. Last response: "
                    f"{last_raw_response[:200]!r}"
                ),
                attempts=self.max_retries,
                last_exc=None,
                context={
                    "agent_id": agent.agent_id,
                    "kind": "parse_failure",
                    "last_response": last_raw_response[:500],
                },
            )

        choice = self._build_choice(chosen, agent, environment, zone_options, prices)
        if self._cache is not None:
            self._cache.put(agent.agent_id, prices, choice)
            self._flush_cache()
        return choice

    # ------------------------------------------------------------------
    # Batch decision (async, the main performance path)
    # ------------------------------------------------------------------

    def decide_batch(
        self,
        agents: list[Agent],
        environment: Environment,
        zone_options: list[str],
        prices: dict[str, float],
    ) -> list[ZoneChoice]:
        if not agents:
            return []

        results: list[Optional[ZoneChoice]] = [None] * len(agents)
        pending_indices: list[int] = []
        prompt_by_index: dict[int, str] = {}

        # Phase 1: cache lookups
        for i, agent in enumerate(agents):
            if self._cache is not None:
                cached = self._cache.get(agent.agent_id, prices)
                if cached is not None:
                    results[i] = cached
                    continue
            pending_indices.append(i)
            prompt_by_index[i] = _build_zone_choice_prompt(
                agent, environment, zone_options, prices
            )

        # Phase 2: async batch for the cache misses.
        # No utility fallback here — if the LLM call fails after all
        # retries (handled by the underlying RetryingClient or
        # MultiProviderClient), LLMCallFailedError propagates and aborts
        # the simulation. This preserves "pure LLM" semantics.
        if pending_indices:
            attempts = max(1, self.max_retries)
            current_pending = list(pending_indices)
            failed_parses: list[tuple[int, str]] = []

            for attempt_idx in range(attempts):
                progress_cb = None
                if self.verbose:
                    from agent_urban_planning.llm.async_client import make_progress_printer

                    cache_info = (
                        f" ({len(agents) - len(pending_indices)} cached)"
                        if len(pending_indices) < len(agents)
                        else ""
                    )
                    retry_info = f" retry {attempt_idx + 1}/{attempts}" if attempt_idx > 0 else ""
                    progress_cb = make_progress_printer(
                        f"LLM calls{cache_info}{retry_info}"
                    )

                prompts = [prompt_by_index[i] for i in current_pending]
                responses = self._async_client.complete_many(
                    prompts, on_progress=progress_cb,
                )

                next_pending: list[int] = []
                failed_parses = []
                for idx_in_pending, raw_response in enumerate(responses):
                    agent_idx = current_pending[idx_in_pending]
                    agent = agents[agent_idx]
                    chosen = _parse_zone_response(raw_response, zone_options)
                    if chosen is None:
                        next_pending.append(agent_idx)
                        failed_parses.append((agent.agent_id, raw_response))
                        continue

                    choice = self._build_choice(
                        chosen, agent, environment, zone_options, prices
                    )
                    results[agent_idx] = choice
                    if self._cache is not None:
                        self._cache.put(agent.agent_id, prices, choice)

                self._flush_cache()

                if not next_pending:
                    current_pending = []
                    break
                current_pending = next_pending

            # If any calls still produced unparseable content after all
            # parse retries, raise a batch-level error naming the first
            # few failures.
            if current_pending:
                sample = failed_parses[:3]
                raise LLMCallFailedError(
                    message=(
                        f"{len(current_pending)} of {len(pending_indices)} LLM "
                        f"responses were unparseable after {attempts} attempts. "
                        f"First failures: {sample}"
                    ),
                    attempts=len(pending_indices) * attempts,
                    last_exc=None,
                    context={
                        "kind": "parse_failure",
                        "failed_agent_ids": [agents[i].agent_id for i in current_pending],
                    },
                )

        assert all(r is not None for r in results)
        return [r for r in results]  # type: ignore[list-item]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_choice(
        self,
        chosen: Optional[str],
        agent: Agent,
        environment: Environment,
        zone_options: list[str],
        prices: dict[str, float],
    ) -> ZoneChoice:
        """Construct a ZoneChoice from the LLM's chosen zone.

        NOTE: ``chosen`` MUST be a valid zone name. Callers are responsible
        for raising ``LLMCallFailedError`` when the LLM cannot produce a
        valid choice — this method will raise if called with ``chosen=None``
        to preserve "pure LLM" semantics.

        The ``UtilityEngine`` is invoked here only to compute the per-zone
        utility values for the record (so downstream analysis can compare
        what the utility model would have scored each zone). The
        ``zone_name`` field always reflects the LLM's choice.
        """
        if chosen is None:
            raise LLMCallFailedError(
                message=(
                    f"_build_choice called with chosen=None for agent "
                    f"{agent.agent_id}. This indicates a bug in the caller — "
                    f"pure LLM mode must never fall back to utility decisions."
                ),
                attempts=0,
                context={"agent_id": agent.agent_id, "kind": "internal"},
            )
        # Compute utility scores for the record only — not for selection.
        util_record = self._fallback.decide(agent, environment, zone_options, prices)
        return ZoneChoice(
            zone_name=chosen,
            utility=util_record.zone_utilities.get(chosen, 0.0),
            zone_utilities=util_record.zone_utilities,
            workplace=agent.job_location,
        )
