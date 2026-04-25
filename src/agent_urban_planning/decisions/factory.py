"""Helpers for constructing decision engines and LLM clients."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from agent_urban_planning.core.agents import Agent
from agent_urban_planning.decisions.base import DecisionEngine
from agent_urban_planning.llm.clients import (
    ClaudeCodeClient,
    CodexCliClient,
    LLMEngine,
    LLMPreferenceElicitor,
    MultiProviderClient,
    RetryingClient,
    ZaiCodingClient,
)
from agent_urban_planning.decisions.estimated_utility import EstimatedUtilityEngine
from agent_urban_planning.decisions._legacy_singapore_utility import UtilityEngine

ENGINE_CHOICES = ("utility", "hybrid", "llm", "estimated", "ahlfeldt")
LLM_PROVIDER_CHOICES = (
    "zai-coding",
    "anthropic",
    "openai",
    "codex-cli",
    "claude-code",
    "multi",
)

# Per-provider recommended concurrency limits for SINGLE-provider runs
# (each value is the measured sweet spot when that provider is the only
# one on the machine — use these for `--llm-provider <name> --llm-concurrency N`).
PROVIDER_CONCURRENCY_DEFAULTS = {
    "zai-coding": 5,     # Z.ai Coding Plan rate-limits sporadically even at 7; 5 is conservative
    "anthropic": 10,     # Tier 1 safe
    "openai": 10,        # Tier 1 safe
    "codex-cli": 15,     # Subprocess-bound, sweet spot at 15 (~3.2 calls/s)
    "claude-code": 20,   # Subprocess-bound; measured sweet spot at c=20 → ~5.5 calls/s (haiku)
}

# Per-provider concurrency defaults for MULTI-provider runs. Scaled down
# from single-provider sweet spots because running multiple subprocess-
# based providers simultaneously on one machine causes OS-level resource
# contention (subprocess spawn serialization, fd limits, process
# scheduling variance).
#
# Measured on a Mac with 24 cores, haiku model, N=60 per run, 3 runs each:
#   * claude-code alone c=20   → mean 5.48/s (baseline, stable)
#   * multi naive c=40 (5+15+20) → mean 5.21/s, high variance (2.41–8.82/s)
#   * multi scaled c=30 (5+10+15) → mean 6.73/s, stable (5.85–8.43/s)
#
# Total target: ≈30 across the three subscription providers (zai + codex
# + claude). Non-subprocess providers (zai-coding is HTTP) keep their solo
# values since they do not contend with OS subprocess resources.
#
# Providers absent from this table fall back to `PROVIDER_CONCURRENCY_DEFAULTS`.
MULTI_PROVIDER_CONCURRENCY_DEFAULTS = {
    "zai-coding": 5,     # HTTP — no subprocess contention, same as solo
    "codex-cli": 10,     # scaled from solo 15 to reduce subprocess contention
    "claude-code": 15,   # scaled from solo 20 to reduce subprocess contention
}


def _multi_default_concurrency(provider: str) -> int:
    """Recommended concurrency for ``provider`` inside a multi-provider run.

    Consults ``MULTI_PROVIDER_CONCURRENCY_DEFAULTS`` first, then falls back
    to ``PROVIDER_CONCURRENCY_DEFAULTS``, then to 10 for unknown providers.
    """
    if provider in MULTI_PROVIDER_CONCURRENCY_DEFAULTS:
        return MULTI_PROVIDER_CONCURRENCY_DEFAULTS[provider]
    return PROVIDER_CONCURRENCY_DEFAULTS.get(provider, 10)


@dataclass
class EngineSetup:
    """Configured decision-engine stack for a simulation run."""

    decision_engine: DecisionEngine
    label: str
    preference_elicitor: Optional[LLMPreferenceElicitor] = None


class AnthropicClient:
    """Anthropic SDK adapter implementing the local LLMClient protocol."""

    def __init__(self, model: Optional[str] = None):
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("Install anthropic SDK: pip install anthropic") from exc

        self.client = anthropic.Anthropic()
        self.model = model or "claude-haiku-4-5-20251001"

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


class OpenAIClient:
    """OpenAI SDK adapter implementing the local LLMClient protocol."""

    def __init__(self, model: Optional[str] = None):
        try:
            import openai
        except ImportError as exc:
            raise ImportError("Install openai SDK: pip install openai") from exc

        self.client = openai.OpenAI()
        self.model = model or "gpt-4o-mini"

    def complete(self, prompt: str, system: str = "") -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=500,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content


def create_llm_client(provider: str, model: Optional[str] = None):
    """Create an LLM client for a supported provider."""
    if provider == "zai-coding":
        return ZaiCodingClient(model=model)
    if provider == "anthropic":
        return AnthropicClient(model=model)
    if provider == "openai":
        return OpenAIClient(model=model)
    if provider == "codex-cli":
        return CodexCliClient(model=model)
    if provider == "claude-code":
        return ClaudeCodeClient(model=model)
    raise ValueError(f"Unsupported LLM provider: {provider}")


def is_provider_available(provider: str) -> bool:
    """Best-effort check for whether a provider can be constructed.

    Looks at environment variables / tools rather than actually probing
    the network. Returns True if the provider *might* work; False if we
    can tell it definitely won't.
    """
    if provider == "zai-coding":
        return bool(os.environ.get("ZAI_API_KEY"))
    if provider == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    if provider == "openai":
        return bool(os.environ.get("OPENAI_API_KEY"))
    if provider == "codex-cli":
        import shutil
        return shutil.which("codex") is not None
    if provider == "claude-code":
        import shutil
        return shutil.which("claude") is not None
    return False


def discover_available_providers() -> list[str]:
    """Return the list of providers whose credentials/tools are present.

    Scans in a deterministic order so the multi-provider client has a
    predictable dispatch pattern across runs.
    """
    order = (
        "zai-coding",
        "codex-cli",
        "claude-code",
        "anthropic",
        "openai",
    )
    return [p for p in order if is_provider_available(p)]


def create_multi_provider_client(
    providers: Optional[list[str]] = None,
    model: Optional[str] = None,
    concurrency_overrides: Optional[dict[str, int]] = None,
    max_retries: int = 5,
    verbose: bool = False,
) -> MultiProviderClient:
    """Build a MultiProviderClient from a list of provider names.

    Args:
        providers: Explicit list of provider names. If None, auto-discovers
            from environment (via ``discover_available_providers``).
        model: Optional model name — applied to all sub-clients that
            accept a ``model`` argument. Use with caution; most providers
            have different model naming conventions.
        concurrency_overrides: Optional per-provider concurrency limits.
            Defaults come from ``MULTI_PROVIDER_CONCURRENCY_DEFAULTS``
            (which is scaled down from solo sweet spots to avoid
            single-machine subprocess contention), falling back to
            ``PROVIDER_CONCURRENCY_DEFAULTS`` for unlisted providers.
        max_retries: Number of full-round retries with exponential backoff
            when all sub-clients fail. Default 5.
        verbose: If True, print which providers were skipped and why.

    Returns:
        A configured ``MultiProviderClient``. Raises ValueError if no
        providers are available.
    """
    if providers is None:
        providers = discover_available_providers()
        if verbose:
            if providers:
                print(f"Multi-provider mode: auto-discovered {providers}")
            else:
                print("Multi-provider mode: no providers available!")

    if not providers:
        raise ValueError(
            "No LLM providers are available. Configure at least one of: "
            "ZAI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, or install the "
            "codex CLI / claude CLI."
        )

    slots: list[tuple[object, int]] = []
    overrides = concurrency_overrides or {}
    for p in providers:
        if not is_provider_available(p):
            if verbose:
                print(f"  Skipping provider '{p}': credentials/tool not found")
            continue
        try:
            client = create_llm_client(p, model=model)
        except Exception as exc:
            if verbose:
                print(f"  Skipping provider '{p}': {exc}")
            continue
        concurrency = overrides.get(p, _multi_default_concurrency(p))
        slots.append((client, concurrency))
        if verbose:
            print(f"  Added '{p}' with concurrency={concurrency}")

    if not slots:
        raise ValueError(
            "No LLM providers could be constructed from the requested list: "
            f"{providers}"
        )

    return MultiProviderClient(slots, max_retries=max_retries, verbose=verbose)


def create_engine_setup(
    mode: str,
    llm_provider: str = "zai-coding",
    model: Optional[str] = None,
    llm_max_retries: int = 3,
    llm_concurrency: int = 10,
    llm_call_retries: int = 5,
    prefs_cache_dir: Optional[str] = ".cache/prefs",
    multi_providers: Optional[list[str]] = None,
    verbose: bool = False,
    ahlfeldt_params=None,
    ahlfeldt_seed: Optional[int] = None,
) -> EngineSetup:
    """Construct the decision-engine configuration for a run.

    Args:
        mode: One of ``utility``, ``hybrid``, or ``llm``.
        llm_provider: Single provider name, or ``"multi"`` to fan out across
            multiple providers.
        multi_providers: When ``llm_provider == "multi"``, an optional
            explicit list of provider names (e.g. ``["zai-coding", "codex-cli"]``).
            If None, auto-discovers from environment.
        llm_concurrency: Per-provider concurrency for single-provider mode.
            For multi mode, the effective concurrency is the sum of
            per-provider defaults (see ``PROVIDER_CONCURRENCY_DEFAULTS``).
        llm_max_retries: Number of times to retry parsing a malformed LLM
            response (parse-level retries). Default 3.
        llm_call_retries: Number of times to retry the underlying network
            call with exponential backoff when the LLM errors (network-
            level retries). Default 5. Only applies to single-provider
            mode — multi-provider mode has its own retry loop inside
            ``MultiProviderClient``.
        verbose: Passed through to multi-provider discovery.
    """
    if mode == "utility":
        return EngineSetup(
            decision_engine=UtilityEngine(),
            label="UtilityEngine",
        )

    if mode == "estimated":
        return EngineSetup(
            decision_engine=EstimatedUtilityEngine(
                coefficients_path=model or "config/estimated_coefficients.json",
                budget_constraint=True,
            ),
            label="EstimatedUtilityEngine",
        )

    if mode == "ahlfeldt":
        if ahlfeldt_params is None:
            raise ValueError(
                "engine=ahlfeldt requires scenario.ahlfeldt_params — the scenario "
                "YAML must declare an ahlfeldt_params block (see docs/berlin-replication.md)."
            )
        from agent_urban_planning.decisions.ahlfeldt_utility import AhlfeldtUtilityEngine
        return EngineSetup(
            decision_engine=AhlfeldtUtilityEngine(
                params=ahlfeldt_params,
                seed=ahlfeldt_seed,
            ),
            label="AhlfeldtUtilityEngine",
        )

    # Build the LLM client — needed for both hybrid and llm modes
    if llm_provider == "multi":
        client = create_multi_provider_client(
            providers=multi_providers,
            model=model,
            max_retries=llm_call_retries,
            verbose=verbose,
        )
        effective_concurrency = client.total_concurrency
        label_provider = f"multi [{', '.join(client.provider_names)}]"
    else:
        raw_client = create_llm_client(llm_provider, model=model)
        client = RetryingClient(
            raw_client,
            max_retries=llm_call_retries,
            provider_name=llm_provider,
        )
        effective_concurrency = llm_concurrency
        label_provider = llm_provider

    if mode == "hybrid":
        from agent_urban_planning.decisions.hybrid_engine import HybridUtilityEngine
        elicitor = LLMPreferenceElicitor(client=client, cache_dir=prefs_cache_dir)
        return EngineSetup(
            decision_engine=HybridUtilityEngine(budget_constraint=True),
            label=f"HybridUtilityEngine ({label_provider})",
            preference_elicitor=elicitor,
        )

    if mode == "llm":
        return EngineSetup(
            decision_engine=LLMEngine(
                client=client,
                max_retries=llm_max_retries,
                concurrency=effective_concurrency,
                provider_name=label_provider,
                verbose=verbose,
            ),
            label=f"LLMEngine ({label_provider}, concurrency={effective_concurrency})",
        )

    raise ValueError(f"Unsupported engine mode: {mode}")


def apply_preference_elicitation(
    agents: list[Agent],
    elicitor: LLMPreferenceElicitor,
    verbose: bool = False,
    concurrency: int = 0,
) -> None:
    """Assign LLM-elicited preferences via async batched dispatch.

    Thin wrapper over ``LLMPreferenceElicitor.elicit_batch`` that assigns
    the returned weights to each agent's ``.preferences`` in place.

    Args:
        agents: Agents to elicit for (modified in-place via ``.preferences``).
        elicitor: Has ``.client`` (LLM client) and disk cache methods.
        verbose: Print progress.
        concurrency: Max parallel LLM calls. 0 = auto (``client.total_concurrency``
            for multi-provider, or 10 for single-provider).
    """
    if not agents:
        return

    weights = elicitor.elicit_batch(
        agents,
        concurrency=concurrency,
        verbose=verbose,
    )
    for agent, pw in zip(agents, weights):
        agent.preferences = pw

    if verbose and hasattr(elicitor.client, "_slots"):
        # Per-provider stats for multi-provider clients.
        for slot in elicitor.client._slots:
            print(
                f"    {slot['name']}: {slot['calls']} calls, "
                f"{slot['errors']} errors",
                flush=True,
            )
        print(flush=True)
