"""LLM client wrappers, async batching, and per-clearing cache.

Concrete clients ship with the package:
  * :class:`CodexCliClient` — wraps the local ``codex`` CLI subprocess.
  * :class:`ClaudeCodeClient` — wraps the local ``claude`` CLI subprocess.
  * :class:`AnthropicClient` — direct Anthropic SDK (uses ``ANTHROPIC_API_KEY``).
  * :class:`OpenAIClient` — direct OpenAI SDK (uses ``OPENAI_API_KEY``).
  * :class:`ZaiCodingClient` — Anthropic-compatible Z.ai proxy.
  * :class:`MultiProviderClient` — round-robin / failover composition.
  * :class:`RetryingClient` — wraps any client with retry logic.

Plus:
  * :class:`AsyncLLMClient` — bounded-concurrency wrapper around any client.
  * :class:`LLMCallCache` — per-clearing, price-bucketed cache.
  * :class:`LLMPreferenceElicitor` — V4 pattern: LLM elicits per-agent
    preference weights for the closed-form choice step.
  * :class:`LLMEngine` — V4-style engine that delegates choice to an LLM
    via prompt+response.

All clients implement the :class:`LLMClient` Protocol; users can write
custom providers by implementing ``.complete(user, system="") -> str``.
"""
from __future__ import annotations

from agent_urban_planning.llm.async_client import AsyncLLMClient
from agent_urban_planning.llm.cache import LLMCallCache
from agent_urban_planning.llm.clients import (
    AnthropicClient,
    ClaudeCodeClient,
    CodexCliClient,
    LLMClient,
    LLMCallFailedError,
    LLMEngine,
    LLMPreferenceElicitor,
    MultiProviderClient,
    OpenAIClient,
    RetryingClient,
    ZaiCodingClient,
)

__all__ = [
    "AnthropicClient",
    "AsyncLLMClient",
    "ClaudeCodeClient",
    "CodexCliClient",
    "LLMCallCache",
    "LLMCallFailedError",
    "LLMClient",
    "LLMEngine",
    "LLMPreferenceElicitor",
    "MultiProviderClient",
    "OpenAIClient",
    "RetryingClient",
    "ZaiCodingClient",
]
