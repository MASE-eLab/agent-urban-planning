"""LLM client wrappers, async batching, and per-clearing cache.

Five concrete clients ship with the package:
  * :class:`CodexCliClient` — wraps the local ``codex`` CLI subprocess.
  * :class:`ClaudeCodeClient` — wraps the local ``claude`` CLI subprocess.
  * :class:`ZaiCodingClient` — Anthropic-compatible Z.ai proxy.
  * :class:`AnthropicClient` — direct Anthropic SDK.
  * :class:`OpenAIClient` — direct OpenAI SDK.

Plus:
  * :class:`AsyncLLMClient` — bounded-concurrency wrapper around any client.
  * :class:`LLMCallCache` — per-clearing, price-bucketed cache.
  * :class:`MultiProviderClient` — round-robin / failover composition.

All clients implement the abstract :class:`LLMClient` interface; users can
subclass for custom providers.

Phase 3 of the package extraction populates this subpackage; until then it is
intentionally empty.
"""
