# LLM integration

The library ships with multiple LLM client wrappers behind a common
abstraction so V4 and V5 engines can swap providers without code
changes.

## Available clients

| Client | Auth | Notes |
|---|---|---|
| `aup.llm.CodexCliClient` | OAuth (handled by `codex` CLI) | **Recommended for V5 reproduction**. No API key. Sweet spot at concurrency=15 (~6 calls/sec). |
| `aup.llm.ClaudeCodeClient` | OAuth (handled by `claude` CLI) | No API key. Sweet spot at concurrency=20. Default model: haiku. |
| `aup.llm.AnthropicClient` | `ANTHROPIC_API_KEY` env var | Direct Anthropic SDK. |
| `aup.llm.OpenAIClient` | `OPENAI_API_KEY` env var | Direct OpenAI SDK. |
| `aup.llm.ZaiCodingClient` | `ZAI_API_KEY` env var | Anthropic-compatible Z.ai proxy. Concurrency=5. |

## Selecting a provider

```python
import agent_urban_planning as aup

# codex-cli (preferred for V5 paper reproduction)
client = aup.llm.CodexCliClient()

# claude-code
client = aup.llm.ClaudeCodeClient(model="haiku")

# Anthropic SDK
client = aup.llm.AnthropicClient(api_key=os.environ["ANTHROPIC_API_KEY"])

# Pass to any engine that uses an LLM:
engine = aup.LLMDecisionEngine(params, llm_client=client)
```

## Async + caching

V5 issues 50 cluster × 50 iter × 11 calls/iter ≈ 27,500 LLM calls per
baseline run. Concurrency + caching are essential:

```python
from aup.llm import AsyncLLMClient, LLMCallCache

# Async wrapper around any sync client (bounded concurrency)
async_client = AsyncLLMClient(client, concurrency=15)

# Per-clearing, price-bucketed cache
cache = LLMCallCache(bucket_size=0.20)
```

The {class}`LLMDecisionEngine` automatically wraps its client with
{class}`AsyncLLMClient` and uses {class}`LLMCallCache` internally; users
typically don't need to construct these directly.

## Custom client

To plug in a custom LLM provider, implement the
{class}`agent_urban_planning.llm.LLMClient` protocol:

```python
class MyClient:
    total_concurrency = 10  # max concurrent calls

    def complete(self, user: str, system: str = "") -> str:
        """Return raw string response."""
        # ... your logic ...
        return raw_response

# Use it like any built-in:
engine = aup.LLMDecisionEngine(params, llm_client=MyClient())
```

Anything with a `.complete(user, system="")` method returning a string
works.

## Multi-provider failover

For long V5 runs, the {class}`MultiProviderClient` rotates calls across
configured providers and fails over if any provider rate-limits:

```python
from aup.llm import MultiProviderClient

client = MultiProviderClient([
    aup.llm.CodexCliClient(),
    aup.llm.ClaudeCodeClient(),
    aup.llm.ZaiCodingClient(),
])
```

## Cost guidance for V5 baseline + shock

Approximate cost on each provider (Berlin 96-zone scenario, seed 42, 50
iters, ~27,500 calls per run, 2 runs for baseline + shock):

| Provider | Cost (full V5 baseline + shock) |
|---|---|
| codex-cli | $0 (OAuth, free tier sufficient) |
| claude-code | $0 (OAuth, free tier sufficient) |
| Anthropic SDK | ~$30-50 |
| OpenAI SDK | ~$40-70 |
| Z.ai | ~$20-30 |

For paper reproduction, codex-cli is recommended — it's free under OAuth
and produces deterministic-enough outputs for the bundled cache to
replicate.

## See also

- {doc}`/tutorials/03_full_llm_v5` — V5 deep dive
- {doc}`/api/index` — full API reference
