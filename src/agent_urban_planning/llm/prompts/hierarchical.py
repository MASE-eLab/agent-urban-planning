"""Hierarchical prompt templates + JSON response validator for V5.

Two stages per (cluster, market-iteration):

  Stage 1 — residence: LLM ranks top-5 Ortsteile as residence candidates.
  Stage 2 — workplace given residence: LLM ranks top-5 workplaces
                                        conditional on the chosen residence.

Response schema (both stages):
  {"top_5": [{"zone": "<name>", "score": <0..1>}, ...]}

The `zone` must be a value present in `allowed_zone_names`; otherwise
validation fails. Scores are used as sampling probabilities downstream
(see `AhlfeldtHierarchicalLLMEngine._sample_m_from_distributions` and
`_scores_to_probs` for semantics).

`prompt_version` identifies the prompt template. Bumping it invalidates
the LLM cache (entries are keyed on prompt_version alongside cluster
and price_bucket).

Version history:
  * v5-hierarchical-v1 — initial version; sampling used `softmax(scores, T=1)`
    over the already-normalized top-5 which flattened preferences to
    near-uniform over the top-5. Bug identified during V5.0 post-mortem.
  * v5-hierarchical-v2 — prompt TEXT unchanged, but downstream sampling
    semantics fixed: at T>=1 raw probabilities pass through as-is
    (no double softmax). Cache entries with v1 and v2 may correspond to
    identical LLM responses but semantically different sampled distributions,
    so the version bump prevents accidental mixing.

V5.3 ablation variants (opt-in, NOT default V5 production):
  * build_stage1_prompt_rank_rebalanced / build_stage2_prompt_rank_rebalanced
    — sibling functions using attribute-rank encoding (1..N per attribute
    instead of raw floats) and a validated rebalance instruction that
    neutralizes the observed amenity-seeker bias (r(top5_freq, B):
    +0.64 → +0.06 in the inline pre-flight test). Keyed by the separate
    constant PROMPT_VERSION_V3_RANK so cache entries don't collide with
    the production v2 prompt. Default V5 production uses `build_stage1_prompt`
    / `build_stage2_prompt` above — those functions are unchanged.

V5 ablation variant (opt-in, NOT default V5 production):
  * build_stage1_prompt_score_all — instructs the LLM to emit a score for
    EVERY zone instead of a top-5 ranking. Directly targets the V5.3-
    diagnosed top-5 discretization bottleneck (91/96 zones invisible to
    the market). Paired with `validate_all_scores_response` which tolerates
    partial coverage (default ≥ 60%) and zero-fills missing zones before
    renormalizing. Keyed by PROMPT_VERSION_V4_SCORE_ALL. Keeps V5.3's
    validated rebalance instruction; keeps raw floats (NOT rank encoding).
"""
from __future__ import annotations

import json
import math
from typing import Iterable


PROMPT_VERSION: str = "v5-hierarchical-v2"
PROMPT_VERSION_V3_RANK: str = "v5-hierarchical-v3-rank-rebalance"
PROMPT_VERSION_V4_SCORE_ALL: str = "v5-hierarchical-v4-score-all"

_SYSTEM_MSG: str = (
    "You are a JSON generator representing a Berlin resident making a "
    "residential location decision. Output only valid JSON matching the "
    "requested schema. Choose zones only from the provided list; do not "
    "invent zones or use names outside the list."
)


def _zones_block(zones_info: Iterable[dict], include_commute: bool = False) -> str:
    """Render a block of lines, one per candidate zone."""
    lines = []
    for z in zones_info:
        name = z["name"]
        Q = z.get("Q")
        w = z.get("w")
        B = z.get("B")
        A = z.get("A")
        tau = z.get("commute_min")
        parts = [f"- {name}:"]
        if Q is not None:
            parts.append(f"floor_price={Q:.3f}")
        if w is not None:
            parts.append(f"wage={w:.3f}")
        if B is not None:
            parts.append(f"amenity={B:.3f}")
        if A is not None:
            parts.append(f"productivity={A:.3f}")
        if include_commute and tau is not None:
            parts.append(f"commute_min={tau:.0f}")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def build_stage1_prompt(
    persona: str,
    zones_info: list[dict],
    *,
    prompt_version: str = PROMPT_VERSION,
) -> tuple[str, str]:
    """Stage-1 residence prompt. Returns (system_message, user_message).

    Args:
        persona:  a one-line human-readable persona summary for this
                  cluster's archetype (from `simulator.agent.persona_summary`).
        zones_info: list of dicts with keys `name` (str), `Q` (float),
                  `w` (float), `B` (float), `A` (float). 96 entries expected.
        prompt_version: the active prompt-template version string.

    Example return (user message):
        '''Persona: 38y, 3-person household, has children, mid income, ...

        Candidate Ortsteile (96):
        - Mitte: floor_price=1.120 wage=0.850 amenity=0.670 productivity=1.15
        - Moabit: floor_price=0.980 wage=0.821 amenity=0.540 productivity=0.95
        ...

        Rank the top 5 Ortsteile this persona would consider as RESIDENCE,
        based on affordability, amenities, and fit with their life stage.

        Output strict JSON:
        {"top_5": [{"zone": "<name>", "score": <float in [0,1]>}, ...]}
        Output only the JSON, no other text.'''
    """
    user = (
        f"[prompt_version={prompt_version}]\n"
        f"Persona: {persona}\n\n"
        f"Candidate Ortsteile ({len(zones_info)}):\n"
        f"{_zones_block(zones_info, include_commute=False)}\n\n"
        f"Rank the top 5 Ortsteile this persona would consider as RESIDENCE, "
        f"based on affordability (lower floor_price = cheaper), amenities, "
        f"and fit with their life stage. Score ∈ [0,1]; higher = more preferred.\n\n"
        f'Output strict JSON: {{"top_5": [{{"zone": "<name>", "score": <float in [0,1]>}}, ...]}}\n'
        f"Output only the JSON, no other text."
    )
    return _SYSTEM_MSG, user


def build_stage2_prompt(
    persona: str,
    residence_name: str,
    workplaces_info: list[dict],
    *,
    prompt_version: str = PROMPT_VERSION,
) -> tuple[str, str]:
    """Stage-2 workplace prompt, conditional on chosen residence.

    Args:
        persona: persona summary (same as stage 1).
        residence_name: the Ortsteile name the agent has chosen to live in.
        workplaces_info: list of dicts with `name`, `w` (wage), `commute_min`
                         (travel time from the chosen residence). 96 entries.
    """
    user = (
        f"[prompt_version={prompt_version}]\n"
        f"Persona: {persona}.\n"
        f"This person has chosen to LIVE in {residence_name}. Now rank "
        f"WORKPLACE options.\n\n"
        f"Candidate workplaces ({len(workplaces_info)}) with wage and "
        f"commute time from {residence_name}:\n"
        f"{_zones_block(workplaces_info, include_commute=True)}\n\n"
        f"Rank the top 5 Ortsteile this persona would consider as WORKPLACE, "
        f"based on wage, commute burden (higher commute_min = worse), and "
        f"career considerations. Score ∈ [0,1]; higher = more preferred.\n\n"
        f'Output strict JSON: {{"top_5": [{{"zone": "<name>", "score": <float in [0,1]>}}, ...]}}\n'
        f"Output only the JSON, no other text."
    )
    return _SYSTEM_MSG, user


def _attribute_ranks(
    zones_info: Iterable[dict], attr: str,
) -> dict[str, int]:
    """Return `{zone_name: rank_int}` where rank=1 is the smallest value
    and rank=N is the largest (min-rank tie-breaking, deterministic).

    Zones whose `attr` is missing or non-numeric are assigned the middle
    rank and a ``TypeError`` is avoided so the prompt-builder can survive
    unexpected inputs during tests.
    """
    pairs: list[tuple[str, float]] = []
    for z in zones_info:
        name = z["name"]
        val = z.get(attr)
        if val is None or not isinstance(val, (int, float)):
            pairs.append((name, float("nan")))
        else:
            pairs.append((name, float(val)))
    finite = [(n, v) for n, v in pairs if v == v]  # drop NaNs
    finite.sort(key=lambda x: (x[1], x[0]))        # value asc, name tiebreak
    # min-rank for ties: group equal values and give them the smallest rank.
    ranks: dict[str, int] = {}
    i = 0
    while i < len(finite):
        j = i
        while j < len(finite) and finite[j][1] == finite[i][1]:
            j += 1
        min_rank = i + 1
        for k in range(i, j):
            ranks[finite[k][0]] = min_rank
        i = j
    # NaNs get the median of ranks (or 1 if no finites).
    if len(finite) < len(pairs):
        nan_rank = (len(finite) // 2) + 1 if finite else 1
        for name, val in pairs:
            if name not in ranks:
                ranks[name] = nan_rank
    return ranks


def _rank_hint(rank: int, n: int, high_is_worse: bool) -> str:
    """Short semantic hint at quartile boundaries. For Q: high_is_worse=True
    (expensive=bad). For B/w/A: high_is_worse=False (more=better)."""
    if n <= 0:
        return ""
    frac = rank / n
    if high_is_worse:
        if frac <= 0.25: return " (affordable)"
        if frac >= 0.75: return " (pricey)"
        return ""
    # high_is_better
    if frac <= 0.25: return " (low)"
    if frac >= 0.75: return " (high)"
    return ""


_REBALANCE_INSTRUCTION_STAGE1: str = (
    "IMPORTANT — weight affordability (Q_rank) AT LEAST as heavily as amenity "
    "when ranking. A zone with Q_rank 30 points higher (more expensive "
    "relative to other zones) should drop significantly in your ranking even "
    "if its amenity_rank is slightly higher. Only pick high-Q_rank zones if "
    "they have overwhelmingly better other attributes OR this persona's "
    "income clearly supports them."
)

_REBALANCE_INSTRUCTION_STAGE2: str = (
    "IMPORTANT — weight commute burden (commute_rank) AT LEAST as heavily as "
    "wage when ranking. A workplace with commute_rank 30 points higher (longer "
    "commute relative to other destinations) should drop significantly in your "
    "ranking even if its wage_rank is slightly higher. Only pick high-commute "
    "destinations if they have overwhelmingly better wages OR this persona's "
    "career clearly rewards long commutes."
)


def _zones_block_rank(zones_info: list[dict]) -> str:
    """Render the rank-encoded zones block for stage 1. One line per zone."""
    n = len(zones_info)
    Q_ranks = _attribute_ranks(zones_info, "Q")
    w_ranks = _attribute_ranks(zones_info, "w")
    B_ranks = _attribute_ranks(zones_info, "B")
    A_ranks = _attribute_ranks(zones_info, "A")
    lines: list[str] = []
    for z in zones_info:
        name = z["name"]
        qr = Q_ranks.get(name, 0)
        wr = w_ranks.get(name, 0)
        br = B_ranks.get(name, 0)
        ar = A_ranks.get(name, 0)
        q_hint = _rank_hint(qr, n, high_is_worse=True)
        parts = [
            f"- {name}:",
            f"Q_rank={qr}{q_hint}",
            f"wage_rank={wr}",
            f"amenity_rank={br}",
            f"productivity_rank={ar}",
        ]
        lines.append(" ".join(parts))
    return "\n".join(lines)


def _workplaces_block_rank(workplaces_info: list[dict]) -> str:
    """Render the rank-encoded workplaces block for stage 2. Ranks are
    computed across this origin's candidate destinations (same N as the
    input list)."""
    n = len(workplaces_info)
    w_ranks = _attribute_ranks(workplaces_info, "w")
    c_ranks = _attribute_ranks(workplaces_info, "commute_min")
    lines: list[str] = []
    for z in workplaces_info:
        name = z["name"]
        wr = w_ranks.get(name, 0)
        cr = c_ranks.get(name, 0)
        c_hint = _rank_hint(cr, n, high_is_worse=True)
        parts = [
            f"- {name}:",
            f"wage_rank={wr}",
            f"commute_rank={cr}{c_hint}",
        ]
        lines.append(" ".join(parts))
    return "\n".join(lines)


def build_stage1_prompt_rank_rebalanced(
    persona: str,
    zones_info: list[dict],
    *,
    prompt_version: str = PROMPT_VERSION_V3_RANK,
) -> tuple[str, str]:
    """V5.3 stage-1 prompt: rank-encoded attributes + rebalance instruction.

    Sibling of `build_stage1_prompt`. The existing function is unchanged.

    Attribute encoding: for each of `Q`, `w`, `B`, `A`, values across the
    N-zone input are converted to integer ranks in `[1, N]` (1 = smallest).
    Raw floats are NOT included in the prompt; only ranks + short semantic
    hints at quartile boundaries.

    The "weight affordability AT LEAST as heavily as amenity" instruction
    is included before the JSON-output directive. This instruction was
    validated in a 36-call inline pre-flight: dropped
    `r(top5_freq, B)` from +0.64 to +0.06 and strengthened
    `r(top5_freq, Q)` from −0.25 to −0.40.
    """
    user = (
        f"[prompt_version={prompt_version}]\n"
        f"Persona: {persona}\n\n"
        f"Candidate Ortsteile ({len(zones_info)}), encoded as ranks "
        f"across all zones (1 = smallest value, {len(zones_info)} = largest):\n"
        f"{_zones_block_rank(zones_info)}\n\n"
        f"{_REBALANCE_INSTRUCTION_STAGE1}\n\n"
        f"Rank the top 5 Ortsteile this persona would consider as RESIDENCE. "
        f"Score ∈ [0,1]; higher = more preferred.\n\n"
        f'Output strict JSON: {{"top_5": [{{"zone": "<name>", "score": <float in [0,1]>}}, ...]}}\n'
        f"Output only the JSON, no other text."
    )
    return _SYSTEM_MSG, user


def build_stage2_prompt_rank_rebalanced(
    persona: str,
    residence_name: str,
    workplaces_info: list[dict],
    *,
    prompt_version: str = PROMPT_VERSION_V3_RANK,
) -> tuple[str, str]:
    """V5.3 stage-2 prompt: rank-encoded wage + commute + rebalance instruction.

    Sibling of `build_stage2_prompt`. The existing function is unchanged.

    Wage and commute_min are converted to integer ranks in `[1, N]` across
    the N workplace candidates for this origin. Raw floats are NOT included.
    """
    user = (
        f"[prompt_version={prompt_version}]\n"
        f"Persona: {persona}.\n"
        f"This person has chosen to LIVE in {residence_name}. Now rank "
        f"WORKPLACE options.\n\n"
        f"Candidate workplaces ({len(workplaces_info)}) with wage and commute "
        f"encoded as ranks across destinations (1 = smallest value, "
        f"{len(workplaces_info)} = largest):\n"
        f"{_workplaces_block_rank(workplaces_info)}\n\n"
        f"{_REBALANCE_INSTRUCTION_STAGE2}\n\n"
        f"Rank the top 5 Ortsteile this persona would consider as WORKPLACE. "
        f"Score ∈ [0,1]; higher = more preferred.\n\n"
        f'Output strict JSON: {{"top_5": [{{"zone": "<name>", "score": <float in [0,1]>}}, ...]}}\n'
        f"Output only the JSON, no other text."
    )
    return _SYSTEM_MSG, user


_REBALANCE_INSTRUCTION_SCORE_ALL: str = (
    "IMPORTANT — weight affordability (lower floor_price = better) AT LEAST "
    "as heavily as amenity when scoring. A zone that is 50% more expensive "
    "should receive a noticeably lower score even if its amenity is slightly "
    "higher. Only give high scores to expensive zones if they have "
    "overwhelmingly better other attributes OR this persona's income clearly "
    "supports them."
)


def build_stage1_prompt_score_all(
    persona: str,
    zones_info: list[dict],
    *,
    prompt_version: str = PROMPT_VERSION_V4_SCORE_ALL,
) -> tuple[str, str]:
    """V5 stage-1 prompt: score ALL zones (no top-5 truncation).

    Sibling of `build_stage1_prompt`. The existing function is unchanged.

    Attributes are presented as raw floats (`floor_price`, `wage`, `amenity`,
    `productivity`). The V5.3-validated rebalance instruction is included
    to neutralize the amenity-seeker bias. The JSON schema requests a score
    for every zone in the input list.

    The idea: by removing the 5-of-N selection bottleneck, target zones are
    guaranteed to contribute to the LLM's output distribution. The remaining
    question is whether the LLM produces a coherent distribution or just
    concentrates mass on the same ~5 favourites with 0.01 everywhere else
    (detected by the pre-flight gate's score_spread metric).
    """
    n = len(zones_info)
    user = (
        f"[prompt_version={prompt_version}]\n"
        f"Persona: {persona}\n\n"
        f"Candidate Ortsteile ({n}):\n"
        f"{_zones_block(zones_info, include_commute=False)}\n\n"
        f"{_REBALANCE_INSTRUCTION_SCORE_ALL}\n\n"
        f"Score ALL {n} Ortsteile for this persona as RESIDENCE candidates, "
        f"based on affordability (lower floor_price = cheaper), amenities, "
        f"and fit with their life stage. Each score ∈ [0,1]; higher = more "
        f"preferred. Zones may receive similar scores if they are similarly "
        f"attractive; the scores across all zones DO NOT need to sum to 1 "
        f"(downstream code will normalize).\n\n"
        f'Output strict JSON: {{"scores": [{{"zone": "<name>", "score": <float in [0,1]>}}, ...]}}\n'
        f"The list should contain one entry per zone above (aim for all {n}). "
        f"Output only the JSON, no other text."
    )
    return _SYSTEM_MSG, user


def validate_all_scores_response(
    raw: str,
    allowed_zone_names: set[str] | frozenset[str],
    *,
    min_coverage_ratio: float = 0.60,
) -> list[tuple[str, float]]:
    """Parse and validate a score-all LLM response.

    On success returns a list of `(zone, normalized_score)` tuples covering
    EVERY zone in `allowed_zone_names`. Missing zones are zero-filled.
    Scores are renormalized so the output sums to 1.

    Rules:
    - Response must be JSON `{"scores": [{"zone": <name>, "score": <float>}, ...]}`.
    - Each entry's zone must be in `allowed_zone_names`.
    - Each score must be a finite number in [0, 1].
    - Duplicate zones: first-win (silently drop subsequent entries).
    - If the number of unique valid entries < ceil(min_coverage_ratio * N),
      where N = len(allowed_zone_names), raises ValueError.
    - If the sum of scores is zero after filtering, falls back to uniform.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"response is not valid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError("response must be a JSON object")
    if "scores" not in obj:
        raise ValueError("response missing 'scores' key")
    entries = obj["scores"]
    if not isinstance(entries, list):
        raise ValueError("'scores' must be a list")

    allowed = set(allowed_zone_names)
    n_total = len(allowed)
    min_required = int(math.ceil(min_coverage_ratio * n_total))

    parsed: dict[str, float] = {}
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            raise ValueError(f"entry {i} is not an object")
        if "zone" not in e or "score" not in e:
            raise ValueError(f"entry {i} missing 'zone' or 'score'")
        name = e["zone"]
        if not isinstance(name, str):
            raise ValueError(f"entry {i} zone is not a string")
        if name not in allowed:
            raise ValueError(f"entry {i} zone {name!r} not in allowed zone list")
        try:
            s = float(e["score"])
        except (TypeError, ValueError):
            raise ValueError(f"entry {i} score is not numeric: {e['score']!r}")
        if not math.isfinite(s):
            raise ValueError(f"entry {i} score is not finite: {s}")
        if not (0.0 <= s <= 1.0):
            raise ValueError(f"entry {i} score {s} out of range [0, 1]")
        if name in parsed:
            continue  # first-win dedup
        parsed[name] = s

    if len(parsed) < min_required:
        raise ValueError(
            f"score coverage too low: {len(parsed)}/{n_total} zones "
            f"(< min_coverage_ratio={min_coverage_ratio:.2f} → "
            f"need ≥ {min_required})"
        )

    # Zero-fill missing zones and renormalize.
    total = sum(parsed.values())
    if total <= 0:
        # All-zero fallback: uniform over allowed zones.
        uniform = 1.0 / max(n_total, 1)
        return [(z, uniform) for z in sorted(allowed)]

    out: list[tuple[str, float]] = []
    for z in sorted(allowed):
        out.append((z, parsed.get(z, 0.0) / total))
    return out


def validate_top5_response(
    raw: str,
    allowed_zone_names: set[str] | frozenset[str],
) -> list[tuple[str, float]]:
    """Parse and validate an LLM response into `[(zone, score), ...]`.

    Raises `ValueError` with a specific failure reason on any malformed
    input. On success returns a deduplicated list of 1-to-5 (zone, score)
    tuples where every zone is in `allowed_zone_names` and every score
    is a finite float in [0, 1]. Scores are renormalized to sum to 1.
    """
    # Tolerate leading / trailing whitespace + accidental ``` fences.
    text = raw.strip()
    if text.startswith("```"):
        # Strip a ```json ... ``` fence if present.
        text = text.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"response is not valid JSON: {e}") from e

    if not isinstance(obj, dict):
        raise ValueError("response must be a JSON object")
    if "top_5" not in obj:
        raise ValueError("response missing 'top_5' key")
    entries = obj["top_5"]
    if not isinstance(entries, list):
        raise ValueError("'top_5' must be a list")
    if not (1 <= len(entries) <= 20):
        raise ValueError(f"'top_5' must have 1 to 20 items, got {len(entries)}")

    parsed: list[tuple[str, float]] = []
    seen: set[str] = set()
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            raise ValueError(f"entry {i} is not an object")
        if "zone" not in e or "score" not in e:
            raise ValueError(f"entry {i} missing 'zone' or 'score'")
        name = e["zone"]
        score = e["score"]
        if not isinstance(name, str):
            raise ValueError(f"entry {i} zone is not a string")
        if name not in allowed_zone_names:
            raise ValueError(
                f"entry {i} zone {name!r} not in allowed zone list"
            )
        try:
            s = float(score)
        except (TypeError, ValueError):
            raise ValueError(f"entry {i} score is not numeric: {score!r}")
        if not (0.0 <= s <= 1.0):
            raise ValueError(
                f"entry {i} score {s} out of range [0, 1]"
            )
        if name in seen:
            continue  # dedupe silently (first-win)
        seen.add(name)
        parsed.append((name, s))
        if len(parsed) >= 5:
            break

    if not parsed:
        raise ValueError("no valid entries after deduplication")

    # Normalize scores so they sum to 1.0 (softmax input is handled by caller).
    total = sum(s for _, s in parsed)
    if total <= 0:
        # All-zero scores: fall back to uniform.
        n = len(parsed)
        parsed = [(z, 1.0 / n) for z, _ in parsed]
    else:
        parsed = [(z, s / total) for z, s in parsed]

    return parsed
