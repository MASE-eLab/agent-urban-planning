"""V5-hierarchical: full-LLM decision engine with clustering + hierarchical prompts.

Replaces the utility function V[i,j] entirely with an LLM call made per cluster
per market iteration. The LLM returns a top-5 ranking with scores for each
stage (stage 1: residence; stage 2: workplace given residence). Scores are
converted to sampling probabilities (via `_scores_to_probs`) and sampled
M = num_agents times to build the empirical `last_choice_probabilities` matrix
consumed by AhlfeldtMarket.

Sampling semantics (prompt_version v5-hierarchical-v2, see softmax-fix change):
  * At `softmax_T >= 1.0` (default): use LLM's validator-normalized top-5
    scores DIRECTLY as sampling probabilities. T=1 is the identity pass-
    through; T>1 offers no additional effect (scores already probabilistic).
  * At `softmax_T < 1.0`: apply `exp(log(scores+ε)/T)` to concentrate mass
    on the LLM's top-1 choice. Useful for ablation.

An earlier iteration composed `normalize → softmax(T=1)` which flattened
[0,1]-scale scores to near-uniform over the top-5 — see the
`_scores_to_probs` docstring for the full bug + fix writeup.

Key design decisions (see `openspec/changes/v5-full-llm-hierarchical/design.md`):
  • Clustering: kmeans on one-hot encoded demographics, K=50 default.
  • Stage 1 issues K LLM calls; stage 2 issues up to K × 5 (per unique
    (cluster, residence) pair observed in stage-1 output).
  • Cache: in-memory dict keyed on
      (cluster_id, stage, residence_name_or_None, prompt_version, price_bucket_tuple)
    backed by JSON files under `cache_dir` for cross-run reuse.
  • Softmax(T=1) over the returned scores; seeded numpy RNG draws M samples.

Cache namespace: `.cache/llm_v5_hierarchical/`.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from agent_urban_planning.core.agents import Agent, persona_summary
from agent_urban_planning.data.loaders import AhlfeldtParams
from agent_urban_planning.decisions.ahlfeldt_abm_engine import AhlfeldtABMEngine
from agent_urban_planning.llm.async_client import AsyncLLMClient
from agent_urban_planning.decisions.base import LocationChoice
from agent_urban_planning.llm.prompts.hierarchical import (
    PROMPT_VERSION,
    build_stage1_prompt,
    build_stage2_prompt,
    validate_top5_response,
)


# Type alias for stage-1 prompt builder signature.
# (persona, zones_info, *, prompt_version) → (sys_msg, user_msg)
Stage1Builder = Callable[..., tuple[str, str]]
# Type alias for response validator signature.
# (raw, allowed_zone_names) → list[(name, score)]
ResponseValidator = Callable[..., list[tuple[str, float]]]


logger = logging.getLogger(__name__)


class AhlfeldtHierarchicalLLMEngine(AhlfeldtABMEngine):
    """LLM-as-decision-maker engine with clustering and two-stage prompts.

    The full V5 / V5 engine. Replaces ``V[i, j]`` entirely with an
    LLM call made per cluster per market iteration. Stage 1 selects a
    residence; stage 2 selects a workplace conditional on residence.
    Returned scores become sampling probabilities (via
    ``_scores_to_probs``) and are sampled ``M = num_agents`` times to
    build the empirical ``last_choice_probabilities`` matrix consumed
    by :class:`AhlfeldtMarket`.

    Cache namespace: ``.cache/llm_v5_hierarchical/``. Most users should
    configure this through :class:`LLMDecisionEngine`.

    Args:
        params: Structural Ahlfeldt parameters.
        llm_client: An LLM client object (``.complete(user, system="")``
            returning a string).
        cluster_k: Number of clusters used in stage-1 prompt grouping.
        clustering_algo: Clustering algorithm; only ``"kmeans"`` is
            currently supported.
        zone_name_map: Optional ``synthetic_id -> real_name`` mapping
            for prompt-side zone naming.
        cache_dir: Directory where stage-1 / stage-2 LLM call results
            are persisted.
        softmax_T: Temperature applied to LLM-returned scores.
        max_retries: Retry budget on parse errors per LLM call.
        prompt_version: String identifier baked into cache keys.
        llm_concurrency: Max parallel LLM calls.
        progress_callback: Optional ``(stage, done, total, retries)``
            callback for progress reporting.
        seed: Optional integer seed.
        prompt_builder_stage1: Optional override for the stage-1
            prompt builder.
        response_validator_stage1: Optional override for the stage-1
            response validator.
        stage2_top_k_residences: Optional cap on stage-2 fan-out (used
            in V5 score-all mode).
        **parent_kwargs: Forwarded to :class:`AhlfeldtABMEngine`.

    Examples:
        >>> import agent_urban_planning as aup
        >>> # Prefer the public wrapper:
        >>> # engine = aup.LLMDecisionEngine(params, llm_client=client,
        >>> #                                response_format="score_all")
    """

    def __init__(
        self,
        params: AhlfeldtParams,
        llm_client,
        *,
        cluster_k: int = 50,
        clustering_algo: str = "kmeans",
        zone_name_map: Optional[dict[str, str]] = None,
        cache_dir: str | Path = ".cache/llm_v5_hierarchical",
        softmax_T: float = 1.0,
        max_retries: int = 3,
        prompt_version: str = PROMPT_VERSION,
        llm_concurrency: int = 15,
        progress_callback: Optional[Callable[[str, int, int, int], None]] = None,
        seed: Optional[int] = None,
        prompt_builder_stage1: Optional[Stage1Builder] = None,
        response_validator_stage1: Optional[ResponseValidator] = None,
        stage2_top_k_residences: Optional[int] = None,
        **parent_kwargs,
    ):
        # Parent: AhlfeldtABMEngine handles num_agents, batch_size, shock_distribution,
        # seed. We skip the shock path in decide_batch but keep the rest.
        parent_kwargs.setdefault("shock_distribution", "frechet")
        super().__init__(params, seed=seed, **parent_kwargs)

        self.llm_client = llm_client
        self.cluster_k = int(cluster_k)
        self.clustering_algo = clustering_algo
        self.zone_name_map = dict(zone_name_map) if zone_name_map else None
        # Reverse map: real_name → synthetic_id for LLM-output decoding.
        self._real_to_synth: dict[str, str] = {}
        if self.zone_name_map:
            self._real_to_synth = {v: k for k, v in self.zone_name_map.items()}
        self.softmax_T = float(softmax_T)
        self.max_retries = int(max_retries)
        self.prompt_version = prompt_version
        self.llm_concurrency = int(llm_concurrency)
        self.progress_callback = progress_callback
        # Stage-1 prompt builder + validator. Default to production V5 top-5.
        # Pass different values for V5 score-all-96 ablation. Stage-2
        # always uses top-5 (see design §D6 in v5-score-all-96-ablation).
        self.prompt_builder_stage1: Stage1Builder = (
            prompt_builder_stage1 if prompt_builder_stage1 is not None
            else build_stage1_prompt
        )
        self.response_validator_stage1: ResponseValidator = (
            response_validator_stage1 if response_validator_stage1 is not None
            else validate_top5_response
        )
        # V5 cost-control knob. `None` = use every residence in the stage-1
        # distribution (legacy top-5 behaviour: stage-1 top-5 → 5
        # stage-2 calls/cluster). Set to an int K to cap stage-2 fan-out at
        # the top-K residences (by stage-1 probability) per cluster — used
        # in score-all-96 mode to prevent 20x LLM-call blowup. Residences
        # outside top-K get the sampler's uniform-workplace fallback.
        self.stage2_top_k_residences: Optional[int] = (
            int(stage2_top_k_residences)
            if stage2_top_k_residences is not None else None
        )
        # Async client for batched LLM calls. Lazy-initialized (so unit tests
        # that never call decide_batch don't spawn a loop).
        self._async_client: Optional[AsyncLLMClient] = None

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._in_memory_cache: dict[tuple, list[tuple[str, float]]] = {}

        # Clustering state (populated on first decide_batch).
        self._cluster_labels: Optional[np.ndarray] = None   # shape (M_types,)
        self._cluster_personas: list[str] = []              # length K
        self._cluster_weights: Optional[np.ndarray] = None  # shape (K,)
        self._cluster_member_counts: Optional[np.ndarray] = None  # shape (K,)
        self._agents_snapshot_ids: set[int] = set()

        # Per-iteration stats (reset each decide_batch).
        self._iter_stats = {
            "n_stage1_calls_made": 0, "n_stage1_cached": 0,
            "n_stage2_calls_made": 0, "n_stage2_cached": 0,
            "n_parse_failures": 0,
        }

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------
    def ensure_clustering(self, agents: list[Agent]) -> None:
        """Build (or reuse) the cluster assignments for the given agent set.

        Idempotent: re-calling with the same agent set is a no-op.
        On first call the agents are one-hot encoded and clustered
        with the configured algorithm (currently only ``kmeans``);
        ``self._cluster_labels``, ``self._cluster_personas``, and
        ``self._cluster_weights`` are populated.

        Args:
            agents: List of :class:`Agent` instances.

        Returns:
            None.

        Raises:
            ValueError: If ``self.clustering_algo`` is not supported.
            RuntimeError: If the cluster weights sum to zero.

        Examples:
            >>> import agent_urban_planning as aup
            >>> # engine = aup.LLMDecisionEngine(params, llm_client=client)
            >>> # engine.ensure_clustering(list(population))
        """
        ids = {int(a.agent_id) for a in agents}
        if self._agents_snapshot_ids == ids and self._cluster_labels is not None:
            return

        # One-hot encode categorical/ordinal fields. Use graceful-None handling.
        feats = _encode_agent_features(agents)  # (M_types, n_feat)
        K = min(self.cluster_k, len(agents))

        if self.clustering_algo == "kmeans":
            labels, centers = _kmeans_simple(feats, K, seed=self.seed)
        else:
            raise ValueError(
                f"clustering_algo={self.clustering_algo!r} not supported; "
                "this engine ships kmeans only. See docs for ablation knobs."
            )
        self._cluster_labels = labels

        # Weights per cluster = sum of member agent weights.
        weights = np.array([float(a.weight) for a in agents], dtype=np.float64)
        K_actual = int(labels.max()) + 1
        cw = np.zeros(K_actual, dtype=np.float64)
        cc = np.zeros(K_actual, dtype=np.int64)
        for i, lab in enumerate(labels):
            cw[int(lab)] += weights[i]
            cc[int(lab)] += 1
        # Normalize to sum to 1 for within-cluster sampling weights.
        total_w = cw.sum()
        if total_w <= 0:
            raise RuntimeError("cluster weights sum to zero")
        cw_norm = cw / total_w
        self._cluster_weights = cw_norm
        self._cluster_member_counts = cc

        # Generate personas: pick the highest-weight agent in each cluster as the
        # persona representative, then format with persona_summary.
        self._cluster_personas = []
        for c in range(K_actual):
            members = [i for i, lab in enumerate(labels) if int(lab) == c]
            if not members:
                self._cluster_personas.append("(empty cluster)")
                continue
            # Pick representative: highest weight within cluster (first wins on ties).
            rep_idx = max(members, key=lambda i: float(agents[i].weight))
            self._cluster_personas.append(persona_summary(agents[rep_idx]))

        self._agents_snapshot_ids = ids
        logger.info(
            "V5 clustering: K=%d, member counts min/max/mean=%d/%d/%.1f",
            K_actual,
            int(cc.min()), int(cc.max()), float(cc.mean()),
        )

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    # Price-bucket width for cache key. Wider buckets → more cache hits but
    # less price sensitivity in the prompt. 20% is a good default: late-iter
    # Q deltas (< 1%) fall into the same bucket; early-iter Q deltas (up to
    # ~10%) also stay in the same bucket about half the time. Observed in
    # development-phase experiments at seed 42.
    _PRICE_BUCKET_WIDTH = 0.20

    def _cache_key(
        self, cluster_id: int, stage: int, residence_name: Optional[str],
        prices: dict[str, float], wages: dict[str, float],
    ) -> tuple:
        w = self._PRICE_BUCKET_WIDTH
        def bucket(d: dict[str, float]) -> tuple:
            return tuple(sorted((z, round(float(v) / w) * w)
                                for z, v in d.items()))
        return (
            int(cluster_id), int(stage), residence_name,
            self.prompt_version,
            bucket(prices), bucket(wages),
        )

    def _key_hash(self, key: tuple) -> str:
        s = json.dumps(key, sort_keys=True, default=str)
        return hashlib.md5(s.encode()).hexdigest()

    def _cache_get(self, key: tuple) -> Optional[list[tuple[str, float]]]:
        if key in self._in_memory_cache:
            return self._in_memory_cache[key]
        path = self.cache_dir / f"{self._key_hash(key)}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text())
                result = [(str(z), float(s)) for z, s in data["top_5"]]
                self._in_memory_cache[key] = result
                return result
            except Exception:
                return None
        return None

    def _cache_put(self, key: tuple, value: list[tuple[str, float]]) -> None:
        self._in_memory_cache[key] = value
        path = self.cache_dir / f"{self._key_hash(key)}.json"
        try:
            path.write_text(json.dumps({
                "top_5": value,
                "prompt_version": self.prompt_version,
            }))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # LLM call helpers
    # ------------------------------------------------------------------
    def _issue_prompt(self, system: str, user: str,
                      allowed_zone_names: set[str],
                      validator: Optional[ResponseValidator] = None,
                      ) -> list[tuple[str, float]]:
        """Call the LLM once (synchronously), validate, retry on parse errors.

        Returns [(zone_name, normalized_score), ...]. Raises on persistent failure.
        Used for single-shot paths (unit tests); prefer `_batch_issue_prompts`
        in the main `decide_batch` codepath.

        `validator` defaults to `validate_top5_response` for backward compat
        with existing callers. Pass a different validator (e.g.,
        `validate_all_scores_response`) for V5 score-all mode.
        """
        if validator is None:
            validator = validate_top5_response
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self.llm_client.complete(user, system=system)
            except Exception as e:
                last_exc = e
                continue
            try:
                parsed = validator(resp, allowed_zone_names)
                return parsed
            except ValueError as e:
                last_exc = e
                self._iter_stats["n_parse_failures"] += 1
                continue
        raise RuntimeError(
            f"LLM call failed after {self.max_retries} retries: {last_exc}"
        )

    def _get_async_client(self) -> AsyncLLMClient:
        if self._async_client is None:
            self._async_client = AsyncLLMClient(
                self.llm_client, concurrency=self.llm_concurrency,
            )
        return self._async_client

    def _batch_issue_prompts(
        self, prompts: list[tuple[str, str]],
        allowed_zone_names: set[str],
        progress_stage_label: str = "",
        validator: Optional[ResponseValidator] = None,
    ) -> list[list[tuple[str, float]]]:
        """Fire all prompts concurrently via `AsyncLLMClient.complete_many`,
        validate each response, retry the malformed/failed ones in follow-up
        batches up to `max_retries`.

        prompts: list of (system, user) tuples.
        validator: defaults to `validate_top5_response`. Pass a different
          validator (e.g., `validate_all_scores_response`) for V5 score-all
          mode. Stage-2 calls always pass the default.
        Returns: list of parsed results, same order as input.
        """
        if validator is None:
            validator = validate_top5_response
        n = len(prompts)
        if n == 0:
            return []

        results: list[Optional[list[tuple[str, float]]]] = [None] * n
        pending_indices = list(range(n))
        async_client = self._get_async_client()

        for attempt in range(self.max_retries):
            if not pending_indices:
                break
            user_prompts = [prompts[i][1] for i in pending_indices]
            system_prompts = [prompts[i][0] for i in pending_indices]

            # Per-call progress via on_progress wrapped in our stage label.
            def _on_progress(done, total, _stage=progress_stage_label,
                             _base=n - len(pending_indices)):
                if self.progress_callback is not None:
                    # `done` here is relative to the current attempt's batch.
                    # We report absolute completed count for this stage.
                    absolute_done = _base + done
                    # "cached" isn't known at batch time — pass 0 here; the
                    # outer decide_batch updates the iter_stats counters
                    # after the batch returns.
                    self.progress_callback(_stage, absolute_done, n, 0)

            responses = async_client.complete_many(
                user_prompts, systems=system_prompts,
                on_progress=_on_progress,
            )

            # Validate each response; on failure add to next-attempt queue.
            next_pending: list[int] = []
            for local_idx, orig_idx in enumerate(pending_indices):
                resp = responses[local_idx]
                try:
                    parsed = validator(resp, allowed_zone_names)
                    results[orig_idx] = parsed
                except ValueError as e:
                    self._iter_stats["n_parse_failures"] += 1
                    next_pending.append(orig_idx)
                except Exception as e:
                    next_pending.append(orig_idx)
            pending_indices = next_pending

        if pending_indices:
            # All retries exhausted for some prompts.
            raise RuntimeError(
                f"LLM call failed after {self.max_retries} retries for "
                f"{len(pending_indices)}/{n} prompts in stage "
                f"{progress_stage_label!r}"
            )
        return results  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # decide_batch — main override
    # ------------------------------------------------------------------
    def decide_batch(
        self,
        agents: list[Agent],
        environment,
        zone_options: list[str],
        prices: dict,
    ) -> list[LocationChoice]:
        """Issue stage-1 + stage-2 LLM calls and sample per-agent (R, W) choices.

        For each cluster, issues one stage-1 LLM call (residence
        ranking / scoring) plus up to ``stage2_top_k_residences``
        stage-2 calls (workplace conditional on residence). Cached
        results are reused across iterations via the in-memory and
        on-disk caches keyed by ``(cluster, stage, residence,
        prompt_version, price_bucket, wage_bucket)``. Returned scores
        are converted to sampling probabilities and sampled
        ``num_agents`` times to populate
        ``self.last_choice_probabilities``.

        Args:
            agents: List of :class:`Agent` instances.
            environment: The :class:`Environment` carrying zones.
            zone_options: Allowed zone names.
            prices: Mapping ``zone -> Q_i``.

        Returns:
            List of :class:`LocationChoice`, one per input agent and
            in the same order.

        Examples:
            >>> import agent_urban_planning as aup
            >>> # engine = aup.LLMDecisionEngine(params, llm_client=client)
            >>> # choices = engine.decide_batch(agents, env, zones, prices)
        """
        if not agents:
            return []

        self.ensure_clustering(agents)

        # Reset per-iter stats.
        self._iter_stats = {
            "n_stage1_calls_made": 0, "n_stage1_cached": 0,
            "n_stage2_calls_made": 0, "n_stage2_cached": 0,
            "n_parse_failures": 0,
        }

        N = len(zone_options)
        zones = list(zone_options)
        synth_to_real = (
            dict(self.zone_name_map) if self.zone_name_map
            else {z: z for z in zones}
        )
        real_names = {synth_to_real[z] for z in zones}
        # For every synthetic zone, the name used in prompts + validation.
        prompt_name_of = {z: synth_to_real[z] for z in zones}
        # Reverse: prompt-name → synthetic id.
        synth_of_prompt_name = {prompt_name_of[z]: z for z in zones}

        # Build zone info (prices, wages, fundamentals).
        wages_source = self._current_wages
        amen_source = self._current_amenity
        def _resolve_B(z):
            if amen_source and z in amen_source and amen_source[z] is not None:
                return float(amen_source[z])
            v = environment.get_zone(z).amenity_B
            return float(v) if v else 1e-6
        def _resolve_A(z):
            v = environment.get_zone(z).productivity_A
            return float(v) if v else 1.0
        def _resolve_w(z):
            v = wages_source.get(z) if wages_source else None
            if v is None:
                v = environment.get_zone(z).wage_observed
            return float(v) if v else 1.0

        zones_info_by_synth: dict[str, dict] = {}
        prices_dict: dict[str, float] = {}
        wages_dict: dict[str, float] = {}
        for z in zones:
            Q = float(prices.get(z, environment.get_zone(z).floor_price_observed))
            w = _resolve_w(z)
            B = _resolve_B(z)
            A = _resolve_A(z)
            zones_info_by_synth[z] = {
                "name": prompt_name_of[z], "Q": Q, "w": w, "B": B, "A": A,
            }
            prices_dict[z] = Q
            wages_dict[z] = w

        # Travel-time matrix (synthetic-id keyed).
        if environment.transport_matrix is not None and environment.transport_matrix_index:
            tt_idx = [environment._matrix_index_map[z] for z in zones]
            tau = environment.transport_matrix[np.ix_(tt_idx, tt_idx)].astype(
                np.float64, copy=False,
            )
        else:
            tau = np.zeros((N, N), dtype=np.float64)
            for i, zi in enumerate(zones):
                for j, zj in enumerate(zones):
                    tau[i, j] = environment.travel_time(zi, zj)

        K = int(self._cluster_weights.shape[0])

        # --- STAGE 1: residence per cluster (batched) ------------------
        stage1_dist_by_cluster: dict[int, list[tuple[str, float]]] = {}
        stage1_synth_dist: dict[int, list[tuple[str, float]]] = {}
        zones_info_list = [zones_info_by_synth[z] for z in zones]

        # Split K clusters into cached vs uncached.
        s1_uncached_clusters: list[int] = []
        s1_uncached_prompts: list[tuple[str, str]] = []
        s1_cache_keys: dict[int, tuple] = {}
        for c in range(K):
            key = self._cache_key(c, 1, None, prices_dict, wages_dict)
            s1_cache_keys[c] = key
            cached = self._cache_get(key)
            if cached is not None:
                stage1_dist_by_cluster[c] = cached
                self._iter_stats["n_stage1_cached"] += 1
            else:
                sys_msg, user_msg = self.prompt_builder_stage1(
                    self._cluster_personas[c], zones_info_list,
                    prompt_version=self.prompt_version,
                )
                s1_uncached_clusters.append(c)
                s1_uncached_prompts.append((sys_msg, user_msg))

        # Batch-dispatch the uncached prompts (stage 1 uses configurable validator).
        if s1_uncached_prompts:
            parsed_list = self._batch_issue_prompts(
                s1_uncached_prompts, real_names,
                progress_stage_label="stage1",
                validator=self.response_validator_stage1,
            )
            for c, parsed in zip(s1_uncached_clusters, parsed_list):
                stage1_dist_by_cluster[c] = parsed
                self._cache_put(s1_cache_keys[c], parsed)
                self._iter_stats["n_stage1_calls_made"] += 1

        # Decode all stage-1 distributions into synthetic-id space.
        for c in range(K):
            stage1_synth_dist[c] = [
                (synth_of_prompt_name[z], s)
                for z, s in stage1_dist_by_cluster[c]
            ]

        # --- STAGE 2: workplace given residence (batched) ---------------
        stage2_dist: dict[tuple[int, str], list[tuple[str, float]]] = {}
        needed_pairs: list[tuple[int, str]] = []
        for c, dist in stage1_synth_dist.items():
            # Optional cap: only query stage-2 for the top-K residences.
            # Used by the V5 score-all-96 variant to prevent stage-2 fan-out
            # from exploding from 5→96 residences per cluster. Residences
            # outside top-K fall back to uniform workplace in the sampler.
            if self.stage2_top_k_residences is not None:
                topk = sorted(dist, key=lambda x: -x[1])[:self.stage2_top_k_residences]
            else:
                topk = dist
            for synth_res, _s in topk:
                needed_pairs.append((c, synth_res))

        s2_uncached_pairs: list[tuple[int, str]] = []
        s2_uncached_prompts: list[tuple[str, str]] = []
        s2_cache_keys: dict[tuple[int, str], tuple] = {}
        for c, synth_res in needed_pairs:
            res_prompt_name = prompt_name_of[synth_res]
            key = self._cache_key(c, 2, res_prompt_name, prices_dict, wages_dict)
            s2_cache_keys[(c, synth_res)] = key
            cached = self._cache_get(key)
            if cached is not None:
                stage2_dist[(c, synth_res)] = cached
                self._iter_stats["n_stage2_cached"] += 1
            else:
                res_idx = zones.index(synth_res)
                wpl_info = [
                    {
                        "name": prompt_name_of[z],
                        "w": zones_info_by_synth[z]["w"],
                        "commute_min": float(tau[res_idx, j]),
                    }
                    for j, z in enumerate(zones)
                ]
                sys_msg, user_msg = build_stage2_prompt(
                    self._cluster_personas[c], res_prompt_name, wpl_info,
                    prompt_version=self.prompt_version,
                )
                s2_uncached_pairs.append((c, synth_res))
                s2_uncached_prompts.append((sys_msg, user_msg))

        if s2_uncached_prompts:
            parsed_list = self._batch_issue_prompts(
                s2_uncached_prompts, real_names,
                progress_stage_label="stage2",
            )
            for (c, synth_res), parsed in zip(s2_uncached_pairs, parsed_list):
                stage2_dist[(c, synth_res)] = parsed
                self._cache_put(s2_cache_keys[(c, synth_res)], parsed)
                self._iter_stats["n_stage2_calls_made"] += 1

        # --- Softmax + MC sampling expand K → M --------------------------
        P_agg, HR_count, HM_count = self._sample_m_from_distributions(
            N, zones, synth_of_prompt_name,
            stage1_synth_dist, stage2_dist,
        )

        # Hand off to the market.
        self.last_choice_probabilities = P_agg.astype(
            self._np_dtype, copy=False,
        )

        # Diagnostics.
        self.last_abm_diagnostics = {
            "num_agents":                   int(self.num_agents),
            "n_clusters":                   K,
            "n_stage1_calls":               self._iter_stats["n_stage1_calls_made"],
            "n_stage1_cached":              self._iter_stats["n_stage1_cached"],
            "n_stage2_calls":               self._iter_stats["n_stage2_calls_made"],
            "n_stage2_cached":              self._iter_stats["n_stage2_cached"],
            "n_parse_failures":             self._iter_stats["n_parse_failures"],
            "n_cells_nonzero":              int((P_agg > 0).sum()),
            "total_cells":                  int(P_agg.size),
            "n_residence_marginal_zero":    int((HR_count == 0).sum()),
            "n_workplace_marginal_zero":    int((HM_count == 0).sum()),
            "HR_min":                       int(HR_count.min()),
            "HR_max":                       int(HR_count.max()),
            "HM_min":                       int(HM_count.min()),
            "HM_max":                       int(HM_count.max()),
            "softmax_T":                    self.softmax_T,
            "prompt_version":               self.prompt_version,
            "cluster_k":                    K,
        }

        # Per-input-agent LocationChoice (population-modal cell).
        flat_mode = int(P_agg.argmax())
        i_mode = flat_mode // N
        j_mode = flat_mode - i_mode * N
        return [
            LocationChoice(
                residence=zones[i_mode],
                workplace=zones[j_mode],
                utility=0.0,
                zone_utilities={},
            )
            for _ in agents
        ]

    # ------------------------------------------------------------------
    def _sample_m_from_distributions(
        self,
        N: int,
        zones: list[str],
        synth_of_prompt_name: dict[str, str],
        stage1_synth_dist: dict[int, list[tuple[str, float]]],
        stage2_dist: dict[tuple[int, str], list[tuple[str, float]]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Expand K-cluster distributions to M MC samples, then aggregate.

        Score → sampling-probability branching (v5-hierarchical-v2 semantics):
        The validator pre-normalizes LLM scores to sum to 1. At softmax_T >= 1.0
        we use those raw probabilities DIRECTLY (T=1 is the no-op identity).
        At softmax_T < 1.0 we apply `exp(log(scores+ε)/T)` renormalization to
        concentrate mass on the LLM's top-ranked zones.

        Previous (buggy) behavior composed `normalize → softmax(scores/T=1)`
        which flattened [0,1]-scale scores into near-uniform over the top-5.
        See openspec/changes/v5-hierarchical-softmax-fix/ for the fix rationale.
        """
        M = int(self.num_agents)
        T = max(float(self.softmax_T), 1e-6)
        K = int(self._cluster_weights.shape[0])
        zone_to_idx = {z: i for i, z in enumerate(zones)}

        rng = np.random.default_rng((self.seed * 2_000_003 + 7) & 0xFFFFFFFF)

        # Per-agent cluster assignment (weighted by cluster_weights).
        cluster_of_m = rng.choice(K, size=M, p=self._cluster_weights)

        # For each cluster, pre-compute sampling probs for residence.
        stage1_probs: dict[int, tuple[np.ndarray, np.ndarray]] = {}  # cluster → (synth_id_array, prob_array)
        for c in range(K):
            dist = stage1_synth_dist.get(c, [])
            if not dist:
                # Empty distribution — fall back to uniform over all zones.
                stage1_probs[c] = (
                    np.array(zones), np.ones(N) / N,
                )
                continue
            names = np.array([z for z, _ in dist])
            scores = np.array([s for _, s in dist], dtype=np.float64)
            probs = _scores_to_probs(scores, T)
            stage1_probs[c] = (names, probs)

        # Pre-compute workplace sampling probs per (cluster, residence).
        stage2_probs: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
        for (c, res_synth), dist in stage2_dist.items():
            if not dist:
                stage2_probs[(c, res_synth)] = (
                    np.array(zones), np.ones(N) / N,
                )
                continue
            # Stage-2 distributions use prompt-name (real) keys; decode to synth.
            names = np.array([synth_of_prompt_name.get(z, z) for z, _ in dist])
            scores = np.array([s for _, s in dist], dtype=np.float64)
            probs = _scores_to_probs(scores, T)
            stage2_probs[(c, res_synth)] = (names, probs)

        HR_count = np.zeros(N, dtype=np.int64)
        HM_count = np.zeros(N, dtype=np.int64)
        P_count = np.zeros((N, N), dtype=np.int64)

        # Sample M agents — group by cluster for efficiency.
        for c in range(K):
            mask = cluster_of_m == c
            m_c = int(mask.sum())
            if m_c == 0:
                continue
            s1_names, s1_probs = stage1_probs[c]
            # Stage 1: draw residences.
            res_idx_in_dist = rng.choice(len(s1_probs), size=m_c, p=s1_probs)
            res_synth_per_agent = s1_names[res_idx_in_dist]

            # For each unique residence picked, sample that sub-batch's workplace.
            unique_res, inverse = np.unique(res_synth_per_agent, return_inverse=True)
            for u_idx, res_synth in enumerate(unique_res):
                sub_mask = inverse == u_idx
                sub_n = int(sub_mask.sum())
                key = (c, str(res_synth))
                if key in stage2_probs:
                    s2_names, s2_probs = stage2_probs[key]
                else:
                    s2_names, s2_probs = np.array(zones), np.ones(N) / N
                wp_idx_in_dist = rng.choice(
                    len(s2_probs), size=sub_n, p=s2_probs,
                )
                wp_synth_per_agent = s2_names[wp_idx_in_dist]
                # Accumulate aggregates.
                res_i = zone_to_idx[str(res_synth)]
                for wp in wp_synth_per_agent:
                    wp_i = zone_to_idx[str(wp)]
                    P_count[res_i, wp_i] += 1
                    HR_count[res_i] += 1
                    HM_count[wp_i] += 1

        P_agg = P_count.astype(np.float64) / float(M)
        return P_agg, HR_count, HM_count

    def cluster_personas(self) -> list[str]:
        """Return a copy of the per-cluster persona strings.

        Each persona is a one-line :func:`persona_summary` of the
        highest-weight agent in the cluster, used as the persona block
        in stage-1 prompts. Useful for diagnostic logging.

        Returns:
            List of persona strings, one per cluster.

        Examples:
            >>> import agent_urban_planning as aup
            >>> # engine.cluster_personas()  # one persona per cluster
        """
        return list(self._cluster_personas)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softmax(scores: np.ndarray, T: float) -> np.ndarray:
    """Numerically-stable softmax with temperature T over raw scores as logits.

    Kept for library use and direct testing; NOT called in the default V5
    sampling path (which uses `_scores_to_probs` instead). See that helper
    for the score-interpretation semantics used by decide_batch.
    """
    x = np.asarray(scores, dtype=np.float64) / max(T, 1e-6)
    x -= x.max()
    e = np.exp(x)
    s = e.sum()
    if s <= 0:
        n = e.size
        return np.ones(n) / n
    return e / s


def _scores_to_probs(scores: np.ndarray, T: float) -> np.ndarray:
    """Convert LLM-returned top-K scores into a sampling distribution.

    The LLM returns values in [0, 1] that the validator has already normalized
    to sum to 1 (so they function as a probability distribution). We therefore
    branch on `T`:

      * T >= 1.0  → return the raw probabilities as-is. This is the identity
        pass-through semantically — we trust the LLM's stated relative
        preferences. Previously the code composed `normalize → softmax(T=1)`
        which flattened [0,1] scores into near-uniform; that was a bug.

      * T < 1.0   → concentrate on the LLM's top choice via a true temperature-
        scaled softmax over log-probabilities: probs ∝ exp(log(score) / T).
        This is the well-known language-model "temperature sampling" form;
        as T → 0 it concentrates on argmax, and T=1 recovers the raw
        probabilities (continuous with the T>=1 branch above).

    Numerical guard: scores of exactly 0 are floored with ε=1e-12 before log
    to avoid -inf. All-zero input falls back to uniform.
    """
    scores = np.asarray(scores, dtype=np.float64)
    total = float(scores.sum())
    if total <= 0:
        n = scores.size
        return np.ones(n) / n
    normalized = scores / total

    if T >= 1.0 - 1e-9:
        return normalized

    # T < 1.0: concentrate via temperature-scaled softmax over log-probs.
    eps = 1e-12
    log_p = np.log(np.maximum(normalized, eps))
    x = log_p / max(T, 1e-6)
    x -= x.max()
    e = np.exp(x)
    s = e.sum()
    if s <= 0:
        n = e.size
        return np.ones(n) / n
    return e / s


def _encode_agent_features(agents: list[Agent]) -> np.ndarray:
    """One-hot encode 10 demographic axes. Missing fields encode as all-zero
    for that axis (graceful handling of non-richer agents)."""
    # Category definitions matching the richer joint labels.
    HH_BUCKETS = (1, 2, 3, 4)
    AGE_BUCKETS = ((24, 34), (34, 44), (44, 54), (54, 64), (64, 99))
    INCOME_BUCKETS = ((0, 1300), (1300, 2400), (2400, 10**6))
    EDU = ("low", "mid", "high")
    MIG = ("none", "EU", "non-EU")
    EMP = ("employed", "self-employed", "unemployed", "retired_or_student")
    TEN = ("owner", "renter")

    def onehot(val, options) -> list[float]:
        return [1.0 if val == o else 0.0 for o in options]

    rows: list[list[float]] = []
    for a in agents:
        row: list[float] = []
        # hh_size
        row.extend(onehot(min(int(a.household_size), 4), HH_BUCKETS))
        # age brackets
        age_oh = [0.0] * len(AGE_BUCKETS)
        for i, (lo, hi) in enumerate(AGE_BUCKETS):
            if lo < int(a.age_head) <= hi:
                age_oh[i] = 1.0
                break
        row.extend(age_oh)
        # income brackets
        inc_oh = [0.0] * len(INCOME_BUCKETS)
        for i, (lo, hi) in enumerate(INCOME_BUCKETS):
            if lo <= float(a.income) < hi:
                inc_oh[i] = 1.0
                break
        row.extend(inc_oh)
        # binary flags
        row.append(1.0 if a.has_children else 0.0)
        row.append(1.0 if a.has_elderly else 0.0)
        row.append(1.0 if a.car_owner else 0.0)
        # richer categoricals (None → all zeros for that field)
        row.extend(onehot(a.education, EDU) if a.education else [0.0] * len(EDU))
        row.extend(onehot(a.migration_background, MIG) if a.migration_background else [0.0] * len(MIG))
        row.extend(onehot(a.employment_status, EMP) if a.employment_status else [0.0] * len(EMP))
        row.extend(onehot(a.tenure, TEN) if a.tenure else [0.0] * len(TEN))
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def _kmeans_simple(
    X: np.ndarray, K: int, seed: Optional[int], max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic k-means with k-means++ initialization.

    Returns (labels[M], centers[K, D]).
    """
    rng = np.random.default_rng(seed if seed is not None else 0)
    M, D = X.shape
    if K >= M:
        # Degenerate: one cluster per agent.
        labels = np.arange(M, dtype=np.int64)
        return labels, X.copy()

    # k-means++ init.
    centers = np.empty((K, D), dtype=X.dtype)
    idx0 = int(rng.integers(0, M))
    centers[0] = X[idx0]
    for c in range(1, K):
        d = np.min(
            ((X[:, None, :] - centers[None, :c, :]) ** 2).sum(axis=2),
            axis=1,
        )
        if d.sum() <= 0:
            # All points identical to selected centers — pick randomly.
            idx = int(rng.integers(0, M))
        else:
            probs = d / d.sum()
            idx = int(rng.choice(M, p=probs))
        centers[c] = X[idx]

    labels = np.zeros(M, dtype=np.int64)
    for _ in range(max_iter):
        # Assign.
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = d2.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        # Update.
        for c in range(K):
            members = X[labels == c]
            if len(members) > 0:
                centers[c] = members.mean(axis=0)
    return labels, centers
