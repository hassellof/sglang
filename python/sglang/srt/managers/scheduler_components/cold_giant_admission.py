"""Cold-giant admission control (OUR patch; no upstream equivalent).

WHY: the decode-aware chunk cap (sibling module
``decode_aware_chunking.py``) only shrinks a giant prefill's per-pass
chunk while SOME rank is decoding -- it is deliberately inert when the
DP group is decode-idle ("a lone giant prefill on an otherwise idle box
runs at full speed and is not taxed"). A cold first-touch giant (e.g. a
fresh ~292K-token agent session) arriving into an idle gap therefore
stalls decode box-wide for its whole drain (~60s on our box), and every
interactive turn that arrives mid-drain waits behind it (observed
2026-08-10; see networks/rigi sage-dp-attention-lockstep-stalls.md
addendum). This module implements the two admission knobs from
networks/rigi/sage-cold-giant-admission-spec.md:

- Knob A, ``SGLANG_COLD_PREFILL_TOKEN_BUDGET`` (per-request): a request
  whose uncached input exceeds the budget is forced through the
  PrefillAdder in budget-sized chunks per pass, regardless of decode
  presence -- the vLLM ``long_prefill_token_threshold`` idea, adapted to
  our DP-joint-step. A giant still gets admitted and progresses; it just
  cannot take more than ``budget`` fresh tokens in one pass.
- Knob B, ``SGLANG_CROSS_RANK_PREFILL_BUDGET`` (cross-rank SUM): caps the
  SUM of uncached prefill tokens being prefilled across all DP ranks in
  the same joint step, so N concurrently-prefilling cold giants cannot
  stack N x into the shared step (the track-9 residual risk; the
  decode-aware cap bounds one rank's contribution, not the sum).

SIGNAL AND STALENESS (Knob B): the per-rank "uncached extend tokens
this step" rides the existing MLP-sync all-gather (tp0_info column 7 of
MLPSyncBatchInfo's local tensor, dp_attn.py) -- a FIELD ADDITION to an
existing collective, NOT a new collective. record_gathered_uncached_extend()
is called on every gather alongside record_gathered_forward_modes() so
the signal can clear. Cross-rank reads are one step stale (the recorded
values are the PREVIOUS iteration's), identical to the decode-aware
cap's documented staleness: at ~1s passes this costs at most one
mis-sized chunk at a giant on/off edge.

COMPOSITION: both knobs are chunk-budget reducers, never admission
gates -- they shrink the pass/request chunk budget via min(), exactly
like the decode-aware cap. They compose with it (independent min()
terms) and with the prefill-delayer (which gates whether a pass prefills
at all, and which chunked-prefill continuation bypasses by design). No
double-counting: each cap only lowers a shared budget variable; min() is
idempotent and order-independent.

HONEST TRADEOFF: Knob A slows a cold giant's total prefill completion in
exchange for keeping interactive turn admission / decode live during the
drain. Knob B bounds the fleet-wide prefill footprint, which can push a
single giant's drain over more passes when OTHER ranks are also
prefilling. Both are off by default (env <= 0), so a swap-in of this
patch is bit-for-bit stock until the operator turns a knob on; the
working values used by the swap-in drill are 1024 (Knob A --
SGLANG_COLD_PREFILL_TOKEN_BUDGET, the decode-aware level, 4096/rank is
the per-rank base chunk on our box so a budget above that is inert) and
16384 (Knob B -- SGLANG_CROSS_RANK_PREFILL_BUDGET, 4 x the base chunk:
the stock fleet-worst-case SUM, an umbrella ceiling that bites when A or
the base chunk lets one rank take more than a small slice of it).

This module is intentionally free of sglang runtime imports (stdlib +
typing only) so the admission decision can be exercised by the pure
python CPU desk harness (see networks/rigi/sage/sglang-admission-desk/)
without a GPU; the scheduler reads the sglang Envs and passes the raw
values in.
"""
from __future__ import annotations

from typing import List, Optional

# Per-DP-rank uncached-extend token totals from the most recent MLP-sync
# all-gather (previous scheduler iteration; same staleness/clearing rule
# as decode_aware_chunking._last_gathered_forward_modes). None until the
# first gather, and forever on deployments without DP-attention MLP sync
# -- treated as "no remote prefill known" by the cross-rank budget.
_last_gathered_uncached_extend: Optional[List[int]] = None


def record_gathered_uncached_extend(totals: List[int]) -> None:
    """Record per-DP-rank uncached-prefill token totals of an MLP-sync gather.

    Called from prepare_mlp_sync_batch_raw on every gather, INCLUDING
    fully-idle ones (local_batch None), so the signal can clear.
    """
    global _last_gathered_uncached_extend
    _last_gathered_uncached_extend = totals


def cross_rank_uncached_sum_excluding(my_rank: int) -> int:
    """Sum of OTHER ranks' recorded in-flight uncached prefill tokens.

    0 when no gather has happened yet or the group is single-rank.
    """
    totals = _last_gathered_uncached_extend
    if not totals or len(totals) <= 1:
        return 0
    return int(sum(totals) - totals[my_rank % len(totals)])


def cold_prefill_token_budget(raw: int, page_size: int) -> Optional[int]:
    """Resolve Knob A to a page-aligned per-request cap.

    ``raw <= 0`` disables the feature (None: stock behavior, bit-for-bit).
    Positive values are aligned DOWN to a page multiple and floored at one
    page: the PrefillAdder truncation path floors trunc_len to a page
    multiple and returns AddReqResult.OTHER at 0, so a sub-page cap would
    stall a chunked prefill instead of shrinking it (same rationale as
    decode_aware_chunk_size).
    """
    if raw <= 0:
        return None
    return max(page_size, raw // page_size * page_size)


def cross_rank_prefill_budget(raw: int) -> Optional[int]:
    """Resolve Knob B. ``raw <= 0`` disables (None). No alignment needed:
    the cap is a SUM bound, not a per-chunk size."""
    if raw <= 0:
        return None
    return raw


def is_cold_giant(uncached_total: int, budget: Optional[int]) -> bool:
    """True when a request's uncached input exceeds the per-request budget."""
    if budget is None:
        return False
    return uncached_total > budget


def cap_request_intake(cand_len: int, budget: Optional[int]) -> int:
    """Knob A, per-request: cap a request's fresh-token intake this pass.

    ``cand_len`` is the request's remaining uncached length (fresh tokens
    it would take this pass). The cap is only binding when the budget is
    enabled; otherwise it returns ``cand_len`` unchanged.
    """
    if budget is None:
        return cand_len
    return cand_len if cand_len <= budget else budget


def cap_cross_rank_chunk(
    chunked_prefill_size: Optional[int],
    cross_budget: Optional[int],
    others_sum: int,
    page_size: int,
) -> Optional[int]:
    """Knob B: shrink this pass's chunk so the fleet-wide uncached-prefill
    SUM (others' in-flight + this rank's chunk) stays under
    SGLANG_CROSS_RANK_PREFILL_BUDGET.

    ``others_sum`` is the one-step-stale sum of OTHER ranks' recorded
    uncached-extend totals (see module docstring). Floored at one page: a
    sub-page cap would stall a chunked prefill (same rationale as
    decode_aware_chunk_size). When the budget is effectively exhausted by
    other ranks (allowed < one page), this rank still gets one page -- the
    strict sum is exceeded by at most a page, matching the one-iteration
    staleness approximation documented above.
    """
    if cross_budget is None or chunked_prefill_size is None:
        return chunked_prefill_size
    allowed = cross_budget - others_sum
    if allowed >= chunked_prefill_size:
        return chunked_prefill_size
    if allowed <= 0:
        return page_size  # floor: never fully starve a chunked prefill
    capped = max(page_size, allowed // page_size * page_size)
    return min(chunked_prefill_size, capped)
