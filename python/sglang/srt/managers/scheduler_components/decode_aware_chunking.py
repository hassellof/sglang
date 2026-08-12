"""Decode-aware adaptive prefill chunking (OUR patch; no upstream equivalent).

WHY: under --enable-dp-attention every scheduler iteration is a JOINT
forward step across all DP attention ranks (prepare_mlp_sync_batch_raw
all-gathers per-rank batch info each step; decode-idle ranks get injected
idle batches), so the joint step's wall-clock is set by the SLOWEST rank.
A single ~1.4M-token chunked prefill on one rank (4096-token chunks at
~1.2 s/step) therefore gates every other rank's decode loop to <1 step/s
for the whole prefill. Observed in production 2026-08-09: decode
throughput on the three sibling ranks collapsed to 0.10-0.48 tok/s for
~7 minutes and recovered on all ranks the second the prefill finished; an
unrelated interactive client waited 600 s for its first byte and was
hard-cancelled.

FIX: when any rank in the DP group has decode work in flight, cap this
pass's chunked-prefill budget at SGLANG_DECODE_AWARE_CHUNK_SIZE tokens
(default 1024) instead of the configured chunked_prefill_size (the
per-rank, already-DP-divided value; 4096/rank on our deployment).
Smaller prefill steps mean proportionally more joint steps per second,
so co-running decodes get ~4x more decode iterations per wall-clock
second at the default sizes. When no decode exists anywhere in the
group, the full configured chunk size is used: a lone giant prefill on
an otherwise idle box runs at full speed and is not taxed.

This is the cross-rank analogue of Sarathi-Serve (OSDI'24) stall-free
batching: Sarathi bounds prefill work per iteration so decode in the
SAME batch is not stalled; here the unit that must not be stalled is the
joint DP-attention step. Upstream has no decode-aware chunk sizing:
v0.5.16's enable_dynamic_chunking is pipeline-parallel-only (gated on
pp_size > 1) and grows chunks LARGER to equalize per-microbatch latency
-- the opposite direction.

SIGNAL AND ITS STALENESS: chunk sizing happens at batch formation
(Scheduler._get_new_batch_prefill_raw), BEFORE this iteration's MLP-sync
all-gather runs, so the cross-rank decode signal is the PREVIOUS
iteration's gather: prepare_mlp_sync_batch_raw records the per-DP-rank
forward modes it already gathers (tp0_info[:, 5], the same list
recv_skipper_forward_mode is derived from) via
record_gathered_forward_modes(). Decode presence is sticky over seconds
while scheduler passes are ~1 s apart, so one step of lag costs at most
one mis-sized chunk at a decode on/off edge (including after an idle
stretch, where the last stored gather may still show the final decode
step: at most one reduced chunk, then the next gather clears it). The
local rank's own decode presence is read fresh from running_batch
instead of the stale gather. DECODE and TARGET_VERIFY both count as
decode: under DSPARK/DFlash speculative decoding the decode-phase batch
is gathered as TARGET_VERIFY (same classification the recv skipper
uses), so spec-decode work is seen without any special-casing -- spec
decode itself is untouched (it runs in decode iterations built from
running_batch, never through the PrefillAdder this cap applies to).

HONEST TRADEOFF: this trades prefill completion time for decode latency
WHEN BOTH COEXIST. A giant prefill now takes ~4x more scheduler passes,
each with per-pass overhead (batch formation, MLP-sync gather, attention
metadata rebuild, launch), so its completion time strictly increases
while any decode is active -- that is the point, but it is not free.
Medium prompts (1-4k fresh tokens) also pick up 1-3 extra passes of TTFT
while decode is active. It does NOT solve:
- admission fairness: a storm of many separate prefills still queues
  decode behind chunk passes (that is the prefill_delayer's job; the cap
  composes with it -- chunked-prefill CONTINUATION bypasses the delayer
  by design in add_chunked_req, and this cap changes chunk SIZE, never
  admission);
- the single-chunk floor: decode still stalls for one reduced-chunk pass
  (~0.3 s at 1024 tokens on our box, vs ~1.2 s at 4096);
- in-batch prefill/decode interleaving on non-DP deployments (upstream
  mixed-chunk / Sarathi territory; without DP attention the cross-rank
  convoy this patch attacks does not exist, though the cap still engages
  on the rank's own running decodes).

Interaction notes: max_prefill_tokens is a separate, batch-wide input
budget and is untouched -- the cap only lowers rem_chunk_tokens, and a
mid-flight chunked request already monopolizes rem_chunk_tokens today,
so per-pass batch composition keeps its shape. With enable_mixed_chunk
(off on our deployment) the mixed-decode-token subtraction applies to
the reduced budget too; keep the cap comfortably above the expected
decode batch size there. With SGLANG_SCHEDULER_SKIP_ALL_GATHER no modes
are ever recorded and only the local-decode signal engages.
"""

from __future__ import annotations

from typing import List, Optional

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode

# Forward modes that mean "a decode stream is live on that rank". Matches
# SchedulerRecvSkipper.derive_forward_mode's decode classification:
# TARGET_VERIFY is the decode-phase batch under speculative decoding.
_DECODE_MODE_VALUES = (ForwardMode.DECODE.value, ForwardMode.TARGET_VERIFY.value)

# Per-DP-rank forward modes from the most recent MLP-sync all-gather
# (previous scheduler iteration; see the staleness note above). None until
# the first gather, and forever on deployments that never gather (no DP
# attention) -- treated as "no remote decode known".
_last_gathered_forward_modes: Optional[List[int]] = None


def record_gathered_forward_modes(modes: List[int]) -> None:
    """Record the per-DP-rank forward modes of an MLP-sync all-gather.

    Called from prepare_mlp_sync_batch_raw on every gather, INCLUDING
    fully-idle ones (local_batch None), so the signal can clear -- decode
    must be seen to disappear, not only to appear.
    """
    global _last_gathered_forward_modes
    _last_gathered_forward_modes = modes


def group_decode_present(local_decode_reqs: int) -> bool:
    """True if decode work is known anywhere in the DP group.

    ``local_decode_reqs`` is the CURRENT pass's running decode request
    count on this rank (fresh, no lag); the gathered modes cover every
    rank -- including this one, one step stale -- so a rank's own decode
    is seen immediately and a remote rank's with one iteration of lag.
    """
    if local_decode_reqs > 0:
        return True
    modes = _last_gathered_forward_modes
    if not modes:
        return False
    return any(mode in _DECODE_MODE_VALUES for mode in modes)


def decode_aware_chunk_size(page_size: int) -> Optional[int]:
    """Resolve SGLANG_DECODE_AWARE_CHUNK_SIZE to a usable per-pass cap.

    <= 0 disables the feature entirely (None: stock behavior, bit-for-bit).
    Positive values are aligned DOWN to a page multiple and floored at one
    page: the PrefillAdder truncation path floors trunc_len to a page
    multiple and returns AddReqResult.OTHER at 0, so a sub-page cap would
    stall a chunked prefill instead of shrinking it.
    """
    size = envs.SGLANG_DECODE_AWARE_CHUNK_SIZE.get()
    if size <= 0:
        return None
    return max(page_size, size // page_size * page_size)


def maybe_cap_chunk_size(
    chunked_prefill_size: Optional[int],
    cap: Optional[int],
    local_decode_reqs: int,
) -> Optional[int]:
    """Apply the decode-aware cap to this pass's chunk budget.

    Pass-through (stock behavior) when the feature is disabled, when
    chunked prefill itself is disabled (the cap only shrinks existing
    chunk budgets; it never introduces chunking the operator turned off),
    or when no decode exists anywhere in the group.
    """
    if cap is None or chunked_prefill_size is None:
        return chunked_prefill_size
    if not group_decode_present(local_decode_reqs):
        return chunked_prefill_size
    return min(chunked_prefill_size, cap)
