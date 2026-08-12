# Phase 5 Cache Patch Resolution Notes

All 5 conflicted patches were manually resolved against the TreeCore rewrite
(#29901). No patches were dropped. Below is what changed and where the logic
moved.

## Structural changes in v0.5.17 (TreeCore rewrite)

- `unified_radix_cache.py` shrunk from ~2136 to ~800 lines (now ~2200 with
  patches). Tree structure, insert/match walks, eviction, and LRU management
  moved to `unified_cache/unified_tree_core.py` (~2275 lines).
- `unified_cache_components/` renamed to `unified_cache/components/`.
- Backup/write-through is now action-based: `BackupKV` actions are built by
  the tree core and executed by `UnifiedRadixCache._apply_cache_action`.
- `_inc_hit_count_and_check` was split into `_maybe_write_backup` (insert
  walk, no hit counting) and `_inc_hit_count` (match walk, counts reuse).

## Patch resolution summary

### 1. pr30393-family-notest.diff (hicache packed/sidecar draft caches)

**8 files conflicted, all resolved.** Applied via `git apply --3way` then
manual conflict resolution. Key merges:

- `scheduler.py`: kept both the rust-server ascend config store init (ours)
  and the `primary_draft_kv_pool` property access (theirs).
- `hybrid_cache_controller.py`: merged pr34002's `sidecar_ok = True` default
  (for pure-MLA follower ranks) with pr30393's `should_backup()` method and
  `backup_thread_func` override.
- `hybrid_pool_assembler.py`: merged `host_size` parameter (DCP support, ours)
  with `mtp_draft_device_pools` parameter (theirs) across all `build_*_stack`
  functions.
- `pool_host/base.py`: kept `dcp_size`/`dcp_rank` params (ours) alongside
  `pool_label` and MTP draft logging (theirs).
- `pool_host/mha.py`: kept `can_use_write_back_jit` staging path (ours) with
  `_resolve_device_transfer_buffers` packed buffer support (theirs).
- `pool_host/mla.py`: kept `dcp_kernel_indices` translation (ours) with
  `host_layer_id`/`device_layer_id` split for draft layers (theirs).
- `unified_radix_cache.py`: kept `release_host_resources` (ours) alongside
  `register_hicache_draft_pools` (theirs).
- `base_spec_worker.py`: kept `__init__` with graph memory/time tracking
  (ours) alongside `HiCacheDraftPlan` and related properties (theirs).

### 2. dsv4-unified-slru-reuse-counting.diff (SLRU reuse counting)

**Rewritten for new file layout.** All changes moved from
`unified_radix_cache.py` to `unified_cache/unified_tree_core.py`.

- `match_prefix` passes `count_reuse=params.req is not None` to
  `_match_prefix_helper`.
- `_match_prefix_helper` accepts `count_reuse` and calls `_inc_hit_count`
  during the walk (both after split and after full match).
- `_inc_hit_count_and_check` split into `_maybe_write_backup` (returns bool,
  no hit counting, used by insert walk) and `_inc_hit_count` (increments
  hit_count, collects backup candidates, used by match walk).
- Backup candidates from the match walk are collected in
  `_pending_match_backups` and drained into `cache_actions` in
  `_match_post_processor`.
- `_add_new_node` sets `hit_count = 1` (creation counts as first hit).

### 3. dsv4-hicache-l2-observability.diff (logging-only)

**Rewritten for new file layout.** Changes split across two files:

- `hybrid_cache_controller.py`: added `_l2_note_alloc_failure` and hooked it
  into `_resolve_pool_transfers_allocation`.
- `unified_radix_cache.py`: added `_l2_backup_ok`, `_l2_loadback_ok`,
  `_l2_backup_fail` counters; `_l2_should_log`, `_l2_exhausted_pool`,
  `_l2_host_state`, `_l2_note_backup_failure` helpers; instrumented
  `_execute_and_commit_kv_backup` (success/failure), `_execute_kv_backup`
  (host eviction shortfall), and `load_back` (success).

### 4. dsv4-hicache-l2-swa-rolling-reclaim.diff (rolling host-SWA reclaim)

**Rewritten for new file layout.** Applied to
`unified_cache/components/swa_component.py` (was
`unified_cache_components/swa_component.py`).

- Added `_host_backed_nodes` OrderedDict registry.
- `_register_host_backed` / `_reclaim_host_swa` methods added.
- Registration hooked into `redistribute_on_node_split`,
  `commit_hicache_transfer(BACKUP_HOST)`, and `_attach_swa_host_value`.
- Unregistration hooked into `evict_component(HOST)`.
- `_reclaim_host_swa` called at end of `drive_host_eviction` when the LRU
  walk hasn't freed enough.
- Adapted `_evict_component_and_detach_lru` call to new signature (requires
  `device_frees`/`host_frees` dicts passed through from caller).

### 5. dsv4-hicache-l2-swa-backup-window.diff (trailing-window SWA backup)

**Rewritten for new file layout.** Changes span 5 files:

- `unified_cache/components/tree_component.py`: added `tail_distance` param
  to `build_hicache_transfers` base signature.
- `unified_cache/components/full_component.py`: same (ignored).
- `unified_cache/components/mamba_component.py`: same (ignored).
- `unified_cache/components/swa_component.py`: added `_backup_window_tokens`,
  `_outside_backup_window`, `_swa_backup_declined` counter; hooked window
  check into `build_hicache_transfers(BACKUP_HOST)`.
- `unified_cache/unified_tree_core.py`: threaded `tail_distance` through
  `_build_backup_kv_action` (stores per-node distances as transient
  `_swa_backup_tail_distance` attribute) and `_build_backup_spec` (reads it
  and passes to component `build_hicache_transfers`).

**Design note**: The original patch threaded `tail_distance` through
`write_backup` and `_maybe_write_backup` on `UnifiedRadixCache`. In the new
architecture, backups are action-based: the tree core builds `BackupKV`
actions that are executed later by the cache. Since `BackupKV` is a frozen
msgspec struct (just `node_ids`), the tail distances are stashed on the
nodes themselves as `_swa_backup_tail_distance` before building the action,
then read by `_build_backup_spec` at execution time.
