"""Session inventory writer for the radixrehome registry feed.

When ``SGLANG_SESSION_INVENTORY_DIR`` is set to a non-empty path (e.g.
``/var/lib/radixrehome``), the tokenizer manager upserts one registry entry
per finished generate request that carries a routing key. Entries match the
radixrehome daemon contract::

    {
      "<session_id>": {
        "tokens": <int>,
        "prompt_ids_file": "<dir>/sessions/<id>.ids.json",
        "routing_key": "<string or null>",
        "home_rank": <int dp rank>
      }
    }

``prompt_ids_file`` holds a JSON array of **server-side templated** token
ids (the same ``input_ids`` the engine dispatched). Park-on-demand matches
against the radix tree with those ids; raw client text does not.

Default off (empty dir) so generic builds are bit-identical. Sage compose
sets the env and mounts the volume on both sglang and radixrehome.

Bounds:
  * ``SGLANG_SESSION_INVENTORY_MAX_ENTRIES`` (default 4096) — LRU eviction
    of oldest sessions from the registry + their ids files.
  * ``SGLANG_SESSION_INVENTORY_MAX_TOKENS`` (default 524288) — skip writing
    inventory for prompts above this ceiling (avoid multi-MB dumps every
    turn of a 1M-token session). Same-session overwrite in place is the
    normal multi-turn path under the ceiling.

Writer is single-process (tokenizer_manager / HTTP serving path). File
updates use an exclusive flock + temp-file rename so a concurrent reader
(the radixrehome sidecar) never sees a partial registry.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_MAX_ENTRIES = 4096
DEFAULT_MAX_TOKENS = 524288

# session_id must be a safe filename component (routing keys are free-form).
_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9._:@+-]+")


def sanitize_session_id(session_id: str) -> str:
    """Map an arbitrary routing key to a filesystem-safe id component."""
    cleaned = _SAFE_ID_RE.sub("_", session_id.strip())
    # Cap length so paths stay well under NAME_MAX; keep a stable tail hash
    # when truncated so distinct long keys don't collide.
    if len(cleaned) <= 180:
        return cleaned or "empty"
    import hashlib

    digest = hashlib.sha1(session_id.encode("utf-8")).hexdigest()[:12]
    return f"{cleaned[:160]}_{digest}"


def resolve_session_id(
    *,
    routing_key: Optional[str],
    conversation_id: Optional[str] = None,
) -> Optional[str]:
    """Pick the inventory key for a finished request.

    Preference order:
      1. ``routing_key`` (``x-smg-routing-key``) — the affinity pin; session
         id equals the routing key for keyed traffic.
      2. ``conversation_id`` when present (OpenAI conversation tracking).
      3. ``None`` — unkeyed, no conversation id: skip inventory. Migration
         only helps affinity-pinned sessions; unkeyed traffic is round-robin
         and has no stable home rank across turns.
    """
    if routing_key:
        key = str(routing_key).strip()
        if key:
            return key
    if conversation_id:
        key = str(conversation_id).strip()
        if key:
            return key
    return None


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Mode 0o644 so the radixrehome sidecar (non-root uid) can read the
    # registry + ids files written by the sglang container (typically root).
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(data)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.chmod(tmp_name, 0o644)
        os.replace(tmp_name, path)
        # replace may preserve mode on some FS; enforce again on final path
        try:
            os.chmod(path, 0o644)
        except OSError:
            pass
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _atomic_write_json(path: Path, obj: Any) -> None:
    payload = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    _atomic_write_bytes(path, payload)


class SessionInventory:
    """Durable registry + per-session prompt-ids files under ``root``."""

    def __init__(
        self,
        root: str,
        *,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        if not root:
            raise ValueError("SessionInventory root must be a non-empty path")
        self.root = Path(root)
        self.sessions_dir = self.root / "sessions"
        self.registry_path = self.root / "registry.json"
        self.lock_path = self.root / ".registry.lock"
        self.max_entries = max(1, int(max_entries))
        self.max_tokens = max(0, int(max_tokens))
        # In-process LRU of session ids (mirrors on-disk order for eviction).
        self._order: "OrderedDict[str, None]" = OrderedDict()
        self._ensure_layout()
        self._hydrate_order()

    # -- construction helpers -------------------------------------------------

    def _ensure_layout(self) -> None:
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            _atomic_write_json(self.registry_path, {})
        if not self.lock_path.exists():
            self.lock_path.touch()

    def _hydrate_order(self) -> None:
        try:
            reg = self._read_registry_unlocked()
        except Exception as e:  # noqa: BLE001
            logger.warning("session inventory: failed to hydrate registry: %s", e)
            return
        for sid in reg.keys():
            self._order[sid] = None

    def _read_registry_unlocked(self) -> Dict[str, Any]:
        if not self.registry_path.exists():
            return {}
        with open(self.registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return data

    # -- public API -----------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Return a copy of the current registry (for admin introspection)."""
        with self._locked():
            return dict(self._read_registry_unlocked())

    def upsert(
        self,
        *,
        session_id: str,
        input_ids: Sequence[int],
        home_rank: int,
        routing_key: Optional[str] = None,
        tokens: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Write/overwrite one session entry. Returns the entry or None if skipped."""
        if not session_id:
            return None
        ids = list(input_ids) if input_ids is not None else []
        if not ids:
            return None
        n_tokens = int(tokens) if tokens is not None else len(ids)
        if self.max_tokens > 0 and n_tokens > self.max_tokens:
            logger.info(
                "session inventory: skip %s — %d tokens > max_tokens=%d",
                session_id,
                n_tokens,
                self.max_tokens,
            )
            return None
        if home_rank is None:
            return None

        safe = sanitize_session_id(session_id)
        ids_path = self.sessions_dir / f"{safe}.ids.json"
        # ids file first (readers that see the registry entry can open it).
        _atomic_write_json(ids_path, [int(x) for x in ids])

        entry = {
            "tokens": n_tokens,
            "prompt_ids_file": str(ids_path),
            "routing_key": routing_key if routing_key is not None else session_id,
            "home_rank": int(home_rank),
        }

        with self._locked():
            reg = self._read_registry_unlocked()
            # Preserve migration history / cooldown fields the daemon writes.
            prev = reg.get(session_id)
            if isinstance(prev, dict):
                for keep in ("migrations",):
                    if keep in prev and keep not in entry:
                        entry[keep] = prev[keep]
            reg[session_id] = entry

            # LRU touch
            if session_id in self._order:
                self._order.move_to_end(session_id)
            else:
                self._order[session_id] = None

            # Evict oldest beyond capacity
            while len(reg) > self.max_entries and self._order:
                victim, _ = self._order.popitem(last=False)
                victim_entry = reg.pop(victim, None)
                if victim_entry and isinstance(victim_entry, dict):
                    victim_ids = victim_entry.get("prompt_ids_file")
                    if victim_ids:
                        try:
                            Path(victim_ids).unlink(missing_ok=True)
                        except OSError:
                            pass

            _atomic_write_json(self.registry_path, reg)

        return entry

    def maybe_record_finished(
        self,
        *,
        obj: Any,
        input_ids: Optional[Sequence[int]],
        home_rank: Optional[int],
        prompt_tokens: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Convenience hook for the tokenizer finish path.

        Pulls routing_key / conversation_id off the request object. No-ops
        (returns None) when the request is not inventory-eligible.
        """
        if home_rank is None or input_ids is None:
            return None
        routing_key = getattr(obj, "routing_key", None)
        conversation_id = getattr(obj, "conversation_id", None)
        session_id = resolve_session_id(
            routing_key=routing_key, conversation_id=conversation_id
        )
        if session_id is None:
            return None
        try:
            return self.upsert(
                session_id=session_id,
                input_ids=input_ids,
                home_rank=int(home_rank),
                routing_key=routing_key if routing_key else None,
                tokens=prompt_tokens,
            )
        except Exception as e:  # noqa: BLE001
            # Inventory must never break serving.
            logger.warning(
                "session inventory upsert failed for %s: %s", session_id, e, exc_info=True
            )
            return None

    # -- locking --------------------------------------------------------------

    class _LockCtx:
        def __init__(self, path: Path) -> None:
            self.path = path
            self._fh = None

        def __enter__(self):
            self._fh = open(self.path, "a+", encoding="utf-8")
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
            return self

        def __exit__(self, exc_type, exc, tb):
            try:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            finally:
                self._fh.close()
                self._fh = None

    def _locked(self) -> "_LockCtx":
        return SessionInventory._LockCtx(self.lock_path)


def inventory_dir_from_env() -> str:
    """Read the kill-switch env without importing the envs registry.

    Empty / unset → disabled. Used at TokenizerManager init and by tests.
    Prefer ``envs.SGLANG_SESSION_INVENTORY_DIR.get()`` when the registry is
    already imported (production path).
    """
    return (os.environ.get("SGLANG_SESSION_INVENTORY_DIR") or "").strip()


def make_inventory_from_env() -> Optional[SessionInventory]:
    """Construct a SessionInventory when the kill-switch dir is set, else None."""
    root = inventory_dir_from_env()
    if not root:
        return None
    max_entries = int(
        os.environ.get("SGLANG_SESSION_INVENTORY_MAX_ENTRIES") or DEFAULT_MAX_ENTRIES
    )
    max_tokens = int(
        os.environ.get("SGLANG_SESSION_INVENTORY_MAX_TOKENS") or DEFAULT_MAX_TOKENS
    )
    try:
        inv = SessionInventory(root, max_entries=max_entries, max_tokens=max_tokens)
        logger.info(
            "session inventory enabled: root=%s max_entries=%d max_tokens=%d",
            root,
            max_entries,
            max_tokens,
        )
        return inv
    except Exception as e:  # noqa: BLE001
        logger.warning("session inventory disabled (init failed): %s", e, exc_info=True)
        return None
