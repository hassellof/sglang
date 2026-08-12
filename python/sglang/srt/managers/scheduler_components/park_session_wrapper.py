import logging
from typing import Callable, Optional

from sglang.srt.managers.io_struct import ParkSessionReqInput, ParkSessionReqOutput

logger = logging.getLogger(__name__)


class SchedulerParkSessionWrapper:
    """Handle ParkSessionReqInput on a DP scheduler: force the session's prefix
    chain to L3 storage on demand (park-on-demand; see sage-session-migration-
    spec.md). Each DP scheduler runs this against its own radix tree; the rank
    whose tree holds the prefix backs it up, the rest return 0. The rebalancer
    calls this BEFORE migrating a hot unparked session under write_back."""

    def __init__(self, *, park_session_prefix: Callable[[list], int]) -> None:
        self._park_session_prefix = park_session_prefix

    def handle(self, recv_req: ParkSessionReqInput) -> Optional[ParkSessionReqOutput]:
        try:
            parked = self._park_session_prefix(recv_req.token_ids)
        except Exception as e:  # noqa: BLE001
            logger.warning("park_session_prefix failed: %s", e, exc_info=True)
            return ParkSessionReqOutput(
                success=False, tokens_parked=0, message=f"park failed: {e}"
            )
        return ParkSessionReqOutput(
            success=True,
            tokens_parked=parked,
            message=f"parked {parked} tokens to storage",
        )
