"""Unit test: kv-canary is disabled for DSPARK ragged-verify 'compact'.

The canary's committed-token chain-hash model cannot represent DSPARK
compact-verify's legitimate draft-token SWA writes: it false-flags every such
write (verify_chain_hash, expected_token=-1) and, at high concurrency, the
violation storm OOB-crashes the server (reproduced at bench_serving conc64/128
on sage 2026-08-12). It must therefore be gated off for that config, while
remaining active under 'static' verify.
"""
import os
import unittest
from unittest import mock

from sglang.srt.kv_canary.config import CanaryConfig, CanaryMode
from sglang.srt.server_args import ServerArgs


def _args(**kw):
    base = {
        "model_path": "dummy",
        "kv_canary": "log",
        "speculative_algorithm": "DSPARK",
    }
    base.update(kw)
    return ServerArgs(**base)


class TestCanaryCompactGate(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("SGLANG_RAGGED_VERIFY_MODE", None)

    def test_compact_disables_canary(self):
        os.environ["SGLANG_RAGGED_VERIFY_MODE"] = "compact"
        cfg = CanaryConfig.from_env(_args())
        self.assertEqual(cfg.mode, CanaryMode.NONE)

    def test_static_keeps_canary(self):
        os.environ["SGLANG_RAGGED_VERIFY_MODE"] = "static"
        cfg = CanaryConfig.from_env(_args())
        self.assertEqual(cfg.mode, CanaryMode.LOG)

    def test_non_dspark_keeps_canary_under_compact(self):
        os.environ["SGLANG_RAGGED_VERIFY_MODE"] = "compact"
        cfg = CanaryConfig.from_env(_args(speculative_algorithm="EAGLE"))
        self.assertEqual(cfg.mode, CanaryMode.LOG)

    def test_explicit_none_stays_none(self):
        os.environ["SGLANG_RAGGED_VERIFY_MODE"] = "compact"
        cfg = CanaryConfig.from_env(_args(kv_canary="none"))
        self.assertEqual(cfg.mode, CanaryMode.NONE)


if __name__ == "__main__":
    unittest.main()