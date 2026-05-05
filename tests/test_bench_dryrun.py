"""Smoke tests for the benchmark CLIs.

We don't actually run the benchmark (that would need a real ChromaDB
and live LLM keys). Instead we verify the scripts:
  - parse and import cleanly
  - expose a --help argparse interface
  - exit 0 on --help

This catches the class of bugs where someone breaks the CLI surface
without realizing it (e.g., renamed a flag, removed an arg, broke an import).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


# Only includes scripts that expose an argparse CLI surface — the rescore_*
# and build_index_bge scripts run with hardcoded paths and have no --help.
ARGPARSE_SCRIPTS = [
    "scripts/rag_bench.py",
    "scripts/bench_multihop.py",
    "scripts/build_knowledge_graph.py",
]


@pytest.mark.parametrize("script", ARGPARSE_SCRIPTS)
def test_script_help_exits_zero(script):
    """python <script> --help must succeed (validates argparse + imports)."""
    path = ROOT / script
    assert path.exists(), f"missing expected script: {script}"
    result = subprocess.run(
        [sys.executable, str(path), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        env={"PYTHONPATH": str(ROOT / "api"), **__import__("os").environ},
    )
    assert result.returncode == 0, (
        f"{script} --help exited {result.returncode}\n"
        f"stdout: {result.stdout[:500]}\n"
        f"stderr: {result.stderr[:500]}"
    )
    assert "usage" in (result.stdout + result.stderr).lower(), \
        f"{script} --help did not print usage"
