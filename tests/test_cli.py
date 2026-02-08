"""Tests for the predted CLI binary.

These tests require the CLI binary to be built first (make cli).
They are skipped if the binary is not found.
"""

import os
import subprocess
import shutil
from pathlib import Path

import pytest

# Try to find the CLI binary
CLI_BIN = shutil.which("predted")
if CLI_BIN is None:
    _local = Path(__file__).resolve().parent.parent / "bin" / "predted"
    if _local.exists() and os.access(str(_local), os.X_OK):
        CLI_BIN = str(_local)

SKIP_CLI = CLI_BIN is None
SKIP_REASON = "predted CLI binary not found (run 'make cli' first)"


def run_cli(*args: str, stdin: str = "") -> subprocess.CompletedProcess:
    """Run the predted CLI with the given arguments."""
    return subprocess.run(
        [CLI_BIN, *args],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=30,
    )


# ---------------------------------------------------------------------------
# Version & help
# ---------------------------------------------------------------------------

@pytest.mark.skipif(SKIP_CLI, reason=SKIP_REASON)
class TestCLIBasic:
    def test_version(self):
        result = run_cli("--version")
        assert result.returncode == 0
        assert "0.1.0" in result.stdout or "0.1.0" in result.stderr

    def test_help(self):
        result = run_cli("--help")
        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert "usage" in output.lower() or "predted" in output.lower()


# ---------------------------------------------------------------------------
# Prediction output
# ---------------------------------------------------------------------------

@pytest.mark.skipif(SKIP_CLI, reason=SKIP_REASON)
class TestCLIPrediction:
    def test_two_structures_text_output(self):
        stdin = "((..))\n(())..\n"
        result = run_cli(stdin=stdin)
        assert result.returncode == 0
        # Should produce some numeric output
        lines = result.stdout.strip().split("\n")
        assert len(lines) >= 1

    def test_three_structures_matrix(self):
        stdin = "((..))\n(())..\n...((..))\n"
        result = run_cli(stdin=stdin)
        assert result.returncode == 0
        lines = result.stdout.strip().split("\n")
        # Should produce at least 3 lines of output (3x3 matrix)
        assert len(lines) >= 3

    def test_upper_only(self):
        stdin = "((..))\n(())..\n...((..))\n"
        result = run_cli("--upper-only", stdin=stdin)
        assert result.returncode == 0

    def test_float_output(self):
        stdin = "((..))\n(())..\n"
        result = run_cli("--float", stdin=stdin)
        assert result.returncode == 0
        # Float output should contain decimal points
        assert "." in result.stdout

    def test_empty_input(self):
        result = run_cli(stdin="")
        # Should handle empty input gracefully (exit 0 or 1, no crash)
        assert result.returncode in (0, 1)

    def test_single_structure(self):
        stdin = "((..))\n"
        result = run_cli(stdin=stdin)
        # Single structure → 1x1 matrix (just "0")
        assert result.returncode == 0
