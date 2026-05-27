"""CLI coverage for the elastic-looped Transformer entry point."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_cli_help_lists_train_elt_command():
    completed = subprocess.run(
        [sys.executable, "-B", "cli.py", "--help"],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "train-elt" in completed.stdout
    assert "elastic-looped Transformer" in completed.stdout
