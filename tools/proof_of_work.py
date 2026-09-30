"""Capture this repository's own commands and commit the transcript.

The command list is the README's claims, in the order the README makes them. If
a claim changes, the list changes with it, and the capture fails the build when
a command does not exit the way the README says it does, so all four documented
exit codes are demonstrated rather than asserted.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from capture_terminal import Step, capture  # noqa: E402

PY = sys.executable

STEPS = [
    Step(
        label="the suite, with branch coverage",
        argv=[PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--cov=lac",
              "--cov-branch", "--cov-report=term-missing", "--cov-fail-under=100"],
        max_lines=16,
        tail_lines=4,
    ),
    Step(
        label="the comparison the accuracy delta cannot make: a finding",
        argv=[PY, "-m", "lac.cli", "compare"],
        expect=3,
        max_lines=17,
        tail_lines=4,
    ),
    Step(
        label="where the labels moved, by class",
        argv=[PY, "-m", "lac.cli", "churn"],
        expect=3,
        max_lines=13,
    ),
    Step(
        label="the items that went from right to wrong",
        argv=[PY, "-m", "lac.cli", "regressions"],
        expect=3,
        max_lines=11,
    ),
    Step(
        label="an evaluation set too small for the discordance asked for: a refusal",
        argv=[PY, "-m", "lac.cli", "compare", "--require-discordant", "99999"],
        expect=4,
        max_lines=5,
        tail_lines=2,
    ),
    Step(
        label="a version file that is not there",
        argv=[PY, "-m", "lac.cli", "compare", "--before", "does-not-exist.json"],
        expect=2,
        max_lines=3,
    ),
    Step(
        label="the recorded comparisons, from a run log with no tracker behind it",
        argv=[PY, "-m", "lac.cli", "runs"],
        expect=0,
        max_lines=8,
    ),
    Step(
        label="every published figure re-checked against the prose",
        argv=[PY, "tools/check_numbers.py"],
        max_lines=3,
    ),
]


def main() -> int:
    return capture(
        STEPS,
        title="llm-label-churn: captured by tools/proof_of_work.py, exit codes as returned",
        out_svg=ROOT / "docs" / "img" / "proof-of-work.svg",
        out_json=ROOT / "docs" / "proof-of-work.json",
    )


if __name__ == "__main__":
    raise SystemExit(main())
