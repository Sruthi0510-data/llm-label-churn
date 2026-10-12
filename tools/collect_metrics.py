"""Re-measure every published figure and write docs/metrics.json.

Nothing is typed by hand. The test count and the coverage come from pytest's machine readable
reports rather than from a parsed progress line, and every other figure comes from an
experiment's JSON, which is produced by running the experiment.

Anchors are the other half. A checker that searches for digits passes while the sentence
around them has gone stale, so each metric carries the two or three words a document must use
with the value substituted in. Anchors are short, because a phrase longer than a few words
straddles a wrapped line and then matches nothing while the prose it guards is correct. They
are also specific: "{} questions" would match every sentence in the repository that counts
questions and report a correct figure as stale a dozen times over.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

sys.path.insert(0, str(Path(__file__).resolve().parent))

from render_tables import table_values  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EXPERIMENTS = ROOT / "experiments"
EXPERIMENT_OUTPUT = DOCS / "experiments"
JUNIT = ROOT / "junit.xml"
COVERAGE_JSON = ROOT / "coverage.json"
METRICS_PATH = DOCS / "metrics.json"


def checked_documents() -> list[str]:
    """Every prose document in the repository, found rather than listed.

    An earlier version carried a hardcoded list, and deleting a document from
    the repository then failed the build with "listed as a checked document and
    not present". Removing a document is an ordinary editorial act, and a
    receipts mechanism that treats it as a build failure is a mechanism people
    learn to edit around.

    Discovery makes the set self-maintaining in both directions: a document
    added to docs/ is checked from the moment it exists, and a document deleted
    stops being checked without anyone having to remember a second place.

    Sorted so the output is stable, and relative so the committed metrics file
    does not record the path this happened to be built at.
    """
    found = [ROOT / "README.md"] if (ROOT / "README.md").exists() else []
    found += sorted(p for p in (ROOT / "docs").rglob("*.md") if p.is_file())
    return [str(p.relative_to(ROOT)) for p in found]


ANCHORS: dict[str, list[str]] = {
    "tests_total": ["{} tests,"],
    "coverage_line_pct": ["{}% line"],
    "coverage_branch_pct": ["{}% branch"],
    "corpus_items": ["{} items,"],
    "labelled_items": ["{} of them have"],
    "coverage_pct": ["{}% of the corpus"],
    "accuracy_delta_points": ["delta of {} points", "up {} points"],
    "held_right": ["{} items both versions"],
    "held_wrong": ["{} both got wrong"],
    "regressed": ["{} items got worse"],
    "improved": ["against {} that got"],
    "net_gain": ["net gain of {} items"],
    "labels_changed": ["{} labels moved"],
    "unevaluated_changed": ["{} of those moves"],
    "churn_per_net_gain": ["{} moved for every"],
    "paired_p": ["the paired test returns {}"],
    "unpaired_p": ["the unpaired test returns {}"],
    "v3_accuracy_delta_points": ["worth {} points"],
    "v3_paired_p": ["the paired test says {}"],
    "v3_unpaired_p": ["the unpaired test says {}"],
    "largest_class_shift": ["shifts by {} labels"],
    "largest_class_name": ["the {} class"],
    "discordance_rate_pct": ["discordant rate of {}%"],
    "pairs_needed_modest": ["needs {} discordant pairs"],
    "labelled_needed_modest": ["{} labelled items, which"],
    "eval_set_multiple": ["{} times the current"],
    "first_significant_coverage_pct": ["resolves at {}% coverage"],
    "unjudgeable_at_half_coverage": ["still {} moves"],
}

def run_tests() -> bool:
    """Run pytest with machine readable reports. Returns whether it was green.

    Deliberately does not raise on a red suite, and that is a fix rather than a
    looseness. The suite contains a test that runs the number checker, the
    number checker compares the documents against the figures this script
    writes, and this script is how a stale figure gets corrected. Refusing to
    measure until the suite is green therefore made a stale figure unfixable:
    the only tool that could repair it declined to run until it had already been
    repaired.

    pytest writes its junit and coverage reports whether it passes or fails, so
    the measurement is available either way. A red suite is a fact to print
    loudly, not a reason to refuse to measure.
    """
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
            "--cov=lac", "--cov-branch", f"--cov-report=json:{COVERAGE_JSON}", f"--junitxml={JUNIT}",
        ],
        cwd=ROOT,
        check=False,
    )
    if result.returncode != 0:
        print(
            "  the suite is red; measuring anyway so a stale figure can be "
            "corrected, then run it again"
        )
    return result.returncode == 0


def require_report(path: Path, produced_by: str) -> Path:
    """Fail with the reason rather than a traceback when a report is absent.

    --skip-tests reads the reports a previous pytest run left behind, and those
    reports are not tracked. A fresh checkout that has not run the suite does
    not have them, which is exactly what a continuous integration job that
    builds the artefacts on a second runner is. The FileNotFoundError this used
    to raise named a path and nothing else, and a path looks like a bug in the
    tool rather than a missing step in the caller.
    """
    if not path.exists():
        raise SystemExit(
            f"{path.name} is not here, and --skip-tests reads it instead of "
            f"running the suite. Run `{produced_by}` first, or drop --skip-tests."
        )
    return path


def read_test_count() -> int:
    """Read the test total from the junit report rather than from stdout."""
    root = ET.parse(require_report(JUNIT, "make cover")).getroot()
    suite = root if root.tag == "testsuite" else root[0]
    return int(suite.get("tests", 0))


def read_coverage() -> tuple[float, float]:
    """Read line and branch coverage from the JSON report.

    A report written without --cov-branch carries no branch data, and the
    fallback here used to be 100.0. That is the worst available default: the
    tool published a perfect branch figure that nothing had measured, in a
    repository whose whole argument is that every published figure was. An
    absent measurement is not a passing one, so it refuses instead.
    """
    totals = json.loads(require_report(COVERAGE_JSON, "make cover").read_text())["totals"]
    line = round(float(totals["percent_covered"]), 1)
    branches = totals.get("num_branches", 0)
    if not branches:
        raise SystemExit(
            "coverage.json carries no branch data, so no branch percentage can "
            "be read from it. It was written without --cov-branch."
        )
    return line, round(100.0 * totals.get("covered_branches", 0) / branches, 1)


def run_experiments() -> dict[str, Any]:
    """Run every experiment from scratch and load the JSON each one writes."""
    results: dict[str, Any] = {}
    for script in sorted(EXPERIMENTS.glob("exp*.py")):
        subprocess.run(
            [sys.executable, str(script)],
            cwd=ROOT,
            check=True,
            env={**os.environ, "PYTHONPATH": str(EXPERIMENTS)},
            stdout=subprocess.DEVNULL,
        )
        results[script.stem] = json.loads((EXPERIMENT_OUTPUT / f"{script.stem}.json").read_text())
    return results


def build_metrics(skip_tests: bool) -> dict[str, Any]:
    green = True
    if not skip_tests:
        green = run_tests()
    line_pct, branch_pct = read_coverage()
    experiments = run_experiments()

    hidden = experiments["exp01_what_the_delta_hides"]["summary"]
    tests = experiments["exp02_the_wrong_test"]["records"]
    coverage = experiments["exp03_the_unevaluable_majority"]
    sizing = experiments["exp04_how_big_an_eval_set"]
    classes = experiments["exp05_where_the_labels_went"]

    third = next(r for r in tests if r["before"] == "prompt-v2")
    modest = next(
        r for r in sizing["records"] if r["improving_share_of_flips"] == 0.60
    )
    half = coverage["records"][-1]

    metrics = {
        "tests_total": read_test_count(),
        "coverage_line_pct": line_pct,
        "coverage_branch_pct": branch_pct,
        "corpus_items": hidden["items"],
        "labelled_items": hidden["evaluated"],
        "coverage_pct": round(100 * hidden["coverage"], 1),
        "accuracy_delta_points": hidden["accuracy_delta_points"],
        "held_right": hidden["held_right"],
        "held_wrong": hidden["held_wrong"],
        "regressed": hidden["regressed"],
        "improved": hidden["improved"],
        "net_gain": hidden["net_evaluated_change"],
        "labels_changed": hidden["changed"],
        "unevaluated_changed": hidden["unevaluated_changed"],
        "churn_per_net_gain": hidden["churn_per_net_gain"],
        "paired_p": round(hidden["paired_p"], 4),
        "unpaired_p": round(hidden["unpaired_p"], 4),
        "v3_accuracy_delta_points": third["accuracy_delta_points"],
        "v3_paired_p": round(third["paired_p"], 4),
        "v3_unpaired_p": round(third["unpaired_p"], 4),
        "largest_class_shift": classes["largest_class_shift"]["net"],
        "largest_class_name": classes["largest_class_shift"]["class"],
        "discordance_rate_pct": sizing["discordance_rate_pct"],
        "pairs_needed_modest": modest["discordant_pairs_needed"],
        "labelled_needed_modest": modest["labelled_items_needed"],
        "eval_set_multiple": modest["multiple_of_current"],
        "first_significant_coverage_pct": round(
            100 * coverage["first_significant_coverage"], 1
        ),
        "unjudgeable_at_half_coverage": half["unevaluated_changed"],
    }
    return {
        "metrics": metrics,
        "table_values": table_values(),
        "anchors": ANCHORS,
        "checked_documents": checked_documents(),
        "measured_despite_a_red_suite": not green,
        "note": (
            "every value here is produced by running the suite and the experiments; "
            "tools/check_numbers.py fails the build when a document quotes a figure the code "
            "no longer produces"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="re-measure every published figure")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    payload = build_metrics(skip_tests=args.skip_tests)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {METRICS_PATH.relative_to(ROOT)} with {len(payload['metrics'])} metrics")
    for name, value in sorted(payload["metrics"].items()):
        print(f"  {name:<34} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
