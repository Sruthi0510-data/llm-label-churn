"""What a single accuracy delta leaves out, on the change a team would ship.

The delta is positive, so the version ships. This experiment prints the same run in the terms
the delta was computed from: how many items got better, how many got worse, how many labels
moved across the whole corpus, and how many of those moves nobody is in a position to judge.
"""

from __future__ import annotations

from lac.compare import compare
from lac.labels import load_corpus, load_label_set

from _common import FIXTURES, print_table, write_result

BEFORE, AFTER = "prompt-v1", "prompt-v2"


def main() -> None:
    corpus = load_corpus(FIXTURES / "corpus.json")
    comparison = compare(
        corpus,
        load_label_set(FIXTURES / f"{BEFORE}.json"),
        load_label_set(FIXTURES / f"{AFTER}.json"),
    )
    matrix = comparison.matrix
    summary = comparison.summary()

    print_table(
        ["quantity", "value"],
        [
            ["accuracy before", f"{summary['accuracy_before']:.3f}"],
            ["accuracy after", f"{summary['accuracy_after']:.3f}"],
            ["the delta a slide would carry", f"{summary['accuracy_delta_points']:+} points"],
            ["items that got better", matrix.improved],
            ["items that got worse", matrix.regressed],
            ["net items gained", matrix.net_evaluated_change],
            ["labels that moved, whole corpus", matrix.changed],
            ["of those, with no ground truth", matrix.unevaluated_changed],
            ["labels moved per net item gained", matrix.churn_per_net_gain],
        ],
        [36, 18],
    )
    print()
    print(
        f"the delta is made of {matrix.improved} improvements and {matrix.regressed} "
        f"regressions; the subtraction is the whole of it"
    )
    print(
        f"{matrix.unevaluated_changed} labels moved on items nobody has looked at, which is "
        f"{round(100 * matrix.unevaluated_changed / matrix.total, 1)}% of the corpus"
    )
    write_result("exp01_what_the_delta_hides", {"summary": summary})


if __name__ == "__main__":
    main()
