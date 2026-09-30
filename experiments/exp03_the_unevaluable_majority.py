"""How much of the corpus a change moves that nobody is in a position to judge.

Ground truth exists for a sample; the pipeline labels everything. Churn needs only two label
sets, so it is measurable on the whole corpus. Correctness needs a human, so it is measurable
only where one looked.

The sweep regenerates the corpus at each coverage rather than extending the shipped ground
truth. The first version of this experiment extended it using the earlier version's own labels
as a stand-in for truth, which made that version correct by construction on every added item
and biased the comparison against the change under test. The generator draws its ground truth
sample from a separate random stream, so raising the coverage changes which items a human
looked at and changes nothing else, and the points on this sweep are comparable.
"""

from __future__ import annotations

from lac.compare import compare
from lac.generator import GeneratorConfig, generate

from _common import print_table, write_result

COVERAGES = (0.002, 0.01, 0.04, 0.1, 0.25, 0.5)
ITEMS = 25_000
SHIPPED_COVERAGE = 0.04


def main() -> None:
    rows, records = [], []
    for coverage in COVERAGES:
        produced = generate(
            GeneratorConfig(items=ITEMS, labelled=max(2, round(coverage * ITEMS)))
        )
        comparison = compare(
            produced.corpus, produced.versions["prompt-v1"], produced.versions["prompt-v2"]
        )
        matrix = comparison.matrix
        record = {
            "coverage": coverage,
            "labelled": matrix.evaluated,
            "discordant": matrix.discordant,
            "unevaluated_changed": matrix.unevaluated_changed,
            "changed": matrix.changed,
            "paired_p": round(comparison.verdict.paired_p, 5),
            "paired_significant": comparison.verdict.paired_significant,
        }
        records.append(record)
        rows.append(
            [
                f"{round(100 * coverage, 1)}%",
                matrix.evaluated,
                matrix.discordant,
                matrix.unevaluated_changed,
                f"{comparison.verdict.paired_p:.4f}",
            ]
        )
    print_table(
        ["coverage", "labelled", "discordant", "unjudgeable moves", "paired p"],
        rows,
        [11, 11, 13, 20, 11],
    )
    print()
    shipped = next(r for r in records if r["coverage"] == SHIPPED_COVERAGE)
    print(
        f"at the shipped coverage of {round(100 * SHIPPED_COVERAGE, 1)}%, "
        f"{shipped['unevaluated_changed']} label moves have no ground truth behind them"
    )
    first_significant = next((r for r in records if r["paired_significant"]), None)
    if first_significant:
        print(
            f"the paired test first resolves this change at "
            f"{round(100 * first_significant['coverage'], 1)}% coverage, which is "
            f"{first_significant['labelled']} labelled items"
        )
    print("a larger labelled sample buys confidence; it does not buy sight of the corpus")
    write_result(
        "exp03_the_unevaluable_majority",
        {
            "items": ITEMS,
            "shipped_coverage": SHIPPED_COVERAGE,
            "records": records,
            "first_significant_coverage": (
                first_significant["coverage"] if first_significant else None
            ),
        },
    )


if __name__ == "__main__":
    main()
