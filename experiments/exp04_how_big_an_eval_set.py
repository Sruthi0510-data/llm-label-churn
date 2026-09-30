"""How large an evaluation set has to be before it can resolve the change you care about.

Only discordant pairs carry information, so the sample size that matters is not the number of
labelled items but the number of items the two versions disagreed about and got differently
right. That is a small fraction of the labelled set, and the conversion between the two is the
number nobody computes when they decide how many items to annotate.
"""

from __future__ import annotations

from lac.compare import compare
from lac.labels import load_corpus, load_label_set
from lac.stats import discordant_pairs_needed

from _common import FIXTURES, print_table, write_result

EFFECTS = (0.55, 0.60, 0.65, 0.70, 0.80)
BEFORE, AFTER = "prompt-v1", "prompt-v2"


def main() -> None:
    corpus = load_corpus(FIXTURES / "corpus.json")
    comparison = compare(
        corpus,
        load_label_set(FIXTURES / f"{BEFORE}.json"),
        load_label_set(FIXTURES / f"{AFTER}.json"),
    )
    matrix = comparison.matrix
    discordance = matrix.discordant / matrix.evaluated

    rows, records = [], []
    for effect in EFFECTS:
        pairs = discordant_pairs_needed(effect)
        labelled = round(pairs / discordance)
        records.append(
            {
                "improving_share_of_flips": effect,
                "discordant_pairs_needed": pairs,
                "labelled_items_needed": labelled,
                "multiple_of_current": round(labelled / matrix.evaluated, 1),
            }
        )
        rows.append(
            [
                f"{round(100 * effect)}%",
                pairs,
                f"{labelled:,}",
                f"{round(labelled / matrix.evaluated, 1)}x",
            ]
        )
    print_table(
        ["improvements among the flips", "pairs needed", "items to label", "of current"],
        rows,
        [30, 15, 17, 13],
    )
    print()
    print(
        f"this corpus produces {matrix.discordant} discordant pairs from "
        f"{matrix.evaluated} labelled items, a rate of "
        f"{round(100 * discordance, 1)}%"
    )
    modest = next(r for r in records if r["improving_share_of_flips"] == 0.60)
    print(
        f"resolving a change where six flips in ten are improvements needs "
        f"{modest['discordant_pairs_needed']} pairs, which is "
        f"{modest['labelled_items_needed']:,} labelled items: "
        f"{modest['multiple_of_current']} times the current set"
    )
    write_result(
        "exp04_how_big_an_eval_set",
        {
            "discordance_rate_pct": round(100 * discordance, 2),
            "current_labelled": matrix.evaluated,
            "current_discordant": matrix.discordant,
            "records": records,
        },
    )


if __name__ == "__main__":
    main()
