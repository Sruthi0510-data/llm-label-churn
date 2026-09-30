"""Both tests on both changes, and the two different ways the usual one misleads.

The comparison is paired: every item was labelled by both versions. The two-proportion z test
assumes two independent samples, so it treats the agreement between the versions as noise, and
its standard error is much larger than the design deserves. It therefore loses power.

That is not a subtlety. On the second change here the paired test finds a real improvement and
the unpaired test does not, so a team running the test their stats memory offers throws the
improvement away and keeps the prompt that was worse.
"""

from __future__ import annotations

from lac.compare import compare
from lac.labels import load_corpus, load_label_set

from _common import FIXTURES, print_table, write_result

PAIRS = (("prompt-v1", "prompt-v2"), ("prompt-v2", "prompt-v3"), ("prompt-v1", "prompt-v3"))


def main() -> None:
    corpus = load_corpus(FIXTURES / "corpus.json")
    rows, records = [], []
    disagreements = 0
    for before, after in PAIRS:
        comparison = compare(
            corpus,
            load_label_set(FIXTURES / f"{before}.json"),
            load_label_set(FIXTURES / f"{after}.json"),
        )
        summary = comparison.summary()
        records.append({"before": before, "after": after, **summary})
        if summary["tests_disagree"]:
            disagreements += 1
        rows.append(
            [
                f"{before} to {after}",
                f"{summary['accuracy_delta_points']:+}",
                summary["discordant"],
                f"{summary['paired_p']:.4f}",
                f"{summary['unpaired_p']:.4f}",
                "yes" if summary["tests_disagree"] else "no",
            ]
        )
    print_table(
        ["change", "delta", "discordant", "paired p", "unpaired p", "disagree"],
        rows,
        [28, 9, 13, 12, 13, 10],
    )
    print()
    print(f"the two tests reach different conclusions on {disagreements} of {len(PAIRS)} changes")
    for record in records:
        if record["tests_disagree"]:
            print(
                f"  {record['before']} to {record['after']}: the paired test finds the change "
                f"and the unpaired test does not"
            )
    write_result(
        "exp02_the_wrong_test", {"records": records, "disagreements": disagreements}
    )


if __name__ == "__main__":
    main()
