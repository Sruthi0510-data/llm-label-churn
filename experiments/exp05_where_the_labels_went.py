"""Where the labels actually moved, by class, and the systematic shift an average removes.

A net accuracy change of a handful of items can sit on top of a taxonomy-wide reallocation: one
class draining into its neighbour across thousands of documents. Every downstream filter,
dashboard and join keyed on that class sees the reallocation; the accuracy delta does not
mention it.
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
    churn = matrix.per_class_churn()

    rows, records = [], []
    for name, movement in churn.items():
        net = movement["in"] - movement["out"]
        records.append({"class": name, **movement, "net": net})
        rows.append([name, movement["out"], movement["in"], f"{net:+}"])
    print_table(["class", "labels out", "labels in", "net"], rows, [22, 13, 12, 10])

    biggest = max(records, key=lambda record: abs(record["net"]))
    print()
    print(
        f"{biggest['class']} moves by {biggest['net']:+} labels across the corpus, on a net "
        f"evaluated change of {matrix.net_evaluated_change} items"
    )
    print(
        f"every downstream filter keyed on {biggest['class']} sees that shift; the accuracy "
        f"delta does not mention it"
    )
    write_result(
        "exp05_where_the_labels_went",
        {
            "records": records,
            "largest_class_shift": biggest,
            "net_evaluated_change": matrix.net_evaluated_change,
        },
    )


if __name__ == "__main__":
    main()
