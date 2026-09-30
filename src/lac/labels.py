"""Items, label sets, and the split that makes the second finding possible.

An enrichment pipeline labels everything and a human labels a sample. Those two populations
are different sizes by orders of magnitude, and almost every evaluation quietly reports on the
smaller one as though it described the larger.

So `LabelSet` carries labels for every item, and ground truth is a separate, partial mapping.
Churn can be measured on the whole corpus, because it only needs two label sets. Correctness
can be measured only where a human looked. The gap between those two populations is not an
inconvenience to be averaged over: it is the set of changes nobody can evaluate, and this
package reports its size rather than pretending it is not there.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lac.errors import DataError


@dataclass(frozen=True)
class Item:
    """One thing being labelled, with enough context for a human to review a regression."""

    item_id: str
    text: str
    segment: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"item_id": self.item_id, "text": self.text, "segment": self.segment}


@dataclass(frozen=True)
class LabelSet:
    """One pipeline version's labels for a corpus, keyed by item id."""

    name: str
    labels: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.name:
            raise DataError("a label set needs a name")
        if not self.labels:
            raise DataError(f"{self.name}: a label set cannot be empty")
        for item_id, label in self.labels.items():
            if not label:
                raise DataError(f"{self.name}: item {item_id!r} has an empty label")

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def item_ids(self) -> frozenset[str]:
        return frozenset(self.labels)

    @property
    def classes(self) -> frozenset[str]:
        return frozenset(self.labels.values())

    def get(self, item_id: str) -> str:
        try:
            return self.labels[item_id]
        except KeyError as exc:
            raise DataError(f"{self.name} has no label for item {item_id!r}") from exc


@dataclass(frozen=True)
class Corpus:
    """Items, two label sets over them, and the partial ground truth a human produced."""

    items: tuple[Item, ...]
    truth: Mapping[str, str]

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for item in self.items:
            if item.item_id in seen:
                raise DataError(f"duplicate item_id {item.item_id!r}")
            seen.add(item.item_id)
        unknown = sorted(set(self.truth) - seen)
        if unknown:
            raise DataError(f"ground truth names items not in the corpus: {unknown[:3]}")

    @property
    def item_ids(self) -> frozenset[str]:
        return frozenset(item.item_id for item in self.items)

    @property
    def labelled_ids(self) -> frozenset[str]:
        return frozenset(self.truth)

    @property
    def unlabelled_ids(self) -> frozenset[str]:
        return self.item_ids - self.labelled_ids

    @property
    def coverage(self) -> float:
        """Share of the corpus a human has actually looked at."""
        if not self.items:
            return 0.0
        return round(len(self.truth) / len(self.items), 4)

    def by_id(self) -> dict[str, Item]:
        return {item.item_id: item for item in self.items}


def check_alignment(corpus: Corpus, *label_sets: LabelSet) -> None:
    """Refuse a comparison where the label sets do not cover the same items.

    A missing item in one version is not a small problem to be dropped quietly: it means the
    two runs did not process the same input, and every difference downstream could be that
    rather than the prompt.
    """
    for label_set in label_sets:
        missing = corpus.item_ids - label_set.item_ids
        extra = label_set.item_ids - corpus.item_ids
        if missing:
            raise DataError(
                f"{label_set.name} is missing {len(missing)} item(s) the corpus has, "
                f"for example {sorted(missing)[0]!r}"
            )
        if extra:
            raise DataError(
                f"{label_set.name} labels {len(extra)} item(s) the corpus does not have, "
                f"for example {sorted(extra)[0]!r}"
            )


def load_corpus(path: Path) -> Corpus:
    """Read a corpus file: items plus the partial ground truth."""
    payload = _read_json(path)
    for key in ("items", "truth"):
        if key not in payload:
            raise DataError(f"{path} has no {key!r} key")
    items = tuple(
        Item(
            item_id=str(raw["item_id"]),
            text=str(raw["text"]),
            segment=str(raw.get("segment", "")),
        )
        for raw in payload["items"]
    )
    return Corpus(items=items, truth={str(k): str(v) for k, v in payload["truth"].items()})


def load_label_set(path: Path) -> LabelSet:
    """Read one pipeline version's labels."""
    payload = _read_json(path)
    for key in ("name", "labels"):
        if key not in payload:
            raise DataError(f"{path} has no {key!r} key")
    return LabelSet(
        name=str(payload["name"]),
        labels={str(k): str(v) for k, v in payload["labels"].items()},
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise DataError(f"file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataError(f"{path} is not valid JSON: {exc}") from exc


def write_json(path: Path, payload: Any, compact: bool = False) -> Path:
    """Write JSON. Compact for the committed fixtures, indented for anything a human reads.

    The corpus holds tens of thousands of items and nobody reads it by eye, so indenting it
    costs several megabytes in the repository and buys nothing. A run log is read by eye and
    stays indented.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    separators = (",", ":") if compact else None
    body = json.dumps(payload, indent=None if compact else 1, sort_keys=True,
                      separators=separators)
    path.write_text(body + "\n", encoding="utf-8")
    return path


def accuracy(label_set: LabelSet, truth: Mapping[str, str]) -> float:
    """Share of the labelled sample this version gets right. The marginal everything hides in."""
    if not truth:
        return 0.0
    correct = sum(1 for item_id, expected in truth.items() if label_set.get(item_id) == expected)
    return correct / len(truth)


def classes_of(*label_sets: LabelSet, truth: Iterable[str] = ()) -> tuple[str, ...]:
    """Every class appearing anywhere, sorted, so a report's rows are stable."""
    seen: set[str] = set(truth)
    for label_set in label_sets:
        seen |= set(label_set.classes)
    return tuple(sorted(seen))
