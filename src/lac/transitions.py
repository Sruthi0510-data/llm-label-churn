"""The per-item accounting an accuracy delta sums away.

Two versions labelling the same item can land in one of four places once ground truth is
known: both right, both wrong, right then wrong, or wrong then right. Accuracy is the sum of
the first and the last two counted from one side, and the subtraction destroys the only
distinction anyone acts on.

A fifth state exists and has no correctness attached. Most items in an enrichment corpus have
no ground truth, because a human labelled a sample and the pipeline labelled everything. Those
items can still change label, and that change can still break a downstream join or a dashboard
filter. They are counted separately and never folded into an accuracy figure, because doing so
would report a number about the sample as though it described the corpus.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from lac.labels import Corpus, LabelSet


class Transition(StrEnum):
    """Where one item landed between two versions."""

    HELD_RIGHT = "held_right"
    HELD_WRONG = "held_wrong"
    REGRESSED = "regressed"
    IMPROVED = "improved"
    UNEVALUATED_STABLE = "unevaluated_stable"
    UNEVALUATED_CHANGED = "unevaluated_changed"


EVALUATED = frozenset(
    {
        Transition.HELD_RIGHT,
        Transition.HELD_WRONG,
        Transition.REGRESSED,
        Transition.IMPROVED,
    }
)
"""Transitions on items a human labelled, where correctness is knowable."""

DISCORDANT = frozenset({Transition.REGRESSED, Transition.IMPROVED})
"""The pairs McNemar's test uses. Everything else carries no information about the change."""


@dataclass(frozen=True)
class ItemTransition:
    """One item's before and after, with its verdict."""

    item_id: str
    before: str
    after: str
    truth: str | None
    transition: Transition

    @property
    def changed(self) -> bool:
        return self.before != self.after

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "before": self.before,
            "after": self.after,
            "truth": self.truth,
            "transition": str(self.transition),
            "changed": self.changed,
        }


def classify(before: str, after: str, truth: str | None) -> Transition:
    """Place one item, given its two labels and whatever ground truth exists."""
    if truth is None:
        return (
            Transition.UNEVALUATED_CHANGED
            if before != after
            else Transition.UNEVALUATED_STABLE
        )
    was_right, is_right = before == truth, after == truth
    if was_right and is_right:
        return Transition.HELD_RIGHT
    if was_right and not is_right:
        return Transition.REGRESSED
    if not was_right and is_right:
        return Transition.IMPROVED
    return Transition.HELD_WRONG


@dataclass(frozen=True)
class TransitionMatrix:
    """Every item placed, with the counts and rates the report and the tests both read."""

    before_name: str
    after_name: str
    transitions: tuple[ItemTransition, ...]
    truth: Mapping[str, str]

    @property
    def total(self) -> int:
        return len(self.transitions)

    def count(self, *wanted: Transition) -> int:
        chosen = set(wanted)
        return sum(1 for entry in self.transitions if entry.transition in chosen)

    @property
    def evaluated(self) -> int:
        return self.count(*EVALUATED)

    def count_held_right(self) -> int:
        return self.count(Transition.HELD_RIGHT)

    def count_held_wrong(self) -> int:
        return self.count(Transition.HELD_WRONG)

    @property
    def regressed(self) -> int:
        return self.count(Transition.REGRESSED)

    @property
    def improved(self) -> int:
        return self.count(Transition.IMPROVED)

    @property
    def discordant(self) -> int:
        return self.count(*DISCORDANT)

    @property
    def unevaluated_changed(self) -> int:
        """Items whose label moved with no ground truth to say whether that was good."""
        return self.count(Transition.UNEVALUATED_CHANGED)

    @property
    def changed(self) -> int:
        return sum(1 for entry in self.transitions if entry.changed)

    @property
    def correct_before(self) -> int:
        return self.count(Transition.HELD_RIGHT, Transition.REGRESSED)

    @property
    def correct_after(self) -> int:
        return self.count(Transition.HELD_RIGHT, Transition.IMPROVED)

    @property
    def accuracy_before(self) -> float:
        return 0.0 if self.evaluated == 0 else self.correct_before / self.evaluated

    @property
    def accuracy_after(self) -> float:
        return 0.0 if self.evaluated == 0 else self.correct_after / self.evaluated

    @property
    def accuracy_delta(self) -> float:
        return self.accuracy_after - self.accuracy_before

    @property
    def churn_rate(self) -> float:
        """Share of the whole corpus whose label moved, evaluated or not."""
        return 0.0 if self.total == 0 else self.changed / self.total

    @property
    def net_evaluated_change(self) -> int:
        """Improved minus regressed. This is the entire content of an accuracy delta."""
        return self.improved - self.regressed

    @property
    def churn_per_net_gain(self) -> float | None:
        """How many labels moved for each net item gained. None when nothing was gained.

        The number that ends an argument about whether a change is worth shipping. A version
        that moves four hundred labels to gain twelve items has a blast radius nobody costed.
        """
        if self.net_evaluated_change <= 0:
            return None
        return round(self.changed / self.net_evaluated_change, 1)

    def regressions(self) -> tuple[ItemTransition, ...]:
        """The items a human should look at before shipping."""
        return tuple(e for e in self.transitions if e.transition is Transition.REGRESSED)

    def per_class_churn(self) -> dict[str, dict[str, int]]:
        """Where labels went, by class, so a systematic shift is visible rather than averaged."""
        movement: dict[str, dict[str, int]] = {}
        for entry in self.transitions:
            if not entry.changed:
                continue
            movement.setdefault(entry.before, {"out": 0, "in": 0})["out"] += 1
            movement.setdefault(entry.after, {"out": 0, "in": 0})["in"] += 1
        return dict(sorted(movement.items()))

    def summary(self) -> dict[str, Any]:
        return {
            "before": self.before_name,
            "after": self.after_name,
            "items": self.total,
            "evaluated": self.evaluated,
            "coverage": round(self.evaluated / self.total, 4) if self.total else 0.0,
            "accuracy_before": round(self.accuracy_before, 4),
            "accuracy_after": round(self.accuracy_after, 4),
            "accuracy_delta_points": round(100 * self.accuracy_delta, 2),
            "held_right": self.count(Transition.HELD_RIGHT),
            "held_wrong": self.count(Transition.HELD_WRONG),
            "regressed": self.regressed,
            "improved": self.improved,
            "discordant": self.discordant,
            "net_evaluated_change": self.net_evaluated_change,
            "changed": self.changed,
            "churn_rate_pct": round(100 * self.churn_rate, 2),
            "unevaluated_changed": self.unevaluated_changed,
            "churn_per_net_gain": self.churn_per_net_gain,
        }


def build_matrix(corpus: Corpus, before: LabelSet, after: LabelSet) -> TransitionMatrix:
    """Place every item in the corpus. Item order is sorted so reports are stable."""
    entries = []
    for item in sorted(corpus.items, key=lambda item: item.item_id):
        truth = corpus.truth.get(item.item_id)
        before_label = before.get(item.item_id)
        after_label = after.get(item.item_id)
        entries.append(
            ItemTransition(
                item_id=item.item_id,
                before=before_label,
                after=after_label,
                truth=truth,
                transition=classify(before_label, after_label, truth),
            )
        )
    return TransitionMatrix(
        before_name=before.name,
        after_name=after.name,
        transitions=tuple(entries),
        truth=corpus.truth,
    )
