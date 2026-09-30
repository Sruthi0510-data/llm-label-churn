"""One comparison: the matrix, both tests, and a verdict that names what it is based on."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lac.errors import ConfigError, UnderpoweredError
from lac.labels import Corpus, LabelSet, check_alignment
from lac.runs import Run, RunStore
from lac.stats import (
    Verdict,
    discordant_pairs_needed,
    mcnemar_exact,
    two_proportion_p,
    wilson_interval,
)
from lac.transitions import TransitionMatrix, build_matrix


@dataclass(frozen=True)
class Comparison:
    """Everything one version-to-version comparison established."""

    matrix: TransitionMatrix
    verdict: Verdict
    confidence: float

    @property
    def shippable_on_accuracy_alone(self) -> bool:
        """What a team decides when the only number on the slide is the accuracy delta."""
        return self.matrix.accuracy_delta > 0

    @property
    def supported_by_the_paired_test(self) -> bool:
        return self.verdict.paired_significant and self.matrix.net_evaluated_change > 0

    @property
    def decision_would_differ(self) -> bool:
        """True when the accuracy delta and the paired test point different ways."""
        return self.shippable_on_accuracy_alone != self.supported_by_the_paired_test

    def accuracy_interval(self, after: bool) -> tuple[float, float]:
        correct = self.matrix.correct_after if after else self.matrix.correct_before
        return wilson_interval(correct, self.matrix.evaluated, self.confidence)

    def pairs_needed(self, effect: float = 0.65) -> int:
        """Discordant pairs this comparison would need to resolve a modest real effect."""
        return discordant_pairs_needed(effect, alpha=1 - self.confidence)

    def summary(self) -> dict[str, Any]:
        matrix = self.matrix.summary()
        before_low, before_high = self.accuracy_interval(after=False)
        after_low, after_high = self.accuracy_interval(after=True)
        return {
            **matrix,
            **self.verdict.as_dict(),
            "accuracy_before_interval": [round(before_low, 4), round(before_high, 4)],
            "accuracy_after_interval": [round(after_low, 4), round(after_high, 4)],
            "intervals_overlap": before_high >= after_low,
            "shippable_on_accuracy_alone": self.shippable_on_accuracy_alone,
            "supported_by_the_paired_test": self.supported_by_the_paired_test,
            "decision_would_differ": self.decision_would_differ,
            "discordant_pairs_needed": self.pairs_needed(),
        }


def compare(
    corpus: Corpus,
    before: LabelSet,
    after: LabelSet,
    alpha: float = 0.05,
    confidence: float = 0.95,
    require_discordant: int | None = None,
) -> Comparison:
    """Run one comparison end to end."""
    if not 0.0 < alpha < 1.0:
        raise ConfigError("alpha must be strictly between 0 and 1")
    if before.name == after.name:
        raise ConfigError("the two label sets have the same name; a comparison needs two versions")
    check_alignment(corpus, before, after)

    matrix = build_matrix(corpus, before, after)
    if require_discordant is not None and matrix.discordant < require_discordant:
        raise UnderpoweredError(
            f"the labelled sample produced {matrix.discordant} discordant pair(s), below the "
            f"{require_discordant} the caller required; only discordant pairs carry information "
            f"about the change"
        )
    verdict = Verdict(
        paired_p=mcnemar_exact(matrix.regressed, matrix.improved),
        unpaired_p=two_proportion_p(
            matrix.correct_before, matrix.evaluated, matrix.correct_after, matrix.evaluated
        ),
        alpha=alpha,
        discordant=matrix.discordant,
    )
    return Comparison(matrix=matrix, verdict=verdict, confidence=confidence)


def record(comparison: Comparison, store: RunStore, run_id: str | None = None) -> Run:
    """Write the comparison to the run log in the shape a tracker uses."""
    matrix = comparison.matrix
    run = store.start(
        f"{matrix.before_name}-to-{matrix.after_name}", run_id=run_id, started_at=0.0
    )
    run.log_param("before", matrix.before_name)
    run.log_param("after", matrix.after_name)
    run.log_param("items", matrix.total)
    run.log_param("evaluated", matrix.evaluated)
    run.log_param("alpha", comparison.verdict.alpha)
    for key, value in comparison.summary().items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        run.log_metric(key, value)
    run.set_tag("paired_test", "mcnemar_exact")
    run.set_tag("decision_would_differ", str(comparison.decision_would_differ))
    run.log_artifact(
        "regressions", [entry.as_dict() for entry in matrix.regressions()]
    )
    run.log_artifact("per_class_churn", matrix.per_class_churn())
    return run
