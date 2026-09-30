"""The paired test, the unpaired one teams actually run, and why they disagree.

Comparing two pipeline versions on the same items is a paired design. Every item was labelled
by both, so the two accuracies are not independent samples and the usual two-proportion z test
does not apply to them. Teams run it anyway, because it is the test their stats memory offers
and the numbers going into it look like two proportions.

The right test is McNemar's, and it uses only the discordant pairs: the items one version got
right and the other got wrong. Under the null that the change is neutral, each discordant pair
is a coin flip, so the exact test is binomial on b successes out of b + c.

The two tests disagree in a specific and consequential direction. The unpaired test treats all
the agreement as noise, so its standard error is much larger and it loses power: a real
improvement reads as not significant. The paired test can also refuse a change the unpaired
test would wave through, when a large accuracy gain rests on a handful of discordant pairs. The
experiments publish both, which is the only way to show they are not the same question.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from lac.errors import ConfigError


def _lower_binomial_tail(successes: int, trials: int) -> float:
    """Exact lower binomial tail at p of one half, summed through log gamma for stability.

    Lower tail only, and no guard for zero trials. Both were here and branch coverage showed
    neither could be reached: the caller takes the smaller of the two counts, so the tail is
    always the lower one, and it returns before this function when there are no discordant
    pairs. A branch nothing can reach is not defensive, it is a claim the tests cannot check.
    """
    log_half = trials * math.log(0.5)
    return min(
        1.0,
        sum(
            math.exp(
                math.lgamma(trials + 1)
                - math.lgamma(k + 1)
                - math.lgamma(trials - k + 1)
                + log_half
            )
            for k in range(successes + 1)
        ),
    )


def mcnemar_exact(right_to_wrong: int, wrong_to_right: int) -> float:
    """Two-sided exact McNemar p value. Uses the discordant pairs and nothing else.

    Returns 1.0 when there are no discordant pairs, which is the honest answer: two versions
    that never disagree on a labelled item provide no evidence either way, however many items
    they agreed on.
    """
    if right_to_wrong < 0 or wrong_to_right < 0:
        raise ConfigError("discordant counts cannot be negative")
    discordant = right_to_wrong + wrong_to_right
    if discordant == 0:
        return 1.0
    lower = _lower_binomial_tail(min(right_to_wrong, wrong_to_right), discordant)
    return min(1.0, 2.0 * lower)


def two_proportion_p(
    correct_a: int, total_a: int, correct_b: int, total_b: int
) -> float:
    """The unpaired two-sided test. Included because it is the one teams run, and it is wrong here.

    Kept in the package rather than described in prose so the experiments can run it and publish
    the disagreement, and so a reader can see it is a real implementation rather than a straw
    man built to lose.
    """
    if total_a < 1 or total_b < 1:
        raise ConfigError("both samples need at least one observation")
    if not 0 <= correct_a <= total_a or not 0 <= correct_b <= total_b:
        raise ConfigError("correct counts must lie within their totals")
    p_a, p_b = correct_a / total_a, correct_b / total_b
    pooled = (correct_a + correct_b) / (total_a + total_b)
    spread = math.sqrt(pooled * (1.0 - pooled) * (1.0 / total_a + 1.0 / total_b))
    if spread == 0.0:
        return 1.0
    z = abs(p_a - p_b) / spread
    return math.erfc(z / math.sqrt(2.0))


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """A Wilson score interval, which behaves near zero and one where the normal one does not.

    The normal approximation interval on a proportion of 0.98 over 200 items runs past 1.0, and
    an accuracy figure with an upper bound above one is the kind of detail that makes a reader
    stop trusting the rest of the page.
    """
    if trials < 1:
        raise ConfigError("trials must be at least 1")
    if not 0 <= successes <= trials:
        raise ConfigError("successes must lie within trials")
    z = {0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}.get(round(confidence, 4))
    if z is None:
        raise ConfigError("confidence must be one of 0.90, 0.95, 0.99")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (proportion + z * z / (2 * trials)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials))
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


@dataclass(frozen=True)
class Verdict:
    """What the paired test says, and what the test the team would have run says instead."""

    paired_p: float
    unpaired_p: float
    alpha: float
    discordant: int

    @property
    def paired_significant(self) -> bool:
        return self.paired_p < self.alpha

    @property
    def unpaired_significant(self) -> bool:
        return self.unpaired_p < self.alpha

    @property
    def tests_disagree(self) -> bool:
        """True when the two tests reach different conclusions about the same data."""
        return self.paired_significant != self.unpaired_significant

    def as_dict(self) -> dict[str, object]:
        return {
            "paired_p": round(self.paired_p, 6),
            "unpaired_p": round(self.unpaired_p, 6),
            "alpha": self.alpha,
            "discordant": self.discordant,
            "paired_significant": self.paired_significant,
            "unpaired_significant": self.unpaired_significant,
            "tests_disagree": self.tests_disagree,
        }


def discordant_pairs_needed(effect: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Discordant pairs needed to detect a given split of them, as a sign test.

    `effect` is the share of discordant pairs that would fall on the improving side under the
    alternative. A change where six of every ten flips are improvements needs a great many more
    discordant pairs than one where nine in ten are, and neither number has anything to do with
    how many items were labelled.
    """
    if not 0.0 < effect < 1.0:
        raise ConfigError("effect must be strictly between 0 and 1")
    if effect == 0.5:
        raise ConfigError("an effect of exactly one half is the null; no sample size detects it")
    quantiles = {0.80: 0.8416, 0.90: 1.2816, 0.95: 1.6449, 0.975: 1.9600, 0.99: 2.3263}
    try:
        z_alpha = quantiles[round(1 - alpha / 2, 4)]
        z_beta = quantiles[round(power, 4)]
    except KeyError as exc:
        raise ConfigError(
            f"unsupported alpha or power; available quantiles: {sorted(quantiles)}"
        ) from exc
    numerator = z_alpha * 0.5 + z_beta * math.sqrt(effect * (1 - effect))
    return math.ceil((numerator / (effect - 0.5)) ** 2)
