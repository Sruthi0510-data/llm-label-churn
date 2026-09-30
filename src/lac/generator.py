"""A seeded enrichment corpus and two pipeline versions, built to behave like a real one.

Three properties matter and one trap had to be avoided.

Ground truth covers a small share of the corpus, because that is what happens: a human labels a
sample and the pipeline labels everything. An evaluation set that covered the corpus would make
the second finding of this repository impossible to see.

The two versions must agree on most items, because two prompts over the same model and the same
inputs do. Independent draws would produce churn near the error rate of both versions combined,
which is not what a prompt change looks like.

The versions must genuinely cross over rather than nest. The trap: if each item has a single
difficulty and a version is correct when difficulty falls below its skill, the better version is
right about every item the worse version was right about, no item ever regresses, and the whole
repository has nothing to measure.

The construction that satisfies all three is an item difficulty every version usually reads,
with each version independently redrawing it for a small share of items. The redraw rate sets
the correlation and leaves the marginal alone: a version is correct with probability exactly its
skill either way. The first attempt mixed the shared difficulty with per-version noise as a
weighted average, which controlled the correlation and quietly changed the variance, so raising
the correlation lowered the accuracy of a version whose skill had gone up. That is the kind of
generator bug that produces a repository arguing the opposite of what it meant to.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from lac.errors import ConfigError
from lac.labels import Corpus, Item, LabelSet

CLASSES = (
    "billing",
    "compliance",
    "incident",
    "onboarding",
    "partnership",
    "product-feedback",
    "security",
    "support",
)

CONFUSIONS = {
    "billing": "support",
    "support": "billing",
    "compliance": "security",
    "security": "compliance",
    "incident": "security",
    "onboarding": "support",
    "partnership": "product-feedback",
    "product-feedback": "partnership",
}
"""Which class a version drifts into when it is wrong.

Not uniform, because a classifier's errors are not uniform: billing and support look alike,
compliance and security look alike, and a wrong label almost always lands on the neighbour.
Uniform errors would make the per-class churn table meaningless.
"""

SUBJECTS = (
    "invoice discrepancy", "retention request", "outage postmortem", "access review",
    "contract renewal", "feature request", "credential rotation", "escalation thread",
    "data export", "vendor assessment", "pricing query", "incident timeline",
)

SEGMENTS = ("enterprise", "mid-market", "self-serve")


@dataclass(frozen=True)
class GeneratorConfig:
    """Shape of the corpus and of the two versions."""

    items: int = 25_000
    labelled: int = 1_000
    seed: int = 20260930
    independent_draw_rate: float = 0.14
    base_skill: float = 0.86
    improved: tuple[str, ...] = ("billing", "compliance", "onboarding")
    degraded: tuple[str, ...] = ("partnership", "security")
    improvement: float = 0.07
    degradation: float = 0.05
    third_version_gain: float = 0.022
    third_version_independent_draw_rate: float = 0.03

    def __post_init__(self) -> None:
        if self.items < 1:
            raise ConfigError("items must be at least 1")
        if not 1 <= self.labelled <= self.items:
            raise ConfigError("labelled must be between 1 and the corpus size")
        if not 0.0 <= self.independent_draw_rate <= 1.0:
            raise ConfigError("independent_draw_rate must be between 0 and 1")
        if not 0.0 < self.base_skill < 1.0:
            raise ConfigError("base_skill must be strictly between 0 and 1")
        if not 0.0 <= self.third_version_independent_draw_rate <= 1.0:
            raise ConfigError("third_version_independent_draw_rate must be between 0 and 1")
        for name in ("improvement", "degradation", "third_version_gain"):
            value = getattr(self, name)
            if not 0.0 <= value < 1.0:
                raise ConfigError(f"{name} must be at least 0 and below 1")
        overlap = set(self.improved) & set(self.degraded)
        if overlap:
            raise ConfigError(f"a class cannot be both improved and degraded: {sorted(overlap)}")
        unknown = (set(self.improved) | set(self.degraded)) - set(CLASSES)
        if unknown:
            raise ConfigError(f"unknown class(es) {sorted(unknown)}")


@dataclass
class Generated:
    """Everything one run of the generator produced."""

    corpus: Corpus
    versions: dict[str, LabelSet]
    skill: dict[str, tuple[float, float, float]] = field(default_factory=dict)


def _skill_table(config: GeneratorConfig) -> dict[str, tuple[float, float, float]]:
    """Per-class skill for each of the three versions.

    Three rather than two, because the repository needs both failures. The second version is
    the one that looks like a win and is not: it trades classes against each other, so the
    marginal moves a little and a great many labels move a lot. The third is a real, modest,
    uniform improvement, which is the case where the test a team actually runs throws the
    improvement away.
    """
    table: dict[str, tuple[float, float, float]] = {}
    for index, name in enumerate(CLASSES):
        first = config.base_skill - 0.04 + 0.01 * (index % 5)
        second = first
        if name in config.improved:
            second = min(0.995, first + config.improvement)
        elif name in config.degraded:
            second = max(0.05, first - config.degradation)
        third = min(0.995, second + config.third_version_gain)
        table[name] = (round(first, 4), round(second, 4), round(third, 4))
    return table


def generate(config: GeneratorConfig | None = None) -> Generated:
    """Build a corpus, two label sets and the partial ground truth."""
    config = config or GeneratorConfig()
    rng = random.Random(config.seed)
    skill = _skill_table(config)

    items: list[Item] = []
    truth: dict[str, str] = {}
    drafts: list[dict[str, str]] = [{}, {}, {}]

    labelled_ids = set(rng.sample(range(config.items), config.labelled))

    for index in range(config.items):
        item_id = f"doc-{index:06d}"
        actual = rng.choice(CLASSES)
        items.append(
            Item(
                item_id=item_id,
                text=f"{rng.choice(SUBJECTS)} ({rng.choice(SEGMENTS)})",
                segment=rng.choice(SEGMENTS),
            )
        )
        if index in labelled_ids:
            truth[item_id] = actual

        difficulty = rng.random()
        flip_rates = (
            config.independent_draw_rate,
            config.independent_draw_rate,
            config.third_version_independent_draw_rate,
        )
        for slot, (version_skill, flip) in enumerate(zip(skill[actual], flip_rates, strict=True)):
            score = rng.random() if rng.random() < flip else difficulty
            drafts[slot][item_id] = actual if score < version_skill else CONFUSIONS[actual]

    names = ("prompt-v1", "prompt-v2", "prompt-v3")
    return Generated(
        corpus=Corpus(items=tuple(items), truth=truth),
        versions={
            name: LabelSet(name=name, labels=labels)
            for name, labels in zip(names, drafts, strict=True)
        },
        skill=skill,
    )
