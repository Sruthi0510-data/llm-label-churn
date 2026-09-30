"""Generator properties, including the pin for the construction that inverted the result."""

from __future__ import annotations

import pytest

from lac.errors import ConfigError
from lac.generator import CLASSES, CONFUSIONS, GeneratorConfig, generate
from lac.labels import accuracy
from lac.transitions import build_matrix


def test_the_same_seed_produces_an_identical_corpus():
    left = generate(GeneratorConfig(items=400, labelled=80, seed=3))
    right = generate(GeneratorConfig(items=400, labelled=80, seed=3))
    assert left.corpus.items == right.corpus.items
    assert left.versions["prompt-v1"].labels == right.versions["prompt-v1"].labels


def test_a_different_seed_produces_a_different_corpus():
    left = generate(GeneratorConfig(items=400, labelled=80, seed=1))
    right = generate(GeneratorConfig(items=400, labelled=80, seed=2))
    assert left.versions["prompt-v1"].labels != right.versions["prompt-v1"].labels


def test_changing_the_labelled_size_changes_nothing_but_the_truth():
    """Pins the split random streams that make the coverage sweep comparable.

    Drawing the ground truth sample from the label stream shifted every subsequent label, so a
    sweep over coverage was comparing a different corpus at every point while appearing to
    compare the same one. The first version of experiment three did exactly that.
    """
    few = generate(GeneratorConfig(items=800, labelled=50, seed=9))
    many = generate(GeneratorConfig(items=800, labelled=400, seed=9))
    for name in few.versions:
        assert few.versions[name].labels == many.versions[name].labels
    assert len(few.corpus.truth) == 50
    assert len(many.corpus.truth) == 400


def test_the_truth_agrees_wherever_two_coverages_overlap():
    few = generate(GeneratorConfig(items=800, labelled=50, seed=9))
    many = generate(GeneratorConfig(items=800, labelled=400, seed=9))
    shared = set(few.corpus.truth) & set(many.corpus.truth)
    assert shared
    assert all(few.corpus.truth[k] == many.corpus.truth[k] for k in shared)


def test_three_versions_are_produced():
    assert set(generate(GeneratorConfig(items=200, labelled=40)).versions) == {
        "prompt-v1", "prompt-v2", "prompt-v3"
    }


def test_every_item_is_labelled_by_every_version():
    produced = generate(GeneratorConfig(items=300, labelled=60, seed=4))
    for label_set in produced.versions.values():
        assert label_set.item_ids == produced.corpus.item_ids


def test_every_label_is_a_known_class():
    produced = generate(GeneratorConfig(items=300, labelled=60, seed=5))
    for label_set in produced.versions.values():
        assert label_set.classes <= set(CLASSES)


def test_a_wrong_label_lands_on_the_confusable_neighbour():
    """Uniform errors would make the per-class churn table meaningless."""
    produced = generate(GeneratorConfig(items=600, labelled=600, seed=6))
    truth = produced.corpus.truth
    for item_id, actual in truth.items():
        predicted = produced.versions["prompt-v1"].get(item_id)
        assert predicted in (actual, CONFUSIONS[actual])


def test_the_versions_agree_on_most_items():
    """Two prompts over the same model and inputs do, and independent draws would not."""
    produced = generate(GeneratorConfig(items=2000, labelled=200, seed=7))
    first, second = produced.versions["prompt-v1"], produced.versions["prompt-v2"]
    agreement = sum(1 for k in first.labels if first.get(k) == second.get(k)) / len(first)
    assert agreement > 0.8


def test_the_versions_genuinely_cross_over_rather_than_nesting():
    """The trap the construction exists to avoid: a strictly better version never regresses."""
    produced = generate(GeneratorConfig(items=4000, labelled=4000, seed=8))
    matrix = build_matrix(
        produced.corpus, produced.versions["prompt-v2"], produced.versions["prompt-v3"]
    )
    assert matrix.improved > 0
    assert matrix.regressed > 0


def test_raising_a_version_skill_raises_its_accuracy():
    """Pins the bug where the correlation knob changed the variance and inverted the result."""
    produced = generate(GeneratorConfig(items=6000, labelled=6000, seed=12))
    truth = produced.corpus.truth
    second = accuracy(produced.versions["prompt-v2"], truth)
    third = accuracy(produced.versions["prompt-v3"], truth)
    assert third > second


def test_the_skill_table_records_three_values_per_class():
    produced = generate(GeneratorConfig(items=200, labelled=40))
    assert set(produced.skill) == set(CLASSES)
    assert all(len(values) == 3 for values in produced.skill.values())


def test_the_improved_classes_gain_and_the_degraded_ones_lose():
    config = GeneratorConfig(items=200, labelled=40)
    produced = generate(config)
    for name in config.improved:
        assert produced.skill[name][1] > produced.skill[name][0]
    for name in config.degraded:
        assert produced.skill[name][1] < produced.skill[name][0]


def test_every_class_has_a_confusable_neighbour():
    assert set(CONFUSIONS) == set(CLASSES)


@pytest.mark.parametrize(("field", "value"), [("items", 0), ("labelled", 0)])
def test_a_non_positive_size_is_refused(field, value):
    with pytest.raises(ConfigError):
        GeneratorConfig(**{field: value})


def test_labelling_more_than_the_corpus_is_refused():
    with pytest.raises(ConfigError, match="between 1 and the corpus size"):
        GeneratorConfig(items=10, labelled=11)


@pytest.mark.parametrize(
    "field", ["independent_draw_rate", "third_version_independent_draw_rate"]
)
@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_an_out_of_range_rate_is_refused(field, bad):
    with pytest.raises(ConfigError, match="between 0 and 1"):
        GeneratorConfig(**{field: bad})


@pytest.mark.parametrize("bad", [0.0, 1.0])
def test_a_base_skill_outside_the_open_interval_is_refused(bad):
    with pytest.raises(ConfigError, match="strictly between"):
        GeneratorConfig(base_skill=bad)


def test_a_class_cannot_be_both_improved_and_degraded():
    with pytest.raises(ConfigError, match="cannot be both"):
        GeneratorConfig(improved=("billing",), degraded=("billing",))


def test_an_unknown_class_is_refused():
    with pytest.raises(ConfigError, match="unknown class"):
        GeneratorConfig(improved=("no-such-class",))


@pytest.mark.parametrize("field", ["improvement", "degradation", "third_version_gain"])
def test_a_negative_change_rate_is_refused(field):
    with pytest.raises(ConfigError, match="at least 0"):
        GeneratorConfig(**{field: -0.1})
