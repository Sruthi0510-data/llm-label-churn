"""The paired test, the unpaired one, and the interval that does not run past one."""

from __future__ import annotations

import pytest

from lac.errors import ConfigError
from lac.stats import (
    Verdict,
    discordant_pairs_needed,
    mcnemar_exact,
    two_proportion_p,
    wilson_interval,
)


def test_no_discordant_pairs_is_no_evidence_either_way():
    """Two versions that never disagree provide nothing, however many items they agreed on."""
    assert mcnemar_exact(0, 0) == 1.0


def test_an_even_split_of_discordant_pairs_is_maximally_unconvincing():
    assert mcnemar_exact(25, 25) == 1.0


def test_a_lopsided_split_is_significant():
    assert mcnemar_exact(2, 20) < 0.001


def test_the_test_is_symmetric_in_its_two_counts():
    assert mcnemar_exact(5, 25) == pytest.approx(mcnemar_exact(25, 5))


def test_more_evidence_in_the_same_direction_lowers_the_p_value():
    assert mcnemar_exact(4, 20) < mcnemar_exact(2, 10)


def test_a_single_discordant_pair_cannot_be_significant():
    assert mcnemar_exact(0, 1) == 1.0


def test_a_p_value_never_exceeds_one():
    for regressed in range(6):
        for improved in range(6):
            assert 0.0 <= mcnemar_exact(regressed, improved) <= 1.0


@pytest.mark.parametrize("args", [(-1, 5), (5, -1)])
def test_negative_discordant_counts_are_refused(args):
    with pytest.raises(ConfigError, match="cannot be negative"):
        mcnemar_exact(*args)


def test_the_unpaired_test_returns_one_when_the_proportions_match():
    assert two_proportion_p(500, 1000, 500, 1000) == 1.0


def test_the_unpaired_test_returns_one_when_both_samples_are_perfect():
    """A pooled proportion of one has zero spread, and a division would follow."""
    assert two_proportion_p(10, 10, 10, 10) == 1.0


def test_the_unpaired_test_falls_as_the_gap_widens():
    assert two_proportion_p(500, 1000, 600, 1000) < two_proportion_p(500, 1000, 520, 1000)


def test_the_unpaired_test_has_less_power_than_the_paired_one():
    """The finding, as a test. Both describe the same 1000 items and the same 25 flips."""
    paired = mcnemar_exact(3, 22)
    unpaired = two_proportion_p(900, 1000, 919, 1000)
    assert paired < 0.001
    assert unpaired > 0.05


@pytest.mark.parametrize("args", [(0, 0, 1, 1), (1, 1, 0, 0)])
def test_an_empty_sample_is_refused(args):
    with pytest.raises(ConfigError, match="at least one observation"):
        two_proportion_p(*args)


@pytest.mark.parametrize("args", [(11, 10, 5, 10), (-1, 10, 5, 10)])
def test_a_correct_count_outside_its_total_is_refused(args):
    with pytest.raises(ConfigError, match="within their totals"):
        two_proportion_p(*args)


def test_a_wilson_interval_never_runs_past_one():
    """The normal interval on 196 of 200 does, and an accuracy above one loses a reader."""
    low, high = wilson_interval(196, 200)
    assert high <= 1.0
    assert low > 0.9


def test_a_wilson_interval_never_runs_below_zero():
    low, _ = wilson_interval(0, 50)
    assert low >= 0.0


def test_a_wider_confidence_gives_a_wider_interval():
    narrow = wilson_interval(90, 100, 0.90)
    wide = wilson_interval(90, 100, 0.99)
    assert wide[0] < narrow[0]
    assert wide[1] > narrow[1]


def test_a_larger_sample_gives_a_tighter_interval():
    small = wilson_interval(90, 100)
    large = wilson_interval(900, 1000)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_an_interval_needs_at_least_one_trial():
    with pytest.raises(ConfigError, match="at least 1"):
        wilson_interval(0, 0)


def test_successes_beyond_the_trials_are_refused():
    with pytest.raises(ConfigError, match="within trials"):
        wilson_interval(11, 10)


def test_an_unsupported_confidence_is_refused():
    with pytest.raises(ConfigError, match="0.90"):
        wilson_interval(5, 10, 0.995)


def test_a_more_lopsided_effect_needs_fewer_discordant_pairs():
    assert discordant_pairs_needed(0.8) < discordant_pairs_needed(0.6)


def test_a_subtler_effect_needs_many_more_pairs():
    assert discordant_pairs_needed(0.55) > 500


def test_an_effect_of_exactly_half_is_the_null_and_is_refused():
    with pytest.raises(ConfigError, match="the null"):
        discordant_pairs_needed(0.5)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.2])
def test_an_effect_outside_the_open_interval_is_refused(bad):
    with pytest.raises(ConfigError, match="strictly between"):
        discordant_pairs_needed(bad)


def test_an_unsupported_alpha_names_the_available_quantiles():
    with pytest.raises(ConfigError, match="available quantiles"):
        discordant_pairs_needed(0.7, alpha=0.037)


def test_a_verdict_reports_each_test_against_its_own_threshold():
    verdict = Verdict(paired_p=0.01, unpaired_p=0.2, alpha=0.05, discordant=40)
    assert verdict.paired_significant is True
    assert verdict.unpaired_significant is False


def test_a_verdict_notices_when_the_two_tests_disagree():
    assert Verdict(0.01, 0.2, 0.05, 40).tests_disagree is True
    assert Verdict(0.01, 0.02, 0.05, 40).tests_disagree is False


def test_a_verdict_serialises_every_published_field():
    payload = Verdict(0.01, 0.2, 0.05, 40).as_dict()
    for field in ("paired_p", "unpaired_p", "discordant", "paired_significant",
                  "unpaired_significant", "tests_disagree"):
        assert field in payload
