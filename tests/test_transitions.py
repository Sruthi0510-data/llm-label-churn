"""The per-item accounting, including the state that has no correctness attached."""

from __future__ import annotations

import pytest

from lac.transitions import (
    DISCORDANT,
    EVALUATED,
    Transition,
    build_matrix,
    classify,
)


def test_right_then_right_is_held_right():
    assert classify("x", "x", "x") is Transition.HELD_RIGHT


def test_wrong_then_wrong_is_held_wrong():
    assert classify("y", "z", "x") is Transition.HELD_WRONG


def test_right_then_wrong_is_a_regression():
    assert classify("x", "y", "x") is Transition.REGRESSED


def test_wrong_then_right_is_an_improvement():
    assert classify("y", "x", "x") is Transition.IMPROVED


def test_an_unlabelled_item_that_did_not_move_is_stable():
    assert classify("x", "x", None) is Transition.UNEVALUATED_STABLE


def test_an_unlabelled_item_that_moved_has_no_correctness_attached():
    """Measurable churn, unmeasurable correctness. It must never enter an accuracy figure."""
    assert classify("x", "y", None) is Transition.UNEVALUATED_CHANGED


def test_the_evaluated_set_is_exactly_the_four_labelled_states():
    assert {
        Transition.HELD_RIGHT,
        Transition.HELD_WRONG,
        Transition.REGRESSED,
        Transition.IMPROVED,
    } == EVALUATED


def test_only_the_two_discordant_states_carry_evidence():
    assert {Transition.REGRESSED, Transition.IMPROVED} == DISCORDANT


def test_the_discordant_states_are_a_subset_of_the_evaluated_ones():
    assert DISCORDANT < EVALUATED


def test_the_matrix_places_every_item(tiny):
    matrix = build_matrix(*tiny)
    assert matrix.total == 6


def test_the_matrix_counts_each_transition(tiny):
    matrix = build_matrix(*tiny)
    assert matrix.count(Transition.HELD_RIGHT) == 1
    assert matrix.regressed == 1
    assert matrix.improved == 1
    assert matrix.count(Transition.HELD_WRONG) == 1
    assert matrix.unevaluated_changed == 1


def test_accuracy_is_computed_only_over_the_labelled_items(tiny):
    matrix = build_matrix(*tiny)
    assert matrix.evaluated == 4
    assert matrix.accuracy_before == 0.5
    assert matrix.accuracy_after == 0.5


def test_two_versions_can_agree_on_accuracy_exactly_and_disagree_about_items(tiny):
    """The whole argument, in one assertion."""
    matrix = build_matrix(*tiny)
    assert matrix.accuracy_delta == 0.0
    assert matrix.discordant == 2
    assert matrix.changed > 0


def test_the_net_change_is_the_entire_content_of_the_delta(tiny):
    matrix = build_matrix(*tiny)
    assert matrix.net_evaluated_change == matrix.improved - matrix.regressed
    assert matrix.accuracy_delta == matrix.net_evaluated_change / matrix.evaluated


def test_churn_is_measured_over_the_whole_corpus_not_the_sample(tiny):
    matrix = build_matrix(*tiny)
    assert matrix.churn_rate == matrix.changed / matrix.total
    assert matrix.total > matrix.evaluated


def test_there_is_no_churn_per_gain_when_nothing_was_gained(tiny):
    assert build_matrix(*tiny).churn_per_net_gain is None


def test_churn_per_gain_is_reported_when_there_is_a_gain(small):
    matrix = build_matrix(small.corpus, small.versions["prompt-v1"], small.versions["prompt-v3"])
    assert matrix.net_evaluated_change > 0
    assert matrix.churn_per_net_gain == round(matrix.changed / matrix.net_evaluated_change, 1)


def test_the_regression_list_holds_exactly_the_regressed_items(tiny):
    matrix = build_matrix(*tiny)
    assert len(matrix.regressions()) == matrix.regressed
    assert all(e.transition is Transition.REGRESSED for e in matrix.regressions())


def test_a_regression_carries_the_truth_a_reviewer_needs(tiny):
    entry = build_matrix(*tiny).regressions()[0]
    assert entry.truth is not None
    assert entry.before == entry.truth
    assert entry.after != entry.truth


def test_per_class_churn_balances_out(tiny):
    """Every label that left one class arrived at another, so the totals must match."""
    churn = build_matrix(*tiny).per_class_churn()
    assert sum(m["out"] for m in churn.values()) == sum(m["in"] for m in churn.values())


def test_per_class_churn_ignores_items_that_did_not_move(small):
    matrix = build_matrix(small.corpus, small.versions["prompt-v1"], small.versions["prompt-v2"])
    churn = matrix.per_class_churn()
    assert sum(m["out"] for m in churn.values()) == matrix.changed


def test_the_matrix_is_ordered_by_item_id_so_reports_are_stable(small):
    matrix = build_matrix(small.corpus, small.versions["prompt-v1"], small.versions["prompt-v2"])
    ids = [entry.item_id for entry in matrix.transitions]
    assert ids == sorted(ids)


def test_an_entry_serialises_every_published_field(tiny):
    payload = build_matrix(*tiny).transitions[0].as_dict()
    for field in ("item_id", "before", "after", "truth", "transition", "changed"):
        assert field in payload


def test_the_summary_carries_every_field_the_documents_quote(tiny):
    summary = build_matrix(*tiny).summary()
    for field in ("accuracy_delta_points", "regressed", "improved", "discordant",
                  "changed", "churn_rate_pct", "unevaluated_changed", "churn_per_net_gain"):
        assert field in summary


def test_an_empty_matrix_reports_zero_rather_than_dividing():
    from lac.transitions import TransitionMatrix

    empty = TransitionMatrix(before_name="a", after_name="b", transitions=(), truth={})
    assert empty.accuracy_before == 0.0
    assert empty.accuracy_after == 0.0
    assert empty.churn_rate == 0.0
    assert empty.summary()["coverage"] == 0.0


def test_transition_values_are_stable_strings():
    assert str(Transition.REGRESSED) == "regressed"


@pytest.mark.parametrize("pair", [("prompt-v1", "prompt-v2"), ("prompt-v2", "prompt-v3")])
def test_the_shipped_fixture_produces_all_four_evaluated_states(shipped, pair):
    corpus, versions = shipped
    matrix = build_matrix(corpus, versions[pair[0]], versions[pair[1]])
    for state in EVALUATED:
        assert matrix.count(state) > 0
