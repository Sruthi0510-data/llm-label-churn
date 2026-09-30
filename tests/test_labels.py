"""The label model, and the alignment check that refuses a comparison of different runs."""

from __future__ import annotations

from pathlib import Path

import pytest

from lac.errors import DataError
from lac.labels import (
    Corpus,
    Item,
    LabelSet,
    accuracy,
    check_alignment,
    classes_of,
    load_corpus,
    load_label_set,
    write_json,
)
from tests.conftest import corpus_of, labels_of


def test_a_label_set_reports_its_size_and_classes():
    label_set = labels_of("v1", a="x", b="y")
    assert len(label_set) == 2
    assert label_set.classes == frozenset({"x", "y"})


def test_a_label_set_needs_a_name():
    with pytest.raises(DataError, match="needs a name"):
        LabelSet(name="", labels={"a": "x"})


def test_an_empty_label_set_is_refused():
    with pytest.raises(DataError, match="cannot be empty"):
        LabelSet(name="v1", labels={})


def test_an_empty_label_is_refused():
    """A blank label is a pipeline failure, not a class, and must not be compared as one."""
    with pytest.raises(DataError, match="empty label"):
        LabelSet(name="v1", labels={"a": ""})


def test_asking_for_a_missing_item_names_the_version():
    with pytest.raises(DataError, match="v1 has no label"):
        labels_of("v1", a="x").get("b")


def test_a_corpus_refuses_duplicate_item_ids():
    with pytest.raises(DataError, match="duplicate item_id"):
        Corpus(items=(Item("a", "t"), Item("a", "t")), truth={})


def test_ground_truth_naming_an_unknown_item_is_refused():
    with pytest.raises(DataError, match="not in the corpus"):
        Corpus(items=(Item("a", "t"),), truth={"z": "x"})


def test_coverage_is_the_labelled_share():
    assert corpus_of(("a", "x"), ("b", None), ("c", None), ("d", None)).coverage == 0.25


def test_an_empty_corpus_has_zero_coverage():
    assert Corpus(items=(), truth={}).coverage == 0.0


def test_the_unlabelled_set_is_the_complement_of_the_labelled_one():
    corpus = corpus_of(("a", "x"), ("b", None))
    assert corpus.labelled_ids == {"a"}
    assert corpus.unlabelled_ids == {"b"}


def test_alignment_passes_when_both_versions_cover_the_corpus():
    corpus = corpus_of(("a", "x"), ("b", None))
    check_alignment(corpus, labels_of("v1", a="x", b="x"), labels_of("v2", a="y", b="x"))


def test_a_version_missing_an_item_is_refused_rather_than_dropped():
    """A missing item means the two runs did not process the same input."""
    corpus = corpus_of(("a", "x"), ("b", None))
    with pytest.raises(DataError, match="missing 1 item"):
        check_alignment(corpus, labels_of("v1", a="x"))


def test_a_version_labelling_an_unknown_item_is_refused():
    corpus = corpus_of(("a", "x"))
    with pytest.raises(DataError, match="does not have"):
        check_alignment(corpus, labels_of("v1", a="x", z="x"))


def test_accuracy_counts_agreement_with_the_truth():
    assert accuracy(labels_of("v1", a="x", b="y"), {"a": "x", "b": "x"}) == 0.5


def test_accuracy_over_an_empty_truth_is_zero_rather_than_an_error():
    assert accuracy(labels_of("v1", a="x"), {}) == 0.0


def test_classes_are_collected_across_versions_and_truth():
    assert classes_of(labels_of("v1", a="x"), labels_of("v2", a="y"), truth=["z"]) == (
        "x", "y", "z",
    )


def test_a_corpus_round_trips_through_json(tmp_path: Path):
    corpus = corpus_of(("a", "x"), ("b", None))
    path = tmp_path / "c.json"
    write_json(path, {"items": [i.as_dict() for i in corpus.items], "truth": dict(corpus.truth)})
    reloaded = load_corpus(path)
    assert reloaded.items == corpus.items
    assert reloaded.truth == corpus.truth


def test_a_label_set_round_trips_through_json(tmp_path: Path):
    path = tmp_path / "v.json"
    write_json(path, {"name": "v1", "labels": {"a": "x"}})
    assert load_label_set(path).labels == {"a": "x"}


def test_the_compact_writer_produces_a_smaller_file(tmp_path: Path):
    """The corpus is tens of thousands of items and nobody reads it by eye."""
    payload = {"items": [{"item_id": f"i{n}", "text": "t"} for n in range(50)], "truth": {}}
    indented = write_json(tmp_path / "a.json", payload)
    compact = write_json(tmp_path / "b.json", payload, compact=True)
    assert compact.stat().st_size < indented.stat().st_size


def test_a_missing_file_names_the_path(tmp_path: Path):
    with pytest.raises(DataError, match="file not found"):
        load_corpus(tmp_path / "absent.json")


def test_invalid_json_is_reported_as_such(tmp_path: Path):
    path = tmp_path / "c.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(DataError, match="not valid JSON"):
        load_corpus(path)


@pytest.mark.parametrize("missing", ["items", "truth"])
def test_a_corpus_missing_a_key_is_refused(tmp_path: Path, missing):
    payload = {"items": [], "truth": {}}
    payload.pop(missing)
    path = tmp_path / "c.json"
    write_json(path, payload)
    with pytest.raises(DataError, match=missing):
        load_corpus(path)


@pytest.mark.parametrize("missing", ["name", "labels"])
def test_a_label_set_missing_a_key_is_refused(tmp_path: Path, missing):
    payload = {"name": "v1", "labels": {"a": "x"}}
    payload.pop(missing)
    path = tmp_path / "v.json"
    write_json(path, payload)
    with pytest.raises(DataError, match=missing):
        load_label_set(path)
