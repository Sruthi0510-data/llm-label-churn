"""Shared builders, so the corpus schema lives in one place."""

from __future__ import annotations

from pathlib import Path

import pytest

from lac.generator import GeneratorConfig, generate
from lac.labels import Corpus, Item, LabelSet, load_corpus, load_label_set

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def corpus_of(*rows: tuple[str, str | None]) -> Corpus:
    """Build a corpus from (item_id, truth) pairs. A truth of None means unlabelled."""
    items = tuple(Item(item_id=item_id, text=item_id) for item_id, _ in rows)
    truth = {item_id: value for item_id, value in rows if value is not None}
    return Corpus(items=items, truth=truth)


def labels_of(name: str, **labels: str) -> LabelSet:
    return LabelSet(name=name, labels=dict(labels))


@pytest.fixture
def tiny():
    """Four labelled items covering all four evaluated transitions, plus two unlabelled."""
    corpus = corpus_of(
        ("a", "x"), ("b", "x"), ("c", "x"), ("d", "x"), ("e", None), ("f", None)
    )
    before = labels_of("v1", a="x", b="x", c="y", d="y", e="x", f="x")
    after = labels_of("v2", a="x", b="y", c="x", d="y", e="x", f="y")
    return corpus, before, after


@pytest.fixture
def small():
    """A deterministic 600-item corpus, fast enough for per-test comparison."""
    return generate(GeneratorConfig(items=600, labelled=200, seed=11))


@pytest.fixture(scope="session")
def shipped():
    """The committed corpus and the three versions the README quotes."""
    corpus = load_corpus(FIXTURES / "corpus.json")
    versions = {
        name: load_label_set(FIXTURES / f"{name}.json")
        for name in ("prompt-v1", "prompt-v2", "prompt-v3")
    }
    return corpus, versions
