"""The comparison engine, the run log, and the optional tracker's import ordering."""

from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from lac.compare import compare, record
from lac.errors import ConfigError, DataError, UnderpoweredError
from lac.runs import Run, RunStore, mirror_to_mlflow
from tests.conftest import corpus_of, labels_of


def test_a_comparison_reports_both_tests(tiny):
    summary = compare(*tiny).summary()
    assert "paired_p" in summary
    assert "unpaired_p" in summary


def test_comparing_a_version_with_itself_is_refused():
    corpus = corpus_of(("a", "x"))
    with pytest.raises(ConfigError, match="same name"):
        compare(corpus, labels_of("v1", a="x"), labels_of("v1", a="x"))


def test_a_misaligned_version_is_refused_before_any_arithmetic():
    corpus = corpus_of(("a", "x"), ("b", "x"))
    with pytest.raises(DataError, match="missing"):
        compare(corpus, labels_of("v1", a="x"), labels_of("v2", a="x", b="x"))


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1])
def test_an_out_of_range_alpha_is_refused(tiny, bad):
    with pytest.raises(ConfigError, match="alpha"):
        compare(*tiny, alpha=bad)


def test_requiring_more_discordant_pairs_than_exist_is_underpowered(tiny):
    """Only discordant pairs carry information, so that is the count worth guarding."""
    with pytest.raises(UnderpoweredError, match="discordant"):
        compare(*tiny, require_discordant=50)


def test_requiring_a_satisfiable_number_of_pairs_passes(tiny):
    assert compare(*tiny, require_discordant=1) is not None


def test_a_positive_delta_alone_would_ship_the_change(small):
    comparison = compare(
        small.corpus, small.versions["prompt-v1"], small.versions["prompt-v3"]
    )
    assert comparison.shippable_on_accuracy_alone is True


def test_the_decision_differs_when_the_delta_and_the_test_point_different_ways(shipped):
    corpus, versions = shipped
    comparison = compare(corpus, versions["prompt-v1"], versions["prompt-v2"])
    assert comparison.shippable_on_accuracy_alone is True
    assert comparison.supported_by_the_paired_test is False
    assert comparison.decision_would_differ is True


def test_the_two_tests_disagree_on_the_third_version(shipped):
    """The other failure: a real improvement the unpaired test throws away."""
    corpus, versions = shipped
    comparison = compare(corpus, versions["prompt-v2"], versions["prompt-v3"])
    assert comparison.verdict.paired_significant is True
    assert comparison.verdict.unpaired_significant is False
    assert comparison.verdict.tests_disagree is True


def test_the_accuracy_intervals_are_reported_for_both_versions(shipped):
    corpus, versions = shipped
    comparison = compare(corpus, versions["prompt-v1"], versions["prompt-v2"])
    low, high = comparison.accuracy_interval(after=False)
    assert low < comparison.matrix.accuracy_before < high


def test_overlapping_intervals_are_reported_as_such(shipped):
    corpus, versions = shipped
    summary = compare(corpus, versions["prompt-v1"], versions["prompt-v2"]).summary()
    assert summary["intervals_overlap"] is True


def test_the_comparison_reports_the_pairs_a_modest_effect_would_need(tiny):
    assert compare(*tiny).pairs_needed() > 0


def test_a_run_needs_a_name():
    with pytest.raises(ConfigError, match="needs a name"):
        Run(name="")


def test_a_metric_must_be_a_number():
    run = Run(name="r")
    with pytest.raises(ConfigError, match="must be a number"):
        run.log_metric("x", "not a number")


def test_a_boolean_is_not_accepted_as_a_metric():
    """True would silently log as 1.0 and read as a measurement."""
    with pytest.raises(ConfigError, match="must be a number"):
        Run(name="r").log_metric("x", True)


def test_a_run_records_params_metrics_tags_and_artifacts():
    run = Run(name="r")
    run.log_param("a", 1)
    run.log_metric("b", 2.0)
    run.set_tag("c", "three")
    run.log_artifact("d", [1, 2])
    payload = run.as_dict()
    assert payload["params"] == {"a": 1}
    assert payload["metrics"] == {"b": 2.0}
    assert payload["tags"] == {"c": "three"}
    assert payload["artifacts"] == {"d": [1, 2]}


def test_an_unstarted_run_cannot_be_saved(tmp_path: Path):
    with pytest.raises(ConfigError, match="must be started"):
        RunStore(tmp_path).save(Run(name="r"))


def test_a_run_round_trips_through_the_store(tmp_path: Path):
    store = RunStore(tmp_path)
    run = store.start("r", run_id="abc123", started_at=0.0)
    run.log_metric("x", 1.5)
    path = store.save(run)
    assert store.load(path).metrics == {"x": 1.5}


def test_the_store_lists_runs_in_a_stable_order(tmp_path: Path):
    store = RunStore(tmp_path)
    for name in ("b", "a", "c"):
        store.save(store.start(name, run_id="0", started_at=0.0))
    assert [p.stem for p in store.list_runs()] == ["a-0", "b-0", "c-0"]


def test_an_empty_store_lists_nothing(tmp_path: Path):
    assert RunStore(tmp_path / "absent").list_runs() == []


def test_loading_a_missing_run_names_the_path(tmp_path: Path):
    with pytest.raises(ConfigError, match="run not found"):
        RunStore(tmp_path).load(tmp_path / "absent.json")


def test_loading_invalid_json_is_reported_as_such(tmp_path: Path):
    path = tmp_path / "r.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid JSON"):
        RunStore(tmp_path).load(path)


def test_recording_a_comparison_logs_the_metrics_and_the_regressions(tiny, tmp_path: Path):
    run = record(compare(*tiny), RunStore(tmp_path), run_id="fixed")
    assert run.metrics
    assert "regressions" in run.artifacts
    assert "per_class_churn" in run.artifacts


def test_a_recorded_run_carries_no_boolean_as_a_metric(tiny, tmp_path: Path):
    run = record(compare(*tiny), RunStore(tmp_path), run_id="fixed")
    assert all(isinstance(value, float) for value in run.metrics.values())


def test_a_recorded_run_tags_whether_the_decision_would_differ(tiny, tmp_path: Path):
    run = record(compare(*tiny), RunStore(tmp_path), run_id="fixed")
    assert "decision_would_differ" in run.tags


@pytest.fixture
def no_mlflow(monkeypatch):
    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name.startswith("mlflow"):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)


def test_an_unstarted_run_is_refused_before_the_optional_import(no_mlflow):
    """A caller with a bad run must not be told to install a package they may not need."""
    with pytest.raises(ConfigError, match="must be started"):
        mirror_to_mlflow(Run(name="r"))


def test_a_run_with_no_metrics_is_refused_before_the_optional_import(no_mlflow):
    with pytest.raises(ConfigError, match="no metrics"):
        mirror_to_mlflow(Run(name="r", run_id="abc"))


def test_a_genuinely_missing_extra_reports_the_install_hint(no_mlflow):
    run = Run(name="r", run_id="abc", metrics={"x": 1.0})
    with pytest.raises(ConfigError, match="pip install"):
        mirror_to_mlflow(run)


@pytest.mark.parametrize(
    "module", ["lac.labels", "lac.transitions", "lac.stats", "lac.compare", "lac.runs"]
)
def test_the_core_imports_without_the_optional_extra(no_mlflow, module):
    import importlib

    assert importlib.import_module(module) is not None
