"""The command line surface, with the exit codes treated as the interface they are."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from lac import exit_codes
from lac.cli import main
from tests.conftest import FIXTURES


def run(*argv: str) -> tuple[int, str]:
    out = io.StringIO()
    code = main(list(argv), out=out)
    return code, out.getvalue()


def args_for(before: str = "prompt-v1", after: str = "prompt-v2") -> list[str]:
    return [
        "--corpus", str(FIXTURES / "corpus.json"),
        "--before", str(FIXTURES / f"{before}.json"),
        "--after", str(FIXTURES / f"{after}.json"),
    ]


def test_generate_writes_a_corpus_and_three_versions(tmp_path: Path):
    code, text = run("generate", "--out", str(tmp_path), "--items", "200", "--labelled", "40")
    assert code == exit_codes.OK
    assert "3 versions" in text
    assert (tmp_path / "corpus.json").exists()
    assert (tmp_path / "prompt-v3.json").exists()


def test_generate_is_reproducible(tmp_path: Path):
    for name in ("a", "b"):
        run("generate", "--out", str(tmp_path / name), "--items", "120", "--labelled", "30",
            "--seed", "5")
    assert (tmp_path / "a" / "corpus.json").read_text() == (
        tmp_path / "b" / "corpus.json"
    ).read_text()


def test_compare_returns_the_regression_code_when_items_got_worse():
    code, text = run("compare", *args_for())
    assert code == exit_codes.REGRESSION
    assert json.loads(text)["regressed"] > 0


def test_compare_reports_both_tests():
    _, text = run("compare", *args_for())
    summary = json.loads(text)
    assert "paired_p" in summary
    assert "unpaired_p" in summary


def test_compare_flags_when_the_delta_and_the_test_point_different_ways():
    _, text = run("compare", *args_for("prompt-v1", "prompt-v2"))
    assert json.loads(text)["decision_would_differ"] is True


def test_compare_flags_when_the_two_tests_disagree():
    _, text = run("compare", *args_for("prompt-v2", "prompt-v3"))
    assert json.loads(text)["tests_disagree"] is True


def test_compare_writes_its_summary_to_json(tmp_path: Path):
    target = tmp_path / "nested" / "c.json"
    run("compare", *args_for(), "--json", str(target))
    assert json.loads(target.read_text())["items"] > 0


def test_compare_records_a_run_when_asked(tmp_path: Path):
    code, text = run("compare", *args_for(), "--record", str(tmp_path))
    assert code == exit_codes.REGRESSION
    assert "recorded run at" in text
    assert list(tmp_path.glob("*.json"))


def test_requiring_more_discordant_pairs_than_exist_returns_underpowered():
    code, text = run("compare", *args_for(), "--require-discordant", "100000")
    assert code == exit_codes.UNDERPOWERED
    assert "discordant" in text


def test_a_regression_and_a_failure_to_answer_have_different_exit_codes(tmp_path: Path):
    finding, _ = run("compare", *args_for())
    failure, _ = run("compare", "--corpus", str(tmp_path / "absent.json"))
    assert finding == exit_codes.REGRESSION
    assert failure == exit_codes.USAGE
    assert finding != failure


def test_a_missing_file_is_reported_by_path(tmp_path: Path):
    code, text = run("compare", "--corpus", str(tmp_path / "absent.json"))
    assert code == exit_codes.USAGE
    assert "file not found" in text


def test_an_out_of_range_alpha_is_a_usage_error():
    code, text = run("compare", *args_for(), "--alpha", "2")
    assert code == exit_codes.USAGE
    assert "alpha" in text


def test_regressions_lists_the_items_a_human_should_review():
    code, text = run("regressions", *args_for(), "--limit", "3")
    assert code == exit_codes.REGRESSION
    assert "went from right to wrong" in text
    assert "more" in text


def test_regressions_shows_the_item_text_so_a_reviewer_can_judge():
    _, text = run("regressions", *args_for(), "--limit", "3")
    assert "(" in text and ")" in text


def test_regressions_shows_every_item_when_the_limit_is_generous():
    _, text = run("regressions", *args_for(), "--limit", "10000")
    assert "more" not in text


def test_churn_tabulates_where_labels_moved():
    code, text = run("churn", *args_for())
    assert code == exit_codes.REGRESSION
    assert "no ground truth to judge by" in text
    assert "billing" in text


def test_report_writes_a_self_contained_html_file(tmp_path: Path):
    target = tmp_path / "nested" / "report.html"
    code, text = run("report", *args_for(), "--out", str(target))
    assert code == exit_codes.REGRESSION
    assert "25000 items" in text
    body = target.read_text()
    assert body.startswith("<!doctype html>")
    assert "<link" not in body


def test_runs_reports_an_empty_directory_plainly(tmp_path: Path):
    code, text = run("runs", "--dir", str(tmp_path))
    assert code == exit_codes.OK
    assert "no runs recorded" in text


def test_runs_lists_a_recorded_comparison(tmp_path: Path):
    run("compare", *args_for(), "--record", str(tmp_path))
    code, text = run("runs", "--dir", str(tmp_path))
    assert code == exit_codes.OK
    assert "prompt-v1-to-prompt-v2" in text
    assert "paired p" in text


def test_no_subcommand_is_a_usage_error():
    with pytest.raises(SystemExit):
        main([])


def test_an_unknown_subcommand_is_a_usage_error():
    with pytest.raises(SystemExit):
        main(["frobnicate"])
