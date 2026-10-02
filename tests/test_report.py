"""The HTML report, including the class collision a passing unit suite cannot see."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from lac.compare import compare
from lac.report import CLASSES, STYLESHEET, build_report


@pytest.fixture(scope="module")
def differing_report(shipped) -> str:
    corpus, versions = shipped
    return build_report(compare(corpus, versions["prompt-v1"], versions["prompt-v2"]))


@pytest.fixture(scope="module")
def disagreeing_report(shipped) -> str:
    corpus, versions = shipped
    return build_report(compare(corpus, versions["prompt-v2"], versions["prompt-v3"]))


@pytest.fixture(scope="module")
def agreeing_report(shipped) -> str:
    corpus, versions = shipped
    return build_report(compare(corpus, versions["prompt-v1"], versions["prompt-v3"]))


def _classes_used(html: str) -> set[str]:
    used: set[str] = set()
    for match in re.finditer(r"class=['\"]([^'\"]+)['\"]", html):
        used.update(match.group(1).split())
    return used


def test_the_report_is_a_complete_html_document(differing_report):
    assert differing_report.startswith("<!doctype html>")
    assert differing_report.rstrip().endswith("</html>")


def test_the_report_inlines_its_stylesheet(differing_report):
    assert "<style>" in differing_report
    assert "<link" not in differing_report


def test_a_decision_conflict_gets_the_finding_verdict(differing_report):
    assert 'class="verdict-finding"' in differing_report
    assert "ACCURACY DELTA AND THE PAIRED TEST DISAGREE" in differing_report


def test_a_test_conflict_gets_its_own_headline(disagreeing_report):
    assert 'class="verdict-finding"' in disagreeing_report
    assert "PAIRED AND UNPAIRED TESTS DISAGREE" in disagreeing_report


def test_full_agreement_gets_the_clear_verdict(agreeing_report):
    assert 'class="verdict-clear"' in agreeing_report
    assert "ALL AGREE" in agreeing_report


def test_the_two_verdicts_never_appear_together(differing_report, agreeing_report):
    assert 'class="verdict-clear"' not in differing_report
    assert 'class="verdict-finding"' not in agreeing_report


@pytest.mark.parametrize("heading", ["summary", "transitions", "tests"])
def test_every_heading_the_screenshot_tool_frames_by_is_present(heading, differing_report):
    assert f'id="{heading}"' in differing_report


def test_the_transition_table_says_which_rows_carry_evidence(differing_report):
    assert "a discordant pair, and evidence" in differing_report
    assert "no information about the change" in differing_report


def test_the_report_names_the_unjudgeable_population(differing_report):
    assert "measurable churn, unmeasurable correctness" in differing_report


def test_the_report_names_both_tests_and_what_they_are(differing_report):
    assert "McNemar exact, paired" in differing_report
    assert "the test teams run" in differing_report


def test_every_class_in_the_markup_is_declared_and_styled(
    differing_report, disagreeing_report, agreeing_report
):
    used = (
        _classes_used(differing_report)
        | _classes_used(disagreeing_report)
        | _classes_used(agreeing_report)
    )
    assert sorted(used - CLASSES) == []
    assert sorted(name for name in used if f".{name}" not in STYLESHEET) == []


def test_every_declared_class_has_a_rule_so_the_list_cannot_rot():
    assert sorted(name for name in CLASSES if f".{name}" not in STYLESHEET) == []


def test_the_two_verdict_classes_have_different_backgrounds():
    finding = re.search(r"\.verdict-finding \{[^}]*background: (#[0-9a-f]+)", STYLESHEET)
    clear = re.search(r"\.verdict-clear \{[^}]*background: (#[0-9a-f]+)", STYLESHEET)
    assert finding and clear
    assert finding.group(1) != clear.group(1)


def test_the_flow_colours_are_distinguishable():
    colours = {
        re.search(rf"\.flow-{state} \{{ color: (#[0-9a-f]+)", STYLESHEET).group(1)
        for state in ("good", "bad", "flat")
    }
    assert len(colours) == 3


def test_version_names_are_escaped_rather_than_interpolated_raw(shipped):
    from dataclasses import replace

    corpus, versions = shipped
    hostile = replace(versions["prompt-v2"], name="<script>x</script>")
    html = build_report(compare(corpus, versions["prompt-v1"], hostile))
    assert "<script>x</script>" not in html
    assert "&lt;script&gt;" in html


ROOT = Path(__file__).resolve().parents[1]


def test_the_proof_of_work_transcript_exists_and_covers_several_exit_codes():
    """A transcript that only ever shows a zero proves the least interesting thing."""
    payload = json.loads((ROOT / "docs" / "proof-of-work.json").read_text())
    codes = {capture["exit_code"] for capture in payload["captures"]}
    assert len(codes) >= 3


def test_every_captured_command_exited_as_the_readme_claims():
    payload = json.loads((ROOT / "docs" / "proof-of-work.json").read_text())
    for capture in payload["captures"]:
        assert capture["exit_code"] == capture["expected_exit_code"], capture["command"]


def test_every_captured_command_carries_a_hash_of_its_untruncated_output():
    payload = json.loads((ROOT / "docs" / "proof-of-work.json").read_text())
    for capture in payload["captures"]:
        assert len(capture["output_sha256"]) == 64


def test_the_transcript_leaks_no_absolute_path():
    svg = (ROOT / "docs" / "img" / "proof-of-work.svg").read_text(encoding="utf-8")
    assert str(ROOT) not in svg
    assert "/usr/bin/python" not in svg


def test_the_transcript_is_valid_svg_with_a_title_and_description():
    import xml.etree.ElementTree as ET

    namespace = "{http://www.w3.org/2000/svg}"
    root = ET.parse(ROOT / "docs" / "img" / "proof-of-work.svg").getroot()
    assert root.find(f"{namespace}title") is not None
    assert root.find(f"{namespace}desc") is not None


def test_the_readme_embeds_the_transcript():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/img/proof-of-work.svg" in text
