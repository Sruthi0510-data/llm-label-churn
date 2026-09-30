"""A self-contained HTML report with stable heading ids and one declared class list.

The screenshot tool frames by heading, so a renamed heading fails loudly rather than producing
half a table. Every class in the markup is declared in CLASSES and has a rule in STYLESHEET,
and a test asserts the two agree.
"""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from lac.compare import Comparison

CLASSES = frozenset(
    {
        "page", "masthead", "subtitle", "verdict-finding", "verdict-clear", "section",
        "grid", "cell", "cell-label", "cell-value", "cell-note", "data-table", "numeric",
        "flow-good", "flow-bad", "flow-flat", "test-agree", "test-disagree", "footnote",
    }
)

STYLESHEET = """
:root { color-scheme: light; }
body { margin: 0; background: #f9f9f7; color: #0b0b0b;
       font-family: -apple-system, Helvetica, Arial, sans-serif; }
.page { max-width: 900px; margin: 0 auto; padding: 32px 24px 64px; }
.masthead { font-size: 26px; font-weight: 700; margin: 0 0 6px; }
.subtitle { color: #52514e; font-size: 14px; margin: 0 0 20px; }
.verdict-finding { display: inline-block; padding: 8px 16px; border-radius: 999px;
                   background: #8a3b12; color: #ffffff; font-weight: 700; font-size: 15px; }
.verdict-clear { display: inline-block; padding: 8px 16px; border-radius: 999px;
                 background: #0b6b5a; color: #ffffff; font-weight: 700; font-size: 15px; }
.section { margin-top: 36px; }
.section h2 { font-size: 18px; margin: 0 0 12px; padding-bottom: 6px;
              border-bottom: 1px solid #e1e0d9; }
.grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
.cell { background: #fcfcfb; border: 1px solid #e1e0d9; border-radius: 8px; padding: 14px 16px; }
.cell-label { color: #52514e; font-size: 12px; text-transform: uppercase;
              letter-spacing: 0.04em; }
.cell-value { font-size: 24px; font-weight: 700; margin-top: 4px; }
.cell-note { color: #52514e; font-size: 12px; margin-top: 4px; }
.data-table { width: 100%; border-collapse: collapse; background: #fcfcfb;
              border: 1px solid #e1e0d9; font-size: 13px; }
.data-table th { text-align: left; padding: 9px 11px; border-bottom: 1px solid #e1e0d9;
                 color: #52514e; font-weight: 600; font-size: 11px; text-transform: uppercase; }
.data-table td { padding: 9px 11px; border-bottom: 1px solid #eaeef2; }
.numeric { text-align: right; font-variant-numeric: tabular-nums; }
.flow-good { color: #0b6b5a; font-weight: 700; }
.flow-bad { color: #b42318; font-weight: 700; }
.flow-flat { color: #52514e; }
.test-agree { color: #0b6b5a; font-weight: 700; }
.test-disagree { color: #8a3b12; font-weight: 700; }
.footnote { color: #52514e; font-size: 12px; margin-top: 28px; line-height: 1.6; }
"""


def _cell(label: str, value: str, note: str) -> str:
    return (
        f'<div class="cell"><div class="cell-label">{escape(label)}</div>'
        f'<div class="cell-value">{escape(value)}</div>'
        f'<div class="cell-note">{escape(note)}</div></div>'
    )


def _summary(comparison: Comparison) -> str:
    matrix = comparison.matrix
    per_gain = matrix.churn_per_net_gain
    cells = [
        _cell(
            "accuracy delta",
            f"{matrix.summary()['accuracy_delta_points']:+} points",
            "the number a slide would carry",
        ),
        _cell(
            "items that got worse",
            str(matrix.regressed),
            f"against {matrix.improved} that got better, on {matrix.evaluated} labelled items",
        ),
        _cell(
            "labels moved",
            f"{matrix.changed:,}",
            f"{matrix.unevaluated_changed:,} of them with no ground truth to judge by",
        ),
        _cell(
            "labels moved per net item gained",
            "no net gain" if per_gain is None else f"{per_gain}",
            "the blast radius of a change nobody costed",
        ),
    ]
    return (
        '<section class="section"><h2 id="summary">Summary</h2>'
        f'<div class="grid">{"".join(cells)}</div></section>'
    )


def _transitions(comparison: Comparison) -> str:
    matrix = comparison.matrix
    rows = [
        ("held right", matrix.count_held_right(), "flow-flat", "no information about the change"),
        ("held wrong", matrix.count_held_wrong(), "flow-flat", "no information about the change"),
        ("improved", matrix.improved, "flow-good", "a discordant pair, and evidence"),
        ("regressed", matrix.regressed, "flow-bad", "a discordant pair, and evidence"),
        (
            "changed with no ground truth",
            matrix.unevaluated_changed,
            "flow-flat",
            "measurable churn, unmeasurable correctness",
        ),
    ]
    body = "".join(
        f'<tr><td><span class="{css}">{escape(label)}</span></td>'
        f'<td class="numeric">{count:,}</td><td>{escape(note)}</td></tr>'
        for label, count, css, note in rows
    )
    return (
        '<section class="section"><h2 id="transitions">Where every item went</h2>'
        '<table class="data-table"><thead><tr><th>transition</th>'
        '<th class="numeric">items</th><th>what it tells you</th></tr></thead>'
        f"<tbody>{body}</tbody></table>"
        '<p class="cell-note">Only the two discordant rows carry information about the change. '
        "An accuracy delta is those two rows subtracted from each other.</p></section>"
    )


def _tests(comparison: Comparison) -> str:
    verdict = comparison.verdict
    css = "test-disagree" if verdict.tests_disagree else "test-agree"
    rows = [
        (
            "McNemar exact, paired",
            f"{verdict.paired_p:.4f}",
            "significant" if verdict.paired_significant else "not significant",
            "the right test: uses the discordant pairs",
        ),
        (
            "two-proportion z, unpaired",
            f"{verdict.unpaired_p:.4f}",
            "significant" if verdict.unpaired_significant else "not significant",
            "the test teams run: treats the agreement as noise",
        ),
    ]
    body = "".join(
        f"<tr><td>{escape(name)}</td><td class=\"numeric\">{p}</td>"
        f'<td><span class="{css}">{escape(call)}</span></td><td>{escape(note)}</td></tr>'
        for name, p, call, note in rows
    )
    conclusion = (
        "The two tests reach different conclusions about the same data."
        if verdict.tests_disagree
        else "The two tests agree here, which they will not always."
    )
    return (
        '<section class="section"><h2 id="tests">What the tests say</h2>'
        '<table class="data-table"><thead><tr><th>test</th><th class="numeric">p</th>'
        "<th>call</th><th>what it is</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
        f'<p class="cell-note">{escape(conclusion)} '
        f"{verdict.discordant} discordant pairs carried all of the evidence.</p></section>"
    )


def build_report(comparison: Comparison) -> str:
    """Render one comparison as a self-contained HTML page."""
    matrix = comparison.matrix
    finding = comparison.decision_would_differ or comparison.verdict.tests_disagree
    verdict_class = "verdict-finding" if finding else "verdict-clear"
    if comparison.decision_would_differ:
        headline = "THE ACCURACY DELTA AND THE PAIRED TEST DISAGREE"
    elif comparison.verdict.tests_disagree:
        headline = "THE PAIRED AND UNPAIRED TESTS DISAGREE"
    else:
        headline = "ACCURACY, THE PAIRED TEST AND THE UNPAIRED TEST ALL AGREE"
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>label churn report</title>"
        f"<style>{STYLESHEET}</style></head><body><div class='page'>"
        '<h1 class="masthead">Label churn</h1>'
        f'<p class="subtitle">{escape(matrix.before_name)} against '
        f"{escape(matrix.after_name)}, {matrix.total:,} items, {matrix.evaluated:,} with "
        f"ground truth</p>"
        f'<span class="{verdict_class}">{escape(headline)}</span>'
        f"{_summary(comparison)}{_transitions(comparison)}{_tests(comparison)}"
        '<p class="footnote">Accuracy is a marginal. Two versions can agree on it exactly and '
        "disagree about a tenth of the items, and a change that raises it can be made of items "
        "that got worse cancelling items that got better. The rows above are what the "
        "subtraction removes.</p>"
        "</div></body></html>"
    )
