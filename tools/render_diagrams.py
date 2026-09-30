"""Render the committed SVGs from the experiment JSON, so a picture cannot outlive its numbers.

Hand-emitted rather than Mermaid, for the four reasons ADR-005 records. The bar chart follows
the visualisation rules the repository was built against: one axis, categorical hues in fixed
order, a legend plus a labelled threshold so identity is never carried by colour alone, thin
marks with a surface gap between adjacent fills, and a recessive grid. The palette is the
validated default set.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "docs" / "metrics.json"
TESTS = ROOT / "docs" / "experiments" / "exp02_the_wrong_test.json"
OUT = ROOT / "docs" / "diagrams"

SURFACE = "#fcfcfb"
PLANE = "#f9f9f7"
INK = "#0b0b0b"
SECOND = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#b42318"


def _text(x, y, content, size=12, fill=INK, weight="normal", anchor="start"):
    return (
        f'<text x="{x}" y="{y}" font-family="Helvetica,Arial,sans-serif" font-size="{size}" '
        f'fill="{fill}" font-weight="{weight}" text-anchor="{anchor}">{escape(str(content))}</text>'
    )


def _document(width, height, body):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{PLANE}"/>'
        f"{body}</svg>"
    )


def flow(metrics: dict) -> str:
    """Where every item goes, and which two boxes the accuracy delta is made of."""
    parts: list[str] = []
    width, height = 760, 430
    parts.append(_text(24, 34, "What the delta is made of", 17, INK, "bold"))
    parts.append(
        _text(24, 56,
              f"{metrics['corpus_items']:,} items, {metrics['labelled_items']:,} of them with "
              f"ground truth", 12, SECOND)
    )

    boxes = [
        ("held right", metrics["held_right"], MUTED, "carries no information"),
        ("held wrong", _held_wrong(metrics), MUTED, "carries no information"),
        ("improved", metrics["improved"], AQUA, "a discordant pair"),
        ("regressed", metrics["regressed"], RED, "a discordant pair"),
    ]
    top = 96
    for index, (label, count, colour, note) in enumerate(boxes):
        y = top + index * 54
        parts.append(
            f'<rect x="24" y="{y}" width="330" height="44" rx="6" fill="{SURFACE}" '
            f'stroke="{colour}" stroke-width="1.5"/>'
        )
        parts.append(_text(42, y + 20, label, 13, colour, "bold"))
        parts.append(_text(42, y + 36, note, 10, MUTED))
        parts.append(_text(336, y + 27, f"{count:,}", 16, INK, "bold", anchor="end"))

    parts.append(
        f'<rect x="386" y="{top}" width="344" height="98" rx="6" fill="{SURFACE}" '
        f'stroke="{GRID}" stroke-width="1"/>'
    )
    parts.append(_text(404, top + 26, "the accuracy delta", 13, INK, "bold"))
    parts.append(
        _text(404, top + 48,
              f"{metrics['improved']} improved minus {metrics['regressed']} regressed", 11, SECOND)
    )
    parts.append(
        _text(404, top + 68,
              f"= {metrics['net_gain']} items, which is "
              f"{metrics['accuracy_delta_points']:+} points", 12, INK, "bold")
    )
    parts.append(
        _text(404, top + 86, "the two grey boxes contribute nothing to it", 10, MUTED)
    )

    parts.append(
        f'<rect x="386" y="{top + 116}" width="344" height="98" rx="6" fill="{SURFACE}" '
        f'stroke="{ORANGE}" stroke-width="1.5"/>'
    )
    parts.append(_text(404, top + 142, "and what it does not mention", 13, ORANGE, "bold"))
    parts.append(
        _text(404, top + 164, f"{metrics['labels_changed']:,} labels moved across the corpus",
              11, INK)
    )
    parts.append(
        _text(404, top + 182,
              f"{metrics['unevaluated_changed']:,} of those with no ground truth", 11, INK)
    )
    parts.append(
        _text(404, top + 200,
              f"{metrics['churn_per_net_gain']} labels moved per net item gained", 11, INK)
    )

    parts.append(
        _text(24, 340,
              "Only the two coloured boxes carry evidence about the change. The delta is those "
              "two", 11, SECOND)
    )
    parts.append(
        _text(24, 358, "subtracted from each other, and the subtraction is where the rest goes.",
              11, SECOND)
    )
    return _document(width, height, "".join(parts))


def _held_wrong(metrics: dict) -> int:
    return metrics["held_wrong"]


def tests_chart(metrics: dict, records: list[dict]) -> str:
    """Both p values per change, on a log axis, against the threshold a team would use."""
    parts: list[str] = []
    width, height = 760, 400
    left, right, top, bottom = 150, 600, 110, 290

    parts.append(_text(24, 34, "The test a team runs, against the test that applies", 17, INK,
                       "bold"))
    parts.append(
        _text(24, 56, "the comparison is paired; the unpaired test treats the agreement as noise",
              12, SECOND)
    )
    parts.append(
        f'<rect x="{left - 126}" y="{top - 26}" width="{right - left + 260}" '
        f'height="{bottom - top + 74}" rx="6" fill="{SURFACE}" stroke="{GRID}" '
        f'stroke-width="1"/>'
    )

    for index, (colour, label) in enumerate(((BLUE, "McNemar, paired"),
                                             (ORANGE, "two-proportion, unpaired"))):
        x = 24 + index * 210
        parts.append(f'<rect x="{x}" y="{82}" width="11" height="9" fill="{colour}"/>')
        parts.append(_text(x + 17, 91, label, 11, SECOND))

    floor = -4.0

    def px(p: float) -> float:
        value = max(math.log10(max(p, 1e-4)), floor)
        return left + (value - floor) / (0.0 - floor) * (right - left)

    for power in range(-4, 1):
        x = px(10.0**power)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top - 10}" x2="{x:.1f}" y2="{bottom}" stroke="{GRID}" '
            f'stroke-width="1"/>'
        )
        parts.append(_text(x, bottom + 18, f"1e{power}" if power else "1", 10, MUTED,
                           anchor="middle"))

    alpha_x = px(0.05)
    parts.append(
        f'<line x1="{alpha_x:.1f}" y1="{top - 16}" x2="{alpha_x:.1f}" y2="{bottom + 2}" '
        f'stroke="{RED}" stroke-width="1.5" stroke-dasharray="5 3"/>'
    )
    parts.append(_text(alpha_x, top - 22, "alpha", 10, RED, "bold", anchor="middle"))

    row_height = (bottom - top) / len(records)
    for index, record in enumerate(records):
        y = top + index * row_height + 10
        before = record["before"].replace("prompt-", "")
        after = record["after"].replace("prompt-", "")
        label = f"{before} to {after}"
        parts.append(_text(left - 14, y + 18, label, 11, INK, anchor="end"))
        for offset, (key, colour) in enumerate((("paired_p", BLUE), ("unpaired_p", ORANGE))):
            bar_y = y + offset * 14
            parts.append(
                f'<rect x="{left}" y="{bar_y:.1f}" width="{max(2.0, px(record[key]) - left):.1f}" '
                f'height="10" rx="2" fill="{colour}"/>'
            )
            parts.append(
                _text(px(record[key]) + 8, bar_y + 9, f"{record[key]:.4f}", 10, colour, "bold")
            )

    parts.append(
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="{AXIS}" '
        f'stroke-width="1.5"/>'
    )
    parts.append(
        _text((left + right) / 2, bottom + 40, "p value, log scale", 11, MUTED, anchor="middle")
    )
    parts.append(
        _text(24, bottom + 74,
              f"on v2 to v3 the paired test says {metrics['v3_paired_p']} and the unpaired test "
              f"says {metrics['v3_unpaired_p']}:", 11, INK)
    )
    parts.append(
        _text(24, bottom + 92,
              "a real improvement the test a team reaches for would have thrown away.", 11, INK)
    )
    return _document(width, height, "".join(parts))


def main() -> int:
    metrics = json.loads(METRICS.read_text())["metrics"]
    records = json.loads(TESTS.read_text())["records"]
    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    for name, svg in (("flow", flow(metrics)), ("tests", tests_chart(metrics, records))):
        path = OUT / f"{name}.svg"
        path.write_text(svg, encoding="utf-8")
        written.append(path)
    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "diagrams": [p.name for p in written],
                "source": "tools/render_diagrams.py",
                "palette": {"categorical": [BLUE, ORANGE, AQUA], "surface": SURFACE},
                "note": "figures are read from the experiment JSON, so a diagram cannot go stale",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for path in written:
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
