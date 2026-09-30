"""The command line surface. Exit codes are an interface, published in the README."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from lac import exit_codes
from lac.compare import compare, record
from lac.errors import ConfigError, DataError, LacError, UnderpoweredError
from lac.generator import GeneratorConfig, generate
from lac.labels import load_corpus, load_label_set, write_json
from lac.runs import RunStore

DEFAULT_CORPUS = Path("fixtures/corpus.json")
DEFAULT_BEFORE = Path("fixtures/prompt-v1.json")
DEFAULT_AFTER = Path("fixtures/prompt-v2.json")
DEFAULT_RUNS = Path("runs")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lac",
        description=(
            "Show what an aggregate accuracy delta hides: which items a change broke, and "
            "whether the change is distinguishable from noise."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="write the deterministic corpus and three versions")
    gen.add_argument("--out", type=Path, default=Path("fixtures"))
    gen.add_argument("--items", type=int, default=25_000)
    gen.add_argument("--labelled", type=int, default=1_000)
    gen.add_argument("--seed", type=int, default=20260930)

    for name, help_text in (
        ("compare", "run one version-to-version comparison"),
        ("regressions", "list the items that went from right to wrong"),
        ("churn", "show where labels moved, by class"),
        ("report", "write a self-contained HTML report"),
    ):
        node = sub.add_parser(name, help=help_text)
        node.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
        node.add_argument("--before", type=Path, default=DEFAULT_BEFORE)
        node.add_argument("--after", type=Path, default=DEFAULT_AFTER)
        node.add_argument("--alpha", type=float, default=0.05)
        if name == "compare":
            node.add_argument("--json", type=Path, default=None)
            node.add_argument("--record", type=Path, default=None)
            node.add_argument("--require-discordant", type=int, default=None)
        if name == "regressions":
            node.add_argument("--limit", type=int, default=10)
        if name == "report":
            node.add_argument("--out", type=Path, default=Path("docs/report.html"))

    runs = sub.add_parser("runs", help="list recorded comparisons")
    runs.add_argument("--dir", type=Path, default=DEFAULT_RUNS)
    return parser


def _comparison(args: argparse.Namespace):
    corpus = load_corpus(args.corpus)
    before = load_label_set(args.before)
    after = load_label_set(args.after)
    return compare(
        corpus,
        before,
        after,
        alpha=args.alpha,
        require_discordant=getattr(args, "require_discordant", None),
    )


def _cmd_generate(args: argparse.Namespace, out) -> int:
    produced = generate(
        GeneratorConfig(items=args.items, labelled=args.labelled, seed=args.seed)
    )
    write_json(
        args.out / "corpus.json",
        {
            "items": [item.as_dict() for item in produced.corpus.items],
            "truth": dict(produced.corpus.truth),
        },
        compact=True,
    )
    for name, label_set in produced.versions.items():
        write_json(
            args.out / f"{name}.json",
            {"name": name, "labels": dict(label_set.labels)},
            compact=True,
        )
    print(
        f"wrote {len(produced.corpus.items)} items, {len(produced.corpus.truth)} labelled, "
        f"and {len(produced.versions)} versions to {args.out}",
        file=out,
    )
    return exit_codes.OK


def _cmd_compare(args: argparse.Namespace, out) -> int:
    comparison = _comparison(args)
    summary = comparison.summary()
    print(json.dumps(summary, indent=2), file=out)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.record:
        run = record(comparison, RunStore(args.record))
        path = RunStore(args.record).save(run)
        print(f"recorded run at {path}", file=out)
    return exit_codes.REGRESSION if comparison.matrix.regressed else exit_codes.OK


def _cmd_regressions(args: argparse.Namespace, out) -> int:
    comparison = _comparison(args)
    items = load_corpus(args.corpus).by_id()
    regressions = comparison.matrix.regressions()
    print(f"{len(regressions)} item(s) went from right to wrong", file=out)
    for entry in regressions[: args.limit]:
        item = items[entry.item_id]
        print(
            f"  {entry.item_id}  {entry.truth} -> {entry.after}  ({item.text})",
            file=out,
        )
    if len(regressions) > args.limit:
        print(f"  ... {len(regressions) - args.limit} more", file=out)
    return exit_codes.REGRESSION if regressions else exit_codes.OK


def _cmd_churn(args: argparse.Namespace, out) -> int:
    comparison = _comparison(args)
    matrix = comparison.matrix
    print(f"{'class':<20}{'out':>8}{'in':>8}{'net':>8}", file=out)
    print("-" * 44, file=out)
    for name, movement in matrix.per_class_churn().items():
        net = movement["in"] - movement["out"]
        print(f"{name:<20}{movement['out']:>8}{movement['in']:>8}{net:>+8}", file=out)
    print(
        f"\n{matrix.changed} of {matrix.total} labels moved, "
        f"{matrix.unevaluated_changed} of them with no ground truth to judge by",
        file=out,
    )
    return exit_codes.REGRESSION if matrix.regressed else exit_codes.OK


def _cmd_report(args: argparse.Namespace, out) -> int:
    from lac.report import build_report

    comparison = _comparison(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_report(comparison), encoding="utf-8")
    print(f"wrote {args.out} for {comparison.matrix.total} items", file=out)
    return exit_codes.REGRESSION if comparison.matrix.regressed else exit_codes.OK


def _cmd_runs(args: argparse.Namespace, out) -> int:
    store = RunStore(args.dir)
    paths = store.list_runs()
    if not paths:
        print(f"no runs recorded in {args.dir}", file=out)
        return exit_codes.OK
    for path in paths:
        run = store.load(path)
        delta = run.metrics.get("accuracy_delta_points", 0.0)
        paired = run.metrics.get("paired_p", 1.0)
        print(f"{run.name:<28}{delta:+7.2f}pts  paired p {paired:.4f}  {path.name}", file=out)
    return exit_codes.OK


HANDLERS = {
    "generate": _cmd_generate,
    "compare": _cmd_compare,
    "regressions": _cmd_regressions,
    "churn": _cmd_churn,
    "report": _cmd_report,
    "runs": _cmd_runs,
}


def main(argv: Sequence[str] | None = None, out=None) -> int:
    """Entry point. Returns a code rather than calling sys.exit, so tests can drive it."""
    out = out or sys.stdout
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        return HANDLERS[args.command](args, out)
    except UnderpoweredError as exc:
        print(f"underpowered: {exc}", file=out)
        return exit_codes.UNDERPOWERED
    except (ConfigError, DataError) as exc:
        print(f"error: {exc}", file=out)
        return exit_codes.USAGE
    except LacError as exc:  # pragma: no cover - defensive, no current path reaches it
        print(f"error: {exc}", file=out)
        return exit_codes.USAGE


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
