"""A file-backed run log, in the shape an experiment tracker uses, with no tracker required.

Every comparison is a run: it has parameters, metrics, and a set of items worth keeping. That
is exactly what MLflow records, and importing MLflow to get it would put a server, a database
and a version pin between a reviewer and the numbers on the README.

So the log is a directory of JSON files with the fields a tracker uses, and the optional
`mlflow` extra mirrors the same run into a tracking server for a team that already has one. The
shape is deliberately the tracker's rather than something bespoke: a run id, a start time,
params, metrics, tags, and artifacts, so the mirror is a field-for-field copy rather than a
translation that has to be maintained.

The import of `mlflow` sits below the validation, and a test makes the module unimportable to
pin that ordering, because a machine with the extra installed can never notice the bug.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lac.errors import ConfigError

MLFLOW_HINT = "install the optional extra: pip install 'llm-label-churn[mlflow]'"


@dataclass
class Run:
    """One recorded comparison."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    run_id: str = ""
    started_at: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigError("a run needs a name")

    def log_param(self, key: str, value: Any) -> None:
        self.params[key] = value

    def log_metric(self, key: str, value: float) -> None:
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise ConfigError(f"metric {key!r} must be a number, got {type(value).__name__}")
        self.metrics[key] = float(value)

    def set_tag(self, key: str, value: str) -> None:
        self.tags[key] = str(value)

    def log_artifact(self, key: str, payload: Any) -> None:
        self.artifacts[key] = payload

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "name": self.name,
            "started_at": self.started_at,
            "params": self.params,
            "metrics": self.metrics,
            "tags": self.tags,
            "artifacts": self.artifacts,
        }


class RunStore:
    """A directory of runs. Deterministic ids on request, so a fixture can be committed."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def start(self, name: str, run_id: str | None = None, started_at: float | None = None) -> Run:
        """Open a run. `run_id` and `started_at` are injectable so a test is not a clock test."""
        return Run(
            name=name,
            run_id=run_id or uuid.uuid4().hex[:12],
            started_at=time.time() if started_at is None else started_at,
        )

    def save(self, run: Run) -> Path:
        """Write the run and return where it went."""
        if not run.run_id:
            raise ConfigError("a run must be started before it can be saved")
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{run.name}-{run.run_id}.json"
        path.write_text(
            json.dumps(run.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return path

    def load(self, path: Path) -> Run:
        if not path.exists():
            raise ConfigError(f"run not found: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{path} is not valid JSON: {exc}") from exc
        return Run(
            name=payload["name"],
            params=payload.get("params", {}),
            metrics=payload.get("metrics", {}),
            tags=payload.get("tags", {}),
            artifacts=payload.get("artifacts", {}),
            run_id=payload.get("run_id", ""),
            started_at=payload.get("started_at", 0.0),
        )

    def list_runs(self) -> list[Path]:
        if not self.directory.exists():
            return []
        return sorted(self.directory.glob("*.json"))


def mirror_to_mlflow(run: Run, tracking_uri: str = "") -> str:  # pragma: no cover - needs extra
    """Copy a recorded run into an MLflow tracking server, field for field.

    Validation runs before the import so a caller who passes an unstarted run is told that,
    rather than being told to install a package they may not need.
    """
    if not run.run_id:
        raise ConfigError("a run must be started before it can be mirrored")
    if not run.metrics:
        raise ConfigError("refusing to mirror a run with no metrics")
    try:
        import mlflow
    except ModuleNotFoundError as exc:
        raise ConfigError(f"mirroring needs the optional extra; {MLFLOW_HINT}") from exc
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(run_name=run.name) as active:
        mlflow.log_params(run.params)
        mlflow.log_metrics(run.metrics)
        mlflow.set_tags(run.tags)
        return active.info.run_id
