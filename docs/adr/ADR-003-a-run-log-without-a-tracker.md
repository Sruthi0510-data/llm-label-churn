# ADR-003: A run log without a tracker

Status: accepted
Date: 2026-09-26

## Context

Every comparison is a run: it has parameters, metrics, and a set of items worth keeping. That is
exactly the shape MLflow records, and reaching for MLflow is the obvious move.

It is also a server, a backing database, a version pin and a tracking URI between a reviewer and
the numbers on the README. For a repository whose whole promise is that a reader clones it and
reproduces every figure with one command and no network, that is a large price for a run id.

## Decision

A directory of JSON files, with the fields a tracker uses: a run id, a start time, params,
metrics, tags and artifacts. The optional `mlflow` extra mirrors the same run into a tracking
server for a team that already runs one.

The shape is deliberately the tracker's rather than something bespoke, so the mirror is a
field-for-field copy rather than a translation somebody has to maintain. `mirror_to_mlflow`
is twelve lines because of that choice.

Two details are load bearing. The run id and start time are injectable, so a test asserts what a
run records rather than what the clock did. And metrics refuse anything that is not a number,
including booleans: `True` would log as 1.0 and read as a measurement.

## Alternatives considered

**MLflow as a required dependency.** A server between the reader and the numbers, and a schema
migration the first time the pin moves.

**MLflow behind an extra, used when available.** Two code paths that can disagree about what was
recorded, which is a second correctness problem in a repository that already has opinions about
tools reporting different things.

**No run log at all: print and move on.** Loses the comparison between comparisons, which is the
thing a team actually wants after the third prompt version.

**A SQLite file.** Queryable and opaque in a diff. JSON in a directory is reviewable in a pull
request, which is where a prompt change is discussed.

## Consequences

The log is readable without any tool, and `lac runs` is fifteen lines.

A team with MLflow gets the same fields in their server through one function, and the import of
`mlflow` sits below the validation so a caller with an unstarted run is told that rather than
told to install a package they may not need. A test makes the module unimportable to pin the
ordering, because a machine with the extra installed can never notice the bug.
