# ADR-004: Exit codes separate a finding from a failure

Status: accepted
Date: 2026-09-27

## Context

A comparison tool has three genuinely different outcomes and the Unix convention has room for
two. "I ran and nothing regressed", "I ran and found regressions" and "I could not run" all
collapse into zero or non-zero, and the collapse has a predictable consequence: a CI step that
treats every non-zero code as a crash reports a real regression as a broken tool, somebody marks
the step `continue-on-error`, and nothing is checked from then on.

There is a fourth outcome specific to an evaluation tool. A comparison can complete, report a
clean result, and rest on too few discordant pairs for that result to mean anything.

## Decision

Four codes, published in the README as an interface.

| code | meaning |
| --- | --- |
| 0 | ran, no item went from right to wrong |
| 3 | ran, found regressed items. The tool worked |
| 2 | could not answer: bad arguments, unreadable label set, mismatched items |
| 4 | the labelled sample cannot support the guarantee the caller asked for |

Three rather than one, deliberately: one is what a crashed interpreter returns, and the gap
makes the distinction visible in a CI log rather than merely documented.

Code 4 is opt in, behind `--require-discordant`. A check that fails by default gets weakened
within a week and then nothing is checked, which is the same failure arriving by another road.
The flag guards the discordant count rather than the labelled count, because only discordant
pairs carry information and guarding the wrong number would be a check that passes while the
comparison rests on four flips.

## Alternatives considered

**Zero unless the tool crashed.** Forces every caller to parse output to learn whether to act,
and the parser is what breaks when the output improves.

**Non-zero for any regression, code 1.** The common choice and the one that produces the
`continue-on-error` outcome.

**Exit non-zero when the paired test is not significant.** Considered and rejected: a
non-significant result is a fact about the evidence, not a defect, and failing a build on it
would make the tool an obstacle rather than an instrument.

## Consequences

A CI step reads `if [ $code -eq 3 ]` and knows a finding from a crash without parsing anything.

The codes are a compatibility surface. The constants live in one module with the meanings in
their docstrings, and tests assert that a regression and a failure to answer return different
codes rather than asserting the literals, so the property survives a renumbering.
