# ADR-001: Ground truth is partial and stays partial

Status: accepted
Date: 2026-09-24

## Context

An enrichment pipeline labels every document it sees. A human labels a sample, because labelling
is the expensive part and nobody funds labelling everything. Those two populations differ by
orders of magnitude: the shipped corpus has twenty-five thousand items and a thousand of them
have a human label.

Every evaluation harness reports on the smaller population. That is correct for accuracy, which
is undefined without ground truth. It is silently wrong for everything else, because the
decision being made is about the corpus and the number on the slide is about the sample.

## Decision

Model them separately and never merge them. `LabelSet` carries a label for every item.
`Corpus.truth` is a partial mapping, and the transition taxonomy has a state for items outside
it.

Churn is therefore measurable on the whole corpus, because it needs two label sets and nothing
else. Correctness is measurable only where a human looked. The gap is reported as a count in
its own right rather than averaged over or inferred.

The state is named `UNEVALUATED_CHANGED` rather than something softer, because the honest
description of those items is that their label moved and nobody is in a position to say whether
that was good. On the shipped comparison there are more than two thousand of them, against
thirteen net items gained on the sample.

## Alternatives considered

**Evaluate on the labelled sample and report that as the result.** Universal, and the reason
this repository exists. It answers a question about a thousand items and is read as an answer
about twenty-five thousand.

**Extrapolate the sample's accuracy to the corpus.** Valid for accuracy under random sampling
and useless for the thing that matters: extrapolating gives an expected number of corpus items
that got worse, and no way to look at any of them.

**Use the earlier version's labels as ground truth where none exists.** Tempting, and the first
version of the coverage experiment did exactly that. It makes the earlier version correct by
construction on every extended item and biases the comparison against the change under test.

**Refuse to report anything about unlabelled items.** Consistent and it throws away the
finding. The churn on those items is real, measurable, and the largest thing the change did.

## Consequences

Two populations appear in every output, and readers ask which one a number is about, which is
the right question and one a single accuracy figure does not prompt.

The generator draws its ground truth sample from a separate random stream, so changing how much
is labelled changes which items a human looked at and nothing else. Without that the coverage
sweep would compare a different corpus at every point while appearing to compare the same one.
