# ADR-005: What this repository cannot measure, and what it would take

Status: accepted
Date: 2026-09-29

## Context

Every figure here comes from a seeded generator and closed-form arithmetic, running offline.
That buys reproducibility, which is the right trade for an artefact a reviewer clones and runs
in two minutes. It does not buy realism, and the difference belongs where a reader will find it.

## Decision

State the boundary claim by claim.

**Claims that hold for any corpus.** That accuracy is a marginal, so two versions can agree on
it exactly and disagree about many items. That only discordant pairs carry information about a
paired change. That a two-proportion z test on paired data treats the agreement as noise and
loses power. That churn is measurable without ground truth and correctness is not. These are
arithmetic and logic.

**Claims that are about this corpus.** Every count and rate. They demonstrate that the
phenomenon has a size, not that it has this size for you. The generator's class skills, drift
rate and confusion map are documented so a reader can set their own and re-measure.

**No model is called.** The label sets are the input. This tool evaluates a comparison between
two sets of labels, not a prompt, and it has no opinion about why the labels differ. Anything
said here about prompts is said about the labels they produced.

**Ground truth is assumed correct.** Every transition is defined against it, so a mislabelled
item is classified confidently and wrongly. An annotation quality pass is a different tool and
would change several figures here.

**The unjudgeable population is counted and never assessed.** The repository reports how many
labels moved with no ground truth behind them and stops. It cannot say whether those moves were
improvements, and the honest position is that nobody can without labelling them.

**Significance is not importance.** The paired test answers whether an improvement is real. It
does not answer whether thirteen net items justify moving two thousand labels, and no test does.
The churn-per-gain figure exists because that question needs a different kind of answer.

**The correlation between versions is a modelling choice.** The generator's redraw rate sets how
much two versions agree, and the real figure depends on the model, the prompt and the taxonomy.
The construction fixes the marginal exactly, so the accuracies are what they claim to be at any
correlation, and the churn figures move with it.

## Alternatives considered

**Ship labels captured from a real pipeline.** More convincing, not distributable, not
reproducible, and it would tie the repository to one taxonomy.

**Call a model in CI to produce the labels.** Adds an API key, a bill and a non-deterministic
result to a repository whose argument does not need one, since the tool starts from labels.

**Present the corpus rates as general.** The standard move, and the reason most portfolio
numbers are discounted on sight.

## Consequences

The README leads with what the tool cannot establish, which costs part of the first screen and
buys the only thing that makes the rest worth reading.

The future work has defined jobs: per-segment breakdowns, a sequential test so annotation can
stop when the question is settled, intervals on the churn figures, and an agreement model that
would let the unjudgeable population be priced rather than only counted.
