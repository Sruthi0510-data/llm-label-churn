# ADR-002: The comparison is paired, and the usual test is not

Status: accepted
Date: 2026-09-25

## Context

Comparing two prompt versions means labelling the same items twice. Every item contributes an
observation to both accuracies, so the two proportions are not independent samples, and the
two-proportion z test does not apply to them.

Teams run it anyway. The numbers going in look like two proportions, the test is the one most
people can recall, and nothing about the output announces that the design was paired.

## Decision

Implement McNemar's exact test as the verdict, and implement the two-proportion z test as well
so the comparison can be published rather than described.

McNemar uses only the discordant pairs: items one version got right and the other got wrong.
Everything else is agreement and carries no information about the change, however much of it
there is. Under the null that the change is neutral, each discordant pair is a coin flip, so
the exact test is binomial on the smaller count out of the total.

Exact rather than the chi-squared approximation, because a real evaluation set produces tens of
discordant pairs rather than hundreds, which is the regime where the approximation is least
reliable and the answer matters most.

## Alternatives considered

**Only the paired test, with the unpaired one described in prose.** Cleaner and unconvincing. A
reader asked to believe that the familiar test loses power should be shown it losing power on
the same data, and on the shipped fixture it loses enough to discard a real improvement.

**A chi-squared McNemar with continuity correction.** Standard and approximate, in the regime
where approximation is worst. The exact version is a binomial tail in log space and costs
nothing at these counts.

**A bootstrap over items.** Correct, and it would give a confidence interval rather than a p
value. Deferred: it needs a decision about what to resample when most items have no ground
truth, and that decision is exactly the thing ADR-001 says not to paper over.

**Report the accuracy interval and let a reader compare.** Included, and insufficient on its
own. On the shipped comparison the two intervals overlap while the paired test is decisive in
the other direction, which is the well known trap of reading overlap as non-significance.

## Consequences

The tool publishes two p values and says which one applies. On one of the three shipped changes
they reach different conclusions, which is the only way to make the point that they are not
answering the same question.

The sample size that matters becomes the discordant count rather than the labelled count, and
the sizing experiment converts between them. On this corpus the discordant rate is under a
tenth, so an evaluation set is roughly ten times smaller than it looks.
