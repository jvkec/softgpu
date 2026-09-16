# Decision records

One file per stage. Each record follows the same shape so the project reads
as a series of measured design decisions rather than a changelog.

```
# NNN — Title

## Context
What the previous stage's numbers showed. Link the results JSON.

## Problem
The specific bottleneck or missing capability, stated in measurable terms.

## Options considered
Two or three real alternatives with their tradeoffs. Say which one real
drivers pick and why.

## Decision
What we built and why. Interface changes, if any, called out explicitly.

## Results
Before/after table from results/<tag>.json. Confirm or refute the hypotheses.

## Consequences
What got worse or more complex, what this enables next, open questions.
```
