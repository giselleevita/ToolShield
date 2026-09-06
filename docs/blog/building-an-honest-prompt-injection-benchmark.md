# Building an honest prompt-injection benchmark

*A design note on the evaluation choices behind [ToolShield](https://github.com/giselleevita/ToolShield).
The numbers referenced here come from the committed results; where a result is still
provisional, this note says so.*

Most prompt-injection detectors are reported on a single number — accuracy or ROC-AUC
on a random train/test split — and most of those numbers are too good. Not because
the models are bad, but because the evaluation quietly lets the model see at test time
what it already saw in training. ToolShield is a bachelor's-thesis benchmark built to
not do that. This is what "not doing that" actually required.

## 1. Random splits leak, and the leak flatters the model

The dataset is generated from versioned templates: a fixed set of attack and benign
patterns, instantiated with different tools and payloads. If you split that dataset
randomly, near-identical instantiations of the *same template* land on both sides of
the split. A model can then score well by recognising the template, not by
generalising to the attack.

ToolShield replaces the random split with three protocols:

| Protocol | What is held out | What it simulates |
|---|---|---|
| `S_random` | nothing (stratified random) | the industry-standard number, kept for comparison |
| `S_attack_holdout` | entire attack families | a novel attack class appears in production |
| `S_tool_holdout` | entire tools | the agent gets a tool the detector was never trained on |

The gap between `S_random` and the holdout protocols is the finding. A TF-IDF model
that looks essentially perfect on `S_random` picks up a large false-positive penalty
at the same detection rate once attack families are held out (see
[`EXEC_SUMMARY.md`](../../EXEC_SUMMARY.md) and the
[results explorer](https://giselleevita.github.io/ToolShield/) for the current
figures). That penalty is the generalisation gap a deployer actually inherits, and a
random-split evaluation hides all of it.

Template isolation is enforced in the split code and checked by tests, so a future
change that reintroduces leakage fails CI rather than silently inflating the results.

## 2. Accuracy is the wrong axis; operational metrics are the right one

A detector that blocks 99% of attacks is useless if it also blocks 20% of legitimate
tool calls. ToolShield reports:

- **FPR @ TPR = 0.90 / 0.95** — false-positive rate at a *fixed* detection rate, so
  two models are compared at the same security level rather than at whatever operating
  point makes each look best.
- **ASR reduction** — attack success rate before vs. after the guard, which is the
  number a security reviewer actually cares about.
- **FPR budgets (1% / 3% / 5%)** — thresholds are chosen to fit a stated
  false-positive budget, because in production you pick the budget first and the
  threshold follows.
- **Latency P50 / P95** — a guard on the tool-call path is in the request critical
  path; if it is slow, it will be removed.

## 3. Say what went wrong

The committed neural-model results were produced with only one effective training
seed: an experiment-runner key did not reach the model, so the multi-seed loop ran
but every neural run got the same seed. The classical models were unaffected.

Two ways to handle that. One is to quietly re-run and replace the numbers. The other
is to fix the defect, add a regression test for seed propagation, label the affected
figures as point estimates, and leave the disclosure in the README until a full
multi-seed run has actually been repeated. ToolShield does the second. A benchmark
whose headline claim is "other people's evaluations are too optimistic" does not get
to be sloppy about its own.

## 4. The detector is not the enforcement

ToolShield measures whether a signal *can* be produced before a tool call. It does
not decide policy, hold authorisation, sandbox anything, or handle multi-turn agent
state. Turning a detection score into a safe allow/block decision — with approvals,
audit, and fail-closed behaviour when the signal is uncertain — is a separate system;
[Agent Security Gate](https://github.com/giselleevita/agent-security-gate) is that
side of the boundary. Keeping the two apart is deliberate: a benchmark that also
tried to be a runtime would be honest about neither job.

## Takeaways for anyone evaluating a guard model

1. If your split is random, assume your robustness number is optimistic until proven
   otherwise. Hold out attack types and tools, not just rows.
2. Report false positives at a fixed true-positive rate, not accuracy.
3. Publish the operating-point selection rule (the FPR budget), not just a curve.
4. Write down what broke in your own pipeline. It is the fastest way for a reviewer
   to trust the parts that didn't.
