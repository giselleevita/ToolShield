# ToolShield in 90 seconds

**0:00–0:15 — Problem.** Open the [results explorer](https://giselleevita.github.io/ToolShield/) and explain that prompt filters must be evaluated on unseen attacks and unseen tools, not only random splits.

**0:15–0:40 — Evidence.** Switch between `S_random`, `S_attack_holdout`, and `S_tool_holdout`. Compare ROC-AUC, false-positive behavior, latency, and the number of genuinely independent seeds. Point out the explicit single-seed limitation for neural results.

**0:40–1:05 — Decision boundary.** Set the score to `0.55`. Show how the same signal can allow or review a read while blocking a privileged tool. The policy hash makes the decision configuration identifiable.

**1:05–1:20 — Reproducibility.** Show the report schema, experiment commit, configuration hash, policy hash, and verified split-file count.

**1:20–1:30 — Boundary.** ToolShield evaluates detector behavior; [Agent Security Gate](https://github.com/giselleevita/agent-security-gate) demonstrates enforcement. The static demo uses no model key, backend, tracking, or submitted prompts.

## Interview prompts this supports

- How would you evaluate distribution shift in a security classifier?
- Why is a detector score not itself an authorization decision?
- How do you avoid overstating an experiment with incomplete seed evidence?
