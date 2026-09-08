# Three-minute recruiter demonstration

1. Open the static results explorer; emphasize that it has no backend, model key, or prompt collection.
2. Switch from S_random to S_tool_holdout and compare the model bars.
3. Explain why unseen-tool performance is a more honest generalization signal.
4. Move the detector-score slider near 0.55.
5. Show how the same score produces different decisions for read, write, and privileged tools.
6. Finish at **Evidence fingerprints**: experiment commit, configuration hash, policy hash, and split inventory.

The central design point is separation of concerns: the detector estimates risk; a deterministic,
versioned policy decides whether a particular tool action may proceed.
