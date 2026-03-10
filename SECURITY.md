## Security Policy

ToolShield is a **research-grade prototype** for detecting prompt injection attacks against tool-using LLM agents. It is not intended to be deployed as a standalone production security appliance.

### Scope

ToolShield focuses on:

- Classifying prompts as **benign vs. prompt-injection attacks** for tool-using LLM agents.
- Evaluating different detection strategies (heuristics, TF–IDF + LR, transformers, context-aware models).
- Studying the impact of context truncation and long tool schemas on detection performance.

Out of scope:

- Hardening of the surrounding LLM stack (model hosting, network, infrastructure).
- Protection against compromised tool backends, data stores, or supply-chain attacks.
- Guaranteeing complete coverage of all real-world prompt injection techniques.

### Intended usage

- As a **research and benchmarking framework** for prompt-injection detection methods.
- As a **signal source** in a broader, defense-in-depth architecture where additional logging, monitoring, and access controls exist.

If you adapt ToolShield for production, you should:

- Perform your own **threat modeling** for the end-to-end system.
- Add **monitoring, alerting, and rate-limiting** around any automated blocking decisions.
- Validate detection performance on **your own data and attack library**.

### Reporting vulnerabilities

If you believe you have found a vulnerability that could:

- Cause ToolShield to misclassify prompts in a way that is not already captured by the documented limitations, or
- Expose sensitive data when running the demo API or experiments,

please open a **private issue** or contact the maintainer directly via GitHub instead of filing a public exploit first. Provide:

- A minimal reproducible example (input prompt, configuration, and expected vs. actual behavior).
- Any relevant environment details (ToolShield version, Python version, OS).

Given this is a research repository, response times may vary, but security-relevant reports will be prioritized where possible.

