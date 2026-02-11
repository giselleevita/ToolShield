# Bibliography TODOs for ToolShield Thesis

All current entries in `references.bib` are placeholders. The keys listed below are **used in the thesis text** (via `\cite{...}`) and must be replaced with real bibliographic data before submission.

For each key, you should provide at least:
- **Entry type** (e.g., `@article`, `@inproceedings`, `@misc`)
- **Author(s)**
- **Full title**
- **Venue** (journal/conference/tech report/source)
- **Year**
- **DOI or URL** (if available)

## Keys used in the thesis

- `CITATION_PROMPT_INJECTION_SURVEY`  
  - Used in: `chapters/methodology.tex` (attack family overview)  
  - Need a recent survey or formalization paper on prompt injection attacks.

- `prompt_injection_survey`  
  - Used in: `chapters/related_work.tex` (prompt injection definition/overview)  
  - Likely a general survey of prompt injection / jailbreak attacks.

- `jailbreak_taxonomy`  
  - Used in: `chapters/related_work.tex` (jailbreak techniques)  
  - Should point to a taxonomy paper on jailbreak-style attacks.

- `prompt_injection_taxonomy`  
  - Used in: `chapters/related_work.tex` (attack strategy taxonomy)  
  - Should cite work that classifies prompt injection strategies.

- `tool_use_security`  
  - Used in: `chapters/related_work.tex` (tool-use security risks)  
  - Should cite a paper on security of tool-using / function-calling LLMs.

- `function_calling_risks`  
  - Used in: `chapters/related_work.tex` (function calling risks)  
  - Should reference work describing vulnerabilities in function-calling APIs.

- `indirect_prompt_injection`  
  - Used in: `chapters/related_work.tex` (indirect injections from external content)  
  - Should cite work on indirect prompt injection or data poisoning via external sources.

- `safety_classifier_overview`  
  - Used in: `chapters/related_work.tex` (safety filters / classifiers)  
  - Should cite a survey or representative paper on LLM safety classifiers.

- `harmful_prompt_detection`  
  - Used in: `chapters/related_work.tex` (classifier-based defenses)  
  - Should reference a specific harmful prompt / jailbreak detection model.

- `instruction_hierarchy`  
  - Used in: `chapters/related_work.tex` (instruction hierarchy concept)  
  - Should cite work that formalizes or discusses instruction hierarchies in LLM prompts.

- `contextual_defenses`  
  - Used in: `chapters/related_work.tex` (context-aware defenses)  
  - Should point to a paper that incorporates system/tool context into defenses.

- `context_window_limitations`  
  - Used in: `chapters/related_work.tex` (context window / truncation issues)  
  - Should cite work on performance degradation from truncation / long contexts.

- `truncation_bias`  
  - Used in: `chapters/related_work.tex` (truncation bias)  
  - Should reference a paper that studies truncation-induced biases or failures.

- `token_budgeting_methods`  
  - Used in: `chapters/related_work.tex` (token budgeting strategies)  
  - Should cite work on token budget allocation / context selection.

- `prompt_injection_benchmarks`  
  - Used in: `chapters/related_work.tex` (prompt injection benchmarks)  
  - Should reference benchmark datasets or evaluation frameworks for prompt injection.

- `benchmark_limitations`  
  - Used in: `chapters/related_work.tex` (limitations of synthetic benchmarks)  
  - Should cite work critiquing benchmark design or external validity.

## How to update `references.bib`

For each key above:

1. Find an appropriate paper or resource.
2. Replace the existing placeholder entry in `thesis/references.bib` with a proper biblatex entry, e.g.:

```bibtex
@article{prompt_injection_survey,
  author  = {...},
  title   = {...},
  journal = {...},
  year    = {...},
  volume  = {...},
  number  = {...},
  pages   = {...},
  doi     = {...},
}
```

3. Rebuild the thesis:

```bash
cd thesis
latexmk -lualatex main.tex
```

As long as the keys remain the same, the compiled PDF will automatically pick up the updated references without further LaTeX changes.

