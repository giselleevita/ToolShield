# Submission Readiness Report

Date: 2026-02-12
Project: ToolShield (TU Darmstadt BSc Thesis)
Scope: Safe, non-fabricated submission-readiness fixes only.

## Requirement Status (PASS/FAIL)

| Requirement | Status | Notes |
|---|---|---|
| Placeholder scan completed (`thesis/**/*.tex`, `thesis/**/*.bib`, `README.md`) | PASS | Scan performed with `rg` for TODO/FIXME/TBD/placeholder/??/Supervisor/Second Reviewer/`\\today`. |
| Reviewer/date placeholders made non-embarrassing without inventing data | PASS | Replaced visible TODO tokens with `(to be filled)` placeholders. |
| Real reviewer names present | FAIL | Manual input still required. |
| Real submission/exam dates present | FAIL | Manual input still required. |
| Declaration include mechanism configured (`pdfpages` + optional included PDF) | PASS | `\\usepackage{pdfpages}` present and declaration block now supports `signed_declaration.pdf`. |
| Signed declaration file present | FAIL | `signed_declaration.pdf` not added by automation. |
| TU logo fallback handling documented | PASS | `accept-missing-logos=true` retained with inline comment on `tuda_logo.pdf` placement and removal step. |
| Placeholder bibliography entries removed/replaced | PASS | No placeholder/TBD bibliography entries detected in current `references.bib`. |
| Build completes | PASS | `latexmk -lualatex` completed successfully. |
| Undefined references | PASS | None found in `main.log`. |
| Undefined citations | PASS | None found in `main.log`. |

## Remaining Manual Actions

- [ ] Fill reviewer names in `thesis/main.tex` (`\\SupervisorName`, `\\SecondReviewerName`).
- [ ] Fill real dates in `thesis/main.tex` (`\\SubmissionDate`, `\\ExamDate`).
- [ ] Add signed declaration PDF at `thesis/signed_declaration.pdf`.
- [ ] Ensure the official TU logo file is present as `thesis/tuda_logo.pdf`, then remove `accept-missing-logos=true` from `thesis/main.tex`.
- [ ] Keep bibliography placeholder-free before final submission (current scan found no placeholder/TBD entries).

## Build Command and Warning Summary

Build command used:

```bash
cd thesis && latexmk -lualatex
```

Summary from `thesis/main.log`:

- LaTeX errors: `0`
- Undefined references: `0`
- Undefined citations: `0`
- Overfull `\\hbox` warnings: `18` (cosmetic)
- TUDa IMRAD label warnings: `4` (non-fatal)
- Babel warning: `1` (non-fatal)
