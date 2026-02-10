# TU Darmstadt Thesis Format Compliance Report

**Thesis:** ToolShield: Prompt Injection Detection for Tool-Using LLM Agents  
**Author:** Giselle Evita  
**Type:** Bachelor Thesis, Department of Computer Science  
**Date:** February 2026  

## Compliance Checklist

| # | Requirement | Expected | Status | Notes |
|---|-------------|----------|--------|-------|
| 1 | Document class | `tudapub.cls` with `thesis={type=bachelor}` | **PASS** | Migrated from `\documentclass{article}` to `tudapub` |
| 2 | Compiler | LuaLaTeX (recommended by TUDa-CI for PDF/A) | **PASS** | Build uses `latexmk -lualatex` |
| 3 | Language option | `english` | **PASS** | Set in class options |
| 4 | Font size | 11pt (TUDa-CI default) | **PASS** | `fontsize=11pt` |
| 5 | Title page | TUDa-CI generated via `\maketitle` | **PASS** | Uses `\title`, `\author`, `\reviewer`, `\department{inf}`, `\submissiondate`, `\examdate` |
| 6 | TU Logo | `tuda_logo.pdf` | **MANUAL** | Logo only available via TU intranet; `accept-missing-logos=true` used. **User must download logo before final submission.** |
| 7 | Eigenständigkeitserklärung | Signed declaration per APB §22(7) | **PASS** | Placeholder declaration page included; must be replaced with signed scan before submission |
| 8 | Abstract | Required | **PASS** | Added using `\begin{abstract}...\end{abstract}` |
| 9 | Table of Contents | Required | **PASS** | `\tableofcontents` present |
| 10 | Chapter structure | `\chapter` (KOMA-Script `scrreprt` via `class=report`) | **PASS** | All 8 chapter files converted from `\section` to `\chapter`, subsections shifted accordingly |
| 11 | Bibliography | `biblatex` (TUDa-CI standard) | **PASS** | Migrated from `natbib`/`plainnat` to `biblatex` with `\printbibliography` |
| 12 | Bibliography backend | `biber` | **PASS** | Default backend for biblatex, invoked automatically by latexmk |
| 13 | Page layout / margins | TUDa-CI managed (`custommargins=true`) | **PASS** | Replaced manual `\geometry{margin=2.5cm}` with TUDa-CI margin management |
| 14 | Headers / footers | TUDa-CI managed (KOMA-Script `scrlayer-scrpage`) | **PASS** | Default TUDa-CI headers/footers active |
| 15 | Font family | XCharter + Roboto (TUDa-CI default) | **PASS** | Loaded automatically by `tudapub` under LuaLaTeX |
| 16 | Accent color | TU Darmstadt color 9c | **PASS** | `accentcolor=9c` |
| 17 | PDF/A compliance | PDF/A-2b recommended for digital submission | **PARTIAL** | LuaLaTeX + tudapub support PDF/A; full validation requires external tool (e.g., veraPDF). Recommend verifying before submission. |
| 18 | Appendix | `\appendix` with chapter-level sections | **PASS** | Appendix contains "Experimental Tables and Figures" and "Reproducibility Notes" as chapters |
| 19 | Undefined references | 0 | **PASS** | Verified in build log |
| 20 | Undefined citations | 0 | **PASS** | All 16 `\cite{}` keys resolve (placeholder entries in `references.bib`) |
| 21 | TODO markers | None in output | **PASS** | No `TODO` markers remain in rendered PDF |

## Build Summary

- **Compiler:** `latexmk -lualatex`
- **Output:** `main.pdf` — 72 pages
- **Errors:** 0
- **Undefined refs/citations:** 0
- **Overfull hbox warnings:** 17 (mostly from long `\texttt{}` variable names; cosmetic only)
- **Underfull hbox warnings:** 0

## Action Items Before Final Submission

1. **Download TU logo**: Obtain `tuda_logo.pdf` from the TU Darmstadt intranet and place it in the `thesis/` directory. Remove `accept-missing-logos=true` from class options.
2. **Supervisor / Reviewer names**: Replace `[Supervisor Name TBD]` and `[Second Reviewer TBD]` in `\reviewer{}` with actual names.
3. **Research Group**: Replace `[Research Group TBD]` in `\group{}` with the actual group name.
4. **Signed declaration**: Replace the placeholder Eigenständigkeitserklärung page with a signed and scanned version of the official APB §22(7) form.
5. **Real bibliography entries**: Replace placeholder `@misc` entries in `references.bib` with actual publication data.
6. **PDF/A validation**: Run the final PDF through veraPDF or a similar tool to confirm PDF/A-2b compliance.
7. **Submission/Exam dates**: Update `\submissiondate` and `\examdate` from `\today` to the actual dates.

## Sources

- [TUDa-CI template repository](https://github.com/tudace/tuda_latex_templates)
- [Systems group thesis guidelines](https://www.informatik.tu-darmstadt.de/systems/teach/write_your_thesis/index.en.jsp)
- [APB §22(7) declaration](https://www.tu-darmstadt.de/studieren/studierende_tu/studienorganisation_und_tucan/hilfe_und_faq/artikel_details_de_en_37824.de.jsp)
- [PDF/A submission requirement](https://www.tu-darmstadt.de/studieren/studierende_tu/studienorganisation_und_tucan/hilfe_und_faq/artikel_details_de_en_26885.en.jsp)
