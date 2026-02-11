# Signed Declaration (Eigenständigkeitserklärung)

TU Darmstadt requires a signed declaration according to APB §22(7).

Workflow:

1. Download the official declaration template (.docx) from the TU website:  
   https://www.tu-darmstadt.de/studieren/studierende_tu/studienorganisation_und_tucan/hilfe_und_faq/artikel_details_de_en_37824.de.jsp
2. Fill in your details, print the document, sign it, and scan it.
3. Save the scan as `signed_declaration.pdf` in this directory:  
   `thesis/declaration/signed_declaration.pdf`.
4. Rebuild the thesis with:

```bash
cd thesis
latexmk -lualatex main.tex
```

The main thesis file `thesis/main.tex` includes the declaration via:

```latex
\IfFileExists{declaration/signed_declaration.pdf}{%
  \includepdf[pages=-]{declaration/signed_declaration.pdf}%
}{%
  % Fallback warning page in the PDF
}
```

If the file is missing, the thesis will still compile but will include a visible warning page reminding you to add the signed declaration.

