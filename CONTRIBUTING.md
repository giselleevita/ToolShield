# Contributing

ToolShield is a research-grade prompt injection detection system developed as a bachelor's thesis project.
Contributions that improve reproducibility, extend attack families, or strengthen evaluation methodology are welcome.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
make install-dev
```

Run the full pipeline:
```bash
make pipeline
```

Run tests:
```bash
make test
```

Lint:
```bash
ruff check .
```

## Branch naming

| Type | Pattern | Example |
|---|---|---|
| Feature | `feat/description` | `feat/attack-family-af5` |
| Bug fix | `fix/description` | `fix/split-leakage-check` |
| Docs | `docs/description` | `docs/dataset-schema` |
| Experiment | `exp/description` | `exp/cross-encoder-rerank` |

## Issue labels

- `bug` — something is broken
- `enhancement` — new feature or attack family
- `reproducibility` — seed, config, or output consistency
- `docs` — documentation gap
- `experiment` — new ablation or evaluation run

## PR checklist

- [ ] `make test` passes
- [ ] New attack families have test coverage in `tests/`
- [ ] Seeds are fixed and documented in config files
- [ ] No model weights or raw dataset files committed
- [ ] README updated if CLI or schema changes

## Reproducibility note

All random operations use fixed seeds:
- Dataset generation: `seed=1337`
- Split generation: `seed=2026`

Do not change seeds in shared configs without updating documentation.

## License

MIT License.
