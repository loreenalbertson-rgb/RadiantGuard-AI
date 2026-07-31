# Contributing

Thank you for helping improve RadiantGuard AI.

Before proposing a feature, consider whether it could be mistaken for clinical advice or increase unjustified confidence in an AI output. Safety, reproducibility, privacy, and transparent limitations take priority over visual novelty.

## Development checks

```bash
pytest -q
python -m compileall app.py src tests
```

## Pull request expectations

- Explain the user or research problem
- Describe safety implications
- Add or update tests
- Document model/data licenses when applicable
- Avoid real patient data and identifying metadata
- Update `DESIGN.md` for meaningful architectural decisions
