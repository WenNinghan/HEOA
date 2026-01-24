# Repository Guidelines

## Project Structure and Modules
- `heoa_repro/`: reproducible experiment scripts and generated outputs.
  - `heoa_truss_repro.py`: main experiment script; entry point.
  - `heoa_truss_results.json` / `heoa_truss_results.csv`: generated results; keep in sync with code.
- `HEOA_*.txt`: extracted notes and intermediate text; treat as source references.
- `HEOA_page*.png`: figure snapshots from the paper.
- `Human Evolutionary Optimization Algorithm.txt`: reference material for the algorithm.
- `.idea/`: local IDE metadata; avoid editing unless you intend to change project config.

## Build, Run, and Development Commands
- Set up deps (NumPy only):
  - `python -m pip install numpy`
- Run the reproduction script from repo root:
  - `python heoa_repro/heoa_truss_repro.py`
  - Writes/overwrites `heoa_repro/heoa_truss_results.json` and `heoa_repro/heoa_truss_results.csv` and prints a summary.

## Code Style and Naming
- Python: 4-space indentation, PEP 8 style.
- Use `snake_case` for functions and variables; keep constants in `UPPER_SNAKE`.
- Keep experiments deterministic by threading a `seed` into RNG usage.
- Prefer NumPy vector operations; avoid mixing Python lists in numeric loops unless necessary.

## Testing Guidance
- No test suite is present in this repository.
- If you add tests, place them under `tests/` and name files `test_*.py`.
- Suggested runner: `pytest -q` (add `pytest` to your environment if needed).

## Data and Reproducibility Notes
- Treat `heoa_truss_results.*` as generated artifacts; do not hand edit.
- When changing algorithm parameters or objective/constraints, rerun the script and commit updated results.
- If you change defaults (e.g., `runs`, `N`, `Maxiter`), document the new values in the commit/PR description.

## Commit and Review Notes
- No established commit convention found; use concise imperative subject lines.
- For PRs, include: purpose of the change, how to reproduce results, and whether `heoa_truss_results.*` was regenerated.
