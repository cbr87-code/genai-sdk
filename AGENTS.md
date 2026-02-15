# AGENTS

Scope: entire repository.

## Purpose
- Keep this SDK minimal, readable, and easy to extend.

## Rules
- Prefer small, composable interfaces over framework-heavy abstractions.
- Preserve async-first APIs; sync wrappers are convenience only.
- Keep provider adapters isolated behind `providers.base.Provider`.
- Add tests for behavior changes.
- Update `README.md` when public API or setup commands change.

## Local Validation
- `python -m unittest discover -s tests -p 'test_*.py' -v`
- `pytest -q`

## Example Execution Note
- Running `examples/*.py` may require unsandboxed command execution if network access is restricted in the sandbox (for example, DNS/socket blocks).
- If an example hangs or fails with connection/DNS errors, rerun it outside sandbox permissions.
