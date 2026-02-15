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

## Git Workflow
- Before committing, inspect changes with `git status --short`.
- Commit all requested changes with a clear message (for example: `git add -A && git commit -m "<message>"`).
- Push to the active branch's remote after commit (for example on `main`: `git push origin main`).
- If `git push` fails in sandbox due DNS/network resolution errors, rerun push with unsandboxed permissions.
