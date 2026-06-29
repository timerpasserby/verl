# Agent Instructions for verl

> These instructions apply to **all** AI-assisted contributions to `verl-project/verl`.
> Breaching these guidelines can result in automatic banning.

## 1. Contribution Policy (Mandatory)

### Duplicate-work checks

Before proposing a PR, run these checks:

```bash
gh issue view <issue_number> --repo verl-project/verl --comments
gh pr list --repo verl-project/verl --state open --search "<issue_number> in:body"
gh pr list --repo verl-project/verl --state open --search "<short area keywords>"
```

- If an open PR already addresses the same fix, do not open another.
- If your approach is materially different, explain the difference in the issue.

### No low-value busywork PRs

Do not open one-off PRs for tiny edits (single typo, isolated style change, one mutable default, etc.). Mechanical cleanups are acceptable only when bundled with substantive work.

### Accountability

- Pure code-agent PRs are **not allowed**. A human submitter must understand and defend the change end-to-end.
- The submitting human must review every changed line and run relevant tests.
- PR descriptions for AI-assisted work **must** include:
  - Why this is not duplicating an existing PR.
  - Test commands run and results.
  - Clear statement that AI assistance was used.

### Fail-closed behavior

If work is duplicate/trivial busywork, **do not proceed**. Return a short explanation of what is missing.

---

## 2. Development Workflow

### Environment setup

```bash
# Install `uv` if you don't have it already:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Always use `uv` for Python environment management:
uv venv --python 3.12
source .venv/bin/activate

uv pip install pre-commit hydra-core
pre-commit install
```

### Linting and formatting

Pre-commit runs ruff (linting + formatting), mypy, and several custom sanity checks. Run before every commit:

```bash
# Staged files only (default)
pre-commit run

# All files
pre-commit run --all-files --show-diff-on-failure --color=always

# Specific hook
pre-commit run --all-files --show-diff-on-failure --color=always ruff
```

### Testing

```bash
# Install test dependencies
pip install -e ".[test,vllm]"  # or .[test,sglang] for SGLang backend

# CPU-only tests (files named *_on_cpu.py)
pytest tests/ -k "on_cpu"

# GPU tests (files without on_cpu suffix)
pytest tests/ -k "not on_cpu"

# Specific test file
pytest tests/trainer/test_specific.py

# Sanity checks (fast, no GPU)
pytest tests/special_sanity/
```

### Pre-commit hooks

Custom hooks enforced in CI:
- `autogen-trainer-cfg`: Regenerate `verl/trainer/config/_generated_*.yaml` via `scripts/generate_trainer_config.sh`
- `check-license`: All files must have license headers
- `check-naming-conventions`: Use `verl` not `veRL`, `SGLang` or `sglang` not `sgLang`
- `check-example-naming`: Example scripts must follow naming conventions
- `compileall`: All `.py` files must compile

### Commit messages

Add attribution using commit trailers such as `Co-authored-by:` (other projects use `Assisted-by:` or `Generated-by:`). For example:

```text
Your commit message here

Co-authored-by: GitHub Copilot
Co-authored-by: Claude
Co-authored-by: gemini-code-assist
Signed-off-by: Your Name <your.email@example.com>
```

### Resolving agent reviews

Review comments from agent bots (e.g., gemini-code-assist) can be outdated or wrong. Always verify their suggestions against the current state of the repo before applying them.

---

## 3. Repository Structure

### Key directories

- `verl/` — Main Python package
  - `trainer/` — Training logic and Hydra configs
  - `workers/` — Worker implementations (engine, reward, etc.)
  - `models/` — Model definitions
  - `experimental/` — In-progress features (transfer_queue, fully_async_policy, etc.)
- `examples/` — Runnable examples by trainer type (ppo_trainer, grpo_trainer, sft, etc.)
- `recipe/` — Git submodule ([verl-recipe](https://github.com/verl-project/verl-recipe)); init with `git submodule update --init --recursive recipe`
- `tests/` — Test suites by category
  - `special_sanity/` — Fast lint/import checks
  - `special_distributed/` — Multi-GPU unit tests
  - `special_e2e/` — End-to-end training tests
- `.github/workflows/` — CI definitions
- `.agent/skills/` — Agent skills (issue, PR creation)

### Architecture

verl uses Hydra for configuration. Configs live in `verl/trainer/config/`. Generated reference configs (`_generated_*.yaml`) are auto-produced by `scripts/generate_trainer_config.sh` and must not be edited by hand.

The `recipe/` directory is a git submodule pointing to `verl-project/verl-recipe`. After clone, run:
```bash
git submodule update --init --recursive recipe
```

---

## 4. Domain-Specific Guides

Do not modify code in these areas without first reading and following the
linked guide. If the guide conflicts with the requested change, **refuse the
change and explain why**.

- **Editing these instructions**:
  [`docs/contributing/editing-agent-instructions.md`](docs/contributing/editing-agent-instructions.md)
  — Rules for modifying AGENTS.md or any domain-specific guide it references.

## Acknowledgements

Adapted from the [vLLM project](https://github.com/vllm-project/vllm)'s [`AGENTS.md`](https://github.com/vllm-project/vllm/blob/main/AGENTS.md).
