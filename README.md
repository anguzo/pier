# pier

Pier is a [Harbor](https://www.harborframework.com/docs/tasks)-compatible framework for evaluating coding agents in sandboxed environments. It reads Harbor's task format and runs trials against it.

```bash
pier run -p path/to/task --agent claude-code --env modal
```

## Why pier

Pier is a fork. We wanted a smaller, more opinionated base to build on, starting with one thing Harbor couldn't do: run installed agents (Claude Code, Codex, etc.) in air-gapped tasks (`allow_internet = false`).

When the agent runs *inside* the sandbox, both the install step and the inference call need the network. Pier fixes that by letting agents declare their install scripts and a network allowlist, which environments then honor when setting up the sandbox.

More is planned from here, and pier will keep drifting from Harbor as we go.

## What works today

- **Task format:** Harbor-compatible.
- **Environments:** `docker`, `modal`.
- **Agents:** `nop`, `oracle`, `claude-code`, `codex`, `gemini-cli`, `opencode`, `mini-swe-agent`.
- **Datasets:** local Harbor-format task directories via `-p` / `--path`.

Pier does not currently resolve or download Harbor registry datasets directly. Use Harbor to export a dataset first, then point Pier at the local directory.

If your tasks allow internet, or your agent runs outside the sandbox, Harbor is the broader, more mature choice and you should probably use it.

## Install

```bash
uv tool install pier
# or
pip install pier
```

## Run

```bash
export ANTHROPIC_API_KEY=...
pier run -p path/to/task --agent claude-code --env modal --env-file .env
```

Run a local dataset:

```bash
pier run -p path/to/dataset --agent claude-code --env modal
```

Run a deterministic random subset of a local dataset:

```bash
pier run -p path/to/dataset --n-tasks 10 --sample-seed 0
```

### Agent Models And Backends

Pier separates model identity from runtime wiring:

- `--model` / `agent.model_name` is Pier's canonical model id, used for tracking and backend compatibility. Use `provider/model` format, such as `anthropic/claude-sonnet-4-5`, `openai/gpt-5.5`, or `google/gemini-2.5-pro`.
- `--backend` / `agent.backend` selects the agent-specific inference integration. Each agent declares its own supported backends (valid values differ per agent):
  - `claude-code`: `anthropic` (default), `respan`, `vertex_ai`
  - `codex`: `openai` (default), `respan`
  - `gemini-cli`: `gemini` (default), `vertex_ai`, `respan`
  - `mini-swe-agent`: `native` (default; dispatches to LiteLLM)
  - `opencode`: `native` (default; dispatches to the agent's own provider layer)
- `--runtime-model` / `agent.runtime_model_name` is an advanced override for the exact model string passed to the agent runtime. Leave it unset unless the backend needs a different provider-specific model id.

For example, Claude Code through Respan still records the canonical model as Anthropic Claude, while the backend maps credentials and endpoint details for Respan:

```bash
pier run -p path/to/task \
  --agent claude-code \
  --model anthropic/claude-sonnet-4-5 \
  --backend respan
```

If a backend requires a runtime id Pier cannot derive, pass it explicitly:

```bash
pier run -p path/to/task \
  --agent gemini-cli \
  --model google/gemini-2.5-pro \
  --backend vertex_ai \
  --runtime-model gemini-2.5-pro
```

`--backend` and `--runtime-model` are agent-specific, so if your job config has multiple agents you must either edit the YAML directly or narrow the job with `--agent`.

To use a Harbor registry dataset, download it with Harbor first. For example, from `~/code/harbor`:

```bash
uv run harbor download swebenchpro -o ~/code/pier/datasets
```

Then run it with Pier:

```bash
cd ~/code/pier
uv run pier run -p datasets/swebenchpro --n-tasks 10 --sample-seed 0
```

See `pier run --help` and `pier job --help` for everything else. Trials land under `jobs/<timestamp>/<trial_id>/`.
