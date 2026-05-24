# vision-analyze plugin for Claude Code

Skill that teaches Claude how to drive the `vision-analyze` CLI for visual screenshot analysis via a local multimodal LLM (Qwen2-VL, SmolVLM, etc.) on top of `llama.cpp`.

## Install

```
/plugin marketplace add AndVl1/claude-plugin
/plugin install vision-analyze@andvl1-plugins
```

Or directly from this repo:

```
/plugin install github:AndVl1/vision-analyze?path=claude-plugin
```

## Requires

- `vision-analyze` binary on `$PATH` → `brew install AndVl1/tap/vision-analyze`
- `llama-server` (or `llama-mtmd-cli`) → `brew install llama.cpp`

## What it adds

- **Skill `vision-analyze`** — activates when the user asks to analyze a screenshot, find UI elements, detect visual defects, compare screens, etc. Knows the CLI flags, presets, warm/cold modes, exit codes, sampling gotchas.

## Source

Skill lives in the same repo as the CLI: https://github.com/AndVl1/vision-analyze under `claude-plugin/`.
