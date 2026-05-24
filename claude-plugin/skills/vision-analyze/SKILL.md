---
name: vision-analyze
description: Use vision-analyze CLI to inspect screenshots via local multimodal LLM (llama.cpp + Qwen2-VL/SmolVLM). Activate when the user asks to analyze a screenshot, find UI elements, detect visual defects, compare before/after screens, check for error dialogs, or verify scroll/state changes — and prefers a local model over cloud vision APIs.
---

# vision-analyze

CLI wrapper over `llama.cpp` for visual analysis of screenshots. Runs a local multimodal model (Qwen2-VL, SmolVLM, etc.) — no cloud calls, no data leaves the machine.

## When to use

- "Что на этом скриншоте?", "опиши UI"
- "Найди кнопку X / поле Y"
- "Что изменилось между этими двумя скринами?"
- "Есть ли на экране ошибка / краш-диалог?"
- "Прокрутился ли список после свайпа?"
- Manual-QA loops where you'd otherwise screenshot → describe-by-eye

Do **not** use for: OCR-heavy text extraction (tesseract is better), pixel-perfect diff (`compare` from ImageMagick), accessibility tree (use platform a11y APIs).

## Prerequisites — verify once per session

```bash
which vision-analyze    # must return a path
which llama-server      # backend; bundle: brew install llama.cpp
```

If `vision-analyze` missing: `brew install AndVl1/tap/vision-analyze`. If `llama-server` missing: `brew install llama.cpp`.

## Two backends

| Backend | When picked | Pros | Cons |
|---------|-------------|------|------|
| `server` | `llama-server` is up on `127.0.0.1:8080` | Fast warm start, auto-pulls model with `-hf` | Needs separate process |
| `cli` | server unavailable | No daemon | Cold start each call (model reload) |

Auto-detect: tries server first, falls back to cli. Override with `--backend server|cli`.

## Starting the model server (one-time per session)

```bash
# Lightweight (CI/dev): SmolVLM 256M, ~200MB
llama-server -hf ggml-org/SmolVLM-256M-Instruct-GGUF -c 2048 --port 8080 &

# Better quality: Qwen2-VL 2B, ~1.5GB
llama-server -hf ggml-org/Qwen2-VL-2B-Instruct-GGUF -c 4096 --port 8080 \
  --repeat-penalty 1.15 --repeat-last-n 64 --temp 0.3 --top-p 0.9 &
```

Wait for `server is listening on http://127.0.0.1:8080` before first call.

> **Why the sampling flags for Qwen2-VL:** without them the model loops ("ChatBot ChatBot ChatBot..."). SmolVLM doesn't need them.

## Invocation patterns

### Free-form prompt

```bash
vision-analyze /path/to/screen.png "что на экране?"
```

### Presets (recommended — JSON output, deterministic schema)

| Preset | Purpose | Output schema |
|--------|---------|---------------|
| `ui` | Enumerate visible UI elements + visual defects | `{elements[], visual_defects[], overall_status}` |
| `ui-compare` | Diff two screenshots | `{changed[], added[], removed[], identical}` |
| `error` | Detect error/crash/empty states | `{error_found, error_type, severity}` |
| `element-find` | Locate one element by description | `{found, bounds, approximate_center}` |
| `scroll-check` | Verify scroll happened between two screens | `{scrolled, direction, reached_end}` |

```bash
# Single-image presets
vision-analyze --preset ui screen.png
vision-analyze --preset error screen.png
vision-analyze --preset element-find screen.png "Login button"

# Two-image presets (need --image-b)
vision-analyze --preset ui-compare before.png --image-b after.png
vision-analyze --preset scroll-check before.png --image-b after.png
```

### JSON output (for piping to `jq`)

```bash
vision-analyze --format json --preset ui screen.png | jq '.elements[].label'
```

## Warm mode — keep model in RAM

For repeated calls (manual QA loops, watch scripts):

```bash
vision-analyze --warm                # spawns daemon, returns immediately
vision-analyze --preset ui s1.png    # uses daemon (no reload)
vision-analyze --preset ui s2.png    # fast
vision-analyze --stop                # kill daemon
vision-analyze --health              # check daemon/backend status
```

Daemon socket: `/tmp/vision-analyze.sock`. Exits cleanly on `--stop` or SIGTERM.

## Exit codes

| Code | Meaning | Action |
|------|---------|--------|
| 0 | OK | parse stdout |
| 1 | General error (bad image, oversized, parse fail, etc.) | check stderr |
| 2 | Backend unavailable / model not found | start `llama-server` or pass `--model` |

Always check `$?` before parsing stdout.

## Common workflows

### Manual QA: "did the screen change after my action?"

```bash
adb exec-out screencap -p > before.png
# ... user action ...
adb exec-out screencap -p > after.png
vision-analyze --preset ui-compare before.png --image-b after.png
```

### CI smoke: "does the login screen render without errors?"

```bash
vision-analyze --warm
vision-analyze --preset error login.png --format json | jq -e '.error_found == false'
vision-analyze --stop
```

### Find a specific element (no a11y tree available)

```bash
vision-analyze --preset element-find screen.png "красная кнопка Submit внизу"
# → {"found": true, "bounds": {"x1": 120, "y1": 800, "x2": 480, "y2": 880}, ...}
```

## Custom presets

User overrides live in `~/.vision-analyze/presets.json` (same schema as built-ins). Useful when default `ui` is too verbose for a specific app.

## Pitfalls

- **Wrong sampling = garbage output.** Qwen2-VL loops without `--repeat-penalty`. Either configure llama-server, or use SmolVLM.
- **`--mmproj-url auto` is invalid.** Don't pass it — `-hf` auto-loads mmproj from the GGUF repo.
- **32 MiB image cap.** Exceeded → exit 1 with `ImageTooLarge`. Resize first (`sips -Z 2048` on macOS).
- **Model output may be wrapped in ```json fences.** Strip before `jq`: `... | sed 's/^```json//; s/```$//'`.
- **Cold mode needs local `.gguf` files** (CLI doesn't pull from HF). Pass `--model` + `--mmproj`.

## Cross-references

Repo: https://github.com/AndVl1/vision-analyze
Install: `brew install AndVl1/tap/vision-analyze`
