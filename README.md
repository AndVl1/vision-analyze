# vision-analyze

CLI tool for visual screenshot analysis via [llama.cpp](https://github.com/ggml-org/llama.cpp). Local vision LLM as a fallback when accessibility-tree-based UI inspection (e.g. `claude-in-mobile analyze-screen`) cannot answer — visual bugs, colors, layout glitches, custom-rendered widgets without accessibility metadata.

## Positioning

```
claude-in-mobile = screenshots + UI tree + taps/swipes (no AI, fast)
vision-analyze   = vision (AI, medium-speed, local llama.cpp)
Claude Code      = orchestrator (text model)
```

`vision-analyze` does NOT wrap `claude-in-mobile`. They are siblings.

## Install

### Homebrew

```bash
brew tap AndVl1/tap
brew install vision-analyze
```

Optional: `brew install llama.cpp` for the runtime.

### From source

```bash
git clone https://github.com/AndVl1/vision-analyze
cd vision-analyze
cargo install --path .
```

## Usage

```bash
vision-analyze [FLAGS] <image-path> <prompt>
```

- **stdout** — model response (plain text or JSON)
- **stderr** — logs / metrics
- **exit code** — `0` ok, `1` analysis error, `2` backend/model unavailable

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model <path>` | `$VISION_MODEL_PATH` or `~/.vision-analyze/model.gguf` | GGUF model path |
| `--mmproj <path>` | `$VISION_MMPROJ_PATH` or alongside model | Vision projector |
| `--format <text\|json>` | `text` | Output format |
| `--max-tokens <n>` | `1024` | Token limit |
| `--temperature <f>` | `0.0` | Sampling temperature |
| `--timeout <ms>` | `30000` | Inference timeout |
| `--preset <name>` | — | Use a preset prompt (`ui`, `ui-compare`, `error`, `element-find`, `scroll-check`, or user-defined) |
| `--image-b <path>` | — | Second image (for `two_images` presets like `ui-compare`) |
| `--backend <auto\|server\|cli>` | `auto` | Force backend selection |
| `--warm` | — | Launch daemon (warm mode) on unix socket |
| `--stop` | — | Stop warm daemon |
| `--health` | — | Health check |

### Modes

#### Cold (default)

```bash
vision-analyze screen.png "describe the UI"
# loads model → inference → unloads → stdout
```

#### Warm (`--warm`)

```bash
vision-analyze --warm                         # launches daemon, listens on /tmp/vision-analyze.sock
vision-analyze screen.png "any errors?"       # fast — model is already loaded
vision-analyze --stop                         # shuts down daemon
```

#### Hybrid

If the socket is unreachable (daemon down / crashed), commands automatically fall back to cold execution.

### Presets

Built-in presets are in [`presets/default.json`](presets/default.json). User overrides go into `~/.vision-analyze/presets.json` (same shape — user entries replace built-ins of the same name).

- `ui` — describe what is visually on the screen
- `ui-compare` — diff two screenshots
- `error` — detect error indicators
- `element-find` — locate element by visual description
- `scroll-check` — verify scroll direction/progress

```bash
vision-analyze screen.png "" --preset ui
vision-analyze before.png "" --image-b after.png --preset ui-compare
```

### Environment

```bash
VISION_MODEL_PATH=/path/to/qwen2-vl-7b.gguf
VISION_MMPROJ_PATH=/path/to/mmproj.gguf
VISION_LLAMA_SERVER=http://127.0.0.1:8080
VISION_LLAMA_BIN=/usr/local/bin/llama-mtmd-cli
VISION_DEFAULT_FORMAT=json
VISION_MAX_TOKENS=1024
VISION_TIMEOUT=30000
```

## Backend autodetect

```
1. probe VISION_LLAMA_SERVER/health (200 ms timeout)
   → if 200 OK → backend = server
2. else: which("llama-mtmd-cli") || which("llama-cli")
   → backend = cli
3. else: exit 2
```

Override with `--backend server|cli|auto`.

## Recommended models

| Model | VRAM | Quality | Speed (Apple M2) |
|-------|------|---------|------------------|
| Qwen2-VL 2B | ~2 GB | basic | ~2 s |
| Qwen2-VL 7B | ~6 GB | good | ~5 s |
| MiniCPM-V 8B | ~6 GB | good | ~5 s |

For "is the screen non-empty / no crash" — 2B is enough. For full UI analysis — 7B.

## Integration example

```bash
# 1. Screenshot via claude-in-mobile
claude-in-mobile screenshot android -o /tmp/screen.png

# 2. Quick AT-tree check (no AI)
claude-in-mobile find "Profile"

# 3. Fallback to vision when AT-tree missed the element
vision-analyze /tmp/screen.png "Is a user profile visible with avatar and name?" --preset ui
```

## License

MIT — see [LICENSE](./LICENSE).
