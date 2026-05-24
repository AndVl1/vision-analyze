# Changelog

All notable changes documented here. Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versioning follows [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - ReleaseDate

### Added
- Initial CLI scaffolding (`vision-analyze <image> <prompt>`)
- Cold + warm (unix-socket daemon) modes with auto-fallback
- llama.cpp backend: HTTP `llama-server` (preferred) + `llama-mtmd-cli`/`llama-cli` fallback, autodetect
- 5 built-in presets: `ui`, `ui-compare`, `error`, `element-find`, `scroll-check`
- User preset overrides via `~/.vision-analyze/presets.json`
- Output formats: `text` (default), `json`
- Environment configuration: `VISION_MODEL_PATH`, `VISION_MMPROJ_PATH`, `VISION_LLAMA_SERVER`, `VISION_DEFAULT_FORMAT`, `VISION_MAX_TOKENS`, `VISION_TIMEOUT`, `VISION_LLAMA_BIN`
- Exit codes: `0` ok / `1` analysis error / `2` model/backend not available
- GitHub Actions: CI (fmt + clippy + test on macos + linux), CD (matrix build + GH release + Homebrew formula push)
- release-plz integration for automated version bumping via Conventional Commits
