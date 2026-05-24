# Changelog

All notable changes documented here. Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versioning follows [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - ReleaseDate

## [0.1.0](https://github.com/AndVl1/vision-analyze/releases/tag/v0.1.0) - 2026-05-24

### Added

- initial vision-analyze implementation

### Fixed

- address Phase 6.5 concurrency regressions (I1+I2)
- address Phase 6 review findings (critical+high+security)

### Other

- ignore .claude local cache
- ignore .local/ handoff transcripts

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
