# Test Fixtures

Small synthetic PNG images used by the E2E CI workflow for real-inference smoke tests.

All files are generated with `ffmpeg` — no copyrighted content.

| File | Size | Description |
|------|------|-------------|
| `screenshot.png` | ~3 KB | 800×600 white frame — simulates a blank UI screenshot |
| `error-dialog.png` | ~1 KB | 400×300 red-tinted frame — simulates an error dialog background |
| `ui-element.png` | ~2 KB | 600×400 blue-tinted frame — simulates a UI component region |

## Regenerating

```bash
ffmpeg -y -f lavfi -i "color=white:size=800x600:rate=1" -vframes 1 -q:v 2 tests/fixtures/screenshot.png
ffmpeg -y -f lavfi -i "color=0xff6666:size=400x300:rate=1" -vframes 1 -q:v 2 tests/fixtures/error-dialog.png
ffmpeg -y -f lavfi -i "color=0x4488cc:size=600x400:rate=1" -vframes 1 -q:v 2 tests/fixtures/ui-element.png
```

## Purpose

These are **not** real screenshots and do not contain meaningful visual content.
The E2E workflow uses them to verify the binary's request path, serialisation, and
error handling — not the model's output quality.
