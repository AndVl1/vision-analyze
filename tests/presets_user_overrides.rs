/// Integration tests for user preset overrides.
///
/// `PresetStore::load_from_dir(dir)` merges built-in presets with a
/// `presets.json` found under `dir`.  This is the same logic as
/// `PresetStore::load()` but takes an explicit config directory so tests
/// do not depend on `$HOME` or the OS user-dirs implementation.
use std::fs;
use tempfile::tempdir;
use vision_analyze::presets::PresetStore;

// ── helpers ───────────────────────────────────────────────────────────────────

/// Write `<config_dir>/presets.json` with the given JSON content.
fn write_user_presets(config_dir: &std::path::Path, json: &str) {
    fs::create_dir_all(config_dir).expect("create config dir");
    fs::write(config_dir.join("presets.json"), json).expect("write presets.json");
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[test]
fn user_preset_overrides_builtin_ui_prompt() {
    let dir = tempdir().expect("tempdir");
    let config_dir = dir.path().join(".vision-analyze");

    let custom_json =
        r#"{"ui": {"prompt": "CUSTOM_UI_PROMPT", "format": "text", "two_images": false}}"#;
    write_user_presets(&config_dir, custom_json);

    let store = PresetStore::load_from_dir(&config_dir).expect("load_from_dir");
    let preset = store.get("ui").expect("ui preset");
    assert_eq!(preset.prompt, "CUSTOM_UI_PROMPT");
}

#[test]
fn user_can_add_new_preset() {
    let dir = tempdir().expect("tempdir");
    let config_dir = dir.path().join(".vision-analyze");

    let custom_json = r#"{"my-custom": {"prompt": "describe the sky", "two_images": false}}"#;
    write_user_presets(&config_dir, custom_json);

    let store = PresetStore::load_from_dir(&config_dir).expect("load_from_dir");
    let preset = store.get("my-custom").expect("my-custom preset");
    assert_eq!(preset.prompt, "describe the sky");
}

#[test]
fn user_override_does_not_remove_other_builtins() {
    let dir = tempdir().expect("tempdir");
    let config_dir = dir.path().join(".vision-analyze");

    // Only override `ui` — all other built-ins must survive.
    let custom_json = r#"{"ui": {"prompt": "OVERRIDE", "two_images": false}}"#;
    write_user_presets(&config_dir, custom_json);

    let store = PresetStore::load_from_dir(&config_dir).expect("load_from_dir");
    for name in ["ui-compare", "error", "element-find", "scroll-check"] {
        store
            .get(name)
            .unwrap_or_else(|_| panic!("built-in preset '{name}' should survive user override"));
    }
}

#[test]
fn missing_user_presets_file_loads_builtins_only() {
    // Point at a directory that exists but has no presets.json.
    let dir = tempdir().expect("tempdir");
    let config_dir = dir.path().join(".vision-analyze");
    fs::create_dir_all(&config_dir).expect("create dir");

    let store = PresetStore::load_from_dir(&config_dir).expect("load_from_dir");
    // All built-in presets should load without error.
    store.get("ui").expect("ui built-in");
    store.get("error").expect("error built-in");
}

#[test]
fn nonexistent_config_dir_loads_builtins_only() {
    // Completely absent directory — must gracefully fall back to built-ins.
    let dir = tempdir().expect("tempdir");
    let absent = dir.path().join("does_not_exist");

    let store = PresetStore::load_from_dir(&absent).expect("load_from_dir with absent dir");
    store.get("ui").expect("ui built-in");
    store.get("error").expect("error built-in");
}

#[test]
fn user_override_format_field_propagates() {
    let dir = tempdir().expect("tempdir");
    let config_dir = dir.path().join(".vision-analyze");

    // Override `error` format to `text` (built-in uses `json`).
    let custom_json =
        r#"{"error": {"prompt": "find errors", "format": "text", "two_images": false}}"#;
    write_user_presets(&config_dir, custom_json);

    let store = PresetStore::load_from_dir(&config_dir).expect("load_from_dir");
    let preset = store.get("error").expect("error preset");
    assert_eq!(
        preset.output_format(),
        Some(vision_analyze::config::OutputFormat::Text)
    );
}

#[test]
fn user_override_two_images_flag() {
    let dir = tempdir().expect("tempdir");
    let config_dir = dir.path().join(".vision-analyze");

    // Built-in `ui` has two_images=false; override to true.
    let custom_json = r#"{"ui": {"prompt": "compare", "two_images": true}}"#;
    write_user_presets(&config_dir, custom_json);

    let store = PresetStore::load_from_dir(&config_dir).expect("load_from_dir");
    assert!(store.get("ui").unwrap().two_images);
}
