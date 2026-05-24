use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::config::OutputFormat;
use crate::error::{Result, VisionError};

const DEFAULT_PRESETS_JSON: &str = include_str!("../presets/default.json");

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Preset {
    pub prompt: String,
    #[serde(default)]
    pub format: Option<String>,
    #[serde(default)]
    pub two_images: bool,
}

impl Preset {
    pub fn output_format(&self) -> Option<OutputFormat> {
        self.format.as_deref().and_then(OutputFormat::parse)
    }
}

#[derive(Debug, Clone)]
pub struct PresetStore {
    presets: HashMap<String, Preset>,
}

impl PresetStore {
    /// Load built-in presets, then overlay user-defined presets from
    /// `~/.vision-analyze/presets.json` (if present). User entries override
    /// built-ins of the same name.
    pub fn load() -> Result<Self> {
        let mut presets: HashMap<String, Preset> = serde_json::from_str(DEFAULT_PRESETS_JSON)
            .map_err(|e| VisionError::Analysis(format!("default presets parse: {e}")))?;

        if let Some(path) = user_presets_path() {
            if path.exists() {
                let bytes = std::fs::read_to_string(&path)?;
                let user: HashMap<String, Preset> = serde_json::from_str(&bytes)
                    .map_err(|e| VisionError::Analysis(format!("user presets parse: {e}")))?;
                presets.extend(user);
            }
        }

        Ok(Self { presets })
    }

    pub fn load_from_str(s: &str) -> Result<Self> {
        let presets = serde_json::from_str(s)
            .map_err(|e| VisionError::Analysis(format!("presets parse: {e}")))?;
        Ok(Self { presets })
    }

    /// Like [`load`] but uses the provided directory as the config dir instead
    /// of resolving `~/.vision-analyze` via `directories::BaseDirs`.
    ///
    /// Useful in tests that need to inject a custom presets file without
    /// relying on `$HOME` being writable or correctly resolved by the OS.
    pub fn load_from_dir(config_dir: &Path) -> Result<Self> {
        let mut presets: HashMap<String, Preset> = serde_json::from_str(DEFAULT_PRESETS_JSON)
            .map_err(|e| VisionError::Analysis(format!("default presets parse: {e}")))?;

        let path = config_dir.join("presets.json");
        if path.exists() {
            let bytes = std::fs::read_to_string(&path)?;
            let user: HashMap<String, Preset> = serde_json::from_str(&bytes)
                .map_err(|e| VisionError::Analysis(format!("user presets parse: {e}")))?;
            presets.extend(user);
        }

        Ok(Self { presets })
    }

    pub fn get(&self, name: &str) -> Result<&Preset> {
        self.presets
            .get(name)
            .ok_or_else(|| VisionError::PresetNotFound(name.to_string()))
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.presets.keys().map(|s| s.as_str())
    }
}

fn user_presets_path() -> Option<std::path::PathBuf> {
    crate::config::EffectiveConfig::config_dir().map(|d| d.join("presets.json"))
}

pub fn user_presets_path_for(dir: &Path) -> std::path::PathBuf {
    dir.join("presets.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_presets_load() {
        let store = PresetStore::load_from_str(DEFAULT_PRESETS_JSON).unwrap();
        for name in ["ui", "ui-compare", "error", "element-find", "scroll-check"] {
            store
                .get(name)
                .unwrap_or_else(|_| panic!("missing preset: {name}"));
        }
    }

    #[test]
    fn two_images_flag_propagates() {
        let store = PresetStore::load_from_str(DEFAULT_PRESETS_JSON).unwrap();
        assert!(store.get("ui-compare").unwrap().two_images);
        assert!(!store.get("ui").unwrap().two_images);
    }

    #[test]
    fn unknown_preset_errors() {
        let store = PresetStore::load_from_str(DEFAULT_PRESETS_JSON).unwrap();
        assert!(matches!(
            store.get("nope"),
            Err(VisionError::PresetNotFound(_))
        ));
    }

    #[test]
    fn preset_format_parsed() {
        let store = PresetStore::load_from_str(DEFAULT_PRESETS_JSON).unwrap();
        assert_eq!(
            store.get("ui").unwrap().output_format(),
            Some(OutputFormat::Json)
        );
    }
}
