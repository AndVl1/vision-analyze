use std::path::PathBuf;

use crate::cli::Args;

/// Effective configuration after merging env vars + CLI flags.
#[derive(Debug, Clone)]
pub struct EffectiveConfig {
    pub model: Option<PathBuf>,
    pub mmproj: Option<PathBuf>,
    pub format: OutputFormat,
    pub max_tokens: u32,
    pub temperature: f32,
    pub timeout_ms: u64,
    pub llama_server_url: String,
    pub llama_bin: Option<PathBuf>,
    pub backend: BackendPreference,
    pub socket_path: PathBuf,
    pub pid_path: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Text,
    Json,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendPreference {
    Auto,
    Server,
    Cli,
}

impl OutputFormat {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "text" | "txt" | "plain" => Some(OutputFormat::Text),
            "json" => Some(OutputFormat::Json),
            _ => None,
        }
    }
}

impl BackendPreference {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Some(BackendPreference::Auto),
            "server" => Some(BackendPreference::Server),
            "cli" => Some(BackendPreference::Cli),
            _ => None,
        }
    }
}

const DEFAULT_LLAMA_SERVER: &str = "http://127.0.0.1:8080";
const DEFAULT_SOCKET: &str = "/tmp/vision-analyze.sock";
const DEFAULT_PID: &str = "/tmp/vision-analyze.pid";

impl EffectiveConfig {
    /// Build config from CLI args + environment. CLI flags take precedence over env.
    pub fn build(args: &Args) -> Self {
        let env = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());

        let model = args
            .model
            .clone()
            .or_else(|| env("VISION_MODEL_PATH").map(PathBuf::from))
            .or_else(default_model_path);

        let mmproj = args
            .mmproj
            .clone()
            .or_else(|| env("VISION_MMPROJ_PATH").map(PathBuf::from))
            .or_else(|| model.as_ref().and_then(|p| infer_mmproj_path(p.as_path())));

        let format = args
            .format
            .as_deref()
            .and_then(OutputFormat::parse)
            .or_else(|| {
                env("VISION_DEFAULT_FORMAT")
                    .as_deref()
                    .and_then(OutputFormat::parse)
            })
            .unwrap_or(OutputFormat::Text);

        let max_tokens = args
            .max_tokens
            .or_else(|| env("VISION_MAX_TOKENS").and_then(|v| v.parse().ok()))
            .unwrap_or(1024);

        let temperature = args.temperature.unwrap_or(0.0);

        let timeout_ms = args
            .timeout
            .or_else(|| env("VISION_TIMEOUT").and_then(|v| v.parse().ok()))
            .unwrap_or(30_000);

        let llama_server_url =
            env("VISION_LLAMA_SERVER").unwrap_or_else(|| DEFAULT_LLAMA_SERVER.to_string());

        let llama_bin = env("VISION_LLAMA_BIN").map(PathBuf::from);

        let backend = args
            .backend
            .as_deref()
            .and_then(BackendPreference::parse)
            .unwrap_or(BackendPreference::Auto);

        let socket_path = env("VISION_SOCKET_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_SOCKET));
        let pid_path = env("VISION_PID_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_PID));

        Self {
            model,
            mmproj,
            format,
            max_tokens,
            temperature,
            timeout_ms,
            llama_server_url,
            llama_bin,
            backend,
            socket_path,
            pid_path,
        }
    }

    pub fn config_dir() -> Option<PathBuf> {
        directories::BaseDirs::new().map(|d| d.home_dir().join(".vision-analyze"))
    }
}

fn default_model_path() -> Option<PathBuf> {
    EffectiveConfig::config_dir().map(|d| d.join("model.gguf"))
}

/// If model is `/path/to/qwen2-vl.gguf`, guess `/path/to/mmproj.gguf` or
/// `/path/to/qwen2-vl.mmproj.gguf` as the projector.
fn infer_mmproj_path(model: &std::path::Path) -> Option<PathBuf> {
    let parent = model.parent()?;
    for name in ["mmproj.gguf", "mmproj-model.gguf"] {
        let candidate = parent.join(name);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    let stem = model.file_stem()?.to_string_lossy().to_string();
    let candidate = parent.join(format!("{stem}.mmproj.gguf"));
    if candidate.exists() {
        return Some(candidate);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_format_parse() {
        assert_eq!(OutputFormat::parse("json"), Some(OutputFormat::Json));
        assert_eq!(OutputFormat::parse("TEXT"), Some(OutputFormat::Text));
        assert_eq!(OutputFormat::parse("yaml"), None);
    }

    #[test]
    fn backend_pref_parse() {
        assert_eq!(
            BackendPreference::parse("auto"),
            Some(BackendPreference::Auto)
        );
        assert_eq!(
            BackendPreference::parse("Server"),
            Some(BackendPreference::Server)
        );
        assert_eq!(BackendPreference::parse("native"), None);
    }
}
