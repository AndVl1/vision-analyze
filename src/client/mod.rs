pub mod cli;
pub mod detect;
pub mod server;

use std::path::PathBuf;
use std::time::Duration;

use crate::config::EffectiveConfig;
use crate::error::{Result, VisionError};

/// One inference request — already includes the resolved prompt + raw image bytes.
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub prompt: String,
    /// Primary image bytes (raw, NOT base64).
    pub image: Vec<u8>,
    /// Optional second image (for `two_images` presets).
    pub image_b: Option<Vec<u8>>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub timeout: Duration,
}

/// Inference response (model output + metrics).
#[derive(Debug, Clone)]
pub struct InferenceResponse {
    pub text: String,
    pub tokens: u32,
    pub time_ms: u64,
    pub model: String,
}

/// Resolved backend ready to execute inference.
///
/// Both variants are cheap to clone: `ServerClient` wraps an `Arc`-backed
/// `reqwest::Client`; `CliClient` wraps `PathBuf` fields.  The `Clone` impl
/// is required by the `ArcSwap`-based concurrency model in `daemon.rs`
/// (I2 fix): inference takes an `Arc` snapshot of the current backend
/// without holding any lock across the await point.
#[derive(Clone)]
pub enum Backend {
    Server(server::ServerClient),
    Cli(cli::CliClient),
}

impl Backend {
    pub async fn infer(&self, req: InferenceRequest) -> Result<InferenceResponse> {
        match self {
            Backend::Server(c) => c.infer(req).await,
            Backend::Cli(c) => c.infer(req).await,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Backend::Server(_) => "server",
            Backend::Cli(_) => "cli",
        }
    }

    pub async fn health(&self) -> Result<()> {
        match self {
            Backend::Server(c) => c.health().await,
            Backend::Cli(c) => c.health().await,
        }
    }
}

pub async fn select_backend(cfg: &EffectiveConfig) -> Result<Backend> {
    detect::select(cfg).await
}

/// Helper: encode bytes to base64 standard alphabet.
pub fn encode_b64(bytes: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// Resolve the model path from config or return a descriptive error.
pub fn resolve_model_path(cfg: &EffectiveConfig) -> Result<PathBuf> {
    cfg.model.clone().ok_or_else(|| {
        VisionError::ModelNotFound(
            "model path not configured (set --model or $VISION_MODEL_PATH)".into(),
        )
    })
}
