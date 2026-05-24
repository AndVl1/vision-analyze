use thiserror::Error;

#[derive(Debug, Error)]
pub enum VisionError {
    #[error("backend unavailable: {0}")]
    BackendUnavailable(String),

    #[error("model not found: {0}")]
    ModelNotFound(String),

    #[error("analysis failed: {0}")]
    Analysis(String),

    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("http: {0}")]
    Http(#[from] reqwest::Error),

    #[error("json: {0}")]
    Json(#[from] serde_json::Error),

    #[error("preset '{0}' not found")]
    PresetNotFound(String),

    #[error("invalid arguments: {0}")]
    InvalidArgs(String),

    #[error("timeout after {0} ms")]
    Timeout(u64),

    #[error("daemon: {0}")]
    Daemon(String),

    #[error("payload too large: {0} bytes (limit {1} bytes)")]
    PayloadTooLarge(usize, usize),

    #[error("image too large: {0} bytes (limit {1} bytes)")]
    ImageTooLarge(u64, u64),
}

impl VisionError {
    /// Maps error variants to CLI exit codes (per spec):
    /// 0 — ok, 1 — analysis error, 2 — backend/model not available.
    pub fn exit_code(&self) -> i32 {
        match self {
            VisionError::BackendUnavailable(_) | VisionError::ModelNotFound(_) => 2,
            _ => 1,
        }
    }
}

pub type Result<T> = std::result::Result<T, VisionError>;
