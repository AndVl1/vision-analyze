use std::path::PathBuf;

use clap::{ArgAction, Parser};

#[derive(Debug, Parser)]
#[command(
    name = "vision-analyze",
    version,
    about = "CLI for visual screenshot analysis via llama.cpp",
    long_about = None,
)]
pub struct Args {
    /// Path to a GGUF model file. Overrides $VISION_MODEL_PATH.
    #[arg(long, value_name = "PATH")]
    pub model: Option<PathBuf>,

    /// Path to the multimodal projector (mmproj) GGUF.
    #[arg(long, value_name = "PATH")]
    pub mmproj: Option<PathBuf>,

    /// Output format: text (default) | json.
    #[arg(long, value_name = "FORMAT")]
    pub format: Option<String>,

    /// Maximum number of output tokens.
    #[arg(long, value_name = "N")]
    pub max_tokens: Option<u32>,

    /// Sampling temperature (default 0.0).
    #[arg(long, value_name = "F")]
    pub temperature: Option<f32>,

    /// Timeout in milliseconds (default 30000).
    #[arg(long, value_name = "MS")]
    pub timeout: Option<u64>,

    /// Use a named preset (overrides prompt + format if matched).
    #[arg(long, value_name = "NAME")]
    pub preset: Option<String>,

    /// Second image (required for two_images presets like ui-compare).
    #[arg(long = "image-b", value_name = "PATH")]
    pub image_b: Option<PathBuf>,

    /// Force backend selection: auto | server | cli.
    #[arg(long, value_name = "MODE")]
    pub backend: Option<String>,

    /// Launch a warm daemon (preloads model, listens on a unix socket).
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["stop", "health"])]
    pub warm: bool,

    /// Stop a running warm daemon.
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["warm", "health"])]
    pub stop: bool,

    /// Health check (warm daemon if running, otherwise backend).
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["warm", "stop"])]
    pub health: bool,

    /// Increase log verbosity (stderr). -v info, -vv debug, -vvv trace.
    #[arg(short = 'v', long = "verbose", action = ArgAction::Count)]
    pub verbose: u8,

    /// Internal: launched by `--warm` to become the daemon process. Not for direct use.
    #[arg(long = "__daemon-internal", hide = true, action = ArgAction::SetTrue)]
    pub daemon_internal: bool,

    /// Image path.
    #[arg(value_name = "IMAGE")]
    pub image: Option<PathBuf>,

    /// Prompt text (ignored when --preset supplies a prompt).
    #[arg(value_name = "PROMPT")]
    pub prompt: Option<String>,
}

impl Args {
    /// True when this invocation should perform inference (vs. control ops like
    /// `--warm` / `--stop` / `--health`).
    pub fn is_inference(&self) -> bool {
        !self.warm && !self.stop && !self.health && !self.daemon_internal
    }
}
