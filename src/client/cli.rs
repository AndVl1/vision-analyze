//! Child-process backend that shells out to `llama-mtmd-cli` (preferred) or
//! `llama-cli` for vision inference. Used when no `llama-server` is reachable.

use std::path::PathBuf;
use std::time::Instant;

use tokio::io::AsyncWriteExt;
use tokio::process::Command;

use crate::client::{InferenceRequest, InferenceResponse};
use crate::config::EffectiveConfig;
use crate::error::{Result, VisionError};

#[derive(Debug, Clone)]
pub struct CliClient {
    binary: PathBuf,
    model: PathBuf,
    mmproj: Option<PathBuf>,
}

impl CliClient {
    pub fn new(binary: PathBuf, model: PathBuf, mmproj: Option<PathBuf>) -> Self {
        Self {
            binary,
            model,
            mmproj,
        }
    }

    /// Resolve the binary from explicit config or `$PATH`. Returns None when
    /// nothing usable is found — caller decides whether that's fatal.
    pub fn locate(cfg: &EffectiveConfig) -> Option<PathBuf> {
        if let Some(p) = cfg.llama_bin.as_ref() {
            if p.exists() {
                return Some(p.clone());
            }
        }
        for candidate in ["llama-mtmd-cli", "llama-cli"] {
            if let Ok(p) = which::which(candidate) {
                return Some(p);
            }
        }
        None
    }

    pub async fn health(&self) -> Result<()> {
        if !self.binary.exists() {
            return Err(VisionError::BackendUnavailable(format!(
                "llama binary not found at {}",
                self.binary.display()
            )));
        }
        if !self.model.exists() {
            return Err(VisionError::ModelNotFound(format!(
                "model not found at {}",
                self.model.display()
            )));
        }
        Ok(())
    }

    pub async fn infer(&self, req: InferenceRequest) -> Result<InferenceResponse> {
        let started = Instant::now();

        // Write images to temporary files. llama-mtmd-cli reads images from disk
        // (`--image PATH`), not stdin/base64.
        let tmp = tempfile::tempdir()?;
        let image_path = tmp.path().join("input-a.png");
        tokio::fs::write(&image_path, &req.image).await?;

        let image_b_path = if let Some(b) = &req.image_b {
            let p = tmp.path().join("input-b.png");
            tokio::fs::write(&p, b).await?;
            Some(p)
        } else {
            None
        };

        let mut cmd = Command::new(&self.binary);
        cmd.arg("-m").arg(&self.model);
        if let Some(mmproj) = self.mmproj.as_ref() {
            cmd.arg("--mmproj").arg(mmproj);
        }
        cmd.arg("--image").arg(&image_path);
        if let Some(p) = image_b_path.as_ref() {
            cmd.arg("--image").arg(p);
        }
        cmd.arg("-n").arg(req.max_tokens.to_string());
        cmd.arg("--temp").arg(format!("{}", req.temperature));
        cmd.arg("-p").arg(&req.prompt);
        cmd.stdin(std::process::Stdio::null());
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        let child = cmd.spawn().map_err(|e| {
            VisionError::BackendUnavailable(format!("spawn {}: {e}", self.binary.display()))
        })?;

        let output = tokio::time::timeout(req.timeout, child.wait_with_output())
            .await
            .map_err(|_| VisionError::Timeout(req.timeout.as_millis() as u64))?
            .map_err(VisionError::Io)?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(VisionError::Analysis(format!(
                "llama-cli exit {:?}: {}",
                output.status.code(),
                stderr.lines().rev().take(3).collect::<Vec<_>>().join(" | ")
            )));
        }

        let raw = String::from_utf8_lossy(&output.stdout).to_string();
        let text = strip_cli_noise(&raw);

        Ok(InferenceResponse {
            text,
            tokens: 0,
            time_ms: started.elapsed().as_millis() as u64,
            model: self
                .model
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "unknown".into()),
        })
    }
}

/// Drop common llama.cpp banner lines that some builds emit on stdout
/// despite logs going to stderr (e.g. `llama_model_loader: ...`).
fn strip_cli_noise(raw: &str) -> String {
    raw.lines()
        .filter(|l| !l.starts_with("llama_") && !l.starts_with("ggml_") && !l.starts_with("main: "))
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

// Suppress unused import warning when `AsyncWriteExt` isn't reached (it's kept
// for future stdin-based protocols).
#[allow(dead_code)]
fn _keep_async_write_ext_in_scope<W: AsyncWriteExt>(_: W) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_noise_removes_llama_banners() {
        let raw =
            "llama_model_loader: loaded\nggml_metal_init: ok\nmain: starting\nThis is the answer.";
        assert_eq!(strip_cli_noise(raw), "This is the answer.");
    }
}
