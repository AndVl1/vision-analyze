//! Backend autodetect: prefer `llama-server` if reachable, else `llama-cli`.

use std::time::Duration;

use crate::client::{cli::CliClient, server::ServerClient, Backend};
use crate::config::{BackendPreference, EffectiveConfig};
use crate::error::{Result, VisionError};

const SERVER_PROBE_TIMEOUT: Duration = Duration::from_millis(200);

pub async fn select(cfg: &EffectiveConfig) -> Result<Backend> {
    match cfg.backend {
        BackendPreference::Server => server(cfg).await,
        BackendPreference::Cli => cli(cfg),
        BackendPreference::Auto => {
            if ServerClient::probe(&cfg.llama_server_url, SERVER_PROBE_TIMEOUT).await {
                Ok(Backend::Server(ServerClient::new(
                    cfg.llama_server_url.clone(),
                )))
            } else {
                cli(cfg).map_err(|_| {
                    VisionError::BackendUnavailable(format!(
                        "neither llama-server (probed {}) nor a local llama-cli binary is available; \
                         install llama.cpp or set --backend / VISION_LLAMA_BIN",
                        cfg.llama_server_url
                    ))
                })
            }
        }
    }
}

async fn server(cfg: &EffectiveConfig) -> Result<Backend> {
    if !ServerClient::probe(&cfg.llama_server_url, SERVER_PROBE_TIMEOUT).await {
        return Err(VisionError::BackendUnavailable(format!(
            "llama-server not reachable at {}",
            cfg.llama_server_url
        )));
    }
    Ok(Backend::Server(ServerClient::new(
        cfg.llama_server_url.clone(),
    )))
}

fn cli(cfg: &EffectiveConfig) -> Result<Backend> {
    let bin = CliClient::locate(cfg).ok_or_else(|| {
        VisionError::BackendUnavailable(
            "no llama-cli/llama-mtmd-cli in PATH (set VISION_LLAMA_BIN)".into(),
        )
    })?;
    let model = crate::client::resolve_model_path(cfg)?;
    Ok(Backend::Cli(CliClient::new(bin, model, cfg.mmproj.clone())))
}
