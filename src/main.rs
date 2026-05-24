use std::time::Duration;

use base64::Engine;
use clap::Parser;
use tracing_subscriber::EnvFilter;

use vision_analyze::cli::Args;
use vision_analyze::client::{self, InferenceRequest};
use vision_analyze::config::EffectiveConfig;
use vision_analyze::daemon;
use vision_analyze::error::{Result, VisionError};
use vision_analyze::output;
use vision_analyze::presets::PresetStore;
use vision_analyze::socket::{SocketRequest, SocketResponse};

/// Maximum image file size accepted from disk (32 MiB).
const MAX_IMAGE_BYTES: u64 = 32 * 1024 * 1024;

fn main() {
    let code = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("tokio runtime")
        .block_on(async { run().await });
    std::process::exit(code);
}

async fn run() -> i32 {
    let args = Args::parse();

    init_tracing(args.verbose);

    let cfg = EffectiveConfig::build(&args);

    match dispatch(&args, cfg).await {
        Ok(code) => code,
        Err(e) => {
            eprintln!("{e}");
            e.exit_code()
        }
    }
}

async fn dispatch(args: &Args, cfg: EffectiveConfig) -> Result<i32> {
    if args.daemon_internal {
        daemon::run(cfg).await?;
        return Ok(0);
    }

    if args.warm {
        daemon::spawn_warm(&cfg).await?;
        eprintln!(
            "vision-analyze daemon started (socket: {})",
            cfg.socket_path.display()
        );
        return Ok(0);
    }

    if args.stop {
        daemon::stop(&cfg).await?;
        return Ok(0);
    }

    if args.health {
        return health_check(&cfg).await;
    }

    inference(args, &cfg).await
}

async fn health_check(cfg: &EffectiveConfig) -> Result<i32> {
    if let Ok(mut c) = daemon::try_client(cfg).await {
        if let Ok(SocketResponse::Health { .. }) = c.send(SocketRequest::Health).await {
            println!("{{\"status\":\"ok\",\"mode\":\"warm\"}}");
            return Ok(0);
        }
    }

    match client::select_backend(cfg).await {
        Ok(b) => {
            b.health().await?;
            println!(
                "{{\"status\":\"ok\",\"mode\":\"cold\",\"backend\":\"{}\"}}",
                b.label()
            );
            Ok(0)
        }
        Err(e) => {
            eprintln!("{e}");
            Ok(e.exit_code())
        }
    }
}

async fn inference(args: &Args, cfg: &EffectiveConfig) -> Result<i32> {
    let image_path = args
        .image
        .as_ref()
        .ok_or_else(|| VisionError::InvalidArgs("image path required".into()))?;

    let store = PresetStore::load()?;

    let preset = args
        .preset
        .as_deref()
        .map(|name| store.get(name).cloned())
        .transpose()?;

    let prompt = if let Some(p) = args.prompt.clone() {
        p
    } else if let Some(p) = preset.as_ref().map(|pr| pr.prompt.clone()) {
        p
    } else {
        return Err(VisionError::InvalidArgs(
            "prompt or --preset required".into(),
        ));
    };

    let output_format = preset
        .as_ref()
        .and_then(|p| p.output_format())
        .unwrap_or(cfg.format);

    // C4: check image size before reading to avoid large allocations.
    let meta = tokio::fs::metadata(image_path).await?;
    if meta.len() > MAX_IMAGE_BYTES {
        return Err(VisionError::ImageTooLarge(meta.len(), MAX_IMAGE_BYTES));
    }
    let image = tokio::fs::read(image_path).await?;

    let two_images_needed = preset.as_ref().map(|p| p.two_images).unwrap_or(false);

    let image_b: Option<Vec<u8>> = match &args.image_b {
        Some(path) => {
            // C4: apply same size limit to the second image.
            let meta_b = tokio::fs::metadata(path).await?;
            if meta_b.len() > MAX_IMAGE_BYTES {
                return Err(VisionError::ImageTooLarge(meta_b.len(), MAX_IMAGE_BYTES));
            }
            Some(tokio::fs::read(path).await?)
        }
        None => None,
    };

    if two_images_needed && image_b.is_none() {
        return Err(VisionError::InvalidArgs(
            "this preset requires two images: provide --image-b".into(),
        ));
    }

    let req = InferenceRequest {
        prompt: prompt.clone(),
        image: image.clone(),
        image_b: image_b.clone(),
        max_tokens: cfg.max_tokens,
        temperature: cfg.temperature,
        timeout: Duration::from_millis(cfg.timeout_ms),
    };

    // Try warm daemon first.
    if let Ok(mut c) = daemon::try_client(cfg).await {
        let socket_req = SocketRequest::Infer {
            prompt: prompt.clone(),
            image: base64::engine::general_purpose::STANDARD.encode(&image),
            image_b: image_b
                .as_deref()
                .map(|b| base64::engine::general_purpose::STANDARD.encode(b)),
            max_tokens: cfg.max_tokens,
            temperature: cfg.temperature,
            timeout_ms: cfg.timeout_ms,
            format: format!("{:?}", output_format).to_lowercase(),
            preset: args.preset.clone(),
        };

        match c.send(socket_req).await {
            Ok(SocketResponse::Ok {
                text,
                tokens,
                time_ms,
                model,
            }) => {
                let resp = vision_analyze::client::InferenceResponse {
                    text,
                    tokens,
                    time_ms,
                    model,
                };
                let out = output::render(&resp, output_format);
                println!("{out}");
                return Ok(0);
            }
            Ok(SocketResponse::Error { error, .. }) => {
                tracing::warn!(err = %error, "daemon inference failed, falling back to cold mode");
                // Fall through to cold path.
            }
            Ok(_) | Err(_) => {
                // Unexpected response — fall through to cold path.
            }
        }
    }

    // Cold path: select backend and run directly.
    let backend = client::select_backend(cfg).await?;
    let resp = backend.infer(req).await?;
    let out = output::render(&resp, output_format);
    println!("{out}");
    Ok(0)
}

fn init_tracing(verbose: u8) {
    let filter = match std::env::var("RUST_LOG").ok().filter(|v| !v.is_empty()) {
        Some(val) => EnvFilter::new(val),
        None => {
            let level = match verbose {
                0 => "warn",
                1 => "info",
                2 => "debug",
                _ => "trace",
            };
            EnvFilter::new(level)
        }
    };

    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(filter)
        .init();
}
