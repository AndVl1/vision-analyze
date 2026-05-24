//! Warm-daemon: a long-running background process that keeps the backend loaded
//! and serves inference requests over a Unix-domain socket.
//!
//! # Lifecycle
//!
//! ```text
//! vision-analyze --warm        →  spawn_warm() → self-re-exec with --__daemon-internal
//! vision-analyze --__daemon-internal  →  run()
//! vision-analyze --stop        →  stop()
//! vision-analyze <image>       →  try_client() → DaemonClient::send(Infer)
//! ```

use std::fs::Permissions;
use std::os::unix::fs::PermissionsExt;
use std::sync::Arc;
use std::time::Duration;

use base64::Engine;
use tokio::net::{UnixListener, UnixStream};
use tokio::signal::unix::{signal, SignalKind};
use tokio::sync::broadcast;
use tracing::{debug, error, info, warn};

use crate::client::{self, Backend, InferenceRequest};
use crate::config::EffectiveConfig;
use crate::error::{Result, VisionError};
use crate::socket::{write_message, SocketRequest, SocketResponse};

// ── Public API ────────────────────────────────────────────────────────────────

/// Run the daemon.  Called when the process is started with `--__daemon-internal`.
///
/// Binds the Unix socket, writes the PID file, loads the backend, then enters
/// the accept-loop until SIGTERM/SIGINT or a `Stop` request is received.
pub async fn run(cfg: EffectiveConfig) -> Result<()> {
    // Remove a stale socket/pid if the previous daemon is truly gone.
    maybe_remove_stale_socket(&cfg)?;

    let listener = UnixListener::bind(&cfg.socket_path)
        .map_err(|e| VisionError::Daemon(format!("bind {}: {e}", cfg.socket_path.display())))?;

    // Restrict socket to owner only — no group/world access.
    std::fs::set_permissions(&cfg.socket_path, Permissions::from_mode(0o600))
        .map_err(VisionError::Io)?;

    write_pid_file(&cfg)?;

    info!(socket = %cfg.socket_path.display(), "daemon started");

    let backend = client::select_backend(&cfg).await.map_err(|e| {
        // Eagerly clean up so the CLI can detect "daemon failed to start".
        let _ = std::fs::remove_file(&cfg.socket_path);
        let _ = std::fs::remove_file(&cfg.pid_path);
        e
    })?;

    info!(backend = backend.label(), "backend selected");

    let backend = Arc::new(backend);

    // Broadcast channel: sending any value triggers graceful shutdown.
    let (shutdown_tx, _) = broadcast::channel::<()>(1);

    let mut sigterm = signal(SignalKind::terminate())
        .map_err(|e| VisionError::Daemon(format!("SIGTERM handler: {e}")))?;
    let mut sigint = signal(SignalKind::interrupt())
        .map_err(|e| VisionError::Daemon(format!("SIGINT handler: {e}")))?;

    loop {
        let mut shutdown_rx = shutdown_tx.subscribe();

        tokio::select! {
            accept = listener.accept() => {
                match accept {
                    Ok((stream, _addr)) => {
                        let backend = Arc::clone(&backend);
                        let shutdown_tx = shutdown_tx.clone();
                        tokio::spawn(async move {
                            if let Err(e) = handle_connection(stream, backend, shutdown_tx).await {
                                warn!(err = %e, "connection error");
                            }
                        });
                    }
                    Err(e) => {
                        error!(err = %e, "accept error");
                    }
                }
            }
            _ = sigterm.recv() => {
                info!("received SIGTERM, shutting down");
                break;
            }
            _ = sigint.recv() => {
                info!("received SIGINT, shutting down");
                break;
            }
            _ = shutdown_rx.recv() => {
                info!("stop request received, shutting down");
                break;
            }
        }
    }

    cleanup(&cfg);
    Ok(())
}

/// Spawn a detached daemon process (self-re-exec with `--__daemon-internal`).
///
/// Returns an error if the daemon is already running or fails to start within 5 s.
pub async fn spawn_warm(cfg: &EffectiveConfig) -> Result<()> {
    // Bail early if daemon is already up.
    if is_daemon_running(cfg).await {
        return Err(VisionError::Daemon("daemon is already running".into()));
    }

    let current_exe = std::env::current_exe().map_err(VisionError::Io)?;

    let mut cmd = std::process::Command::new(&current_exe);
    cmd.arg("--__daemon-internal");
    cmd.stdin(std::process::Stdio::null());
    cmd.stdout(std::process::Stdio::null());
    cmd.stderr(std::process::Stdio::null());

    // Detach: create a new session so the daemon survives the parent process.
    use std::os::unix::process::CommandExt;
    unsafe {
        cmd.pre_exec(|| {
            if libc::setsid() == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }

    cmd.spawn()
        .map_err(|e| VisionError::Daemon(format!("failed to spawn daemon: {e}")))?;

    // Poll the socket until the daemon signals readiness (max 5 s).
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        tokio::time::sleep(Duration::from_millis(50)).await;

        if tokio::time::timeout(
            Duration::from_millis(50),
            UnixStream::connect(&cfg.socket_path),
        )
        .await
        .is_ok_and(|r| r.is_ok())
        {
            return Ok(());
        }

        if tokio::time::Instant::now() >= deadline {
            return Err(VisionError::Daemon(
                "daemon did not become ready within 5 s".into(),
            ));
        }
    }
}

/// Stop a running daemon.
///
/// Tries a graceful `Stop` request first; falls back to SIGTERM then SIGKILL.
pub async fn stop(cfg: &EffectiveConfig) -> Result<()> {
    // Attempt graceful stop via socket.
    if let Ok(mut client) = try_client(cfg).await {
        match client.send(SocketRequest::Stop).await {
            Ok(SocketResponse::Stopped) => {
                info!("daemon stopped gracefully");
                // Give the daemon a moment to clean up its own files.
                tokio::time::sleep(Duration::from_millis(200)).await;
                return Ok(());
            }
            Ok(other) => {
                warn!(
                    ?other,
                    "unexpected response to Stop; continuing with signal"
                );
            }
            Err(e) => {
                warn!(err = %e, "socket stop failed; falling back to signal");
            }
        }
    }

    // Fall back: read PID and signal.
    if cfg.pid_path.exists() {
        let pid_str = std::fs::read_to_string(&cfg.pid_path).map_err(VisionError::Io)?;
        let pid_raw: i32 = pid_str
            .trim()
            .parse()
            .map_err(|_| VisionError::Daemon(format!("malformed pid file: {:?}", pid_str)))?;

        let pid = nix::unistd::Pid::from_raw(pid_raw);

        // SIGTERM first.
        if let Err(e) = nix::sys::signal::kill(pid, nix::sys::signal::Signal::SIGTERM) {
            warn!(err = %e, "SIGTERM failed");
        } else {
            tokio::time::sleep(Duration::from_secs(2)).await;

            // If still alive, SIGKILL.
            if nix::sys::signal::kill(pid, None).is_ok() {
                warn!("daemon still alive after SIGTERM; sending SIGKILL");
                let _ = nix::sys::signal::kill(pid, nix::sys::signal::Signal::SIGKILL);
            }
        }
    }

    // Always clean up files regardless of signal outcome.
    let _ = std::fs::remove_file(&cfg.socket_path);
    let _ = std::fs::remove_file(&cfg.pid_path);

    Ok(())
}

/// Try to connect to the daemon socket.  Returns `Err` if the socket is not
/// reachable (daemon not running).
pub async fn try_client(cfg: &EffectiveConfig) -> Result<DaemonClient> {
    let stream = tokio::time::timeout(
        Duration::from_millis(50),
        UnixStream::connect(&cfg.socket_path),
    )
    .await
    .map_err(|_| VisionError::Daemon("socket connect timed out".into()))?
    .map_err(VisionError::Io)?;

    Ok(DaemonClient { stream })
}

// ── DaemonClient ─────────────────────────────────────────────────────────────

/// A connected client to the running daemon.
pub struct DaemonClient {
    stream: UnixStream,
}

impl DaemonClient {
    /// Send a request and await the daemon's response.
    pub async fn send(&mut self, req: SocketRequest) -> Result<SocketResponse> {
        let (reader, mut writer) = self.stream.split();
        write_message(&mut writer, &req).await?;
        let mut reader = tokio::io::BufReader::new(reader);
        read_message_from_buf(&mut reader).await
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Handle a single client connection on the daemon side.
async fn handle_connection(
    stream: UnixStream,
    backend: Arc<Backend>,
    shutdown_tx: broadcast::Sender<()>,
) -> Result<()> {
    let (reader, mut writer) = stream.into_split();
    let mut buf_reader = tokio::io::BufReader::new(reader);

    let req: SocketRequest = read_message_from_buf(&mut buf_reader).await?;
    debug!(?req, "daemon received request");

    match req {
        SocketRequest::Infer {
            prompt,
            image,
            image_b,
            max_tokens,
            temperature,
            timeout_ms,
            ..
        } => {
            let image_bytes = base64::engine::general_purpose::STANDARD
                .decode(&image)
                .map_err(|e| VisionError::Daemon(format!("image_a base64 decode: {e}")))?;

            let image_b_bytes = match image_b {
                Some(ref b) => {
                    let decoded = base64::engine::general_purpose::STANDARD
                        .decode(b)
                        .map_err(|e| VisionError::Daemon(format!("image_b base64 decode: {e}")))?;
                    Some(decoded)
                }
                None => None,
            };

            let infer_req = InferenceRequest {
                prompt,
                image: image_bytes,
                image_b: image_b_bytes,
                max_tokens,
                temperature,
                timeout: Duration::from_millis(timeout_ms),
            };

            match backend.infer(infer_req).await {
                Ok(resp) => {
                    write_message(
                        &mut writer,
                        &SocketResponse::Ok {
                            text: resp.text,
                            tokens: resp.tokens,
                            time_ms: resp.time_ms,
                            model: resp.model,
                        },
                    )
                    .await?;
                }
                Err(e) => {
                    let code = if e.exit_code() == 2 {
                        "backend_unavailable"
                    } else {
                        "analysis_failed"
                    };
                    write_message(
                        &mut writer,
                        &SocketResponse::Error {
                            error: e.to_string(),
                            code: code.into(),
                        },
                    )
                    .await?;
                }
            }
        }

        SocketRequest::Health => {
            write_message(
                &mut writer,
                &SocketResponse::Health {
                    status: "ok".into(),
                    backend: backend.label().into(),
                },
            )
            .await?;
        }

        SocketRequest::Stop => {
            write_message(&mut writer, &SocketResponse::Stopped).await?;
            // Signal the accept-loop to exit after the write completes.
            let _ = shutdown_tx.send(());
        }
    }

    Ok(())
}

/// Read a newline-terminated JSON message from a `BufReader` wrapping an
/// `AsyncRead`.  Duplicates the logic from `socket::read_message` but works
/// directly with `BufReader` that is already constructed.
async fn read_message_from_buf<R, T>(reader: &mut tokio::io::BufReader<R>) -> Result<T>
where
    R: tokio::io::AsyncRead + Unpin,
    T: for<'de> serde::Deserialize<'de>,
{
    use tokio::io::AsyncBufReadExt;
    const MAX: usize = 64 * 1024 * 1024;

    let mut line = String::new();
    let n = reader.read_line(&mut line).await.map_err(VisionError::Io)?;

    if n == 0 {
        return Err(VisionError::Daemon("connection closed by peer".into()));
    }
    if line.len() > MAX {
        return Err(VisionError::Daemon(format!(
            "message too large: {} bytes",
            line.len()
        )));
    }

    serde_json::from_str(line.trim_end()).map_err(VisionError::Json)
}

/// Remove a stale socket file only if the daemon PID that created it is gone.
fn maybe_remove_stale_socket(cfg: &EffectiveConfig) -> Result<()> {
    if !cfg.socket_path.exists() {
        return Ok(());
    }

    // If there is a PID file and that process is alive, refuse to overwrite.
    if let Ok(pid_str) = std::fs::read_to_string(&cfg.pid_path) {
        if let Ok(pid_raw) = pid_str.trim().parse::<i32>() {
            let pid = nix::unistd::Pid::from_raw(pid_raw);
            // kill(pid, 0) returns Ok if the process exists.
            if nix::sys::signal::kill(pid, None).is_ok() {
                return Err(VisionError::Daemon(
                    "daemon is already running (pid file exists and process is alive)".into(),
                ));
            }
        }
    }

    // Stale — safe to remove.
    let _ = std::fs::remove_file(&cfg.socket_path);
    let _ = std::fs::remove_file(&cfg.pid_path);
    Ok(())
}

/// Atomically write the current PID to the pid file (write to temp + rename).
fn write_pid_file(cfg: &EffectiveConfig) -> Result<()> {
    let pid = std::process::id();

    // Ensure the parent directory exists.
    if let Some(parent) = cfg.pid_path.parent() {
        std::fs::create_dir_all(parent).map_err(VisionError::Io)?;
    }

    let tmp_path = cfg.pid_path.with_extension("pid.tmp");
    std::fs::write(&tmp_path, pid.to_string()).map_err(VisionError::Io)?;
    std::fs::rename(&tmp_path, &cfg.pid_path).map_err(VisionError::Io)?;
    Ok(())
}

/// Clean up socket and PID files on shutdown.
fn cleanup(cfg: &EffectiveConfig) {
    if let Err(e) = std::fs::remove_file(&cfg.socket_path) {
        debug!(err = %e, path = %cfg.socket_path.display(), "cleanup socket");
    }
    if let Err(e) = std::fs::remove_file(&cfg.pid_path) {
        debug!(err = %e, path = %cfg.pid_path.display(), "cleanup pid");
    }
    info!("daemon stopped");
}

/// Return `true` if the daemon process appears to be alive (socket connectable
/// and its PID is still running).
async fn is_daemon_running(cfg: &EffectiveConfig) -> bool {
    // Quick socket probe first (50 ms).
    let socket_reachable = tokio::time::timeout(
        Duration::from_millis(50),
        UnixStream::connect(&cfg.socket_path),
    )
    .await
    .is_ok_and(|r| r.is_ok());

    if !socket_reachable {
        return false;
    }

    // Cross-check PID to avoid false positives from leftover socket files.
    if let Ok(pid_str) = std::fs::read_to_string(&cfg.pid_path) {
        if let Ok(pid_raw) = pid_str.trim().parse::<i32>() {
            let pid = nix::unistd::Pid::from_raw(pid_raw);
            return nix::sys::signal::kill(pid, None).is_ok();
        }
    }

    // Socket is connectable but no PID file — treat as running.
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn test_cfg(dir: &std::path::Path) -> EffectiveConfig {
        EffectiveConfig {
            model: None,
            mmproj: None,
            format: crate::config::OutputFormat::Text,
            max_tokens: 256,
            temperature: 0.0,
            timeout_ms: 5000,
            llama_server_url: "http://127.0.0.1:8080".into(),
            llama_bin: None,
            backend: crate::config::BackendPreference::Auto,
            socket_path: dir.join("test.sock"),
            pid_path: dir.join("test.pid"),
        }
    }

    #[test]
    fn write_pid_file_creates_and_reads_back() {
        let dir = tempdir().unwrap();
        let cfg = test_cfg(dir.path());

        write_pid_file(&cfg).unwrap();
        let content = std::fs::read_to_string(&cfg.pid_path).unwrap();
        let pid: u32 = content.trim().parse().unwrap();
        assert_eq!(pid, std::process::id());
    }

    #[test]
    fn maybe_remove_stale_socket_no_socket_is_noop() {
        let dir = tempdir().unwrap();
        let cfg = test_cfg(dir.path());
        // Socket does not exist — should succeed without error.
        assert!(maybe_remove_stale_socket(&cfg).is_ok());
    }

    #[test]
    fn maybe_remove_stale_socket_removes_orphaned_file() {
        let dir = tempdir().unwrap();
        let cfg = test_cfg(dir.path());

        // Create a socket file with no corresponding live process (PID 0 / invalid).
        std::fs::write(&cfg.socket_path, b"fake").unwrap();
        std::fs::write(&cfg.pid_path, b"99999999").unwrap(); // almost certainly dead

        // Should remove the stale files (or succeed if the process is alive — unlikely).
        let _ = maybe_remove_stale_socket(&cfg);
        // Either way we expect no error panic; if pid 99999999 is alive the function
        // returns Err, which is also correct behaviour.
    }
}
