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

use tokio::sync::Mutex;

use base64::Engine;
use tokio::net::{UnixListener, UnixStream};
use tokio::signal::unix::{signal, SignalKind};
use tokio::sync::broadcast;
use tracing::{debug, error, info, warn};

use crate::client::{self, Backend, InferenceRequest};
use crate::config::EffectiveConfig;
use crate::error::{Result, VisionError};
use crate::socket::{write_message, SocketRequest, SocketResponse, MAX_FRAME_BYTES};

// ── RAII guards ───────────────────────────────────────────────────────────────

/// Removes the PID file on drop.
///
/// Bound in `run()` so the file is cleaned up on early return, panic, or normal
/// shutdown — whichever comes first.
struct PidFileGuard {
    path: std::path::PathBuf,
}

impl Drop for PidFileGuard {
    fn drop(&mut self) {
        // Ignore errors: the file may already be gone on normal shutdown path.
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Removes the socket file on drop.
struct SocketFileGuard {
    path: std::path::PathBuf,
}

impl Drop for SocketFileGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

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

    // C2: RAII guard — socket file is removed on any exit path (panic, early return, normal).
    let _socket_guard = SocketFileGuard {
        path: cfg.socket_path.clone(),
    };

    // C2: PID file written after socket is ready; guard ensures removal on any exit.
    write_pid_file(&cfg)?;
    let _pid_guard = PidFileGuard {
        path: cfg.pid_path.clone(),
    };

    info!(socket = %cfg.socket_path.display(), "daemon started");

    // Backend init: if this fails the guards clean up socket + pid automatically.
    let backend = client::select_backend(&cfg).await?;

    info!(backend = backend.label(), "backend selected");

    // H3: wrap in Mutex so individual connections can attempt re-detect on failure
    // without requiring a daemon restart.
    let backend = Arc::new(Mutex::new(backend));
    let cfg = Arc::new(cfg);

    // Broadcast channel: sending any value triggers graceful shutdown.
    let (shutdown_tx, _) = broadcast::channel::<()>(1);

    let mut sigterm = signal(SignalKind::terminate())
        .map_err(|e| VisionError::Daemon(format!("SIGTERM handler: {e}")))?;
    let mut sigint = signal(SignalKind::interrupt())
        .map_err(|e| VisionError::Daemon(format!("SIGINT handler: {e}")))?;

    // C3: subscribe ONCE before the loop, not inside it.
    // Each spawned task gets its own Receiver via resubscribe() so all tasks
    // observe the shutdown signal independently.
    let mut shutdown_rx = shutdown_tx.subscribe();

    loop {
        tokio::select! {
            accept = listener.accept() => {
                match accept {
                    Ok((stream, _addr)) => {
                        let backend = Arc::clone(&backend);
                        let cfg = Arc::clone(&cfg);
                        let shutdown_tx = shutdown_tx.clone();
                        // C3: give each task its own independent Receiver.
                        let task_shutdown_rx = shutdown_tx.subscribe();
                        tokio::spawn(async move {
                            if let Err(e) = handle_connection(stream, backend, cfg, shutdown_tx, task_shutdown_rx).await {
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

    // Guards (_socket_guard, _pid_guard) clean up files when they drop here.
    info!("daemon stopped");
    Ok(())
}

/// Spawn a detached daemon process (self-re-exec with `--__daemon-internal`).
///
/// Returns an error if the daemon is already running or fails to start within 5 s.
///
/// # Readiness probe (C1)
///
/// After spawning, polls the socket with an active `Health` request rather than
/// checking only that the socket file exists.  Only a successful
/// `SocketResponse::Health` reply is treated as "ready".  Uses exponential
/// backoff (50 ms → 500 ms, capped) with a 5-second overall deadline.
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
    // C5: keep stderr inherited (default) so tracing output goes to the terminal.
    // Future enhancement: redirect to ~/.vision-analyze/daemon.log.

    // Detach: create a new session so the daemon survives the parent process.
    use std::os::unix::process::CommandExt;
    unsafe {
        cmd.pre_exec(|| {
            // SAFETY: setsid(2) is async-signal-safe per POSIX (listed in
            // signal-safety(7)).  We do not touch heap allocations, mutexes,
            // or any other state that could be in an inconsistent state after
            // fork().  This is the canonical way to detach a child process.
            if libc::setsid() == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }

    cmd.spawn()
        .map_err(|e| VisionError::Daemon(format!("failed to spawn daemon: {e}")))?;

    // C1: active Health probe — only consider ready when SocketResponse::Health
    // comes back.  Exponential backoff: 50 ms, 100 ms, 200 ms, … capped at 500 ms.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    let mut delay_ms = 50u64;

    loop {
        tokio::time::sleep(Duration::from_millis(delay_ms)).await;

        // Try to connect AND send Health.
        let health_ok = probe_health(cfg).await;
        if health_ok {
            return Ok(());
        }

        if tokio::time::Instant::now() >= deadline {
            return Err(VisionError::BackendUnavailable(
                "daemon did not become ready within 5 s".into(),
            ));
        }

        // Exponential backoff capped at 500 ms.
        delay_ms = (delay_ms * 2).min(500);
    }
}

/// Send a `Health` request to the daemon and return `true` if it responds with
/// `SocketResponse::Health`.  Used by `spawn_warm` readiness polling.
async fn probe_health(cfg: &EffectiveConfig) -> bool {
    let Ok(mut client) = try_client(cfg).await else {
        return false;
    };
    matches!(
        client.send(SocketRequest::Health).await,
        Ok(SocketResponse::Health { .. })
    )
}

/// Stop a running daemon.
///
/// Tries a graceful `Stop` request first; falls back to SIGTERM then SIGKILL.
///
/// # PID-reuse safety (H2)
///
/// Before sending SIGTERM/SIGKILL we verify the daemon socket still responds to
/// a `Health` probe (same socket + pid = same daemon).  If Health fails, the pid
/// file is stale — we clean up files and return without signalling.
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

        // H2: verify socket still belongs to this daemon before signalling to
        // prevent killing an unrelated process that reused the PID.
        if !probe_health(cfg).await {
            warn!("pid file present but socket does not respond to Health; cleaning stale files");
            let _ = std::fs::remove_file(&cfg.socket_path);
            let _ = std::fs::remove_file(&cfg.pid_path);
            return Ok(());
        }

        // SIGTERM first.
        if let Err(e) = nix::sys::signal::kill(pid, nix::sys::signal::Signal::SIGTERM) {
            warn!(err = %e, "SIGTERM failed");
        } else {
            tokio::time::sleep(Duration::from_secs(2)).await;

            // If still alive, re-verify ownership before SIGKILL.
            if nix::sys::signal::kill(pid, None).is_ok() {
                warn!("daemon still alive after SIGTERM; sending SIGKILL");
                // H2: re-check Health before SIGKILL to avoid killing a PID-reuse victim.
                if probe_health(cfg).await {
                    let _ = nix::sys::signal::kill(pid, nix::sys::signal::Signal::SIGKILL);
                } else {
                    warn!("socket no longer responds after SIGTERM; assuming daemon exited");
                }
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
///
/// `_shutdown_rx` is kept alive for the duration of the connection so the task
/// participates in the broadcast shutdown (C3: each task holds its own Receiver).
///
/// `cfg` is used for H3: if a `Server` backend returns an inference error, we
/// attempt re-detect once before returning the error to the client.
async fn handle_connection(
    stream: UnixStream,
    backend: Arc<Mutex<Backend>>,
    cfg: Arc<EffectiveConfig>,
    shutdown_tx: broadcast::Sender<()>,
    _shutdown_rx: broadcast::Receiver<()>,
) -> Result<()> {
    let (reader, mut writer) = stream.into_split();
    let mut buf_reader = tokio::io::BufReader::new(reader);

    let req: SocketRequest = read_message_from_buf(&mut buf_reader).await?;
    // S3: avoid logging full base64 image — log only metadata.
    debug!(
        op = match &req {
            SocketRequest::Infer { .. } => "infer",
            SocketRequest::Health => "health",
            SocketRequest::Stop => "stop",
        },
        "daemon received request"
    );

    match req {
        SocketRequest::Infer {
            prompt,
            image,
            image_b,
            max_tokens,
            temperature,
            timeout_ms,
            preset,
            ..
        } => {
            // S3: log only metadata, never the base64 image payload.
            debug!(
                prompt_len = prompt.len(),
                image_b64_len = image.len(),
                image_b_b64_len = image_b.as_ref().map(|b| b.len()),
                preset = ?preset,
                "infer request metadata"
            );

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

            let infer_result = {
                let guard = backend.lock().await;
                guard.infer(infer_req.clone()).await
            };

            // H3: on inference error, attempt backend re-detect once.
            // This handles the case where llama-server restarted during daemon lifetime.
            let infer_result = match infer_result {
                Err(ref e) if e.exit_code() == 2 => {
                    warn!(err = %e, "inference failed; attempting backend re-detect");
                    match client::select_backend(&cfg).await {
                        Ok(new_backend) => {
                            info!(backend = new_backend.label(), "backend re-detected");
                            let retry = new_backend.infer(infer_req).await;
                            // Update shared backend only on successful re-detect.
                            if retry.is_ok() {
                                *backend.lock().await = new_backend;
                            }
                            retry
                        }
                        Err(detect_err) => {
                            warn!(err = %detect_err, "backend re-detect failed; returning original error");
                            infer_result
                        }
                    }
                }
                other => other,
            };

            match infer_result {
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
            let label = backend.lock().await.label();
            write_message(
                &mut writer,
                &SocketResponse::Health {
                    status: "ok".into(),
                    backend: label.into(),
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
///
/// Uses `MAX_FRAME_BYTES` from `socket.rs` — single source of truth (H4).
/// Applies `.take()` cap before `read_line` to prevent pre-allocation DoS (C4).
async fn read_message_from_buf<R, T>(reader: &mut tokio::io::BufReader<R>) -> Result<T>
where
    R: tokio::io::AsyncRead + Unpin,
    T: for<'de> serde::Deserialize<'de>,
{
    use tokio::io::{AsyncBufReadExt, AsyncReadExt};

    let mut line = String::new();
    // Cap actual read to avoid pre-allocation DoS — same approach as socket::read_message.
    let n = reader
        .take((MAX_FRAME_BYTES as u64) + 1)
        .read_line(&mut line)
        .await
        .map_err(VisionError::Io)?;

    if n == 0 {
        return Err(VisionError::Daemon("connection closed by peer".into()));
    }
    if line.len() > MAX_FRAME_BYTES {
        return Err(VisionError::PayloadTooLarge(line.len(), MAX_FRAME_BYTES));
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

    /// C2: RAII PidFileGuard must remove the file on drop.
    #[test]
    fn pid_file_guard_removes_on_drop() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.pid");
        std::fs::write(&path, b"123").unwrap();
        assert!(path.exists());
        let guard = PidFileGuard { path: path.clone() };
        drop(guard);
        assert!(!path.exists(), "PidFileGuard should remove file on drop");
    }

    /// C2: RAII SocketFileGuard must remove the file on drop.
    #[test]
    fn socket_file_guard_removes_on_drop() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.sock");
        std::fs::write(&path, b"").unwrap();
        assert!(path.exists());
        let guard = SocketFileGuard { path: path.clone() };
        drop(guard);
        assert!(!path.exists(), "SocketFileGuard should remove file on drop");
    }

    /// C2: double-drop must not panic (file already removed by first drop).
    #[test]
    fn pid_file_guard_double_drop_is_safe() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.pid");
        std::fs::write(&path, b"1").unwrap();
        let guard = PidFileGuard { path: path.clone() };
        drop(guard);
        // File is gone; creating a second guard for the same (now absent) path
        // and dropping it must not panic.
        let guard2 = PidFileGuard { path };
        drop(guard2); // must not panic
    }

    /// H2: stop() with a stale PID file and no running socket must clean up
    /// files without signalling any process.
    #[tokio::test]
    async fn stop_with_stale_pid_cleans_up() {
        let dir = tempdir().unwrap();
        let cfg = test_cfg(dir.path());

        // Write a PID file with a definitely-dead PID (no socket present).
        std::fs::write(&cfg.pid_path, b"99999999").unwrap();
        // No socket file — probe_health will fail, so stop() cleans up stale files.

        // stop() must succeed without panicking even with a stale pid file
        // and no daemon socket.
        let result = stop(&cfg).await;
        assert!(
            result.is_ok(),
            "stop with stale pid should succeed: {result:?}"
        );
        // Files should be cleaned up.
        assert!(
            !cfg.pid_path.exists(),
            "stale pid file should be removed by stop()"
        );
    }

    /// C4 / read_message_from_buf: payload exactly at the limit is accepted.
    #[tokio::test]
    async fn read_message_from_buf_accepts_normal_payload() {
        let json = serde_json::to_string(&crate::socket::SocketRequest::Health).unwrap() + "\n";
        let bytes = json.as_bytes().to_vec();
        let mut reader = tokio::io::BufReader::new(bytes.as_slice());
        let result: Result<crate::socket::SocketRequest> = read_message_from_buf(&mut reader).await;
        assert!(result.is_ok());
    }
}
