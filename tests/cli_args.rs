/// Integration tests for CLI argument parsing and basic flag behaviour.
///
/// These tests launch the real `vision-analyze` binary via `assert_cmd` and check
/// exit codes / stdout / stderr without touching any llama backend.
use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::tempdir;

// ── helpers ───────────────────────────────────────────────────────────────────

fn cmd() -> Command {
    Command::cargo_bin("vision-analyze").expect("binary not found")
}

// ── --version ─────────────────────────────────────────────────────────────────

#[test]
fn version_flag_prints_version_and_exits_zero() {
    cmd()
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains("vision-analyze"));
}

// ── --help ────────────────────────────────────────────────────────────────────

#[test]
fn help_flag_mentions_preset() {
    cmd()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("--preset"));
}

#[test]
fn help_flag_mentions_warm() {
    cmd()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("--warm"));
}

#[test]
fn help_flag_mentions_backend() {
    cmd()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("--backend"));
}

// ── conflicting flags ─────────────────────────────────────────────────────────

/// `--warm` and `--stop` are declared as `conflicts_with_all` in clap, so
/// passing both should produce a non-zero exit and an error message on stderr.
#[test]
fn warm_and_stop_flags_conflict_exits_nonzero() {
    cmd()
        .args(["--warm", "--stop"])
        .assert()
        .failure()
        .stderr(predicate::str::is_empty().not());
}

/// Clap conflict message typically includes the flag names.
#[test]
fn warm_and_stop_conflict_message_mentions_argument() {
    cmd()
        .args(["--warm", "--stop"])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("--warm")
                .or(predicate::str::contains("--stop"))
                .or(predicate::str::contains("cannot be used")),
        );
}

// ── --health with no backends available → exit 2 ─────────────────────────────

/// Point VISION_LLAMA_SERVER at a port that is not listening, set
/// VISION_LLAMA_BIN to a non-existent path so the CLI backend falls through,
/// and use a tempdir for the socket path so no warm daemon is found.
/// The expected outcome is exit code 2 (BackendUnavailable / ModelNotFound).
#[test]
fn health_with_no_backends_exits_2() {
    let dir = tempdir().expect("tempdir");

    cmd()
        .arg("--health")
        .env("VISION_LLAMA_SERVER", "http://127.0.0.1:19999")
        .env("VISION_LLAMA_BIN", "/nonexistent/llama-cli")
        .env("VISION_SOCKET_PATH", dir.path().join("no.sock").to_str().unwrap())
        .env("VISION_PID_PATH", dir.path().join("no.pid").to_str().unwrap())
        // Remove RUST_LOG noise in test output.
        .env("RUST_LOG", "off")
        .timeout(std::time::Duration::from_secs(10))
        .assert()
        .code(2);
}
