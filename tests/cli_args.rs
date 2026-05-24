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
    cmd().args(["--warm", "--stop"]).assert().failure().stderr(
        predicate::str::contains("--warm")
            .or(predicate::str::contains("--stop"))
            .or(predicate::str::contains("cannot be used")),
    );
}

// ── image size limit (C4) ─────────────────────────────────────────────────────

/// Files smaller than 32 MiB must not trigger the size limit error on the file
/// path that is checked before `tokio::fs::read`.  We only test the rejection
/// path (oversize file) and acceptance path (normal file) at the CLI layer.
///
/// For the oversize case we create a sparse file that reports a large size via
/// `metadata().len()` without actually consuming disk space.
#[test]
fn image_too_large_exits_1() {
    use std::os::unix::fs::FileExt;

    let dir = tempdir().expect("tempdir");
    let big_image = dir.path().join("huge.png");

    // Create a sparse file of 33 MiB (1 byte past the 32 MiB limit).
    let file = std::fs::File::create(&big_image).unwrap();
    // Write 1 byte at offset 33 MiB to create a sparse file with that reported size.
    let offset: u64 = 33 * 1024 * 1024;
    file.write_at(b"\x00", offset).unwrap();

    // IMAGE and PROMPT are positional args (not flags).
    cmd()
        .arg(&big_image)
        .arg("test prompt")
        .env("VISION_LLAMA_SERVER", "http://127.0.0.1:19999")
        .env("VISION_LLAMA_BIN", "/nonexistent/llama-cli")
        .env(
            "VISION_SOCKET_PATH",
            dir.path().join("no.sock").to_str().unwrap(),
        )
        .env(
            "VISION_PID_PATH",
            dir.path().join("no.pid").to_str().unwrap(),
        )
        .env("RUST_LOG", "off")
        .timeout(std::time::Duration::from_secs(5))
        .assert()
        // Exit code 1 = analysis/input error (ImageTooLarge maps to code 1).
        .code(1)
        .stderr(
            predicate::str::contains("image too large").or(predicate::str::contains("too large")),
        );
}

/// A small image file (a few bytes) must not be rejected by the size check.
/// The CLI will still fail (no backend) but with exit code 2, not 1.
#[test]
fn small_image_not_rejected_by_size_limit() {
    let dir = tempdir().expect("tempdir");
    let small_image = dir.path().join("small.png");
    std::fs::write(&small_image, b"\x89PNG\r\n\x1a\n").unwrap(); // PNG magic bytes

    // IMAGE and PROMPT are positional args (not flags).
    cmd()
        .arg(&small_image)
        .arg("test prompt")
        .env("VISION_LLAMA_SERVER", "http://127.0.0.1:19999")
        .env("VISION_LLAMA_BIN", "/nonexistent/llama-cli")
        .env(
            "VISION_SOCKET_PATH",
            dir.path().join("no.sock").to_str().unwrap(),
        )
        .env(
            "VISION_PID_PATH",
            dir.path().join("no.pid").to_str().unwrap(),
        )
        .env("RUST_LOG", "off")
        .timeout(std::time::Duration::from_secs(10))
        .assert()
        // Exit code 2 = backend unavailable (not rejected by image size).
        .code(2);
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
        .env(
            "VISION_SOCKET_PATH",
            dir.path().join("no.sock").to_str().unwrap(),
        )
        .env(
            "VISION_PID_PATH",
            dir.path().join("no.pid").to_str().unwrap(),
        )
        // Remove RUST_LOG noise in test output.
        .env("RUST_LOG", "off")
        .timeout(std::time::Duration::from_secs(10))
        .assert()
        .code(2);
}
