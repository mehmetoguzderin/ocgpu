// SPDX-License-Identifier: CC0-1.0

//! Reproducible generation, validation, test, and release-readiness orchestration.

use std::env;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

const HARDWARE_SMOKE_PROCESS_TIMEOUT: Duration = Duration::from_secs(45);
const HARDWARE_SMOKE_REAP_GRACE: Duration = Duration::from_secs(2);

fn main() {
    if let Err(error) = run() {
        eprintln!("xtask: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let root = workspace_root()?;
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    match arguments.as_slice() {
        [command] if command == "generate" => generate(&root),
        [command] if command == "check" => check(&root),
        [command] if command == "test" => test(&root),
        [command] if command == "ci" => ci(&root),
        [command] if command == "c99" => c99(&root),
        [command] if command == "licenses" => dependency_manifest(&root, false),
        [command] if command == "hardware-smoke" => hardware_smoke(&root),
        _ => Err(usage().into()),
    }
}

fn generate(root: &Path) -> Result<(), Box<dyn Error>> {
    cargo(root, &["run", "-p", "ocgpu-oracle", "--", "vendor-union"])?;
    cargo(root, &["run", "-p", "ocgpu-codegen", "--", "generate"])?;
    cargo(root, &["run", "-p", "ocgpu-oracle", "--", "semantics"])?;
    cargo(root, &["run", "-p", "ocgpu-oracle", "--", "classify"])?;
    cargo(root, &["run", "-p", "ocgpu-oracle", "--", "report"])?;
    dependency_manifest(root, false)
}

fn check(root: &Path) -> Result<(), Box<dyn Error>> {
    cargo(
        root,
        &["run", "-p", "ocgpu-oracle", "--", "vendor-union", "--check"],
    )?;
    cargo(root, &["run", "-p", "ocgpu-codegen", "--", "check"])?;
    cargo(
        root,
        &["run", "-p", "ocgpu-oracle", "--", "semantics", "--check"],
    )?;
    cargo(
        root,
        &["run", "-p", "ocgpu-oracle", "--", "classify", "--check"],
    )?;
    dependency_manifest(root, true)?;
    cargo(root, &["run", "-p", "ocgpu-oracle", "--", "check"])
}

fn test(root: &Path) -> Result<(), Box<dyn Error>> {
    check(root)?;
    cargo(root, &["test", "--workspace", "--all-features"])
}

fn ci(root: &Path) -> Result<(), Box<dyn Error>> {
    check(root)?;
    feature_matrix(root)?;
    cargo(root, &["fmt", "--all", "--check"])?;
    cargo(
        root,
        &[
            "clippy",
            "--workspace",
            "--all-targets",
            "--all-features",
            "--",
            "-D",
            "warnings",
        ],
    )?;
    cargo(root, &["test", "--workspace", "--all-features"])?;
    cargo(
        root,
        &["build", "--workspace", "--all-features", "--release"],
    )?;
    cargo(
        root,
        &[
            "build",
            "-p",
            "ocgpu-capi",
            "--release",
            "--no-default-features",
            "--features",
            "flat-c-exports",
        ],
    )?;
    cargo_with_env(
        root,
        &[
            "doc",
            "--workspace",
            "--exclude",
            "ocgpu-cli",
            "--all-features",
            "--no-deps",
        ],
        &[
            ("RUSTDOCFLAGS", "-D warnings"),
            ("CARGO_TARGET_DIR", "target/rustdoc-workspace"),
        ],
    )?;
    cargo_with_env(
        root,
        &["doc", "-p", "ocgpu-cli", "--all-features", "--no-deps"],
        &[
            ("RUSTDOCFLAGS", "-D warnings"),
            ("CARGO_TARGET_DIR", "target/rustdoc-cli"),
        ],
    )
}

fn feature_matrix(root: &Path) -> Result<(), Box<dyn Error>> {
    for features in [
        "cuda",
        "raw-cuda",
        "cuda,explicit-library-path",
        "hip",
        "raw-hip",
        "hip,explicit-library-path",
        "explicit-library-path",
    ] {
        cargo(
            root,
            &[
                "check",
                "-p",
                "ocgpu",
                "--no-default-features",
                "--features",
                features,
            ],
        )?;
    }
    for features in [
        "flat-c-exports",
        "cuda,flat-c-exports",
        "hip,flat-c-exports",
    ] {
        cargo(
            root,
            &[
                "check",
                "-p",
                "ocgpu-capi",
                "--no-default-features",
                "--features",
                features,
            ],
        )?;
    }
    Ok(())
}

fn c99(root: &Path) -> Result<(), Box<dyn Error>> {
    let include = root.join("include");
    let compiler = env::var_os("CC").unwrap_or_else(|| "cc".into());
    for (source_name, source) in [
        ("header_compile", root.join("tests/c99/header_compile.c")),
        ("enumerate", root.join("tests/c99/enumerate.c")),
        (
            "generated_layout",
            root.join("tests/abi/generated_layout.c"),
        ),
    ] {
        let status = compiler_command(&compiler)
            .current_dir(root)
            .args([
                "-std=c99",
                "-pedantic-errors",
                "-Wall",
                "-Wextra",
                "-Werror",
            ])
            .arg("-I")
            .arg(&include)
            .arg("-c")
            .arg(source)
            .arg("-o")
            .arg(root.join(format!("target/c99-{source_name}.o")))
            .stdin(Stdio::null())
            .status()?;
        require_success(
            &format!("{} C99 {source_name} check", Path::new(&compiler).display()),
            status,
        )?;
    }
    for (source_name, source) in [
        (
            "header_compile_flat",
            root.join("tests/c99/header_compile.c"),
        ),
        (
            "generated_layout_flat",
            root.join("tests/abi/generated_layout.c"),
        ),
    ] {
        let status = compiler_command(&compiler)
            .current_dir(root)
            .args([
                "-std=c99",
                "-pedantic-errors",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-DOCGPU_ENABLE_FLAT_C_EXPORTS",
                "-DOCGPU_ENABLE_CUDA",
                "-DOCGPU_ENABLE_HIP",
            ])
            .arg("-I")
            .arg(&include)
            .arg("-c")
            .arg(source)
            .arg("-o")
            .arg(root.join(format!("target/c99-{source_name}.o")))
            .stdin(Stdio::null())
            .status()?;
        require_success(
            &format!(
                "{} flat C99 {source_name} check",
                Path::new(&compiler).display()
            ),
            status,
        )?;
    }
    let cxx = env::var_os("CXX").unwrap_or_else(|| default_cxx(&compiler));
    let mut cxx_command = compiler_command(&cxx);
    cxx_command.current_dir(root).args([
        "-std=c++17",
        "-pedantic-errors",
        "-Wall",
        "-Wextra",
        "-Werror",
    ]);
    if is_clang_compiler(&cxx) {
        cxx_command.arg("-Wreserved-identifier");
    }
    let status = cxx_command
        .arg("-I")
        .arg(&include)
        .arg("-c")
        .arg(root.join("tests/c99/header_cpp_compile.cpp"))
        .arg("-o")
        .arg(root.join("target/header-cpp.o"))
        .stdin(Stdio::null())
        .status()?;
    require_success(
        &format!("{} C++ header check", Path::new(&cxx).display()),
        status,
    )?;
    Ok(())
}

fn compiler_command(compiler: &std::ffi::OsStr) -> Command {
    let mut command = Command::new(compiler);
    if cfg!(windows) && is_clang_compiler(compiler) {
        command.arg("--target=x86_64-pc-windows-msvc");
    }
    command
}

fn is_clang_compiler(compiler: &std::ffi::OsStr) -> bool {
    Path::new(compiler)
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.to_ascii_lowercase().contains("clang"))
}

fn default_cxx(compiler: &std::ffi::OsStr) -> std::ffi::OsString {
    if is_clang_compiler(compiler) {
        let compiler_path = Path::new(compiler);
        if let Some(parent) = compiler_path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            return parent
                .join(if cfg!(windows) {
                    "clang++.exe"
                } else {
                    "clang++"
                })
                .into_os_string();
        }
        return "clang++".into();
    }
    "c++".into()
}

fn hardware_smoke(root: &Path) -> Result<(), Box<dyn Error>> {
    if env::var("OCGPU_RUN_HARDWARE_SMOKE").as_deref() != Ok("1") {
        return Err(
            "hardware smoke is opt-in; set OCGPU_RUN_HARDWARE_SMOKE=1 on a labelled GPU runner"
                .into(),
        );
    }
    let mode = env::var("OCGPU_SMOKE_BACKEND")
        .map_err(|_| "hardware smoke requires an explicit OCGPU_SMOKE_BACKEND mode")?;
    if !matches!(mode.as_str(), "cuda" | "hip" | "all" | "coexistence") {
        return Err(format!("unsupported OCGPU_SMOKE_BACKEND mode {mode:?}").into());
    }
    if mode == "all" && env::var("OCGPU_ALLOW_DUAL_EXECUTION").as_deref() != Ok("1") {
        return Err(
            "all mode additionally requires OCGPU_ALLOW_DUAL_EXECUTION=1 on a reviewed dual-GPU runner"
                .into(),
        );
    }
    if matches!(mode.as_str(), "hip" | "all") {
        let module = env::var_os("OCGPU_HIP_SMOKE_MODULE")
            .map(PathBuf::from)
            .ok_or("HIP execution requires OCGPU_HIP_SMOKE_MODULE")?;
        if !module.is_absolute() {
            return Err("OCGPU_HIP_SMOKE_MODULE must be an absolute path".into());
        }
        if !env::var("OCGPU_HIP_SMOKE_ARCH").is_ok_and(|value| !value.trim().is_empty()) {
            return Err("HIP execution requires a nonempty OCGPU_HIP_SMOKE_ARCH".into());
        }
    }

    // Compilation is deliberately outside the runtime watchdog: it performs
    // no device operation and can legitimately take longer on a cold runner.
    cargo(
        root,
        &[
            "test",
            "-p",
            "ocgpu",
            "--test",
            "hardware_smoke",
            "--all-features",
            "--no-run",
        ],
    )?;
    let executable = hardware_smoke_executable(root)?;
    run_hardware_smoke_child(root, &executable)
}

fn hardware_smoke_executable(root: &Path) -> Result<PathBuf, Box<dyn Error>> {
    let output = Command::new("cargo")
        .current_dir(root)
        .args([
            "test",
            "-p",
            "ocgpu",
            "--test",
            "hardware_smoke",
            "--all-features",
            "--no-run",
            "--message-format=json",
        ])
        .stdin(Stdio::null())
        .stderr(Stdio::inherit())
        .output()?;
    require_success("locate compiled hardware-smoke test", output.status)?;
    let messages = String::from_utf8(output.stdout)?;
    messages
        .lines()
        .filter(|line| {
            line.contains("\"name\":\"hardware_smoke\"") && line.contains("\"kind\":[\"test\"]")
        })
        .filter_map(|line| json_string_field(line, "executable").transpose())
        .next_back()
        .transpose()?
        .map(PathBuf::from)
        .ok_or_else(|| "cargo did not report the compiled hardware-smoke executable".into())
}

fn json_string_field(line: &str, field: &str) -> Result<Option<String>, Box<dyn Error>> {
    let needle = format!("\"{field}\":");
    let Some(value) = line
        .split_once(&needle)
        .map(|(_, value)| value.trim_start())
    else {
        return Ok(None);
    };
    if value.starts_with("null") {
        return Ok(None);
    }
    decode_json_string(value).map(Some).map_err(Into::into)
}

fn decode_json_string(value: &str) -> Result<String, &'static str> {
    let bytes = value.as_bytes();
    if bytes.first() != Some(&b'\"') {
        return Err("cargo executable field is not a JSON string");
    }
    let mut decoded = Vec::new();
    let mut index = 1;
    while index < bytes.len() {
        match bytes[index] {
            b'\"' => return String::from_utf8(decoded).map_err(|_| "invalid UTF-8 in JSON string"),
            b'\\' => {
                index += 1;
                let escaped = *bytes.get(index).ok_or("truncated JSON escape")?;
                match escaped {
                    b'\"' | b'\\' | b'/' => decoded.push(escaped),
                    b'b' => decoded.push(8),
                    b'f' => decoded.push(12),
                    b'n' => decoded.push(b'\n'),
                    b'r' => decoded.push(b'\r'),
                    b't' => decoded.push(b'\t'),
                    b'u' => {
                        let end = index.checked_add(5).ok_or("JSON escape overflow")?;
                        let digits = value
                            .get(index + 1..end)
                            .ok_or("truncated Unicode escape")?;
                        let scalar = u32::from_str_radix(digits, 16)
                            .map_err(|_| "invalid Unicode escape")?;
                        let character =
                            char::from_u32(scalar).ok_or("unsupported surrogate Unicode escape")?;
                        let mut buffer = [0_u8; 4];
                        decoded.extend_from_slice(character.encode_utf8(&mut buffer).as_bytes());
                        index = end - 1;
                    }
                    _ => return Err("invalid JSON escape"),
                }
            }
            byte if byte < 0x20 => return Err("unescaped JSON control character"),
            byte => decoded.push(byte),
        }
        index += 1;
    }
    Err("unterminated JSON string")
}

fn run_hardware_smoke_child(root: &Path, executable: &Path) -> Result<(), Box<dyn Error>> {
    let mut child = Command::new(executable)
        .current_dir(root)
        .args(["--nocapture", "--test-threads=1"])
        .stdin(Stdio::null())
        .spawn()?;
    let deadline = Instant::now() + HARDWARE_SMOKE_PROCESS_TIMEOUT;
    loop {
        if let Some(status) = child.try_wait()? {
            return require_success("bounded hardware-smoke child", status);
        }
        if Instant::now() >= deadline {
            let process_id = child.id();
            child.kill().map_err(|error| {
                format!(
                    "hardware-smoke child {process_id} exceeded {} seconds and could not be terminated: {error}",
                    HARDWARE_SMOKE_PROCESS_TIMEOUT.as_secs()
                )
            })?;
            let reap_deadline = Instant::now() + HARDWARE_SMOKE_REAP_GRACE;
            loop {
                match child.try_wait() {
                    Ok(Some(_)) => break,
                    Ok(None) if Instant::now() < reap_deadline => {
                        thread::sleep(Duration::from_millis(25));
                    }
                    Ok(None) => {
                        return Err(format!(
                            "hardware-smoke child {process_id} exceeded the {}-second process watchdog; termination was requested but the child was not reaped within {} seconds",
                            HARDWARE_SMOKE_PROCESS_TIMEOUT.as_secs(),
                            HARDWARE_SMOKE_REAP_GRACE.as_secs()
                        )
                        .into());
                    }
                    Err(error) => {
                        return Err(format!(
                            "hardware-smoke child {process_id} exceeded the {}-second process watchdog and post-termination status failed: {error}",
                            HARDWARE_SMOKE_PROCESS_TIMEOUT.as_secs()
                        )
                        .into());
                    }
                }
            }
            return Err(format!(
                "hardware-smoke child {process_id} exceeded the {}-second process watchdog and was terminated",
                HARDWARE_SMOKE_PROCESS_TIMEOUT.as_secs()
            )
            .into());
        }
        thread::sleep(Duration::from_millis(100));
    }
}

fn dependency_manifest(root: &Path, check: bool) -> Result<(), Box<dyn Error>> {
    let shell = if cfg!(windows) { "powershell" } else { "pwsh" };
    let mut command = Command::new(shell);
    command
        .current_dir(root)
        .args(["-NoProfile", "-File"])
        .arg(root.join("xtask/dependency-manifest.ps1"));
    if check {
        command.arg("-Check");
    }
    let status = command.stdin(Stdio::null()).status()?;
    require_success("deterministic dependency manifest", status)
}

fn cargo(root: &Path, arguments: &[&str]) -> Result<(), Box<dyn Error>> {
    cargo_with_env(root, arguments, &[])
}

fn cargo_with_env(
    root: &Path,
    arguments: &[&str],
    environment: &[(&str, &str)],
) -> Result<(), Box<dyn Error>> {
    let mut command = Command::new("cargo");
    for (name, value) in environment {
        command.env(name, value);
    }
    let status = command
        .current_dir(root)
        .args(arguments)
        .stdin(Stdio::null())
        .status()?;
    require_success(&format!("cargo {}", arguments.join(" ")), status)
}

fn require_success(description: &str, status: ExitStatus) -> Result<(), Box<dyn Error>> {
    if status.success() {
        Ok(())
    } else {
        Err(format!("{description} failed with {status}").into())
    }
}

fn workspace_root() -> Result<PathBuf, Box<dyn Error>> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| "xtask manifest directory has no workspace parent".into())
}

fn usage() -> &'static str {
    "usage: cargo run -p xtask -- <generate|check|test|ci|c99|licenses|hardware-smoke>"
}

#[cfg(test)]
mod tests {
    use super::{decode_json_string, json_string_field};

    #[test]
    fn cargo_json_path_decoder_handles_windows_escaping() {
        assert_eq!(
            decode_json_string(r#""C:\\work\\hardware_smoke.exe","#).expect("JSON path"),
            r"C:\work\hardware_smoke.exe"
        );
    }

    #[test]
    fn cargo_json_executable_field_handles_null_and_unicode() {
        assert_eq!(
            json_string_field(r#"{"executable":"C:\\work\\\u0073moke.exe"}"#, "executable",)
                .expect("field"),
            Some(r"C:\work\smoke.exe".to_owned())
        );
        assert_eq!(
            json_string_field(r#"{"executable":null}"#, "executable").expect("null field"),
            None
        );
    }
}
