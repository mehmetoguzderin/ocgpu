// SPDX-License-Identifier: CC0-1.0

//! Reproducible generation, validation, test, and release-readiness orchestration.

use std::env;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};

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
    cargo(
        root,
        &[
            "test",
            "-p",
            "ocgpu",
            "--test",
            "hardware_smoke",
            "--all-features",
            "--",
            "--nocapture",
        ],
    )
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
