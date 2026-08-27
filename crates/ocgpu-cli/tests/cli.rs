// SPDX-License-Identifier: CC0-1.0

//! Black-box command-line contract tests.

use serde_json::Value;
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

fn run(arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ocgpu"))
        .args(arguments)
        .output()
        .expect("CLI process starts")
}

fn json_stdout(output: &Output) -> Value {
    assert!(
        output.status.success(),
        "CLI failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("stdout is JSON")
}

struct TemporaryModule(PathBuf);

impl TemporaryModule {
    fn create(label: &str, bytes: &[u8]) -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock is after the Unix epoch")
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("ocgpu-cli-{}-{nonce}-{label}", std::process::id()));
        fs::write(&path, bytes).expect("synthetic module is written");
        Self(path)
    }
}

impl Drop for TemporaryModule {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn synthetic_clang_bundle(entry_id: &str) -> Vec<u8> {
    const MAGIC: &[u8] = b"__CLANG_OFFLOAD_BUNDLE__";
    let payload = b"synthetic";
    let payload_offset = MAGIC.len() + 8 + 24 + entry_id.len();
    let mut bytes = MAGIC.to_vec();
    bytes.extend_from_slice(&1_u64.to_le_bytes());
    bytes.extend_from_slice(&(payload_offset as u64).to_le_bytes());
    bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&(entry_id.len() as u64).to_le_bytes());
    bytes.extend_from_slice(entry_id.as_bytes());
    bytes.extend_from_slice(payload);
    bytes
}

#[test]
fn help_and_version_are_stable() {
    let help = run(&["help"]);
    assert!(help.status.success());
    assert!(String::from_utf8_lossy(&help.stdout).contains("ocgpu doctor"));

    let version = run(&["version"]);
    assert!(version.status.success());
    assert_eq!(
        String::from_utf8_lossy(&version.stdout).trim(),
        concat!("ocgpu ", env!("CARGO_PKG_VERSION"))
    );
}

#[test]
fn abi_and_embedded_coverage_are_machine_readable() {
    let abi = json_stdout(&run(&["abi", "--json"]));
    assert_eq!(abi["abi_version"], 0x0001_0000_u32);
    assert_eq!(abi["pointer_width"], usize::BITS);

    let coverage = json_stdout(&run(&["coverage", "--json"]));
    assert_eq!(coverage["schema_version"], 1);
    assert!(coverage["metrics"].is_array());
}

#[test]
fn module_inspection_recognizes_the_committed_ptx_fixture() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/noop.ptx")
        .canonicalize()
        .expect("PTX fixture exists");
    let output = Command::new(env!("CARGO_BIN_EXE_ocgpu"))
        .arg("module")
        .arg("inspect")
        .arg(fixture)
        .arg("--json")
        .output()
        .expect("CLI process starts");
    let report = json_stdout(&output);
    assert_eq!(report["format"], "ptx");
    assert!(report["size_bytes"].as_u64().is_some_and(|size| size > 0));
}

#[test]
fn module_inspection_renders_amdgpu_bundle_targets() {
    let fixture = TemporaryModule::create(
        "amdgpu-bundle.bin",
        &synthetic_clang_bundle("hipv4-amdgcn-amd-amdhsa--gfx90c:xnack-"),
    );
    let json = Command::new(env!("CARGO_BIN_EXE_ocgpu"))
        .arg("module")
        .arg("inspect")
        .arg(&fixture.0)
        .arg("--json")
        .output()
        .expect("CLI process starts");
    let report = json_stdout(&json);
    assert_eq!(report["format"], "hip_fat_binary");
    assert_eq!(report["amdgpu_target_ids"][0], "gfx90c:xnack-");
    assert_eq!(report["amdgpu_architectures"][0], "gfx90c");

    let text = Command::new(env!("CARGO_BIN_EXE_ocgpu"))
        .arg("module")
        .arg("inspect")
        .arg(&fixture.0)
        .output()
        .expect("CLI process starts");
    assert!(text.status.success());
    let stdout = String::from_utf8_lossy(&text.stdout);
    assert!(stdout.contains("AMDGPU target ID: gfx90c:xnack-"));
    assert!(stdout.contains("AMDGPU architecture: gfx90c"));
}

#[test]
fn module_inspection_rejects_a_truncated_bundle_header() {
    let fixture = TemporaryModule::create("truncated-bundle.bin", b"__CLANG_OFFLOAD_BUNDLE__");
    let output = Command::new(env!("CARGO_BIN_EXE_ocgpu"))
        .arg("module")
        .arg("inspect")
        .arg(&fixture.0)
        .output()
        .expect("CLI process starts");
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("truncated header while reading bundle entry count")
    );
}

#[test]
fn backend_and_doctor_reports_remain_valid_without_an_sdk() {
    let backends = json_stdout(&run(&["backends", "--json"]));
    assert_eq!(backends.as_array().map(Vec::len), Some(2));

    let doctor = json_stdout(&run(&["doctor", "--json"]));
    assert_eq!(doctor["abi_version"], 0x0001_0000_u32);
    assert_eq!(doctor["inventory_schema"], 1);
}

#[test]
fn invalid_arguments_have_a_usage_exit_code() {
    let output = run(&["unknown-command"]);
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("Try `ocgpu help`"));
}
