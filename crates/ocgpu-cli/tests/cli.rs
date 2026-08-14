// SPDX-License-Identifier: CC0-1.0

//! Black-box command-line contract tests.

use serde_json::Value;
use std::path::PathBuf;
use std::process::{Command, Output};

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
