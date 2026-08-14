// SPDX-License-Identifier: CC0-1.0

//! Opt-in bounded hardware validation.
//!
//! These tests never reset a device, alter power/display state, install a
//! driver, or run a stress workload. They allocate 64 bytes and launch one
//! single-thread no-op kernel. Dedicated hardware runners set
//! `OCGPU_RUN_HARDWARE_SMOKE=1`; on other hosts each test records an explicit
//! capability skip without being disabled in the Rust test harness.

#![cfg(any(feature = "cuda", feature = "hip"))]

use ocgpu::{Backend, Driver, LaunchConfig};
use std::ffi::{CString, c_void};
use std::path::{Path, PathBuf};

const BYTES: usize = 64;

#[derive(Clone, Copy)]
enum FixtureKind {
    Text,
    Binary,
}

fn enabled_for(backend: &str) -> bool {
    let enabled = std::env::var_os("OCGPU_RUN_HARDWARE_SMOKE").is_some_and(|value| value == "1");
    let selected = std::env::var("OCGPU_SMOKE_BACKEND").unwrap_or_else(|_| "all".to_owned());
    enabled && (selected == "all" || selected == backend)
}

fn smoke<B: Backend>(backend: &str, fixture: &Path, fixture_kind: FixtureKind) {
    if !enabled_for(backend) {
        eprintln!(
            "{backend} hardware smoke skipped; dedicated runner capability is absent or another backend was selected"
        );
        return;
    }

    let driver = Driver::<B>::load().expect("backend must load on its dedicated runner");
    assert!(driver.driver_version().expect("driver version query") > 0);
    assert!(driver.device_count().expect("device enumeration") > 0);
    let device = driver.device(0).expect("first device");
    assert!(!device.name().expect("device name").is_empty());

    let context = device.create_context(0).expect("bounded smoke context");
    let memory = context.allocate(BYTES).expect("64-byte allocation");
    let source: Vec<u8> = (0_u8..u8::try_from(BYTES).expect("fixture size fits u8")).collect();
    let mut destination = vec![0_u8; BYTES];
    memory.copy_from(&source).expect("host-to-device copy");
    memory
        .copy_to(&mut destination)
        .expect("device-to-host copy");
    assert_eq!(source, destination);

    let stream = context.create_stream(0).expect("stream creation");
    let event = context.create_event(0).expect("event creation");
    event.record(&stream).expect("event recording");
    event.synchronize().expect("event synchronization");

    let bytes = std::fs::read(fixture).expect("committed module fixture");
    let module = match fixture_kind {
        FixtureKind::Text => {
            let text = CString::new(bytes).expect("text module has no embedded NUL");
            // SAFETY: the committed fixture is a complete textual module for
            // this backend and remains readable throughout the synchronous
            // load call.
            unsafe { context.load_module_cstr(&text) }.expect("text module load")
        }
        FixtureKind::Binary => {
            assert!(!bytes.is_empty(), "binary fixture must not be empty");
            // SAFETY: the committed binary is a complete backend code object,
            // remains live throughout the synchronous load call, and its format
            // carries the bounds consumed by the vendor module loader.
            unsafe {
                context
                    .load_module_data(bytes.as_ptr().cast::<c_void>())
                    .expect("binary module load")
            }
        }
    };
    let name = CString::new("ocgpu_noop").expect("literal contains no NUL");
    let function = module.function(&name).expect("no-op function lookup");
    let config = LaunchConfig::new([1, 1, 1], [1, 1, 1], 0).expect("bounded launch config");
    // SAFETY: `ocgpu_noop` has no parameters and the one-thread launch matches
    // the committed fixture's entry-point ABI.
    unsafe {
        function
            .launch(config, Some(&stream), &mut [])
            .expect("single-thread no-op launch");
    }
    stream.synchronize().expect("stream synchronization");
    context.synchronize().expect("context synchronization");
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_bounded_hardware_smoke() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/noop.ptx");
    smoke::<ocgpu::Cuda>("cuda", &fixture, FixtureKind::Text);
}

#[cfg(feature = "hip")]
#[test]
fn hip_bounded_hardware_smoke() {
    if !enabled_for("hip") {
        eprintln!("HIP hardware smoke skipped; dedicated runner capability is absent");
        return;
    }
    let fixture = std::env::var_os("OCGPU_HIP_SMOKE_MODULE")
        .map(PathBuf::from)
        .expect("dedicated AMD runner must provide OCGPU_HIP_SMOKE_MODULE");
    smoke::<ocgpu::Hip>("hip", &fixture, FixtureKind::Binary);
}
