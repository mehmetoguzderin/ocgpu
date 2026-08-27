// SPDX-License-Identifier: CC0-1.0

//! Opt-in, deliberately bounded hardware validation.
//!
//! No test resets a device, changes power/display state, installs a driver, or
//! loops a workload. Execution tests allocate 64 bytes on each selected device
//! and launch exactly one thread once. Dedicated runners must select exactly
//! one mode with `OCGPU_SMOKE_BACKEND`: `cuda`, `hip`, `all` (simultaneous
//! execution), or `coexistence` (simultaneous discovery without GPU work).

#![cfg(any(feature = "cuda", feature = "hip"))]

use ocgpu::{Backend, Driver, LaunchConfig};
#[cfg(feature = "hip")]
use sha2::{Digest, Sha256};
#[cfg(all(feature = "cuda", feature = "hip"))]
use std::any::Any;
use std::ffi::CString;
#[cfg(feature = "hip")]
use std::ffi::c_void;
#[cfg(feature = "hip")]
use std::fs::File;
#[cfg(feature = "hip")]
use std::io::Read;
#[cfg(all(feature = "cuda", feature = "hip"))]
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
#[cfg(all(feature = "cuda", feature = "hip"))]
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
#[cfg(all(feature = "cuda", feature = "hip"))]
use std::thread::{self, JoinHandle};
#[cfg(all(feature = "cuda", feature = "hip"))]
use std::time::{Duration, Instant};

const BYTES: usize = 64;
#[cfg(feature = "hip")]
const MAX_HIP_FIXTURE_BYTES: usize = 8 * 1024 * 1024;
#[cfg(all(feature = "cuda", feature = "hip"))]
const COORDINATION_TIMEOUT: Duration = Duration::from_secs(5);
#[cfg(all(feature = "cuda", feature = "hip"))]
const WORKER_TIMEOUT: Duration = Duration::from_secs(30);
#[cfg(feature = "hip")]
const CLANG_OFFLOAD_BUNDLE_MAGIC: &[u8] = b"__CLANG_OFFLOAD_BUNDLE__";
#[cfg(feature = "hip")]
const ELF_MACHINE_AMDGPU: u16 = 224;
#[cfg(feature = "hip")]
const ELF64_HEADER_BYTES: usize = 64;
#[cfg(feature = "hip")]
const ELF_FLAGS_OFFSET: usize = 48;
#[cfg(feature = "hip")]
const ELF_AMDGPU_MACH_MASK: u32 = 0xff;

enum ModuleFixture {
    #[cfg(feature = "cuda")]
    Text(PathBuf),
    #[cfg(feature = "hip")]
    Hip(HipFixture),
}

#[cfg(feature = "hip")]
struct HipFixture {
    canonical_path: PathBuf,
    bytes: Vec<u8>,
}

#[cfg(all(feature = "cuda", feature = "hip"))]
enum WorkerUpdate {
    Ready(&'static str),
    Finished(&'static str, Result<(), String>),
}

fn selected(mode: &str) -> bool {
    if std::env::var("OCGPU_RUN_HARDWARE_SMOKE").as_deref() != Ok("1") {
        return false;
    }
    let actual = std::env::var("OCGPU_SMOKE_BACKEND").unwrap_or_else(|_| {
        panic!("OCGPU_SMOKE_BACKEND must be set explicitly when hardware smoke is enabled")
    });
    assert!(
        matches!(actual.as_str(), "cuda" | "hip" | "all" | "coexistence"),
        "unsupported OCGPU_SMOKE_BACKEND mode {actual:?}"
    );
    actual == mode
}

#[cfg(all(feature = "cuda", feature = "hip"))]
fn assert_dual_execution_acknowledged() {
    assert_eq!(
        std::env::var("OCGPU_ALLOW_DUAL_EXECUTION").as_deref(),
        Ok("1"),
        "all mode additionally requires OCGPU_ALLOW_DUAL_EXECUTION=1"
    );
}

fn load_driver<B: Backend>(backend: &str) -> Driver<B> {
    Driver::<B>::load().unwrap_or_else(|error| {
        panic!(
            "{backend} runtime load/initialization was rejected before any context or GPU execution: {error}"
        )
    })
}

#[cfg(feature = "cuda")]
fn load_cuda_driver() -> Driver<ocgpu::Cuda> {
    load_driver::<ocgpu::Cuda>("CUDA")
}

#[cfg(feature = "hip")]
fn load_hip_driver() -> Driver<ocgpu::Hip> {
    let driver = load_driver::<ocgpu::Hip>("HIP");
    eprintln!(
        "HIP selected the supported fail-closed runtime profile {}",
        driver.runtime_profile().as_str()
    );
    driver
}

#[cfg(feature = "hip")]
fn bounded_hip_smoke(fixture: HipFixture) {
    let driver = load_hip_driver();
    bounded_smoke("HIP", &driver, ModuleFixture::Hip(fixture));
}

fn inspect_devices<B: Backend>(driver: &Driver<B>, backend: &str) {
    assert!(driver.driver_version().expect("driver version query") > 0);
    let count = driver.device_count().expect("device enumeration");
    assert!(count > 0, "{backend} must expose at least one device");
    for device in driver.devices().expect("device iterator") {
        let device = device.expect("enumerated device");
        assert!(!device.name().expect("device name").trim().is_empty());
    }

    let first = driver.device(0).expect("first device");
    for (name, attribute) in [
        (
            "maximum threads per block",
            ocgpu::sys::OCGPU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        ),
        ("warp size", ocgpu::sys::OCGPU_DEVICE_ATTRIBUTE_WARP_SIZE),
        (
            "multiprocessor count",
            ocgpu::sys::OCGPU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        ),
    ] {
        assert!(
            first.attribute(attribute).expect(name) > 0,
            "{backend} reported a non-positive {name}"
        );
    }
}

fn bounded_smoke<B: Backend>(backend: &str, driver: &Driver<B>, fixture: ModuleFixture) {
    inspect_devices(driver, backend);
    let device = driver.device(0).expect("first device");
    let context = device.create_context(0).expect("bounded smoke context");
    // SAFETY: this thread owns `context`, which remains alive for the complete
    // lifetime of the immediately inspected non-owning view.
    let current = unsafe { driver.current_context() }
        .expect("current-context query")
        .expect("new context must be current on its creating thread");
    assert_eq!(current.raw(), context.raw());

    let memory = context.allocate(BYTES).expect("64-byte allocation");
    let source: Vec<u8> = (0_u8..u8::try_from(BYTES).expect("fixture size fits u8")).collect();
    let mut destination = vec![0_u8; BYTES];
    memory.copy_from(&source).expect("host-to-device copy");
    memory
        .copy_to(&mut destination)
        .expect("device-to-host copy");
    assert_eq!(source, destination);

    let stream = context.create_stream(0).expect("stream creation");
    let launch_start = context.create_event(0).expect("start-event creation");
    let launch_complete = context.create_event(0).expect("completion-event creation");

    let module = match fixture {
        #[cfg(feature = "cuda")]
        ModuleFixture::Text(path) => {
            let bytes = std::fs::read(path).expect("committed CUDA module fixture");
            let text = CString::new(bytes).expect("text module has no embedded NUL");
            // SAFETY: the committed fixture is a complete textual CUDA module
            // and remains readable throughout the synchronous load call.
            unsafe { context.load_module_cstr(&text) }.expect("text module load")
        }
        #[cfg(feature = "hip")]
        ModuleFixture::Hip(fixture) => {
            eprintln!(
                "using preflighted HIP module {} ({} bytes)",
                fixture.canonical_path.display(),
                fixture.bytes.len()
            );
            // SAFETY: preflight established a bounded AMDGPU ELF or structured
            // Clang bundle and the bytes remain live during the synchronous
            // load. Container checks cannot prove kernel semantics: the runner
            // operator must separately review and trust this local fixture.
            unsafe {
                context
                    .load_module_data(fixture.bytes.as_ptr().cast::<c_void>())
                    .expect("HIP module load")
            }
        }
    };
    let name = CString::new("ocgpu_noop").expect("literal contains no NUL");
    let function = module
        .function(&name)
        .expect("reviewed zero-argument smoke entry-point lookup");
    let config = LaunchConfig::new([1, 1, 1], [1, 1, 1], 0).expect("bounded launch config");

    launch_start
        .record(&stream)
        .expect("pre-launch event recording");
    // SAFETY: the runner-local fixture must be reviewed to establish the
    // zero-argument entry-point ABI. The harness deliberately makes no claim
    // about arbitrary code based only on the symbol name or container format.
    unsafe {
        function
            .launch(config, Some(&stream), &mut [])
            .expect("single-thread launch");
    }
    launch_complete
        .record(&stream)
        .expect("post-launch event recording");
    launch_complete
        .synchronize()
        .expect("post-launch event synchronization");
    stream.synchronize().expect("stream synchronization");
    context.synchronize().expect("context synchronization");
}

#[cfg(feature = "hip")]
fn read_u64(bytes: &[u8], cursor: &mut usize) -> Option<u64> {
    let end = cursor.checked_add(8)?;
    let value = u64::from_le_bytes(bytes.get(*cursor..end)?.try_into().ok()?);
    *cursor = end;
    Some(value)
}

#[cfg(feature = "hip")]
fn amdgpu_elf_architecture(bytes: &[u8]) -> Option<&'static str> {
    if bytes.len() < ELF64_HEADER_BYTES
        || !bytes.starts_with(b"\x7fELF")
        || bytes[4] != 2
        || bytes[5] != 1
        || bytes[6] != 1
        || u16::from_le_bytes([bytes[18], bytes[19]]) != ELF_MACHINE_AMDGPU
    {
        return None;
    }
    let flags = u32::from_le_bytes(
        bytes[ELF_FLAGS_OFFSET..ELF_FLAGS_OFFSET + 4]
            .try_into()
            .ok()?,
    );
    amdgpu_machine_architecture(flags & ELF_AMDGPU_MACH_MASK)
}

#[cfg(feature = "hip")]
const fn amdgpu_machine_architecture(machine: u32) -> Option<&'static str> {
    Some(match machine {
        0x020 => "gfx600",
        0x021 => "gfx601",
        0x022 => "gfx700",
        0x023 => "gfx701",
        0x024 => "gfx702",
        0x025 => "gfx703",
        0x026 => "gfx704",
        0x028 => "gfx801",
        0x029 => "gfx802",
        0x02a => "gfx803",
        0x02b => "gfx810",
        0x02c => "gfx900",
        0x02d => "gfx902",
        0x02e => "gfx904",
        0x02f => "gfx906",
        0x030 => "gfx908",
        0x031 => "gfx909",
        0x032 => "gfx90c",
        0x033 => "gfx1010",
        0x034 => "gfx1011",
        0x035 => "gfx1012",
        0x036 => "gfx1030",
        0x037 => "gfx1031",
        0x038 => "gfx1032",
        0x039 => "gfx1033",
        0x03a => "gfx602",
        0x03b => "gfx705",
        0x03c => "gfx805",
        0x03d => "gfx1035",
        0x03e => "gfx1034",
        0x03f => "gfx90a",
        0x041 => "gfx1100",
        0x042 => "gfx1013",
        0x043 => "gfx1150",
        0x044 => "gfx1103",
        0x045 => "gfx1036",
        0x046 => "gfx1101",
        0x047 => "gfx1102",
        0x048 => "gfx1200",
        0x049 => "gfx1250",
        0x04a => "gfx1151",
        0x04c => "gfx942",
        0x04e => "gfx1201",
        0x04f => "gfx950",
        0x050 => "gfx1310",
        0x051 => "gfx9-generic",
        0x052 => "gfx10-1-generic",
        0x053 => "gfx10-3-generic",
        0x054 => "gfx11-generic",
        0x055 => "gfx1152",
        0x057 => "gfx1154",
        0x058 => "gfx1153",
        0x059 => "gfx12-generic",
        0x05a => "gfx1251",
        0x05b => "gfx12-5-generic",
        0x05c => "gfx1172",
        0x05d => "gfx1170",
        0x05e => "gfx1171",
        0x05f => "gfx9-4-generic",
        0x062 => "gfx11-7-generic",
        0x063 => "gfx13-generic",
        0x0eb => "gfx1250-strict",
        _ => return None,
    })
}

#[cfg(feature = "hip")]
fn bundle_architecture(identifier: &str) -> Option<&str> {
    let target = identifier.rsplit("--").next()?;
    let end = target.find([':', '+']).unwrap_or(target.len());
    let architecture = &target[..end];
    (architecture.starts_with("gfx")
        && architecture
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-'))
    .then_some(architecture)
}

#[cfg(feature = "hip")]
fn is_amdgpu_bundle_identifier(identifier: &str) -> bool {
    identifier.contains("-amdgcn-amd-amdhsa-") || identifier.contains("-amdgpu-amd-amdhsa-")
}

#[cfg(feature = "hip")]
fn valid_clang_offload_bundle(bytes: &[u8], expected_architecture: &str) -> bool {
    if !bytes.starts_with(CLANG_OFFLOAD_BUNDLE_MAGIC) {
        return false;
    }
    let mut cursor = CLANG_OFFLOAD_BUNDLE_MAGIC.len();
    let Some(bundle_count) = read_u64(bytes, &mut cursor) else {
        return false;
    };
    if !(1..=64).contains(&bundle_count) {
        return false;
    }

    let mut payloads = Vec::with_capacity(usize::try_from(bundle_count).unwrap_or(64));
    for _ in 0..bundle_count {
        let Some(offset) = read_u64(bytes, &mut cursor) else {
            return false;
        };
        let Some(size) = read_u64(bytes, &mut cursor) else {
            return false;
        };
        let Some(triple_size) = read_u64(bytes, &mut cursor) else {
            return false;
        };
        let Ok(triple_size) = usize::try_from(triple_size) else {
            return false;
        };
        if triple_size == 0 || triple_size > 4096 {
            return false;
        }
        let Some(triple_end) = cursor.checked_add(triple_size) else {
            return false;
        };
        let Some(triple_bytes) = bytes.get(cursor..triple_end) else {
            return false;
        };
        let Ok(triple) = std::str::from_utf8(triple_bytes) else {
            return false;
        };
        if !triple.bytes().all(|byte| byte.is_ascii_graphic())
            || payloads.iter().any(|(_, _, existing)| existing == triple)
        {
            return false;
        }
        let Ok(offset) = usize::try_from(offset) else {
            return false;
        };
        let Ok(size) = usize::try_from(size) else {
            return false;
        };
        let Some(end) = offset.checked_add(size) else {
            return false;
        };
        if size == 0 || end > bytes.len() {
            return false;
        }
        payloads.push((offset, end, triple.to_owned()));
        cursor = triple_end;
    }

    if payloads.iter().any(|(offset, _, _)| *offset < cursor) {
        return false;
    }
    let mut ranges = payloads
        .iter()
        .map(|(offset, end, _)| (*offset, *end))
        .collect::<Vec<_>>();
    ranges.sort_unstable_by_key(|(offset, _)| *offset);
    if ranges.windows(2).any(|pair| pair[1].0 < pair[0].1) {
        return false;
    }
    let mut found_expected = false;
    for (offset, end, identifier) in &payloads {
        if !is_amdgpu_bundle_identifier(identifier) {
            continue;
        }
        let Some(identifier_architecture) = bundle_architecture(identifier) else {
            return false;
        };
        if amdgpu_elf_architecture(&bytes[*offset..*end]) != Some(identifier_architecture) {
            return false;
        }
        found_expected |= identifier_architecture == expected_architecture;
    }
    found_expected
}

#[cfg(feature = "hip")]
fn load_hip_fixture() -> HipFixture {
    let supplied = std::env::var_os("OCGPU_HIP_SMOKE_MODULE")
        .map(PathBuf::from)
        .expect("HIP execution requires OCGPU_HIP_SMOKE_MODULE");
    assert!(
        supplied.is_absolute(),
        "OCGPU_HIP_SMOKE_MODULE must be an absolute path"
    );
    let canonical_path = supplied
        .canonicalize()
        .expect("HIP module path must resolve to a canonical local file");
    assert!(canonical_path.is_absolute());
    let metadata = canonical_path
        .metadata()
        .expect("HIP module metadata query");
    assert!(metadata.is_file(), "HIP module must be a regular file");
    assert!(
        (1..=u64::try_from(MAX_HIP_FIXTURE_BYTES).expect("size cap fits u64"))
            .contains(&metadata.len()),
        "HIP module must be between 1 byte and {MAX_HIP_FIXTURE_BYTES} bytes"
    );

    let mut bytes = Vec::with_capacity(usize::try_from(metadata.len()).expect("bounded size"));
    File::open(&canonical_path)
        .expect("open canonical HIP module")
        .take(u64::try_from(MAX_HIP_FIXTURE_BYTES + 1).expect("size cap fits u64"))
        .read_to_end(&mut bytes)
        .expect("read bounded HIP module");
    assert_eq!(
        u64::try_from(bytes.len()).expect("fixture length fits u64"),
        metadata.len(),
        "HIP module changed during preflight"
    );
    let expected_sha256 = std::env::var("OCGPU_HIP_SMOKE_SHA256")
        .expect("HIP execution requires an explicit OCGPU_HIP_SMOKE_SHA256");
    assert!(
        expected_sha256.len() == 64 && expected_sha256.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "OCGPU_HIP_SMOKE_SHA256 must be exactly 64 hexadecimal digits"
    );
    let actual_sha256 = format!("{:X}", Sha256::digest(&bytes));
    assert_eq!(
        actual_sha256,
        expected_sha256.to_ascii_uppercase(),
        "the exact HIP module bytes read for loading do not match OCGPU_HIP_SMOKE_SHA256"
    );
    let expected_architecture = std::env::var("OCGPU_HIP_SMOKE_ARCH")
        .expect("HIP execution requires an explicit OCGPU_HIP_SMOKE_ARCH");
    assert!(
        amdgpu_elf_architecture(&bytes).is_some_and(|actual| actual == expected_architecture)
            || valid_clang_offload_bundle(&bytes, &expected_architecture),
        "HIP module must be a 64-bit little-endian AMDGPU ELF or structured Clang bundle matching OCGPU_HIP_SMOKE_ARCH={expected_architecture:?}"
    );
    HipFixture {
        canonical_path,
        bytes,
    }
}

#[cfg(all(feature = "cuda", feature = "hip"))]
fn panic_text(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "worker panicked without a string payload".to_owned()
    }
}

#[cfg(all(feature = "cuda", feature = "hip"))]
fn spawn_worker(
    name: &'static str,
    start: Receiver<()>,
    updates: Sender<WorkerUpdate>,
    operation: impl FnOnce() + Send + 'static,
) -> JoinHandle<()> {
    thread::Builder::new()
        .name(format!("ocgpu-{name}-smoke"))
        .spawn(move || {
            if updates.send(WorkerUpdate::Ready(name)).is_err() {
                return;
            }
            if start.recv_timeout(COORDINATION_TIMEOUT).is_err() {
                let _ = updates.send(WorkerUpdate::Finished(
                    name,
                    Err("coordinator did not release worker before timeout".to_owned()),
                ));
                return;
            }
            let result = catch_unwind(AssertUnwindSafe(operation))
                .map_err(|payload| panic_text(payload.as_ref()));
            let _ = updates.send(WorkerUpdate::Finished(name, result));
        })
        .expect("spawn bounded backend worker")
}

#[cfg(all(feature = "cuda", feature = "hip"))]
fn receive_before(receiver: &Receiver<WorkerUpdate>, deadline: Instant) -> WorkerUpdate {
    let remaining = deadline
        .checked_duration_since(Instant::now())
        .unwrap_or(Duration::ZERO);
    assert!(!remaining.is_zero(), "dual-backend coordination timed out");
    match receiver.recv_timeout(remaining) {
        Ok(update) => update,
        Err(RecvTimeoutError::Timeout) => panic!("dual-backend coordination timed out"),
        Err(RecvTimeoutError::Disconnected) => {
            panic!("dual-backend worker channel disconnected")
        }
    }
}

#[cfg(all(feature = "cuda", feature = "hip"))]
fn run_concurrently(
    cuda_operation: impl FnOnce() + Send + 'static,
    hip_operation: impl FnOnce() + Send + 'static,
) {
    let (updates_tx, updates_rx) = mpsc::channel();
    let (cuda_start_tx, cuda_start_rx) = mpsc::channel();
    let (hip_start_tx, hip_start_rx) = mpsc::channel();
    let cuda = spawn_worker("cuda", cuda_start_rx, updates_tx.clone(), cuda_operation);
    let hip = spawn_worker("hip", hip_start_rx, updates_tx, hip_operation);

    let ready_deadline = Instant::now() + COORDINATION_TIMEOUT;
    let mut cuda_ready = false;
    let mut hip_ready = false;
    while !(cuda_ready && hip_ready) {
        match receive_before(&updates_rx, ready_deadline) {
            WorkerUpdate::Ready("cuda") => {
                assert!(!cuda_ready, "CUDA worker reported ready twice");
                cuda_ready = true;
            }
            WorkerUpdate::Ready("hip") => {
                assert!(!hip_ready, "HIP worker reported ready twice");
                hip_ready = true;
            }
            WorkerUpdate::Ready(name) => panic!("unexpected worker {name}"),
            WorkerUpdate::Finished(name, result) => {
                panic!("{name} worker finished before release: {result:?}")
            }
        }
    }
    cuda_start_tx.send(()).expect("release CUDA worker");
    hip_start_tx.send(()).expect("release HIP worker");

    let completion_deadline = Instant::now() + WORKER_TIMEOUT;
    let mut results = Vec::with_capacity(2);
    while results.len() != 2 {
        match receive_before(&updates_rx, completion_deadline) {
            WorkerUpdate::Finished(name, result) => results.push((name, result)),
            WorkerUpdate::Ready(name) => panic!("{name} worker reported ready twice"),
        }
    }
    cuda.join().expect("CUDA worker join");
    hip.join().expect("HIP worker join");
    for (name, result) in results {
        if let Err(error) = result {
            panic!("{name} worker failed: {error}");
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_bounded_hardware_smoke() {
    if !selected("cuda") {
        eprintln!("CUDA single-backend smoke skipped; mode is not cuda");
        return;
    }
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/noop.ptx");
    let driver = load_cuda_driver();
    bounded_smoke("CUDA", &driver, ModuleFixture::Text(fixture));
}

#[cfg(feature = "hip")]
#[test]
fn hip_bounded_hardware_smoke() {
    if !selected("hip") {
        eprintln!("HIP single-backend smoke skipped; mode is not hip");
        return;
    }
    let fixture = load_hip_fixture();
    bounded_hip_smoke(fixture);
}

#[cfg(all(feature = "cuda", feature = "hip"))]
#[test]
fn cuda_and_hip_bounded_hardware_smoke_concurrently() {
    if !selected("all") {
        eprintln!("simultaneous CUDA+HIP execution skipped; mode is not all");
        return;
    }
    assert_dual_execution_acknowledged();
    let cuda_fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/noop.ptx");
    // Fail before either backend initializes if the runner-local AMD fixture
    // does not satisfy the bounded container and architecture preflight.
    let hip_fixture = load_hip_fixture();
    run_concurrently(
        move || {
            let driver = load_cuda_driver();
            bounded_smoke("CUDA", &driver, ModuleFixture::Text(cuda_fixture));
        },
        move || bounded_hip_smoke(hip_fixture),
    );
}

#[cfg(all(feature = "cuda", feature = "hip"))]
#[test]
fn cuda_and_hip_runtime_coexistence_concurrently() {
    if !selected("coexistence") {
        eprintln!("CUDA+HIP coexistence discovery skipped; mode is not coexistence");
        return;
    }
    run_concurrently(
        || {
            let driver = load_cuda_driver();
            inspect_devices(&driver, "CUDA");
        },
        || {
            let driver = load_hip_driver();
            inspect_devices(&driver, "HIP");
        },
    );
}

#[cfg(all(test, feature = "hip"))]
mod preflight_tests {
    use super::*;

    fn amdgpu_elf(machine: u8) -> Vec<u8> {
        let mut elf = vec![0_u8; ELF64_HEADER_BYTES];
        elf[..4].copy_from_slice(b"\x7fELF");
        elf[4] = 2;
        elf[5] = 1;
        elf[6] = 1;
        elf[18..20].copy_from_slice(&ELF_MACHINE_AMDGPU.to_le_bytes());
        elf[ELF_FLAGS_OFFSET..ELF_FLAGS_OFFSET + 4]
            .copy_from_slice(&u32::from(machine).to_le_bytes());
        elf
    }

    #[test]
    fn amdgpu_elf_preflight_requires_concrete_64_bit_little_endian_isa() {
        assert_eq!(amdgpu_elf_architecture(&amdgpu_elf(0x032)), Some("gfx90c"));
        assert_eq!(amdgpu_elf_architecture(&amdgpu_elf(0)), None);

        let mut wrong_machine = amdgpu_elf(0x032);
        wrong_machine[18..20].copy_from_slice(&0_u16.to_le_bytes());
        assert_eq!(amdgpu_elf_architecture(&wrong_machine), None);
    }

    #[test]
    fn clang_bundle_preflight_checks_header_target_and_payload_bounds() {
        let payload = amdgpu_elf(0x032);
        for triple in [
            b"hipv4-amdgcn-amd-amdhsa--gfx90c".as_slice(),
            b"hipv4-amdgpu-amd-amdhsa--gfx90c".as_slice(),
        ] {
            let payload_offset = CLANG_OFFLOAD_BUNDLE_MAGIC.len() + 8 + 24 + triple.len();
            let mut bundle = Vec::new();
            bundle.extend_from_slice(CLANG_OFFLOAD_BUNDLE_MAGIC);
            bundle.extend_from_slice(&1_u64.to_le_bytes());
            bundle.extend_from_slice(
                &u64::try_from(payload_offset)
                    .expect("offset fits")
                    .to_le_bytes(),
            );
            bundle.extend_from_slice(
                &u64::try_from(payload.len())
                    .expect("size fits")
                    .to_le_bytes(),
            );
            bundle.extend_from_slice(
                &u64::try_from(triple.len())
                    .expect("triple length fits")
                    .to_le_bytes(),
            );
            bundle.extend_from_slice(triple);
            bundle.extend_from_slice(&payload);
            assert!(valid_clang_offload_bundle(&bundle, "gfx90c"));
            assert!(!valid_clang_offload_bundle(&bundle, "gfx1100"));

            bundle.truncate(bundle.len() - 1);
            assert!(!valid_clang_offload_bundle(&bundle, "gfx90c"));
        }
    }
}
