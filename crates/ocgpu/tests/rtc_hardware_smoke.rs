// SPDX-License-Identifier: CC0-1.0

//! Opt-in, watchdog-bounded runtime-compilation hardware validation.
//!
//! This test is skipped unless `OCGPU_RUN_RTC_HARDWARE_SMOKE=1`. A selected
//! backend compiles one fixed, headerless, zero-argument no-op kernel inside
//! the test process and launches exactly one thread once. Setting
//! `OCGPU_RTC_COMPILE_ONLY=1` performs a context-free compiler staging run. It
//! never resets a device, changes power/display state, invokes an external
//! compiler, or uses a backend other than CUDA/HIP through ocgpu.

#![cfg(all(feature = "rtc", any(feature = "nvrtc", feature = "hiprtc")))]

use ocgpu::rtc::{Compiler, Error as RtcError, RtcBackend};
use ocgpu::{Backend, Driver, LaunchConfig};
use std::ffi::{CStr, CString, c_void};
use std::path::PathBuf;

#[cfg(all(feature = "nvrtc", feature = "hiprtc"))]
use std::any::Any;
#[cfg(all(feature = "nvrtc", feature = "hiprtc"))]
use std::panic::{AssertUnwindSafe, catch_unwind};
#[cfg(all(feature = "nvrtc", feature = "hiprtc"))]
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
#[cfg(all(feature = "nvrtc", feature = "hiprtc"))]
use std::thread::{self, JoinHandle};
#[cfg(all(feature = "nvrtc", feature = "hiprtc"))]
use std::time::{Duration, Instant};

const COPY_BYTES: usize = 64;
const SOURCE: &CStr = c"extern \"C\" __global__ void ocgpu_noop() {}";
const INVALID_SOURCE: &CStr = c"extern \"C\" __global__ void ocgpu_invalid( {";
const PROGRAM_NAME: &CStr = c"ocgpu_noop.cpp";
const INVALID_PROGRAM_NAME: &CStr = c"ocgpu_invalid.cpp";
const NAME_EXPRESSION: &CStr = c"&ocgpu_noop";

#[cfg(all(feature = "nvrtc", feature = "hiprtc"))]
const COORDINATION_TIMEOUT: Duration = Duration::from_secs(5);
#[cfg(all(feature = "nvrtc", feature = "hiprtc"))]
const WORKER_TIMEOUT: Duration = Duration::from_secs(30);

struct CompiledModule {
    code: Vec<u8>,
    lowered_name: CString,
}

fn required_architecture(variable: &str, vendor: &str) -> String {
    let architecture = std::env::var(variable)
        .unwrap_or_else(|_| panic!("{vendor} RTC smoke requires explicit {variable}"));
    assert!(
        !architecture.is_empty()
            && architecture.len() <= 32
            && architecture
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_'),
        "{variable} must be a short architecture identifier containing only ASCII letters, digits, and underscore"
    );
    architecture
}

#[cfg(feature = "nvrtc")]
fn nvrtc_architecture() -> String {
    let architecture = required_architecture("OCGPU_NVRTC_ARCH", "NVRTC");
    let suffix = architecture
        .strip_prefix("compute_")
        .expect("OCGPU_NVRTC_ARCH must use the compute_<target> form (for example compute_86)");
    assert!(
        (2..=4).contains(&suffix.len()) && suffix.bytes().all(|byte| byte.is_ascii_alphanumeric()),
        "OCGPU_NVRTC_ARCH must name one explicit CUDA virtual architecture"
    );
    architecture
}

#[cfg(feature = "hiprtc")]
fn hiprtc_architecture() -> String {
    let architecture = required_architecture("OCGPU_HIPRTC_ARCH", "HIPRTC");
    let suffix = architecture
        .strip_prefix("gfx")
        .expect("OCGPU_HIPRTC_ARCH must use the gfx<target> form (for example gfx90c)");
    assert!(
        (3..=12).contains(&suffix.len()) && suffix.bytes().all(|byte| byte.is_ascii_alphanumeric()),
        "OCGPU_HIPRTC_ARCH must name one explicit AMDGPU architecture without target features"
    );
    architecture
}

fn explicit_library(variable: &str) -> Option<PathBuf> {
    let supplied = std::env::var_os(variable).map(PathBuf::from)?;
    assert!(
        supplied.is_absolute(),
        "{variable} must be an absolute path when supplied"
    );
    let canonical = supplied
        .canonicalize()
        .unwrap_or_else(|error| panic!("{variable} does not resolve to a local file: {error}"));
    let metadata = canonical
        .metadata()
        .unwrap_or_else(|error| panic!("could not inspect {variable}: {error}"));
    assert!(
        metadata.is_file(),
        "{variable} must resolve to a regular file"
    );
    assert!(metadata.len() > 0, "{variable} must not name an empty file");
    Some(canonical)
}

#[cfg(feature = "nvrtc")]
fn load_nvrtc() -> Compiler<ocgpu::rtc::Nvrtc> {
    if let Some(path) = explicit_library("OCGPU_NVRTC_LIBRARY") {
        #[cfg(feature = "explicit-library-path")]
        {
            // SAFETY: this opt-in hardware harness accepts only a canonical
            // absolute regular file explicitly selected by the runner owner.
            // The runner owner remains responsible for trusting the library
            // and its dependency closure, as required by the API contract.
            return unsafe { Compiler::<ocgpu::rtc::Nvrtc>::load_from_absolute(&path) }
                .unwrap_or_else(|error| {
                panic!(
                    "selected NVRTC library {} could not provide the required common RTC API: {error}",
                    path.display()
                )
                });
        }
        #[cfg(not(feature = "explicit-library-path"))]
        panic!(
            "OCGPU_NVRTC_LIBRARY={} requires compiling the smoke test with explicit-library-path",
            path.display()
        );
    }
    Compiler::<ocgpu::rtc::Nvrtc>::load().unwrap_or_else(|error| {
        panic!("selected CUDA RTC smoke requires a usable NVRTC library: {error}")
    })
}

#[cfg(feature = "hiprtc")]
fn load_hiprtc() -> Compiler<ocgpu::rtc::Hiprtc> {
    if let Some(path) = explicit_library("OCGPU_HIPRTC_LIBRARY") {
        #[cfg(feature = "explicit-library-path")]
        {
            // SAFETY: this opt-in hardware harness accepts only a canonical
            // absolute regular file explicitly selected by the runner owner.
            // The runner owner remains responsible for trusting the library
            // and its dependency closure, as required by the API contract.
            return unsafe { Compiler::<ocgpu::rtc::Hiprtc>::load_from_absolute(&path) }
                .unwrap_or_else(|error| {
                panic!(
                    "selected HIPRTC library {} could not provide the required common RTC API: {error}",
                    path.display()
                )
                });
        }
        #[cfg(not(feature = "explicit-library-path"))]
        panic!(
            "OCGPU_HIPRTC_LIBRARY={} requires compiling the smoke test with explicit-library-path",
            path.display()
        );
    }
    Compiler::<ocgpu::rtc::Hiprtc>::load().unwrap_or_else(|error| {
        panic!("selected HIP RTC smoke requires a usable HIPRTC library: {error}")
    })
}

fn compile_and_exercise<B: RtcBackend>(
    compiler: Compiler<B>,
    architecture: &str,
) -> CompiledModule {
    let (major, minor) = compiler.version().expect("runtime-compiler version query");
    assert!(
        major > 0 && minor >= 0,
        "runtime compiler reported an invalid version {major}.{minor}"
    );
    let success_text = compiler
        .error_string(ocgpu::sys::OCGPU_RTC_SUCCESS)
        .expect("runtime compiler must describe its success result");
    assert!(!success_text.trim().is_empty());

    let architecture_option = CString::new(format!("--gpu-architecture={architecture}"))
        .expect("validated architecture contains no NUL");
    let options = [architecture_option.as_c_str()];

    // Exercise the compilation-error path before creating any driver context.
    let mut invalid = compiler
        .create_program(INVALID_SOURCE, Some(INVALID_PROGRAM_NAME), &[])
        .expect("invalid-source program creation");
    match invalid.compile(&options) {
        Err(RtcError::Compile(failure)) => {
            assert_ne!(failure.rtc.result, ocgpu::sys::OCGPU_RTC_SUCCESS);
            assert!(!failure.rtc.message.trim().is_empty());
            assert!(
                failure.log_error.is_none(),
                "invalid-source compiler log could not be read: {:?}",
                failure.log_error
            );
            assert!(
                !failure.log.is_empty(),
                "invalid source must produce a bounded diagnostic log"
            );
        }
        Ok(()) => panic!("intentionally invalid source unexpectedly compiled"),
        Err(error) => panic!("invalid-source compilation returned the wrong error shape: {error}"),
    }
    let invalid_log = invalid
        .log()
        .expect("explicit invalid-source compiler-log query");
    assert!(!invalid_log.is_empty());
    invalid
        .destroy()
        .expect("explicit invalid-source program destruction");

    let mut program = compiler
        .create_program(SOURCE, Some(PROGRAM_NAME), &[])
        .expect("no-op program creation");
    program
        .add_name_expression(NAME_EXPRESSION)
        .expect("no-op name-expression registration");
    program
        .compile(&options)
        .expect("no-op program compilation");
    assert!(program.is_compiled());
    let log = program.log().expect("successful compiler-log query");
    assert!(log.len() <= program.limits().max_log_bytes);
    let lowered_name = program
        .lowered_name(NAME_EXPRESSION)
        .expect("no-op lowered-name query");
    assert!(!lowered_name.as_bytes().is_empty());
    let code = program.code().expect("compiled-code query");
    assert!(!code.is_empty(), "runtime compiler returned empty code");
    assert!(code.len() <= program.limits().max_code_bytes);
    program
        .destroy()
        .expect("explicit no-op program destruction");

    CompiledModule { code, lowered_name }
}

fn exercise_driver<B: Backend>(
    backend: &str,
    driver: &Driver<B>,
    module_image: &CompiledModule,
    textual_code: bool,
    before_launch: impl FnOnce(),
) {
    assert!(driver.driver_version().expect("driver version query") > 0);
    assert!(
        driver.device_count().expect("device enumeration") > 0,
        "{backend} must expose at least one device"
    );
    let device = driver.device(0).expect("first device");
    assert!(!device.name().expect("device name").trim().is_empty());
    let context = device.create_context(0).expect("RTC smoke context");

    // Keep the transfer bounded and independent of the no-op kernel. In
    // particular, the HIP kernel never receives or writes a memory pointer.
    let memory = context
        .allocate(COPY_BYTES)
        .expect("64-byte device allocation");
    let source = [0xA5_u8; COPY_BYTES];
    let mut destination = [0_u8; COPY_BYTES];
    memory
        .copy_from(&source)
        .expect("64-byte host-to-device copy");
    memory
        .copy_to(&mut destination)
        .expect("64-byte device-to-host copy");
    assert_eq!(source, destination);

    let stream = context.create_stream(0).expect("stream creation");
    let launch_start = context.create_event(0).expect("start-event creation");
    let launch_complete = context.create_event(0).expect("completion-event creation");

    let textual_storage;
    let module = if textual_code {
        let without_trailing_nul = module_image
            .code
            .strip_suffix(&[0])
            .unwrap_or(&module_image.code);
        textual_storage = CString::new(without_trailing_nul)
            .expect("NVRTC PTX must not contain an interior NUL byte");
        // SAFETY: NVRTC returned a complete PTX image and the owned CString
        // remains live until after the module has been loaded and used.
        unsafe { context.load_module_cstr(&textual_storage) }.expect("compiled PTX module load")
    } else {
        // SAFETY: HIPRTC returned its complete native code object and the
        // bounded byte vector remains live until after module use.
        unsafe {
            context
                .load_module_data(module_image.code.as_ptr().cast::<c_void>())
                .expect("compiled HIP code-object module load")
        }
    };
    let function = module
        .function(&module_image.lowered_name)
        .expect("compiled no-op entry-point lookup");
    let launch = LaunchConfig::new([1, 1, 1], [1, 1, 1], 0)
        .expect("one-block, one-thread launch configuration");
    launch_start
        .record(&stream)
        .expect("pre-launch event recording");
    before_launch();
    // SAFETY: the only accepted source is the fixed zero-argument no-op above;
    // the launch uses exactly one block and one thread, once, with zero args.
    unsafe {
        function
            .launch(launch, Some(&stream), &mut [])
            .expect("single no-op launch");
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

#[cfg(feature = "nvrtc")]
fn run_cuda(rtc: Compiler<ocgpu::rtc::Nvrtc>, architecture: &str, before_launch: impl FnOnce()) {
    let module_image = compile_and_exercise(rtc, architecture);
    let driver = Driver::<ocgpu::Cuda>::load().unwrap_or_else(|error| {
        panic!("CUDA driver load/initialization failed before execution: {error}")
    });
    exercise_driver("CUDA", &driver, &module_image, true, before_launch);
}

#[cfg(feature = "hiprtc")]
fn run_hip(rtc: Compiler<ocgpu::rtc::Hiprtc>, architecture: &str, before_launch: impl FnOnce()) {
    let module_image = compile_and_exercise(rtc, architecture);
    let driver = Driver::<ocgpu::Hip>::load().unwrap_or_else(|error| {
        panic!("HIP driver load/initialization failed before execution: {error}")
    });
    exercise_driver("HIP", &driver, &module_image, false, before_launch);
}

#[cfg(all(feature = "nvrtc", feature = "hiprtc"))]
enum WorkerUpdate {
    Ready(&'static str),
    ExecutionReady(&'static str),
    Finished(&'static str, Result<(), String>),
}

#[cfg(all(feature = "nvrtc", feature = "hiprtc"))]
fn panic_text(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "worker panicked without a string payload".to_owned()
    }
}

#[cfg(all(feature = "nvrtc", feature = "hiprtc"))]
fn spawn_worker(
    name: &'static str,
    start: Receiver<()>,
    updates: Sender<WorkerUpdate>,
    operation: impl FnOnce() + Send + 'static,
) -> JoinHandle<()> {
    thread::Builder::new()
        .name(format!("ocgpu-{name}-rtc-smoke"))
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
        .expect("spawn bounded RTC backend worker")
}

#[cfg(all(feature = "nvrtc", feature = "hiprtc"))]
fn receive_before(receiver: &Receiver<WorkerUpdate>, deadline: Instant) -> WorkerUpdate {
    let remaining = deadline
        .checked_duration_since(Instant::now())
        .unwrap_or(Duration::ZERO);
    assert!(
        !remaining.is_zero(),
        "dual RTC backend coordination timed out"
    );
    match receiver.recv_timeout(remaining) {
        Ok(update) => update,
        Err(RecvTimeoutError::Timeout) => panic!("dual RTC backend coordination timed out"),
        Err(RecvTimeoutError::Disconnected) => {
            panic!("dual RTC backend worker channel disconnected")
        }
    }
}

#[cfg(all(feature = "nvrtc", feature = "hiprtc"))]
fn expect_ready(update: WorkerUpdate) -> &'static str {
    match update {
        WorkerUpdate::Ready(name) => name,
        WorkerUpdate::ExecutionReady(name) => {
            panic!("{name} RTC worker reached execution before initial release")
        }
        WorkerUpdate::Finished(name, result) => {
            panic!("{name} RTC worker finished before release: {result:?}")
        }
    }
}

#[cfg(all(feature = "nvrtc", feature = "hiprtc"))]
fn expect_finished(update: WorkerUpdate) -> (&'static str, Result<(), String>) {
    match update {
        WorkerUpdate::Finished(name, result) => (name, result),
        WorkerUpdate::Ready(name) => panic!("{name} RTC worker reported ready twice"),
        WorkerUpdate::ExecutionReady(name) => {
            panic!("{name} RTC worker reported execution-ready twice")
        }
    }
}

#[cfg(all(feature = "nvrtc", feature = "hiprtc"))]
fn expect_execution_ready(update: WorkerUpdate) -> &'static str {
    match update {
        WorkerUpdate::ExecutionReady(name) => name,
        WorkerUpdate::Ready(name) => panic!("{name} RTC worker reported ready twice"),
        WorkerUpdate::Finished(name, result) => {
            panic!("{name} RTC worker failed before simultaneous launch release: {result:?}")
        }
    }
}

#[cfg(all(feature = "nvrtc", feature = "hiprtc"))]
fn run_both(
    cuda_compiler: Compiler<ocgpu::rtc::Nvrtc>,
    cuda_architecture: String,
    hip_compiler: Compiler<ocgpu::rtc::Hiprtc>,
    hip_architecture: String,
) {
    let (updates_tx, updates_rx) = mpsc::channel();
    let (cuda_start_tx, cuda_start_rx) = mpsc::channel();
    let (hip_start_tx, hip_start_rx) = mpsc::channel();
    let (cuda_launch_tx, cuda_launch_rx) = mpsc::channel();
    let (hip_launch_tx, hip_launch_rx) = mpsc::channel();
    let cuda_execution_tx = updates_tx.clone();
    let cuda = spawn_worker("cuda", cuda_start_rx, updates_tx.clone(), move || {
        run_cuda(cuda_compiler, &cuda_architecture, move || {
            cuda_execution_tx
                .send(WorkerUpdate::ExecutionReady("cuda"))
                .expect("report CUDA execution readiness");
            cuda_launch_rx
                .recv_timeout(WORKER_TIMEOUT)
                .expect("coordinator did not release CUDA launch before timeout");
        });
    });
    let hip_execution_tx = updates_tx.clone();
    let hip = spawn_worker("hip", hip_start_rx, updates_tx, move || {
        run_hip(hip_compiler, &hip_architecture, move || {
            hip_execution_tx
                .send(WorkerUpdate::ExecutionReady("hip"))
                .expect("report HIP execution readiness");
            hip_launch_rx
                .recv_timeout(WORKER_TIMEOUT)
                .expect("coordinator did not release HIP launch before timeout");
        });
    });

    let ready_deadline = Instant::now() + COORDINATION_TIMEOUT;
    let first_ready = expect_ready(receive_before(&updates_rx, ready_deadline));
    let second_ready = expect_ready(receive_before(&updates_rx, ready_deadline));
    assert_ne!(
        first_ready, second_ready,
        "one RTC worker reported ready twice"
    );
    cuda_start_tx.send(()).expect("release CUDA RTC worker");
    hip_start_tx.send(()).expect("release HIP RTC worker");

    let completion_deadline = Instant::now() + WORKER_TIMEOUT;
    let first_execution_ready =
        expect_execution_ready(receive_before(&updates_rx, completion_deadline));
    let second_execution_ready =
        expect_execution_ready(receive_before(&updates_rx, completion_deadline));
    assert_ne!(
        first_execution_ready, second_execution_ready,
        "one RTC worker reported execution readiness twice"
    );
    // Both contexts, generated modules, and zero-argument functions are ready.
    // Release the two one-thread launches together without extending the
    // shared 30-second compile/setup/execute deadline.
    cuda_launch_tx.send(()).expect("release CUDA launch");
    hip_launch_tx.send(()).expect("release HIP launch");
    let first_result = expect_finished(receive_before(&updates_rx, completion_deadline));
    let second_result = expect_finished(receive_before(&updates_rx, completion_deadline));
    assert_ne!(
        first_result.0, second_result.0,
        "one RTC worker reported completion twice"
    );
    cuda.join().expect("CUDA RTC worker join");
    hip.join().expect("HIP RTC worker join");
    for (name, result) in [first_result, second_result] {
        if let Err(error) = result {
            panic!("{name} RTC worker failed: {error}");
        }
    }
}

#[test]
fn bounded_rtc_hardware_smoke() {
    if std::env::var("OCGPU_RUN_RTC_HARDWARE_SMOKE").as_deref() != Ok("1") {
        eprintln!("RTC hardware smoke skipped; set OCGPU_RUN_RTC_HARDWARE_SMOKE=1 to opt in");
        return;
    }
    let mode = std::env::var("OCGPU_RTC_SMOKE_BACKEND")
        .unwrap_or_else(|_| panic!("OCGPU_RTC_SMOKE_BACKEND must be set to cuda, hip, or both"));
    let compile_only = match std::env::var("OCGPU_RTC_COMPILE_ONLY") {
        Ok(value) if value == "1" => true,
        Ok(value) => panic!("OCGPU_RTC_COMPILE_ONLY, when set, must equal 1; received {value:?}"),
        Err(std::env::VarError::NotPresent) => false,
        Err(std::env::VarError::NotUnicode(_)) => {
            panic!("OCGPU_RTC_COMPILE_ONLY must contain valid Unicode")
        }
    };
    match mode.as_str() {
        "cuda" => {
            #[cfg(feature = "nvrtc")]
            {
                let compiler = load_nvrtc();
                let architecture = nvrtc_architecture();
                if compile_only {
                    let _ = compile_and_exercise(compiler, &architecture);
                } else {
                    run_cuda(compiler, &architecture, || {});
                }
            }
            #[cfg(not(feature = "nvrtc"))]
            panic!("CUDA RTC smoke requires the nvrtc feature");
        }
        "hip" => {
            #[cfg(feature = "hiprtc")]
            {
                let compiler = load_hiprtc();
                let architecture = hiprtc_architecture();
                if compile_only {
                    let _ = compile_and_exercise(compiler, &architecture);
                } else {
                    run_hip(compiler, &architecture, || {});
                }
            }
            #[cfg(not(feature = "hiprtc"))]
            panic!("HIP RTC smoke requires the hiprtc feature");
        }
        "both" => {
            #[cfg(all(feature = "nvrtc", feature = "hiprtc"))]
            {
                let cuda_architecture = nvrtc_architecture();
                let hip_architecture = hiprtc_architecture();
                // Resolve both selected compiler libraries before either
                // worker creates a driver context or executes GPU work.
                let cuda_compiler = load_nvrtc();
                let hip_compiler = load_hiprtc();
                if compile_only {
                    let _ = compile_and_exercise(cuda_compiler, &cuda_architecture);
                    let _ = compile_and_exercise(hip_compiler, &hip_architecture);
                } else {
                    run_both(
                        cuda_compiler,
                        cuda_architecture,
                        hip_compiler,
                        hip_architecture,
                    );
                }
            }
            #[cfg(not(all(feature = "nvrtc", feature = "hiprtc")))]
            panic!("both RTC smoke requires the nvrtc and hiprtc features");
        }
        _ => panic!("unsupported OCGPU_RTC_SMOKE_BACKEND mode {mode:?}"),
    }
}
