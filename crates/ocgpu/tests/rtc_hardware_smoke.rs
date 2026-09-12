// SPDX-License-Identifier: CC0-1.0

//! Opt-in, watchdog-bounded runtime-compilation hardware validation.
//!
//! This test is skipped unless `OCGPU_RUN_RTC_HARDWARE_SMOKE=1`. A selected
//! backend compiles one fixed, zero-argument no-op kernel with injected headers inside
//! the test process and launches exactly one thread once. Setting
//! `OCGPU_RTC_COMPILE_ONLY=1` performs a context-free compiler staging run. It
//! compiles both backends concurrently when `OCGPU_RTC_SMOKE_BACKEND=both`.
//! Separate fixed-source programs validate optional CUBIN/bitcode output without
//! loading or launching that output. The test never resets a device, changes
//! power/display state, invokes an external compiler, or uses a backend other
//! than CUDA/HIP through ocgpu.

#![cfg(all(feature = "rtc", any(feature = "nvrtc", feature = "hiprtc")))]

use ocgpu::rtc::{Compiler, Error as RtcError, Header, Program, RtcBackend};
use ocgpu::{Backend, Context, Driver, LaunchConfig};
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
const SOURCE: &CStr = c"#include \"ocgpu/noop.h\"\nextern \"C\" __global__ void ocgpu_noop() { static_assert(ocgpu_smoke_value == OCGPU_SMOKE_VALUE, \"injected headers and options\"); }";
const HEADER: &CStr =
    c"#include \"ocgpu/value.h\"\nconstexpr int ocgpu_smoke_value = ocgpu_header_value;";
const VALUE_HEADER: &CStr = c"constexpr int ocgpu_header_value = 17;";
const INVALID_SOURCE: &CStr = c"extern \"C\" __global__ void ocgpu_invalid( {";
const PROGRAM_NAME: &CStr = c"ocgpu_noop.cpp";
const INVALID_PROGRAM_NAME: &CStr = c"ocgpu_invalid.cpp";
const NAME_EXPRESSION: &CStr = c"&ocgpu_noop";
const OPTIONAL_OUTPUT_SOURCE: &CStr =
    c"extern \"C\" __global__ void ocgpu_optional_output_noop() {}";

#[cfg(all(feature = "nvrtc", feature = "hiprtc"))]
const COORDINATION_TIMEOUT: Duration = Duration::from_secs(5);
#[cfg(all(feature = "nvrtc", feature = "hiprtc"))]
const WORKER_TIMEOUT: Duration = Duration::from_secs(30);

struct CompiledModule {
    code: Vec<u8>,
    lowered_name: CString,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WorkerPhase {
    Compile,
    Code,
    Launch,
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

fn exercise_invalid_program<B: RtcBackend>(compiler: Compiler<B>, options: &[&CStr]) {
    // Exercise the compilation-error path before creating any driver context.
    let mut invalid = compiler
        .create_program(INVALID_SOURCE, Some(INVALID_PROGRAM_NAME), &[])
        .expect("invalid-source program creation");
    match invalid.compile(options) {
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
    assert!(invalid.compile_attempted());
    assert!(!invalid.is_compiled());
    assert!(matches!(invalid.code(), Err(RtcError::InvalidState { .. })));
    assert!(matches!(
        invalid.add_name_expression(c"&ocgpu_invalid"),
        Err(RtcError::InvalidState { .. })
    ));
    invalid
        .destroy()
        .expect("explicit invalid-source program destruction");
}

#[cfg(feature = "hiprtc")]
fn report_hip_code_object(code: &[u8]) {
    if code.len() < 64 || !code.starts_with(b"\x7fELF") {
        eprintln!("HIPRTC: output has no complete direct ELF header");
        return;
    }
    let abi = code[8];
    if code[7] == 64 && (2..=4).contains(&abi) {
        eprintln!(
            "HIPRTC: ELF ABI byte 8 = {abi}, AMD HSA code object V{}",
            abi + 2
        );
    } else {
        eprintln!(
            "HIPRTC: ELF OSABI {}, ABI byte 8 = {abi} (unrecognized code-object version)",
            code[7]
        );
    }
}

fn compile_and_exercise<B: RtcBackend>(
    compiler: Compiler<B>,
    architecture: &str,
    extra_options: &[&CStr],
    mut progress: impl FnMut(WorkerPhase),
    inspect_program: impl FnOnce(&Program<B>),
) -> CompiledModule {
    let (major, minor) = compiler.version().expect("runtime-compiler version query");
    assert!(
        major > 0 && minor >= 0,
        "runtime compiler reported an invalid version {major}.{minor}"
    );
    eprintln!(
        "{} {major}.{minor}: {} ({architecture})",
        B::KIND,
        compiler.loaded_path().display()
    );
    let success_text = compiler
        .error_string(ocgpu::sys::OCGPU_RTC_SUCCESS)
        .expect("runtime compiler must describe its success result");
    assert!(!success_text.trim().is_empty());

    let architecture_option = CString::new(format!("--gpu-architecture={architecture}"))
        .expect("validated architecture contains no NUL");
    let mut options = vec![
        architecture_option.as_c_str(),
        c"--std=c++14",
        c"-DOCGPU_SMOKE_VALUE=17",
    ];
    options.extend_from_slice(extra_options);
    exercise_invalid_program(compiler, &options);

    let mut program = compiler
        .create_program(
            SOURCE,
            Some(PROGRAM_NAME),
            &[
                Header::new(HEADER, c"ocgpu/noop.h"),
                Header::new(VALUE_HEADER, c"ocgpu/value.h"),
            ],
        )
        .expect("no-op program creation");
    assert!(!program.is_compiled());
    assert!(!program.compile_attempted());
    assert!(matches!(program.code(), Err(RtcError::InvalidState { .. })));
    assert!(matches!(
        program.lowered_name(NAME_EXPRESSION),
        Err(RtcError::InvalidState { .. })
    ));
    program
        .add_name_expression(NAME_EXPRESSION)
        .expect("no-op name-expression registration");
    // In both mode each independent program is created before either worker
    // enters its successful vendor compile call.
    progress(WorkerPhase::Compile);
    program
        .compile(&options)
        .expect("no-op program compilation");
    assert!(program.is_compiled());
    assert!(program.compile_attempted());
    // Keep both successful programs alive together until both compilers finish.
    // This also makes compilation-only both mode exercise concurrent state.
    progress(WorkerPhase::Code);
    assert!(matches!(
        program.add_name_expression(c"&ocgpu_noop"),
        Err(RtcError::InvalidState { .. })
    ));
    let log = program.log().expect("successful compiler-log query");
    assert!(log.len() <= program.limits().max_log_bytes);
    let lowered_name = program
        .lowered_name(NAME_EXPRESSION)
        .expect("no-op lowered-name query");
    assert!(!lowered_name.as_bytes().is_empty());
    let code = program.code().expect("compiled-code query");
    assert!(!code.is_empty(), "runtime compiler returned empty code");
    #[cfg(feature = "hiprtc")]
    if B::CODE_KIND == ocgpu::rtc::CodeKind::HipCodeObject {
        report_hip_code_object(&code);
    }
    assert!(code.len() <= program.limits().max_code_bytes);
    assert!(matches!(
        program.code_with_limit(code.len() - 1),
        Err(RtcError::OutputTooLarge { .. })
    ));
    assert_eq!(
        program.code().expect("repeated compiled-code query"),
        code,
        "code extraction must leave the program available for another query"
    );
    assert_eq!(
        program
            .lowered_name(NAME_EXPRESSION)
            .expect("repeated lowered-name query"),
        lowered_name
    );
    inspect_program(&program);
    eprintln!(
        "{}: headers, options, diagnostics, name lowering, and {} code bytes validated",
        B::KIND,
        code.len()
    );
    program
        .destroy()
        .expect("explicit no-op program destruction");

    CompiledModule { code, lowered_name }
}

fn exercise_transfers<B: Backend>(context: &Context<'_, B>) {
    let (free, total) = context.memory_info().expect("ocgpuMemGetInfo");
    assert!(total > 0 && free <= total, "inconsistent free/total memory");
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
    let copied = context
        .allocate(COPY_BYTES)
        .expect("second 64-byte device allocation");
    copied.copy_from_device(&memory).expect("ocgpuMemcpyDtoD");
    destination.fill(0);
    copied
        .copy_to(&mut destination)
        .expect("device-to-device result readback");
    assert_eq!(
        source, destination,
        "device-to-device transfer changed data"
    );
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
    exercise_transfers(&context);
    let stream = context.create_stream(0).expect("stream creation");
    let dependent_stream = context
        .create_stream(1)
        .expect("nonblocking dependent stream creation");
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
    // Enqueue the cross-stream dependency before any host synchronization.
    dependent_stream
        .wait_event(&launch_complete)
        .expect("ocgpuStreamWaitEvent");
    dependent_stream
        .synchronize()
        .expect("dependent stream synchronization");
    assert!(
        launch_complete
            .query()
            .expect("dependent stream waited for completion event")
    );
    assert!(dependent_stream.query().expect("dependent stream query"));
    launch_complete
        .synchronize()
        .expect("post-launch event synchronization");
    stream.synchronize().expect("stream synchronization");
    context.synchronize().expect("context synchronization");
    assert!(
        stream.query().expect("ocgpuStreamQuery"),
        "synchronized stream must report completion"
    );
    assert!(
        launch_complete.query().expect("ocgpuEventQuery"),
        "synchronized event must report completion"
    );
    let milliseconds = launch_start
        .elapsed_time(&launch_complete)
        .expect("ocgpuEventElapsedTime");
    assert!(milliseconds.is_finite() && milliseconds >= 0.0);
    eprintln!(
        "{backend}: all six common extensions validated: ocgpuMemGetInfo, ocgpuMemcpyDtoD, ocgpuStreamQuery, ocgpuStreamWaitEvent, ocgpuEventQuery, ocgpuEventElapsedTime; one no-op launch completed"
    );
}

#[cfg(feature = "nvrtc")]
fn exercise_nvrtc_cubin(rtc: Compiler<ocgpu::rtc::Nvrtc>, architecture: &str) {
    let raw = rtc.raw_table();
    if raw.ocgpuNvrtcGetCUBINSize.is_none() || raw.ocgpuNvrtcGetCUBIN.is_none() {
        eprintln!("NVRTC native CUBIN compilation skipped: optional output exports unavailable");
        return;
    }
    // NVRTC 12.4.1's GetCUBIN documentation requires an actual sm_ target;
    // the primary no-op program still uses its selected compute_ target.
    let target = architecture
        .strip_prefix("compute_")
        .expect("NVRTC architecture was validated before compiler loading");
    let option = CString::new(format!("--gpu-architecture=sm_{target}"))
        .expect("validated architecture contains no NUL");
    let mut program = rtc
        .create_program(
            OPTIONAL_OUTPUT_SOURCE,
            Some(c"ocgpu_optional_output.cu"),
            &[],
        )
        .expect("native CUBIN program creation");
    let output = ocgpu::rtc::NvrtcOutput::Cubin;
    assert!(matches!(
        program.native_output(output),
        Err(RtcError::InvalidState { .. })
    ));
    program
        .compile(&[option.as_c_str(), c"--std=c++14"])
        .expect("native CUBIN program compilation");
    let cubin = program.native_output(output).expect("native CUBIN query");
    assert!(!cubin.is_empty(), "actual sm_ target must produce CUBIN");
    assert!(cubin.len() <= program.limits().max_code_bytes);
    assert!(matches!(
        program.native_output_with_limit(output, cubin.len() - 1),
        Err(RtcError::OutputTooLarge { .. })
    ));
    assert_eq!(
        program
            .native_output_with_limit(output, cubin.len())
            .expect("native CUBIN query at its exact size limit"),
        cubin
    );
    assert_eq!(
        program
            .native_output(output)
            .expect("repeated native CUBIN query"),
        cubin
    );
    program
        .destroy()
        .expect("explicit native CUBIN program destruction");
    eprintln!(
        "NVRTC: {} native CUBIN bytes validated for sm_{target} without loading or launching",
        cubin.len()
    );
}

#[cfg(feature = "hiprtc")]
fn exercise_hiprtc_bitcode(rtc: Compiler<ocgpu::rtc::Hiprtc>, architecture: &str) {
    let raw = rtc.raw_table();
    if raw.ocgpuHiprtcGetBitcodeSize.is_none() || raw.ocgpuHiprtcGetBitcode.is_none() {
        eprintln!("HIPRTC bitcode compilation skipped: optional output exports unavailable");
        return;
    }
    // HIP's RTC programming guide specifies -fgpu-rdc for the bitcode getters.
    // This program is separate from the executable no-op code object.
    let option = CString::new(format!("--gpu-architecture={architecture}"))
        .expect("validated architecture contains no NUL");
    let mut program = rtc
        .create_program(
            OPTIONAL_OUTPUT_SOURCE,
            Some(c"ocgpu_optional_output.hip"),
            &[],
        )
        .expect("bitcode program creation");
    assert!(matches!(
        program.bitcode(),
        Err(RtcError::InvalidState { .. })
    ));
    program
        .compile(&[option.as_c_str(), c"--std=c++14", c"-fgpu-rdc"])
        .expect("bitcode program compilation");
    let bitcode = program.bitcode().expect("HIP bitcode query");
    assert!(!bitcode.is_empty(), "-fgpu-rdc must produce bitcode");
    assert!(bitcode.len() <= program.limits().max_code_bytes);
    assert!(matches!(
        program.bitcode_with_limit(bitcode.len() - 1),
        Err(RtcError::OutputTooLarge { .. })
    ));
    assert_eq!(
        program
            .bitcode_with_limit(bitcode.len())
            .expect("HIP bitcode query at its exact size limit"),
        bitcode
    );
    assert_eq!(
        program.bitcode().expect("repeated HIP bitcode query"),
        bitcode
    );
    program
        .destroy()
        .expect("explicit bitcode program destruction");
    eprintln!(
        "HIPRTC: {} bitcode bytes validated for {architecture} without loading or launching",
        bitcode.len()
    );
}

#[cfg(feature = "nvrtc")]
fn run_cuda(
    rtc: Compiler<ocgpu::rtc::Nvrtc>,
    architecture: &str,
    compile_only: bool,
    mut progress: impl FnMut(WorkerPhase),
) {
    match rtc.supported_architectures() {
        Ok(architectures) => {
            assert!(!architectures.is_empty());
            assert!(architectures.iter().all(|&architecture| architecture > 0));
            if let Ok(selected) = architecture.trim_start_matches("compute_").parse::<i32>() {
                assert!(
                    architectures.contains(&selected),
                    "selected NVRTC architecture {selected} missing from reported supported architectures {architectures:?}"
                );
            }
            eprintln!("NVRTC supported architectures: {architectures:?}");
        }
        Err(RtcError::MissingOptionalSymbol { symbol, .. }) => {
            eprintln!("NVRTC architecture enumeration unavailable: {symbol}");
        }
        Err(error) => panic!("NVRTC architecture enumeration failed: {error}"),
    }
    let module_image = compile_and_exercise(rtc, architecture, &[], &mut progress, |program| {
        // A virtual compute_XX target produces PTX; the optional native-binary
        // size query must report zero without invalidating the PTX program.
        match program.native_output(ocgpu::rtc::NvrtcOutput::Cubin) {
            Ok(cubin) => assert!(cubin.is_empty(), "a virtual target must not produce CUBIN"),
            Err(RtcError::MissingOptionalSymbol { symbol, .. }) => {
                eprintln!("NVRTC native CUBIN query unavailable: {symbol}");
            }
            Err(error) => panic!("NVRTC virtual-target CUBIN query failed: {error}"),
        }
    });
    exercise_nvrtc_cubin(rtc, architecture);
    if compile_only {
        return;
    }
    let driver = Driver::<ocgpu::Cuda>::load().unwrap_or_else(|error| {
        panic!("CUDA driver load/initialization failed before execution: {error}")
    });
    exercise_driver("CUDA", &driver, &module_image, true, || {
        progress(WorkerPhase::Launch);
    });
}

#[cfg(feature = "hiprtc")]
fn run_hip(
    rtc: Compiler<ocgpu::rtc::Hiprtc>,
    architecture: &str,
    compile_only: bool,
    mut progress: impl FnMut(WorkerPhase),
) {
    // An older Windows HIP driver imports amd_comgr.dll by basename. Bind its
    // installed dependency before a newer HIPRTC loads its own absolute sibling
    // with the same basename. Compiler-only runs remain independent of drivers.
    let driver = (!compile_only).then(|| {
        Driver::<ocgpu::Hip>::load().unwrap_or_else(|error| {
            panic!("HIP driver load/initialization failed before execution: {error}")
        })
    });
    let code_object_version = std::env::var("OCGPU_HIPRTC_CODE_OBJECT_VERSION");
    let (option, expected_abi) = match code_object_version.as_deref() {
        Ok("4") => (Some(c"-mcode-object-version=4"), Some(2)),
        Ok("5") => (Some(c"-mcode-object-version=5"), Some(3)),
        Ok("6") => (Some(c"-mcode-object-version=6"), Some(4)),
        Err(std::env::VarError::NotPresent) => (None, None),
        _ => panic!("OCGPU_HIPRTC_CODE_OBJECT_VERSION, when set, must equal 4, 5, or 6"),
    };
    let extra_options = option.into_iter().collect::<Vec<_>>();
    let module_image =
        compile_and_exercise(rtc, architecture, &extra_options, &mut progress, |_| {});
    exercise_hiprtc_bitcode(rtc, architecture);
    if let Some(expected_abi) = expected_abi {
        assert!(module_image.code.len() >= 64 && module_image.code.starts_with(b"\x7fELF"));
        assert_eq!(module_image.code[7], 64, "expected AMD HSA ELF OSABI");
        assert_eq!(
            module_image.code[8], expected_abi,
            "HIPRTC must honor the requested code-object version"
        );
    }
    let Some(driver) = driver else { return };
    exercise_driver("HIP", &driver, &module_image, false, || {
        progress(WorkerPhase::Launch);
    });
}

#[cfg(all(feature = "nvrtc", feature = "hiprtc"))]
enum WorkerUpdate {
    Ready(&'static str),
    PhaseReady(&'static str, WorkerPhase),
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
        WorkerUpdate::PhaseReady(name, phase) => {
            panic!("{name} RTC worker reached {phase:?} before initial release")
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
        WorkerUpdate::PhaseReady(name, phase) => {
            panic!("{name} RTC worker unexpectedly reported {phase:?} readiness")
        }
    }
}

#[cfg(all(feature = "nvrtc", feature = "hiprtc"))]
fn expect_phase_ready(update: WorkerUpdate, expected_phase: WorkerPhase) -> &'static str {
    match update {
        WorkerUpdate::PhaseReady(name, phase) => {
            assert_eq!(
                phase, expected_phase,
                "{name} RTC worker reached wrong phase"
            );
            name
        }
        WorkerUpdate::Ready(name) => panic!("{name} RTC worker reported ready twice"),
        WorkerUpdate::Finished(name, result) => {
            panic!("{name} RTC worker failed before {expected_phase:?} release: {result:?}")
        }
    }
}

#[cfg(all(feature = "nvrtc", feature = "hiprtc"))]
fn run_both(
    cuda_compiler: Compiler<ocgpu::rtc::Nvrtc>,
    cuda_architecture: String,
    hip_compiler: Compiler<ocgpu::rtc::Hiprtc>,
    hip_architecture: String,
    compile_only: bool,
) {
    run_coordinated_workers(
        compile_only,
        move |progress| run_cuda(cuda_compiler, &cuda_architecture, compile_only, progress),
        move |progress| run_hip(hip_compiler, &hip_architecture, compile_only, progress),
    );
}

#[cfg(all(feature = "nvrtc", feature = "hiprtc"))]
fn run_coordinated_workers(
    compile_only: bool,
    cuda_operation: impl FnOnce(&mut dyn FnMut(WorkerPhase)) + Send + 'static,
    hip_operation: impl FnOnce(&mut dyn FnMut(WorkerPhase)) + Send + 'static,
) {
    let (updates_tx, updates_rx) = mpsc::channel();
    let (cuda_start_tx, cuda_start_rx) = mpsc::channel();
    let (hip_start_tx, hip_start_rx) = mpsc::channel();
    let (cuda_phase_tx, cuda_phase_rx) = mpsc::channel();
    let (hip_phase_tx, hip_phase_rx) = mpsc::channel();
    let cuda_progress_tx = updates_tx.clone();
    let cuda = spawn_worker("cuda", cuda_start_rx, updates_tx.clone(), move || {
        cuda_operation(&mut |phase| {
            cuda_progress_tx
                .send(WorkerUpdate::PhaseReady("cuda", phase))
                .expect("report CUDA phase readiness");
            let released_phase = cuda_phase_rx
                .recv_timeout(WORKER_TIMEOUT)
                .expect("coordinator did not release CUDA phase before timeout");
            assert_eq!(released_phase, phase, "CUDA phase release mismatch");
        });
    });
    let hip_progress_tx = updates_tx.clone();
    let hip = spawn_worker("hip", hip_start_rx, updates_tx, move || {
        hip_operation(&mut |phase| {
            hip_progress_tx
                .send(WorkerUpdate::PhaseReady("hip", phase))
                .expect("report HIP phase readiness");
            let released_phase = hip_phase_rx
                .recv_timeout(WORKER_TIMEOUT)
                .expect("coordinator did not release HIP phase before timeout");
            assert_eq!(released_phase, phase, "HIP phase release mismatch");
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
    for phase in [WorkerPhase::Compile, WorkerPhase::Code, WorkerPhase::Launch] {
        if compile_only && phase == WorkerPhase::Launch {
            break;
        }
        let first_ready =
            expect_phase_ready(receive_before(&updates_rx, completion_deadline), phase);
        let second_ready =
            expect_phase_ready(receive_before(&updates_rx, completion_deadline), phase);
        assert_ne!(
            first_ready, second_ready,
            "one RTC worker reported {phase:?} twice"
        );
        eprintln!("NVRTC and HIPRTC ready together: {phase:?}");
        // A shared deadline bounds every phase. Compile and code extraction
        // rendezvous also run in context-free mode; only launch is conditional.
        cuda_phase_tx.send(phase).expect("release CUDA phase");
        hip_phase_tx.send(phase).expect("release HIP phase");
    }
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
                run_cuda(compiler, &architecture, compile_only, |_| {});
            }
            #[cfg(not(feature = "nvrtc"))]
            panic!("CUDA RTC smoke requires the nvrtc feature");
        }
        "hip" => {
            #[cfg(feature = "hiprtc")]
            {
                let compiler = load_hiprtc();
                let architecture = hiprtc_architecture();
                run_hip(compiler, &architecture, compile_only, |_| {});
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
                run_both(
                    cuda_compiler,
                    cuda_architecture,
                    hip_compiler,
                    hip_architecture,
                    compile_only,
                );
            }
            #[cfg(not(all(feature = "nvrtc", feature = "hiprtc")))]
            panic!("both RTC smoke requires the nvrtc and hiprtc features");
        }
        _ => panic!("unsupported OCGPU_RTC_SMOKE_BACKEND mode {mode:?}"),
    }
}

#[cfg(all(feature = "nvrtc", feature = "hiprtc"))]
mod coordination_tests {
    use super::{WorkerPhase, run_coordinated_workers};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn check_concurrent_phases(compile_only: bool) {
        let phase_counts = Arc::new([AtomicUsize::new(0), AtomicUsize::new(0)]);
        let make_worker = |index: usize| {
            let phase_counts = Arc::clone(&phase_counts);
            move |progress: &mut dyn FnMut(WorkerPhase)| {
                for (phase_index, phase) in
                    [WorkerPhase::Compile, WorkerPhase::Code, WorkerPhase::Launch]
                        .into_iter()
                        .enumerate()
                {
                    if compile_only && phase == WorkerPhase::Launch {
                        break;
                    }
                    phase_counts[index].store(phase_index + 1, Ordering::SeqCst);
                    progress(phase);
                    assert!(
                        phase_counts[1 - index].load(Ordering::SeqCst) > phase_index,
                        "worker {index} advanced before its peer was ready for {phase:?}"
                    );
                }
            }
        };
        run_coordinated_workers(compile_only, make_worker(0), make_worker(1));
        let expected = if compile_only { 2 } else { 3 };
        for count in phase_counts.iter() {
            assert_eq!(count.load(Ordering::SeqCst), expected);
        }
    }

    #[test]
    fn compile_only_both_keeps_independent_workers_at_each_compiler_phase() {
        check_concurrent_phases(true);
    }

    #[test]
    fn execution_both_coordinates_compilers_and_launches() {
        check_concurrent_phases(false);
    }
}
