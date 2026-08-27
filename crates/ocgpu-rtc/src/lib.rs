// SPDX-License-Identifier: CC0-1.0

//! SDK-free, runtime-loaded NVRTC and HIPRTC access.
//!
//! Raw tables preserve optional vendor exports. Successful construction of a
//! [`Compiler`] validates the complete eleven-call common profile once. Nine
//! declarations use vendor pointers directly; HIPRTC's two pointer-array
//! const-qualification mismatches use reviewed, allocation-free adapters that
//! call the exact raw HIPRTC declarations.

#![cfg_attr(not(any(target_os = "linux", target_os = "windows")), allow(dead_code))]

use ocgpu_abi::{
    OCGPU_ABI_VERSION_1, OCGPU_BACKEND_CUDA, OCGPU_BACKEND_HIP, OCGPU_ERROR_BACKEND_NOT_FOUND,
    OCGPU_ERROR_INTERNAL, OCGPU_ERROR_INVALID_ARGUMENT, OCGPU_ERROR_NOT_SUPPORTED,
    OCGPU_ERROR_SYMBOL_UNAVAILABLE, OCGPU_HIPRTC_ERROR_INTERNAL_ERROR,
    OCGPU_RTC_API_FLAG_CODE_IS_HIP_CODE_OBJECT, OCGPU_RTC_API_FLAG_CODE_IS_PTX, OCGPU_RTC_SUCCESS,
    ocgpuBackend, ocgpuHiprtcApi_v1, ocgpuNvrtcApi_v1, ocgpuResult, ocgpuRtcApi_v1,
    ocgpuRtcProgram, ocgpuRtcResult,
};
use ocgpu_loader::{Backend as LoaderBackend, Library, LoadError};
use std::error::Error as StdError;
use std::ffi::{CStr, CString, c_char, c_void};
use std::fmt;
use std::marker::PhantomData;
use std::mem::{ManuallyDrop, size_of};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::OnceLock;

/// Maximum source or header size accepted by the safe API (64 KiB).
pub const MAX_SOURCE_BYTES: usize = 64 * 1024;
/// Maximum option count accepted by the safe API.
pub const MAX_OPTIONS: usize = 64;
/// Maximum injected-header count accepted by the safe API.
pub const MAX_HEADERS: usize = 64;
/// Default maximum compilation-log allocation (1 MiB).
pub const MAX_LOG_BYTES: usize = 1024 * 1024;
/// Default maximum loadable-code allocation (16 MiB).
pub const MAX_CODE_BYTES: usize = 16 * 1024 * 1024;

/// Resource policy for the safe typed API.
///
/// [`Limits::default`] supplies conservative machine-test budgets. Applications
/// with larger legitimate workloads may pass an explicit policy without using
/// the unsafe raw API.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Limits {
    /// Maximum source/header/name-expression bytes, excluding trailing NUL.
    pub max_source_bytes: usize,
    /// Maximum compiler option count.
    pub max_options: usize,
    /// Maximum injected-header count.
    pub max_headers: usize,
    /// Maximum compiler-log allocation.
    pub max_log_bytes: usize,
    /// Maximum loadable-code allocation.
    pub max_code_bytes: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_source_bytes: MAX_SOURCE_BYTES,
            max_options: MAX_OPTIONS,
            max_headers: MAX_HEADERS,
            max_log_bytes: MAX_LOG_BYTES,
            max_code_bytes: MAX_CODE_BYTES,
        }
    }
}

/// Runtime compiler selected for a typed API object.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompilerKind {
    /// NVIDIA NVRTC, producing PTX for CUDA driver JIT loading.
    Nvrtc,
    /// AMD HIPRTC, producing a HIP code object.
    Hiprtc,
}

impl CompilerKind {
    /// The corresponding execution backend in the public ocgpu ABI.
    #[must_use]
    pub const fn ocgpu_backend(self) -> ocgpuBackend {
        match self {
            Self::Nvrtc => OCGPU_BACKEND_CUDA,
            Self::Hiprtc => OCGPU_BACKEND_HIP,
        }
    }
}

impl fmt::Display for CompilerKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Nvrtc => "NVRTC",
            Self::Hiprtc => "HIPRTC",
        })
    }
}

/// Native code representation returned by a runtime compiler.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CodeKind {
    /// NVIDIA PTX, intended for CUDA driver JIT loading.
    Ptx,
    /// AMD HIP code object, intended for HIP module loading.
    HipCodeObject,
}

/// Marker implemented only by the two supported runtime compilers.
pub trait RtcBackend: private::Sealed + 'static {
    /// Complete nullable raw ABI table for this compiler.
    type RawApi: Copy + 'static;
    /// Runtime compiler identity.
    const KIND: CompilerKind;
    /// Native code representation returned by this compiler.
    const CODE_KIND: CodeKind;
}

/// NVIDIA NVRTC marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct Nvrtc;

/// AMD HIPRTC marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct Hiprtc;

impl private::Sealed for Nvrtc {}
impl private::Sealed for Hiprtc {}

impl RtcBackend for Nvrtc {
    type RawApi = ocgpuNvrtcApi_v1;
    const KIND: CompilerKind = CompilerKind::Nvrtc;
    const CODE_KIND: CodeKind = CodeKind::Ptx;
}

impl RtcBackend for Hiprtc {
    type RawApi = ocgpuHiprtcApi_v1;
    const KIND: CompilerKind = CompilerKind::Hiprtc;
    const CODE_KIND: CodeKind = CodeKind::HipCodeObject;
}

mod private {
    pub trait Sealed {}
}

/// One injected source header and its virtual include name.
#[derive(Clone, Copy, Debug)]
pub struct Header<'a> {
    source: &'a CStr,
    include_name: &'a CStr,
}

impl<'a> Header<'a> {
    /// Creates an injected source header.
    #[must_use]
    pub const fn new(source: &'a CStr, include_name: &'a CStr) -> Self {
        Self {
            source,
            include_name,
        }
    }

    /// Header source text.
    #[must_use]
    pub const fn source(self) -> &'a CStr {
        self.source
    }

    /// Virtual include name used by source code.
    #[must_use]
    pub const fn include_name(self) -> &'a CStr {
        self.include_name
    }
}

/// Native runtime-compiler failure with its stable error string.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RtcFailure {
    /// Runtime compiler that reported the error.
    pub compiler: CompilerKind,
    /// Common operation being attempted.
    pub operation: &'static str,
    /// Unmodified vendor result code.
    pub result: ocgpuRtcResult,
    /// Vendor's stable error string, or a fallback for an invalid null result.
    pub message: String,
}

impl fmt::Display for RtcFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} {} failed with {}: {}",
            self.compiler, self.operation, self.result, self.message
        )
    }
}

impl StdError for RtcFailure {}

/// Compilation failure, including a bounded copy of the program log.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompileFailure {
    /// Native compilation result and error string.
    pub rtc: RtcFailure,
    /// Compiler log, including any vendor-provided trailing NUL byte.
    pub log: Vec<u8>,
    /// Why the log could not be obtained completely, when applicable.
    pub log_error: Option<Box<Error>>,
}

impl fmt::Display for CompileFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.rtc)?;
        if !self.log.is_empty() {
            let without_nul = self.log.strip_suffix(&[0]).unwrap_or(&self.log);
            write!(
                formatter,
                "; compiler log: {}",
                String::from_utf8_lossy(without_nul)
            )?;
        }
        if let Some(error) = &self.log_error {
            write!(formatter, "; compiler log unavailable: {error}")?;
        }
        Ok(())
    }
}

impl StdError for CompileFailure {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        Some(&self.rtc)
    }
}

/// Safe runtime-compilation error.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// Dynamic-library discovery or loading failed.
    Loader(LoadError),
    /// A common table was missing one or more mandatory entries.
    MissingRequiredSymbols {
        /// Runtime compiler being validated.
        compiler: CompilerKind,
        /// Complete list of missing vendor symbol names.
        symbols: Vec<&'static str>,
    },
    /// A public execution-backend value does not select CUDA or HIP.
    InvalidBackend {
        /// Rejected ABI value.
        backend: ocgpuBackend,
    },
    /// A bounded collection exceeded the safe API limit.
    TooManyItems {
        /// Collection kind.
        kind: &'static str,
        /// Supplied item count.
        count: usize,
        /// Maximum accepted item count.
        limit: usize,
    },
    /// Source-like input exceeded the safe API byte limit.
    InputTooLarge {
        /// Input kind.
        kind: &'static str,
        /// Supplied byte count, excluding the trailing NUL.
        bytes: usize,
        /// Maximum accepted byte count.
        limit: usize,
    },
    /// A vendor-reported output exceeded a caller allocation limit.
    OutputTooLarge {
        /// Output kind.
        kind: &'static str,
        /// Vendor-reported byte count.
        bytes: usize,
        /// Caller-selected maximum byte count.
        limit: usize,
    },
    /// A bounded allocation failed.
    AllocationFailed {
        /// Allocation purpose.
        kind: &'static str,
        /// Requested byte count.
        bytes: usize,
    },
    /// A successful vendor call returned a null required output.
    NullOutput {
        /// Operation that produced the invalid output.
        operation: &'static str,
    },
    /// An operation was requested in the wrong program lifecycle state.
    InvalidState {
        /// Rejected operation.
        operation: &'static str,
        /// Required lifecycle state.
        required: &'static str,
    },
    /// A non-compilation vendor call failed.
    Rtc(RtcFailure),
    /// Compilation failed and its bounded log was collected.
    Compile(CompileFailure),
}

impl Error {
    /// Maps the error to an existing ocgpu management/native result code.
    #[must_use]
    pub fn as_ocgpu_result(&self) -> ocgpuResult {
        match self {
            Self::Loader(LoadError::BackendUnavailable { .. }) => OCGPU_ERROR_BACKEND_NOT_FOUND,
            Self::Loader(LoadError::UnsupportedPlatform { .. }) => OCGPU_ERROR_NOT_SUPPORTED,
            Self::Loader(LoadError::SymbolUnavailable { .. })
            | Self::MissingRequiredSymbols { .. } => OCGPU_ERROR_SYMBOL_UNAVAILABLE,
            Self::Loader(
                LoadError::InvalidExplicitPath { .. }
                | LoadError::AlreadyInitialized { .. }
                | LoadError::InvalidSymbolName { .. },
            )
            | Self::InvalidBackend { .. }
            | Self::TooManyItems { .. }
            | Self::InputTooLarge { .. }
            | Self::OutputTooLarge { .. }
            | Self::InvalidState { .. } => OCGPU_ERROR_INVALID_ARGUMENT,
            Self::Rtc(failure) => failure.result,
            Self::Compile(failure) => failure.rtc.result,
            Self::AllocationFailed { .. } | Self::NullOutput { .. } | Self::Loader(_) => {
                OCGPU_ERROR_INTERNAL
            }
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Loader(error) => write!(formatter, "{error}"),
            Self::MissingRequiredSymbols { compiler, symbols } => write!(
                formatter,
                "{compiler} is missing mandatory symbols: {}",
                symbols.join(", ")
            ),
            Self::InvalidBackend { backend } => {
                write!(formatter, "invalid RTC execution backend {backend}")
            }
            Self::TooManyItems { kind, count, limit } => {
                write!(formatter, "{kind} count {count} exceeds limit {limit}")
            }
            Self::InputTooLarge { kind, bytes, limit }
            | Self::OutputTooLarge { kind, bytes, limit } => {
                write!(formatter, "{kind} size {bytes} exceeds limit {limit}")
            }
            Self::AllocationFailed { kind, bytes } => {
                write!(formatter, "could not allocate {bytes} bytes for {kind}")
            }
            Self::NullOutput { operation } => {
                write!(formatter, "{operation} succeeded but returned null")
            }
            Self::InvalidState {
                operation,
                required,
            } => write!(formatter, "{operation} requires program state {required}"),
            Self::Rtc(failure) => write!(formatter, "{failure}"),
            Self::Compile(failure) => write!(formatter, "{failure}"),
        }
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Loader(error) => Some(error),
            Self::Rtc(failure) => Some(failure),
            Self::Compile(failure) => Some(failure),
            _ => None,
        }
    }
}

impl From<LoadError> for Error {
    fn from(error: LoadError) -> Self {
        Self::Loader(error)
    }
}

type GetErrorStringFn = unsafe extern "C" fn(ocgpuRtcResult) -> *const c_char;
type VersionFn = unsafe extern "C" fn(*mut i32, *mut i32) -> ocgpuRtcResult;
type CreateProgramFn = unsafe extern "C" fn(
    *mut ocgpuRtcProgram,
    *const c_char,
    *const c_char,
    i32,
    *const *const c_char,
    *const *const c_char,
) -> ocgpuRtcResult;
type DestroyProgramFn = unsafe extern "C" fn(*mut ocgpuRtcProgram) -> ocgpuRtcResult;
type CompileProgramFn =
    unsafe extern "C" fn(ocgpuRtcProgram, i32, *const *const c_char) -> ocgpuRtcResult;
type GetSizeFn = unsafe extern "C" fn(ocgpuRtcProgram, *mut usize) -> ocgpuRtcResult;
type GetBytesFn = unsafe extern "C" fn(ocgpuRtcProgram, *mut c_char) -> ocgpuRtcResult;
type AddNameExpressionFn = unsafe extern "C" fn(ocgpuRtcProgram, *const c_char) -> ocgpuRtcResult;
type GetLoweredNameFn =
    unsafe extern "C" fn(ocgpuRtcProgram, *const c_char, *mut *const c_char) -> ocgpuRtcResult;

/// Validated common function pointers; every field is non-null.
#[derive(Clone, Copy)]
pub struct CoreFns {
    /// Direct vendor `GetErrorString` pointer.
    pub get_error_string: GetErrorStringFn,
    /// Direct vendor `Version` pointer.
    pub version: VersionFn,
    /// Validated common `CreateProgram` pointer; HIPRTC uses an ABI adapter.
    pub create_program: CreateProgramFn,
    /// Direct vendor `DestroyProgram` pointer.
    pub destroy_program: DestroyProgramFn,
    /// Validated common `CompileProgram` pointer; HIPRTC uses an ABI adapter.
    pub compile_program: CompileProgramFn,
    /// Direct vendor `GetProgramLogSize` pointer.
    pub get_program_log_size: GetSizeFn,
    /// Direct vendor `GetProgramLog` pointer.
    pub get_program_log: GetBytesFn,
    /// Direct vendor `AddNameExpression` pointer.
    pub add_name_expression: AddNameExpressionFn,
    /// Direct vendor `GetLoweredName` pointer.
    pub get_lowered_name: GetLoweredNameFn,
    /// Direct vendor backend-native code-size pointer.
    pub get_code_size: GetSizeFn,
    /// Direct vendor backend-native code pointer.
    pub get_code: GetBytesFn,
}

impl fmt::Debug for CoreFns {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("CoreFns { 11 validated callable function pointers }")
    }
}

impl CoreFns {
    /// Validates all mandatory slots and copies them into a non-null hot table.
    #[allow(clippy::too_many_lines)]
    pub fn from_table(compiler: CompilerKind, table: &ocgpuRtcApi_v1) -> Result<Self, Error> {
        let checks = [
            (table.ocgpuRtcGetErrorString.is_none(), "GetErrorString"),
            (table.ocgpuRtcVersion.is_none(), "Version"),
            (table.ocgpuRtcCreateProgram.is_none(), "CreateProgram"),
            (table.ocgpuRtcDestroyProgram.is_none(), "DestroyProgram"),
            (table.ocgpuRtcCompileProgram.is_none(), "CompileProgram"),
            (
                table.ocgpuRtcGetProgramLogSize.is_none(),
                "GetProgramLogSize",
            ),
            (table.ocgpuRtcGetProgramLog.is_none(), "GetProgramLog"),
            (
                table.ocgpuRtcAddNameExpression.is_none(),
                "AddNameExpression",
            ),
            (table.ocgpuRtcGetLoweredName.is_none(), "GetLoweredName"),
            (table.ocgpuRtcGetCodeSize.is_none(), "GetCodeSize"),
            (table.ocgpuRtcGetCode.is_none(), "GetCode"),
        ];
        let prefix = match compiler {
            CompilerKind::Nvrtc => "nvrtc",
            CompilerKind::Hiprtc => "hiprtc",
        };
        let symbols: Vec<_> = checks
            .into_iter()
            .filter_map(|(missing, suffix)| missing.then_some((prefix, suffix)))
            .map(|(prefix, suffix)| match (prefix, suffix) {
                ("nvrtc", "GetErrorString") => "nvrtcGetErrorString",
                ("nvrtc", "Version") => "nvrtcVersion",
                ("nvrtc", "CreateProgram") => "nvrtcCreateProgram",
                ("nvrtc", "DestroyProgram") => "nvrtcDestroyProgram",
                ("nvrtc", "CompileProgram") => "nvrtcCompileProgram",
                ("nvrtc", "GetProgramLogSize") => "nvrtcGetProgramLogSize",
                ("nvrtc", "GetProgramLog") => "nvrtcGetProgramLog",
                ("nvrtc", "AddNameExpression") => "nvrtcAddNameExpression",
                ("nvrtc", "GetLoweredName") => "nvrtcGetLoweredName",
                ("nvrtc", "GetCodeSize") => "nvrtcGetPTXSize",
                ("nvrtc", "GetCode") => "nvrtcGetPTX",
                ("hiprtc", "GetErrorString") => "hiprtcGetErrorString",
                ("hiprtc", "Version") => "hiprtcVersion",
                ("hiprtc", "CreateProgram") => "hiprtcCreateProgram",
                ("hiprtc", "DestroyProgram") => "hiprtcDestroyProgram",
                ("hiprtc", "CompileProgram") => "hiprtcCompileProgram",
                ("hiprtc", "GetProgramLogSize") => "hiprtcGetProgramLogSize",
                ("hiprtc", "GetProgramLog") => "hiprtcGetProgramLog",
                ("hiprtc", "AddNameExpression") => "hiprtcAddNameExpression",
                ("hiprtc", "GetLoweredName") => "hiprtcGetLoweredName",
                ("hiprtc", "GetCodeSize") => "hiprtcGetCodeSize",
                ("hiprtc", "GetCode") => "hiprtcGetCode",
                _ => "unknown RTC symbol",
            })
            .collect();
        if !symbols.is_empty() {
            return Err(Error::MissingRequiredSymbols { compiler, symbols });
        }

        let Some(get_error_string) = table.ocgpuRtcGetErrorString else {
            unreachable!("all common fields were checked")
        };
        let Some(version) = table.ocgpuRtcVersion else {
            unreachable!("all common fields were checked")
        };
        let Some(create_program) = table.ocgpuRtcCreateProgram else {
            unreachable!("all common fields were checked")
        };
        let Some(destroy_program) = table.ocgpuRtcDestroyProgram else {
            unreachable!("all common fields were checked")
        };
        let Some(compile_program) = table.ocgpuRtcCompileProgram else {
            unreachable!("all common fields were checked")
        };
        let Some(get_program_log_size) = table.ocgpuRtcGetProgramLogSize else {
            unreachable!("all common fields were checked")
        };
        let Some(get_program_log) = table.ocgpuRtcGetProgramLog else {
            unreachable!("all common fields were checked")
        };
        let Some(add_name_expression) = table.ocgpuRtcAddNameExpression else {
            unreachable!("all common fields were checked")
        };
        let Some(get_lowered_name) = table.ocgpuRtcGetLoweredName else {
            unreachable!("all common fields were checked")
        };
        let Some(get_code_size) = table.ocgpuRtcGetCodeSize else {
            unreachable!("all common fields were checked")
        };
        let Some(get_code) = table.ocgpuRtcGetCode else {
            unreachable!("all common fields were checked")
        };

        Ok(Self {
            get_error_string,
            version,
            create_program,
            destroy_program,
            compile_program,
            get_program_log_size,
            get_program_log,
            add_name_expression,
            get_lowered_name,
            get_code_size,
            get_code,
        })
    }
}

struct ApiState<R> {
    common_table: ocgpuRtcApi_v1,
    core: CoreFns,
    raw_table: R,
    loaded_path: PathBuf,
}

static NVRTC_STATE: OnceLock<Result<ApiState<ocgpuNvrtcApi_v1>, Error>> = OnceLock::new();
static HIPRTC_STATE: OnceLock<Result<ApiState<ocgpuHiprtcApi_v1>, Error>> = OnceLock::new();

/// A loaded, validated runtime compiler.
pub struct Compiler<B: RtcBackend> {
    core: &'static CoreFns,
    common_table: &'static ocgpuRtcApi_v1,
    raw_table: &'static B::RawApi,
    loaded_path: &'static Path,
    marker: PhantomData<B>,
}

impl<B: RtcBackend> Clone for Compiler<B> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<B: RtcBackend> Copy for Compiler<B> {}

impl<B: RtcBackend> fmt::Debug for Compiler<B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Compiler")
            .field("kind", &B::KIND)
            .field("loaded_path", &self.loaded_path)
            .finish_non_exhaustive()
    }
}

impl Compiler<Nvrtc> {
    /// Loads NVRTC from secure platform candidates and validates the common API.
    pub fn load() -> Result<Self, Error> {
        let library = ocgpu_loader::load(LoaderBackend::Nvrtc)?;
        Ok(compiler_from_nvrtc_state(nvrtc_state(library)?))
    }

    /// Loads NVRTC from a trusted absolute path and validates the common API.
    ///
    /// # Safety
    ///
    /// The path and its dependency closure must be trusted native code
    /// implementing the NVRTC ABI.
    #[cfg(feature = "explicit-library-path")]
    pub unsafe fn load_from_absolute(path: &Path) -> Result<Self, Error> {
        // SAFETY: forwarded from this function's explicit trust contract.
        let library = unsafe { ocgpu_loader::load_from_absolute(LoaderBackend::Nvrtc, path) }?;
        Ok(compiler_from_nvrtc_state(nvrtc_state(library)?))
    }
}

impl Compiler<Hiprtc> {
    /// Loads HIPRTC from secure platform candidates and validates the common API.
    pub fn load() -> Result<Self, Error> {
        let library = ocgpu_loader::load(LoaderBackend::Hiprtc)?;
        Ok(compiler_from_hiprtc_state(hiprtc_state(library)?))
    }

    /// Loads HIPRTC from a trusted absolute path and validates the common API.
    ///
    /// # Safety
    ///
    /// The path and its dependency closure must be trusted native code
    /// implementing the HIPRTC ABI.
    #[cfg(feature = "explicit-library-path")]
    pub unsafe fn load_from_absolute(path: &Path) -> Result<Self, Error> {
        // SAFETY: forwarded from this function's explicit trust contract.
        let library = unsafe { ocgpu_loader::load_from_absolute(LoaderBackend::Hiprtc, path) }?;
        Ok(compiler_from_hiprtc_state(hiprtc_state(library)?))
    }
}

impl<B: RtcBackend> Compiler<B> {
    /// Runtime compiler identity.
    #[must_use]
    pub const fn kind(self) -> CompilerKind {
        B::KIND
    }

    /// Native code representation produced by this compiler.
    #[must_use]
    pub const fn code_kind(self) -> CodeKind {
        B::CODE_KIND
    }

    /// Loaded shared-library identity.
    #[must_use]
    pub fn loaded_path(self) -> &'static Path {
        self.loaded_path
    }

    /// Validated non-null common function pointers.
    #[must_use]
    pub const fn core(self) -> &'static CoreFns {
        self.core
    }

    /// Negotiated common ABI table; all eleven fields are non-null.
    #[must_use]
    pub const fn common_table(self) -> &'static ocgpuRtcApi_v1 {
        self.common_table
    }

    /// Complete nullable vendor-specific raw table.
    #[must_use]
    pub const fn raw_table(self) -> &'static B::RawApi {
        self.raw_table
    }

    /// Queries the runtime-compiler version through the direct common pointer.
    pub fn version(self) -> Result<(i32, i32), Error> {
        let mut major = 0;
        let mut minor = 0;
        // SAFETY: output pointers are valid for the duration of the call.
        let result = unsafe { (self.core.version)(&raw mut major, &raw mut minor) };
        self.check_result("Version", result)?;
        Ok((major, minor))
    }

    /// Returns the backend error string for any compatible raw result code.
    #[must_use]
    pub fn error_string(self, result: ocgpuRtcResult) -> Option<String> {
        // SAFETY: this function accepts every integer result value by vendor ABI.
        let pointer = unsafe { (self.core.get_error_string)(result) };
        if pointer.is_null() {
            None
        } else {
            // SAFETY: vendor error strings are immutable, static, and NUL-terminated.
            Some(
                unsafe { CStr::from_ptr(pointer) }
                    .to_string_lossy()
                    .into_owned(),
            )
        }
    }

    /// Creates a program after enforcing source/header count and size budgets.
    pub fn create_program(
        self,
        source: &CStr,
        name: Option<&CStr>,
        headers: &[Header<'_>],
    ) -> Result<Program<B>, Error> {
        self.create_program_with_limits(source, name, headers, Limits::default())
    }

    /// Creates a program with a caller-selected bounded resource policy.
    pub fn create_program_with_limits(
        self,
        source: &CStr,
        name: Option<&CStr>,
        headers: &[Header<'_>],
        limits: Limits,
    ) -> Result<Program<B>, Error> {
        check_input_size("source", source, limits.max_source_bytes)?;
        if let Some(name) = name {
            check_input_size("program name", name, limits.max_source_bytes)?;
        }
        check_count("headers", headers.len(), limits.max_headers)?;
        for header in headers {
            check_input_size("header source", header.source, limits.max_source_bytes)?;
            check_input_size(
                "header include name",
                header.include_name,
                limits.max_source_bytes,
            )?;
        }

        let header_count = i32::try_from(headers.len()).map_err(|_| Error::TooManyItems {
            kind: "headers",
            count: headers.len(),
            limit: limits.max_headers,
        })?;
        let pointer_bytes = headers.len().saturating_mul(size_of::<*const c_char>());
        let mut header_sources = Vec::new();
        header_sources
            .try_reserve_exact(headers.len())
            .map_err(|_| Error::AllocationFailed {
                kind: "header source pointer array",
                bytes: pointer_bytes,
            })?;
        let mut include_names = Vec::new();
        include_names
            .try_reserve_exact(headers.len())
            .map_err(|_| Error::AllocationFailed {
                kind: "header include-name pointer array",
                bytes: pointer_bytes,
            })?;
        for header in headers {
            header_sources.push(header.source.as_ptr());
            include_names.push(header.include_name.as_ptr());
        }
        let mut handle = std::ptr::null_mut();
        let header_pointer = if header_sources.is_empty() {
            std::ptr::null()
        } else {
            header_sources.as_ptr()
        };
        let name_pointer = name.map_or(std::ptr::null(), CStr::as_ptr);
        let include_pointer = if include_names.is_empty() {
            std::ptr::null()
        } else {
            include_names.as_ptr()
        };
        // SAFETY: every C string and pointer array remains live through the call;
        // the output handle points to writable storage.
        let result = unsafe {
            (self.core.create_program)(
                &raw mut handle,
                source.as_ptr(),
                name_pointer,
                header_count,
                header_pointer,
                include_pointer,
            )
        };
        if result != OCGPU_RTC_SUCCESS {
            // Neither vendor grants ownership of the output slot on failure.
            // A non-null value is therefore not a documented valid handle.
            return Err(Error::Rtc(self.failure("CreateProgram", result)));
        }
        if handle.is_null() {
            return Err(Error::NullOutput {
                operation: "CreateProgram",
            });
        }
        Ok(Program {
            compiler: self,
            handle,
            compile_attempted: false,
            compiled: false,
            limits,
            not_send_or_sync: PhantomData,
        })
    }

    fn failure(self, operation: &'static str, result: ocgpuRtcResult) -> RtcFailure {
        RtcFailure {
            compiler: B::KIND,
            operation,
            result,
            message: self
                .error_string(result)
                .unwrap_or_else(|| "vendor returned a null error string".to_owned()),
        }
    }

    fn check_result(self, operation: &'static str, result: ocgpuRtcResult) -> Result<(), Error> {
        if result == OCGPU_RTC_SUCCESS {
            Ok(())
        } else {
            Err(Error::Rtc(self.failure(operation, result)))
        }
    }
}

/// Owned runtime-compilation program.
///
/// Programs are deliberately neither `Send` nor `Sync`; vendor program handles
/// have no cross-thread guarantee. Dropping performs best-effort destruction.
pub struct Program<B: RtcBackend> {
    compiler: Compiler<B>,
    handle: ocgpuRtcProgram,
    compile_attempted: bool,
    compiled: bool,
    limits: Limits,
    not_send_or_sync: PhantomData<Rc<()>>,
}

impl<B: RtcBackend> fmt::Debug for Program<B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Program")
            .field("compiler", &B::KIND)
            .field("compile_attempted", &self.compile_attempted)
            .field("compiled", &self.compiled)
            .finish_non_exhaustive()
    }
}

impl<B: RtcBackend> Program<B> {
    /// Runtime compiler that owns the program.
    #[must_use]
    pub const fn compiler(&self) -> Compiler<B> {
        self.compiler
    }

    /// Raw program handle for driver/module integration.
    #[must_use]
    pub const fn as_raw(&self) -> ocgpuRtcProgram {
        self.handle
    }

    /// Whether the most recent compilation completed successfully.
    #[must_use]
    pub const fn is_compiled(&self) -> bool {
        self.compiled
    }

    /// Whether the vendor compiler has been invoked at least once.
    #[must_use]
    pub const fn compile_attempted(&self) -> bool {
        self.compile_attempted
    }

    /// Default resource policy attached when the program was created.
    #[must_use]
    pub const fn limits(&self) -> Limits {
        self.limits
    }

    /// Registers a name expression before compilation.
    pub fn add_name_expression(&mut self, expression: &CStr) -> Result<(), Error> {
        if self.compile_attempted {
            return Err(Error::InvalidState {
                operation: "AddNameExpression",
                required: "created with no compilation attempt",
            });
        }
        check_input_size("name expression", expression, self.limits.max_source_bytes)?;
        // SAFETY: the live program belongs to this table and the C string remains live.
        let result =
            unsafe { (self.compiler.core.add_name_expression)(self.handle, expression.as_ptr()) };
        self.compiler.check_result("AddNameExpression", result)
    }

    /// Compiles the program with at most [`MAX_OPTIONS`] C-string options.
    ///
    /// A native compilation failure includes a log bounded by
    /// [`MAX_LOG_BYTES`] as well as the unmodified result code.
    pub fn compile(&mut self, options: &[&CStr]) -> Result<(), Error> {
        self.compile_with_limits(options, self.limits)
    }

    /// Compiles with a caller-selected bounded resource policy.
    pub fn compile_with_limits(&mut self, options: &[&CStr], limits: Limits) -> Result<(), Error> {
        check_count("options", options.len(), limits.max_options)?;
        for option in options {
            check_input_size("compiler option", option, limits.max_source_bytes)?;
        }
        let option_count = i32::try_from(options.len()).map_err(|_| Error::TooManyItems {
            kind: "options",
            count: options.len(),
            limit: limits.max_options,
        })?;
        let pointer_bytes = options.len().saturating_mul(size_of::<*const c_char>());
        let mut option_pointers = Vec::new();
        option_pointers
            .try_reserve_exact(options.len())
            .map_err(|_| Error::AllocationFailed {
                kind: "compiler option pointer array",
                bytes: pointer_bytes,
            })?;
        option_pointers.extend(options.iter().map(|option| option.as_ptr()));
        let options_pointer = if option_pointers.is_empty() {
            std::ptr::null()
        } else {
            option_pointers.as_ptr()
        };
        // SAFETY: the live handle belongs to this compiler and pointer array
        // remains valid for the call.
        self.compile_attempted = true;
        let result = unsafe {
            (self.compiler.core.compile_program)(self.handle, option_count, options_pointer)
        };
        if result == OCGPU_RTC_SUCCESS {
            self.compiled = true;
            return Ok(());
        }
        self.compiled = false;
        let rtc = self.compiler.failure("CompileProgram", result);
        let (log, log_error) = match self.log_with_limit(limits.max_log_bytes) {
            Ok(log) => (log, None),
            Err(error) => (Vec::new(), Some(Box::new(error))),
        };
        Err(Error::Compile(CompileFailure {
            rtc,
            log,
            log_error,
        }))
    }

    /// Copies the current compiler log with the default 1 MiB bound.
    pub fn log(&self) -> Result<Vec<u8>, Error> {
        self.log_with_limit(self.limits.max_log_bytes)
    }

    /// Copies the current compiler log with a caller-selected allocation bound.
    pub fn log_with_limit(&self, limit: usize) -> Result<Vec<u8>, Error> {
        self.read_bounded(
            "program log",
            "GetProgramLogSize",
            self.compiler.core.get_program_log_size,
            "GetProgramLog",
            self.compiler.core.get_program_log,
            limit,
        )
    }

    /// Resolves a previously registered expression after successful compilation.
    pub fn lowered_name(&self, expression: &CStr) -> Result<CString, Error> {
        if !self.compiled {
            return Err(Error::InvalidState {
                operation: "GetLoweredName",
                required: "successfully compiled",
            });
        }
        check_input_size("name expression", expression, self.limits.max_source_bytes)?;
        let mut lowered = std::ptr::null();
        // SAFETY: handle/expression are live and the output slot is writable.
        let result = unsafe {
            (self.compiler.core.get_lowered_name)(
                self.handle,
                expression.as_ptr(),
                &raw mut lowered,
            )
        };
        self.compiler.check_result("GetLoweredName", result)?;
        if lowered.is_null() {
            return Err(Error::NullOutput {
                operation: "GetLoweredName",
            });
        }
        // SAFETY: on success the vendor returns a program-owned NUL-terminated
        // string. Copying it prevents lifetime escape beyond this program.
        Ok(unsafe { CStr::from_ptr(lowered) }.to_owned())
    }

    /// Copies backend-native loadable code with the default 16 MiB bound.
    pub fn code(&self) -> Result<Vec<u8>, Error> {
        self.code_with_limit(self.limits.max_code_bytes)
    }

    /// Copies backend-native loadable code with a caller-selected allocation bound.
    pub fn code_with_limit(&self, limit: usize) -> Result<Vec<u8>, Error> {
        if !self.compiled {
            return Err(Error::InvalidState {
                operation: "GetCode",
                required: "successfully compiled",
            });
        }
        self.read_bounded(
            "loadable code",
            "GetCodeSize",
            self.compiler.core.get_code_size,
            "GetCode",
            self.compiler.core.get_code,
            limit,
        )
    }

    /// Destroys the vendor program exactly once and reports its result.
    ///
    /// On failure the vendor has not provided a reliable ownership state, so
    /// the handle is not called again by `Drop`.
    pub fn destroy(self) -> Result<(), Error> {
        let mut this = ManuallyDrop::new(self);
        // SAFETY: handle belongs to this compiler; `ManuallyDrop` prevents a
        // second call regardless of the vendor's result.
        let result = unsafe { (this.compiler.core.destroy_program)(&raw mut this.handle) };
        this.handle = std::ptr::null_mut();
        this.compiler.check_result("DestroyProgram", result)
    }

    fn read_bounded(
        &self,
        kind: &'static str,
        size_operation: &'static str,
        size_function: GetSizeFn,
        get_operation: &'static str,
        get_function: GetBytesFn,
        limit: usize,
    ) -> Result<Vec<u8>, Error> {
        let mut bytes = 0_usize;
        // SAFETY: handle is live and size output is writable.
        let result = unsafe { size_function(self.handle, &raw mut bytes) };
        self.compiler.check_result(size_operation, result)?;
        if bytes > limit {
            return Err(Error::OutputTooLarge { kind, bytes, limit });
        }
        if bytes == 0 {
            return Ok(Vec::new());
        }
        let mut output = Vec::new();
        output
            .try_reserve_exact(bytes)
            .map_err(|_| Error::AllocationFailed { kind, bytes })?;
        output.resize(bytes, 0);
        // SAFETY: vector is writable for exactly the vendor-reported byte count.
        let result = unsafe { get_function(self.handle, output.as_mut_ptr().cast()) };
        self.compiler.check_result(get_operation, result)?;
        Ok(output)
    }
}

impl<B: RtcBackend> Drop for Program<B> {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: best-effort RAII cleanup of the live owned handle.
            let _ = unsafe { (self.compiler.core.destroy_program)(&raw mut self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

/// Loads a required common RTC table for a public CUDA/HIP backend value.
///
/// Success guarantees that every one of the eleven fields is non-null.
pub fn load_common(backend: ocgpuBackend) -> Result<ocgpuRtcApi_v1, Error> {
    match backend {
        OCGPU_BACKEND_CUDA => {
            let library = ocgpu_loader::load(LoaderBackend::Nvrtc)?;
            Ok(nvrtc_state(library)?.common_table)
        }
        OCGPU_BACKEND_HIP => {
            let library = ocgpu_loader::load(LoaderBackend::Hiprtc)?;
            Ok(hiprtc_state(library)?.common_table)
        }
        _ => Err(Error::InvalidBackend { backend }),
    }
}

/// Loads the complete nullable NVRTC raw table.
pub fn load_nvrtc_raw() -> Result<ocgpuNvrtcApi_v1, Error> {
    let library = ocgpu_loader::load(LoaderBackend::Nvrtc)?;
    Ok(nvrtc_state(library)?.raw_table)
}

/// Loads the complete nullable HIPRTC raw table.
pub fn load_hiprtc_raw() -> Result<ocgpuHiprtcApi_v1, Error> {
    let library = ocgpu_loader::load(LoaderBackend::Hiprtc)?;
    Ok(hiprtc_state(library)?.raw_table)
}

fn compiler_from_nvrtc_state(state: &'static ApiState<ocgpuNvrtcApi_v1>) -> Compiler<Nvrtc> {
    Compiler {
        core: &state.core,
        common_table: &state.common_table,
        raw_table: &state.raw_table,
        loaded_path: &state.loaded_path,
        marker: PhantomData,
    }
}

fn compiler_from_hiprtc_state(state: &'static ApiState<ocgpuHiprtcApi_v1>) -> Compiler<Hiprtc> {
    Compiler {
        core: &state.core,
        common_table: &state.common_table,
        raw_table: &state.raw_table,
        loaded_path: &state.loaded_path,
        marker: PhantomData,
    }
}

fn nvrtc_state(library: &'static Library) -> Result<&'static ApiState<ocgpuNvrtcApi_v1>, Error> {
    result_ref(NVRTC_STATE.get_or_init(|| build_nvrtc_state(library)))
}

fn hiprtc_state(library: &'static Library) -> Result<&'static ApiState<ocgpuHiprtcApi_v1>, Error> {
    result_ref(HIPRTC_STATE.get_or_init(|| build_hiprtc_state(library)))
}

fn result_ref<R>(
    result: &'static Result<ApiState<R>, Error>,
) -> Result<&'static ApiState<R>, Error> {
    match result {
        Ok(state) => Ok(state),
        Err(error) => Err(error.clone()),
    }
}

fn build_nvrtc_state(library: &'static Library) -> Result<ApiState<ocgpuNvrtcApi_v1>, Error> {
    build_nvrtc_state_from(library, library.loaded_path())
}

fn build_nvrtc_state_from<P: SymbolProvider>(
    provider: &P,
    loaded_path: &Path,
) -> Result<ApiState<ocgpuNvrtcApi_v1>, Error> {
    let mut raw = ocgpuNvrtcApi_v1 {
        struct_size: table_size::<ocgpuNvrtcApi_v1>()?,
        abi_version: OCGPU_ABI_VERSION_1,
        backend: OCGPU_BACKEND_CUDA,
        flags: OCGPU_RTC_API_FLAG_CODE_IS_PTX,
        ..ocgpuNvrtcApi_v1::default()
    };
    resolve_nvrtc_raw(provider, &mut raw)?;
    let mut common = ocgpuRtcApi_v1 {
        struct_size: table_size::<ocgpuRtcApi_v1>()?,
        abi_version: OCGPU_ABI_VERSION_1,
        backend: OCGPU_BACKEND_CUDA,
        flags: OCGPU_RTC_API_FLAG_CODE_IS_PTX,
        ocgpuRtcGetErrorString: raw.ocgpuNvrtcGetErrorString,
        ocgpuRtcVersion: raw.ocgpuNvrtcVersion,
        ocgpuRtcCreateProgram: raw.ocgpuNvrtcCreateProgram,
        ocgpuRtcDestroyProgram: raw.ocgpuNvrtcDestroyProgram,
        ocgpuRtcCompileProgram: raw.ocgpuNvrtcCompileProgram,
        ocgpuRtcGetProgramLogSize: raw.ocgpuNvrtcGetProgramLogSize,
        ocgpuRtcGetProgramLog: raw.ocgpuNvrtcGetProgramLog,
        ocgpuRtcAddNameExpression: raw.ocgpuNvrtcAddNameExpression,
        ocgpuRtcGetLoweredName: raw.ocgpuNvrtcGetLoweredName,
        ocgpuRtcGetCodeSize: raw.ocgpuNvrtcGetPTXSize,
        ocgpuRtcGetCode: raw.ocgpuNvrtcGetPTX,
        ..ocgpuRtcApi_v1::default()
    };
    let core = CoreFns::from_table(CompilerKind::Nvrtc, &common)?;
    let (major, minor) = version_from_core(CompilerKind::Nvrtc, &core)?;
    common.rtc_version_major = major;
    common.rtc_version_minor = minor;
    raw.rtc_version_major = major;
    raw.rtc_version_minor = minor;
    Ok(ApiState {
        common_table: common,
        core,
        raw_table: raw,
        loaded_path: loaded_path.to_path_buf(),
    })
}

fn build_hiprtc_state(library: &'static Library) -> Result<ApiState<ocgpuHiprtcApi_v1>, Error> {
    build_hiprtc_state_from(library, library.loaded_path())
}

fn build_hiprtc_state_from<P: SymbolProvider>(
    provider: &P,
    loaded_path: &Path,
) -> Result<ApiState<ocgpuHiprtcApi_v1>, Error> {
    let mut raw = ocgpuHiprtcApi_v1 {
        struct_size: table_size::<ocgpuHiprtcApi_v1>()?,
        abi_version: OCGPU_ABI_VERSION_1,
        backend: OCGPU_BACKEND_HIP,
        flags: OCGPU_RTC_API_FLAG_CODE_IS_HIP_CODE_OBJECT,
        ..ocgpuHiprtcApi_v1::default()
    };
    resolve_hiprtc_raw(provider, &mut raw)?;
    // Nine common types match HIPRTC exactly. HIPRTC 5.7 spells the outer
    // CreateProgram/CompileProgram arrays mutable despite documenting them as
    // input-only, so those two slots use exact-signature shims below.
    let mut common = ocgpuRtcApi_v1 {
        struct_size: table_size::<ocgpuRtcApi_v1>()?,
        abi_version: OCGPU_ABI_VERSION_1,
        backend: OCGPU_BACKEND_HIP,
        flags: OCGPU_RTC_API_FLAG_CODE_IS_HIP_CODE_OBJECT,
        ocgpuRtcGetErrorString: resolve(provider, b"hiprtcGetErrorString")?,
        ocgpuRtcVersion: resolve(provider, b"hiprtcVersion")?,
        ocgpuRtcCreateProgram: raw
            .ocgpuHiprtcCreateProgram
            .map(|_| hiprtc_common_create_program as CreateProgramFn),
        ocgpuRtcDestroyProgram: resolve(provider, b"hiprtcDestroyProgram")?,
        ocgpuRtcCompileProgram: raw
            .ocgpuHiprtcCompileProgram
            .map(|_| hiprtc_common_compile_program as CompileProgramFn),
        ocgpuRtcGetProgramLogSize: resolve(provider, b"hiprtcGetProgramLogSize")?,
        ocgpuRtcGetProgramLog: resolve(provider, b"hiprtcGetProgramLog")?,
        ocgpuRtcAddNameExpression: resolve(provider, b"hiprtcAddNameExpression")?,
        ocgpuRtcGetLoweredName: resolve(provider, b"hiprtcGetLoweredName")?,
        ocgpuRtcGetCodeSize: resolve(provider, b"hiprtcGetCodeSize")?,
        ocgpuRtcGetCode: resolve(provider, b"hiprtcGetCode")?,
        ..ocgpuRtcApi_v1::default()
    };
    let core = CoreFns::from_table(CompilerKind::Hiprtc, &common)?;
    let (major, minor) = version_from_core(CompilerKind::Hiprtc, &core)?;
    common.rtc_version_major = major;
    common.rtc_version_minor = minor;
    raw.rtc_version_major = major;
    raw.rtc_version_minor = minor;
    Ok(ApiState {
        common_table: common,
        core,
        raw_table: raw,
        loaded_path: loaded_path.to_path_buf(),
    })
}

fn published_hiprtc_state() -> Option<&'static ApiState<ocgpuHiprtcApi_v1>> {
    HIPRTC_STATE.get()?.as_ref().ok()
}

unsafe extern "C" fn hiprtc_common_create_program(
    program: *mut ocgpuRtcProgram,
    source: *const c_char,
    name: *const c_char,
    header_count: i32,
    headers: *const *const c_char,
    include_names: *const *const c_char,
) -> ocgpuRtcResult {
    let Some(state) = published_hiprtc_state() else {
        return OCGPU_HIPRTC_ERROR_INTERNAL_ERROR;
    };
    // SAFETY: this table cannot escape until HIPRTC_STATE is fully published,
    // and the caller owns every pointer for the duration of the call.
    unsafe {
        hiprtc_create_program_from_state(
            state,
            program,
            source,
            name,
            header_count,
            headers,
            include_names,
        )
    }
}

unsafe fn hiprtc_create_program_from_state(
    state: &ApiState<ocgpuHiprtcApi_v1>,
    program: *mut ocgpuRtcProgram,
    source: *const c_char,
    name: *const c_char,
    header_count: i32,
    headers: *const *const c_char,
    include_names: *const *const c_char,
) -> ocgpuRtcResult {
    let Some(function) = state.raw_table.ocgpuHiprtcCreateProgram else {
        return OCGPU_HIPRTC_ERROR_INTERNAL_ERROR;
    };
    // SAFETY: HIPRTC 5.7.1 declares the outer arrays mutable but documents
    // both as `[in]`; only that qualification changes, and every pointer/count
    // is forwarded unchanged to the exact raw declaration.
    unsafe {
        function(
            program,
            source,
            name,
            header_count,
            headers.cast_mut(),
            include_names.cast_mut(),
        )
    }
}

unsafe extern "C" fn hiprtc_common_compile_program(
    program: ocgpuRtcProgram,
    option_count: i32,
    options: *const *const c_char,
) -> ocgpuRtcResult {
    let Some(state) = published_hiprtc_state() else {
        return OCGPU_HIPRTC_ERROR_INTERNAL_ERROR;
    };
    // SAFETY: this table cannot escape until HIPRTC_STATE is fully published,
    // and the caller owns every pointer for the duration of the call.
    unsafe { hiprtc_compile_program_from_state(state, program, option_count, options) }
}

unsafe fn hiprtc_compile_program_from_state(
    state: &ApiState<ocgpuHiprtcApi_v1>,
    program: ocgpuRtcProgram,
    option_count: i32,
    options: *const *const c_char,
) -> ocgpuRtcResult {
    let Some(function) = state.raw_table.ocgpuHiprtcCompileProgram else {
        return OCGPU_HIPRTC_ERROR_INTERNAL_ERROR;
    };
    // SAFETY: HIPRTC 5.7.1 documents `options` as `[in]`; only its outer const
    // qualification changes before the exact raw call.
    unsafe { function(program, option_count, options.cast_mut()) }
}

fn version_from_core(compiler: CompilerKind, core: &CoreFns) -> Result<(i32, i32), Error> {
    let mut major = 0;
    let mut minor = 0;
    // SAFETY: both outputs are writable for the duration of the call.
    let result = unsafe { (core.version)(&raw mut major, &raw mut minor) };
    if result == OCGPU_RTC_SUCCESS {
        Ok((major, minor))
    } else {
        let pointer = unsafe { (core.get_error_string)(result) };
        let message = if pointer.is_null() {
            "vendor returned a null error string".to_owned()
        } else {
            // SAFETY: vendor returns a static NUL-terminated error string.
            unsafe { CStr::from_ptr(pointer) }
                .to_string_lossy()
                .into_owned()
        };
        Err(Error::Rtc(RtcFailure {
            compiler,
            operation: "Version",
            result,
            message,
        }))
    }
}

trait SymbolProvider {
    fn find_symbol(&self, symbol: &[u8]) -> Result<Option<NonNull<c_void>>, Error>;
}

impl SymbolProvider for Library {
    fn find_symbol(&self, symbol: &[u8]) -> Result<Option<NonNull<c_void>>, Error> {
        Ok(self.find(symbol)?)
    }
}

fn resolve_nvrtc_raw<P: SymbolProvider>(
    library: &P,
    table: &mut ocgpuNvrtcApi_v1,
) -> Result<(), Error> {
    macro_rules! fields {
        ($($field:ident => $symbol:literal),+ $(,)?) => {$({
            table.$field = resolve(library, $symbol)?;
        })+};
    }
    fields! {
        ocgpuNvrtcGetErrorString => b"nvrtcGetErrorString",
        ocgpuNvrtcVersion => b"nvrtcVersion",
        ocgpuNvrtcCreateProgram => b"nvrtcCreateProgram",
        ocgpuNvrtcDestroyProgram => b"nvrtcDestroyProgram",
        ocgpuNvrtcCompileProgram => b"nvrtcCompileProgram",
        ocgpuNvrtcGetProgramLogSize => b"nvrtcGetProgramLogSize",
        ocgpuNvrtcGetProgramLog => b"nvrtcGetProgramLog",
        ocgpuNvrtcAddNameExpression => b"nvrtcAddNameExpression",
        ocgpuNvrtcGetLoweredName => b"nvrtcGetLoweredName",
        ocgpuNvrtcGetPTXSize => b"nvrtcGetPTXSize",
        ocgpuNvrtcGetPTX => b"nvrtcGetPTX",
        ocgpuNvrtcGetNumSupportedArchs => b"nvrtcGetNumSupportedArchs",
        ocgpuNvrtcGetSupportedArchs => b"nvrtcGetSupportedArchs",
        ocgpuNvrtcGetCUBINSize => b"nvrtcGetCUBINSize",
        ocgpuNvrtcGetCUBIN => b"nvrtcGetCUBIN",
        ocgpuNvrtcGetNVVMSize => b"nvrtcGetNVVMSize",
        ocgpuNvrtcGetNVVM => b"nvrtcGetNVVM",
        ocgpuNvrtcGetLTOIRSize => b"nvrtcGetLTOIRSize",
        ocgpuNvrtcGetLTOIR => b"nvrtcGetLTOIR",
        ocgpuNvrtcGetOptiXIRSize => b"nvrtcGetOptiXIRSize",
        ocgpuNvrtcGetOptiXIR => b"nvrtcGetOptiXIR",
    }
    Ok(())
}

fn resolve_hiprtc_raw<P: SymbolProvider>(
    library: &P,
    table: &mut ocgpuHiprtcApi_v1,
) -> Result<(), Error> {
    macro_rules! fields {
        ($($field:ident => $symbol:literal),+ $(,)?) => {$({
            table.$field = resolve(library, $symbol)?;
        })+};
    }
    fields! {
        ocgpuHiprtcGetErrorString => b"hiprtcGetErrorString",
        ocgpuHiprtcVersion => b"hiprtcVersion",
        ocgpuHiprtcCreateProgram => b"hiprtcCreateProgram",
        ocgpuHiprtcDestroyProgram => b"hiprtcDestroyProgram",
        ocgpuHiprtcCompileProgram => b"hiprtcCompileProgram",
        ocgpuHiprtcGetProgramLogSize => b"hiprtcGetProgramLogSize",
        ocgpuHiprtcGetProgramLog => b"hiprtcGetProgramLog",
        ocgpuHiprtcAddNameExpression => b"hiprtcAddNameExpression",
        ocgpuHiprtcGetLoweredName => b"hiprtcGetLoweredName",
        ocgpuHiprtcGetCodeSize => b"hiprtcGetCodeSize",
        ocgpuHiprtcGetCode => b"hiprtcGetCode",
        ocgpuHiprtcGetBitcodeSize => b"hiprtcGetBitcodeSize",
        ocgpuHiprtcGetBitcode => b"hiprtcGetBitcode",
        ocgpuHiprtcLinkCreate => b"hiprtcLinkCreate",
        ocgpuHiprtcLinkAddFile => b"hiprtcLinkAddFile",
        ocgpuHiprtcLinkAddData => b"hiprtcLinkAddData",
        ocgpuHiprtcLinkComplete => b"hiprtcLinkComplete",
        ocgpuHiprtcLinkDestroy => b"hiprtcLinkDestroy",
    }
    Ok(())
}

fn resolve<P: SymbolProvider, F: Copy>(library: &P, symbol: &[u8]) -> Result<Option<F>, Error> {
    let address = library.find_symbol(symbol)?;
    Ok(address.map(|address| {
        assert_eq!(size_of::<F>(), size_of::<*mut c_void>());
        // SAFETY: callers assign each fixed vendor symbol to its independently
        // declared exact FFI field type. Function and data pointers share the
        // supported 64-bit platform representation.
        unsafe { std::mem::transmute_copy::<*mut c_void, F>(&address.as_ptr()) }
    }))
}

fn table_size<T>() -> Result<u32, Error> {
    u32::try_from(size_of::<T>()).map_err(|_| Error::AllocationFailed {
        kind: "RTC ABI table metadata",
        bytes: size_of::<T>(),
    })
}

fn check_count(kind: &'static str, count: usize, limit: usize) -> Result<(), Error> {
    if count > limit {
        Err(Error::TooManyItems { kind, count, limit })
    } else {
        Ok(())
    }
}

fn check_input_size(kind: &'static str, input: &CStr, limit: usize) -> Result<(), Error> {
    let bytes = input.to_bytes().len();
    if bytes > limit {
        Err(Error::InputTooLarge { kind, bytes, limit })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ocgpu_abi::{
        OCGPU_HIPRTC_API_V1_SLOT_COUNT, OCGPU_NVRTC_API_V1_SLOT_COUNT, OCGPU_RTC_API_V1_SLOT_COUNT,
        OCGPU_RTC_ERROR_COMPILATION, OCGPU_RTC_ERROR_INVALID_INPUT,
    };
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicI32, AtomicU32, AtomicUsize, Ordering};

    const CALL_ERROR_STRING: u32 = 1 << 0;
    const CALL_VERSION: u32 = 1 << 1;
    const CALL_CREATE: u32 = 1 << 2;
    const CALL_DESTROY: u32 = 1 << 3;
    const CALL_COMPILE: u32 = 1 << 4;
    const CALL_LOG_SIZE: u32 = 1 << 5;
    const CALL_LOG: u32 = 1 << 6;
    const CALL_ADD_NAME: u32 = 1 << 7;
    const CALL_LOWERED_NAME: u32 = 1 << 8;
    const CALL_CODE_SIZE: u32 = 1 << 9;
    const CALL_CODE: u32 = 1 << 10;
    const ALL_COMMON_CALLS: u32 = (1 << 11) - 1;
    const MOCK_LOG: &[u8] = b"mock compiler log\0";
    const MOCK_CODE: &[u8] = b"PTX\0";
    const MOCK_LOWERED_NAME: &[u8] = b"_Z11mock_kernelv\0";
    const MOCK_ERROR: &[u8] = b"mock RTC error\0";

    static TEST_LOCK: Mutex<()> = Mutex::new(());
    static CALLS: AtomicU32 = AtomicU32::new(0);
    static DESTROY_CALLS: AtomicUsize = AtomicUsize::new(0);
    static CREATE_RESULT: AtomicI32 = AtomicI32::new(OCGPU_RTC_SUCCESS);
    static COMPILE_RESULT: AtomicI32 = AtomicI32::new(OCGPU_RTC_SUCCESS);
    static CODE_SIZE: AtomicUsize = AtomicUsize::new(MOCK_CODE.len());
    static LOG_SIZE: AtomicUsize = AtomicUsize::new(MOCK_LOG.len());
    static HIP_HEADER_COUNT: AtomicI32 = AtomicI32::new(0);
    static HIP_HEADERS_POINTER: AtomicUsize = AtomicUsize::new(0);
    static HIP_INCLUDE_NAMES_POINTER: AtomicUsize = AtomicUsize::new(0);
    static HIP_OPTION_COUNT: AtomicI32 = AtomicI32::new(0);
    static HIP_OPTIONS_POINTER: AtomicUsize = AtomicUsize::new(0);

    unsafe extern "C" fn mock_get_error_string(_result: ocgpuRtcResult) -> *const c_char {
        CALLS.fetch_or(CALL_ERROR_STRING, Ordering::SeqCst);
        MOCK_ERROR.as_ptr().cast()
    }

    unsafe extern "C" fn mock_version(major: *mut i32, minor: *mut i32) -> ocgpuRtcResult {
        CALLS.fetch_or(CALL_VERSION, Ordering::SeqCst);
        // SAFETY: tests pass valid writable output pointers.
        unsafe {
            *major = 12;
            *minor = 4;
        }
        OCGPU_RTC_SUCCESS
    }

    unsafe extern "C" fn mock_create_program(
        program: *mut ocgpuRtcProgram,
        _source: *const c_char,
        _name: *const c_char,
        _header_count: i32,
        _headers: *const *const c_char,
        _include_names: *const *const c_char,
    ) -> ocgpuRtcResult {
        CALLS.fetch_or(CALL_CREATE, Ordering::SeqCst);
        // SAFETY: tests pass a valid writable output pointer. The nonzero
        // address is an opaque token and is never dereferenced.
        unsafe {
            *program = std::ptr::without_provenance_mut(1);
        }
        CREATE_RESULT.load(Ordering::SeqCst)
    }

    unsafe extern "C" fn mock_destroy_program(program: *mut ocgpuRtcProgram) -> ocgpuRtcResult {
        CALLS.fetch_or(CALL_DESTROY, Ordering::SeqCst);
        DESTROY_CALLS.fetch_add(1, Ordering::SeqCst);
        // SAFETY: tests pass the owned handle slot.
        unsafe {
            *program = std::ptr::null_mut();
        }
        OCGPU_RTC_SUCCESS
    }

    unsafe extern "C" fn mock_compile_program(
        _program: ocgpuRtcProgram,
        _option_count: i32,
        _options: *const *const c_char,
    ) -> ocgpuRtcResult {
        CALLS.fetch_or(CALL_COMPILE, Ordering::SeqCst);
        COMPILE_RESULT.load(Ordering::SeqCst)
    }

    unsafe extern "C" fn mock_hip_create_program(
        program: *mut ocgpuRtcProgram,
        _source: *const c_char,
        _name: *const c_char,
        header_count: i32,
        headers: *mut *const c_char,
        include_names: *mut *const c_char,
    ) -> ocgpuRtcResult {
        CALLS.fetch_or(CALL_CREATE, Ordering::SeqCst);
        HIP_HEADER_COUNT.store(header_count, Ordering::SeqCst);
        HIP_HEADERS_POINTER.store(headers.addr(), Ordering::SeqCst);
        HIP_INCLUDE_NAMES_POINTER.store(include_names.addr(), Ordering::SeqCst);
        // SAFETY: tests pass a valid writable output slot; the token is opaque.
        unsafe {
            *program = std::ptr::without_provenance_mut(1);
        }
        CREATE_RESULT.load(Ordering::SeqCst)
    }

    unsafe extern "C" fn mock_hip_compile_program(
        _program: ocgpuRtcProgram,
        option_count: i32,
        options: *mut *const c_char,
    ) -> ocgpuRtcResult {
        CALLS.fetch_or(CALL_COMPILE, Ordering::SeqCst);
        HIP_OPTION_COUNT.store(option_count, Ordering::SeqCst);
        HIP_OPTIONS_POINTER.store(options.addr(), Ordering::SeqCst);
        COMPILE_RESULT.load(Ordering::SeqCst)
    }

    unsafe extern "C" fn mock_get_log_size(
        _program: ocgpuRtcProgram,
        bytes: *mut usize,
    ) -> ocgpuRtcResult {
        CALLS.fetch_or(CALL_LOG_SIZE, Ordering::SeqCst);
        // SAFETY: tests pass a valid writable output pointer.
        unsafe {
            *bytes = LOG_SIZE.load(Ordering::SeqCst);
        }
        OCGPU_RTC_SUCCESS
    }

    unsafe extern "C" fn mock_get_log(
        _program: ocgpuRtcProgram,
        output: *mut c_char,
    ) -> ocgpuRtcResult {
        CALLS.fetch_or(CALL_LOG, Ordering::SeqCst);
        // SAFETY: the safe layer allocated the size returned by the paired mock.
        unsafe {
            std::ptr::copy_nonoverlapping(MOCK_LOG.as_ptr(), output.cast(), MOCK_LOG.len());
        }
        OCGPU_RTC_SUCCESS
    }

    unsafe extern "C" fn mock_add_name_expression(
        _program: ocgpuRtcProgram,
        _expression: *const c_char,
    ) -> ocgpuRtcResult {
        CALLS.fetch_or(CALL_ADD_NAME, Ordering::SeqCst);
        OCGPU_RTC_SUCCESS
    }

    unsafe extern "C" fn mock_get_lowered_name(
        _program: ocgpuRtcProgram,
        _expression: *const c_char,
        output: *mut *const c_char,
    ) -> ocgpuRtcResult {
        CALLS.fetch_or(CALL_LOWERED_NAME, Ordering::SeqCst);
        // SAFETY: tests pass a valid writable output pointer.
        unsafe {
            *output = MOCK_LOWERED_NAME.as_ptr().cast();
        }
        OCGPU_RTC_SUCCESS
    }

    unsafe extern "C" fn mock_get_code_size(
        _program: ocgpuRtcProgram,
        bytes: *mut usize,
    ) -> ocgpuRtcResult {
        CALLS.fetch_or(CALL_CODE_SIZE, Ordering::SeqCst);
        // SAFETY: tests pass a valid writable output pointer.
        unsafe {
            *bytes = CODE_SIZE.load(Ordering::SeqCst);
        }
        OCGPU_RTC_SUCCESS
    }

    unsafe extern "C" fn mock_get_code(
        _program: ocgpuRtcProgram,
        output: *mut c_char,
    ) -> ocgpuRtcResult {
        CALLS.fetch_or(CALL_CODE, Ordering::SeqCst);
        // SAFETY: the safe layer allocated the size returned by the paired mock.
        unsafe {
            std::ptr::copy_nonoverlapping(MOCK_CODE.as_ptr(), output.cast(), MOCK_CODE.len());
        }
        OCGPU_RTC_SUCCESS
    }

    unsafe extern "C" fn mock_erased() {}

    #[derive(Clone, Copy)]
    struct MockProvider {
        missing: Option<&'static [u8]>,
    }

    impl MockProvider {
        const fn all() -> Self {
            Self { missing: None }
        }

        const fn missing(symbol: &'static [u8]) -> Self {
            Self {
                missing: Some(symbol),
            }
        }
    }

    impl SymbolProvider for MockProvider {
        fn find_symbol(&self, symbol: &[u8]) -> Result<Option<NonNull<c_void>>, Error> {
            if self.missing.is_some_and(|missing| missing == symbol) {
                return Ok(None);
            }
            let pointer = if symbol.ends_with(b"GetErrorString") {
                mock_get_error_string as *const ()
            } else if symbol.ends_with(b"Version") {
                mock_version as *const ()
            } else if symbol == b"hiprtcCreateProgram" {
                mock_hip_create_program as *const ()
            } else if symbol.ends_with(b"CreateProgram") {
                mock_create_program as *const ()
            } else if symbol.ends_with(b"DestroyProgram") {
                mock_destroy_program as *const ()
            } else if symbol == b"hiprtcCompileProgram" {
                mock_hip_compile_program as *const ()
            } else if symbol.ends_with(b"CompileProgram") {
                mock_compile_program as *const ()
            } else if symbol.ends_with(b"GetProgramLogSize") {
                mock_get_log_size as *const ()
            } else if symbol.ends_with(b"GetProgramLog") {
                mock_get_log as *const ()
            } else if symbol.ends_with(b"AddNameExpression") {
                mock_add_name_expression as *const ()
            } else if symbol.ends_with(b"GetLoweredName") {
                mock_get_lowered_name as *const ()
            } else if symbol.ends_with(b"GetPTXSize") || symbol.ends_with(b"GetCodeSize") {
                mock_get_code_size as *const ()
            } else if symbol.ends_with(b"GetPTX") || symbol.ends_with(b"GetCode") {
                mock_get_code as *const ()
            } else {
                mock_erased as *const ()
            };
            let pointer = pointer.cast_mut().cast::<c_void>();
            Ok(NonNull::new(pointer))
        }
    }

    fn mock_common_table() -> ocgpuRtcApi_v1 {
        ocgpuRtcApi_v1 {
            struct_size: u32::try_from(size_of::<ocgpuRtcApi_v1>()).expect("table fits u32"),
            abi_version: OCGPU_ABI_VERSION_1,
            backend: OCGPU_BACKEND_CUDA,
            flags: OCGPU_RTC_API_FLAG_CODE_IS_PTX,
            rtc_version_major: 12,
            rtc_version_minor: 4,
            ocgpuRtcGetErrorString: Some(mock_get_error_string),
            ocgpuRtcVersion: Some(mock_version),
            ocgpuRtcCreateProgram: Some(mock_create_program),
            ocgpuRtcDestroyProgram: Some(mock_destroy_program),
            ocgpuRtcCompileProgram: Some(mock_compile_program),
            ocgpuRtcGetProgramLogSize: Some(mock_get_log_size),
            ocgpuRtcGetProgramLog: Some(mock_get_log),
            ocgpuRtcAddNameExpression: Some(mock_add_name_expression),
            ocgpuRtcGetLoweredName: Some(mock_get_lowered_name),
            ocgpuRtcGetCodeSize: Some(mock_get_code_size),
            ocgpuRtcGetCode: Some(mock_get_code),
        }
    }

    fn mock_compiler() -> Compiler<Nvrtc> {
        let common_table = Box::leak(Box::new(mock_common_table()));
        let core = Box::leak(Box::new(
            CoreFns::from_table(CompilerKind::Nvrtc, common_table)
                .expect("mock common table is complete"),
        ));
        let raw_table = Box::leak(Box::new(ocgpuNvrtcApi_v1::default()));
        Compiler {
            core,
            common_table,
            raw_table,
            loaded_path: Path::new("mock-nvrtc"),
            marker: PhantomData,
        }
    }

    fn reset_mocks() {
        CALLS.store(0, Ordering::SeqCst);
        DESTROY_CALLS.store(0, Ordering::SeqCst);
        CREATE_RESULT.store(OCGPU_RTC_SUCCESS, Ordering::SeqCst);
        COMPILE_RESULT.store(OCGPU_RTC_SUCCESS, Ordering::SeqCst);
        CODE_SIZE.store(MOCK_CODE.len(), Ordering::SeqCst);
        LOG_SIZE.store(MOCK_LOG.len(), Ordering::SeqCst);
        HIP_HEADER_COUNT.store(0, Ordering::SeqCst);
        HIP_HEADERS_POINTER.store(0, Ordering::SeqCst);
        HIP_INCLUDE_NAMES_POINTER.store(0, Ordering::SeqCst);
        HIP_OPTION_COUNT.store(0, Ordering::SeqCst);
        HIP_OPTIONS_POINTER.store(0, Ordering::SeqCst);
    }

    #[test]
    fn core_validation_is_atomic_and_aggregates_all_missing_slots() {
        let error = CoreFns::from_table(CompilerKind::Nvrtc, &ocgpuRtcApi_v1::default())
            .expect_err("empty table must fail");
        let Error::MissingRequiredSymbols { compiler, symbols } = error else {
            panic!("unexpected error variant")
        };
        assert_eq!(compiler, CompilerKind::Nvrtc);
        assert_eq!(symbols.len(), OCGPU_RTC_API_V1_SLOT_COUNT as usize);
        assert!(symbols.contains(&"nvrtcGetPTX"));

        let core = CoreFns::from_table(CompilerKind::Nvrtc, &mock_common_table())
            .expect("complete table must validate");
        assert_eq!(core.version as usize, mock_version as *const () as usize);
        assert_eq!(core.get_code as usize, mock_get_code as *const () as usize);
    }

    #[test]
    fn failed_create_never_assumes_ownership_of_a_nonnull_output() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset_mocks();
        CREATE_RESULT.store(OCGPU_RTC_ERROR_INVALID_INPUT, Ordering::SeqCst);
        let error = mock_compiler()
            .create_program(c"source", None, &[])
            .expect_err("mock create must fail");
        assert!(matches!(
            error,
            Error::Rtc(RtcFailure {
                operation: "CreateProgram",
                result: OCGPU_RTC_ERROR_INVALID_INPUT,
                ..
            })
        ));
        assert_eq!(DESTROY_CALLS.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn raw_mock_resolvers_cover_every_pinned_slot_and_preserve_optional_missing() {
        let mut nvrtc = ocgpuNvrtcApi_v1::default();
        resolve_nvrtc_raw(&MockProvider::all(), &mut nvrtc).expect("NVRTC mock resolve");
        let nvrtc_present = [
            nvrtc.ocgpuNvrtcGetErrorString.is_some(),
            nvrtc.ocgpuNvrtcVersion.is_some(),
            nvrtc.ocgpuNvrtcCreateProgram.is_some(),
            nvrtc.ocgpuNvrtcDestroyProgram.is_some(),
            nvrtc.ocgpuNvrtcCompileProgram.is_some(),
            nvrtc.ocgpuNvrtcGetProgramLogSize.is_some(),
            nvrtc.ocgpuNvrtcGetProgramLog.is_some(),
            nvrtc.ocgpuNvrtcAddNameExpression.is_some(),
            nvrtc.ocgpuNvrtcGetLoweredName.is_some(),
            nvrtc.ocgpuNvrtcGetPTXSize.is_some(),
            nvrtc.ocgpuNvrtcGetPTX.is_some(),
            nvrtc.ocgpuNvrtcGetNumSupportedArchs.is_some(),
            nvrtc.ocgpuNvrtcGetSupportedArchs.is_some(),
            nvrtc.ocgpuNvrtcGetCUBINSize.is_some(),
            nvrtc.ocgpuNvrtcGetCUBIN.is_some(),
            nvrtc.ocgpuNvrtcGetNVVMSize.is_some(),
            nvrtc.ocgpuNvrtcGetNVVM.is_some(),
            nvrtc.ocgpuNvrtcGetLTOIRSize.is_some(),
            nvrtc.ocgpuNvrtcGetLTOIR.is_some(),
            nvrtc.ocgpuNvrtcGetOptiXIRSize.is_some(),
            nvrtc.ocgpuNvrtcGetOptiXIR.is_some(),
        ];
        assert_eq!(
            nvrtc_present.into_iter().filter(|present| *present).count(),
            OCGPU_NVRTC_API_V1_SLOT_COUNT as usize
        );

        let mut hiprtc = ocgpuHiprtcApi_v1::default();
        resolve_hiprtc_raw(&MockProvider::all(), &mut hiprtc).expect("HIPRTC mock resolve");
        let hiprtc_present = [
            hiprtc.ocgpuHiprtcGetErrorString.is_some(),
            hiprtc.ocgpuHiprtcVersion.is_some(),
            hiprtc.ocgpuHiprtcCreateProgram.is_some(),
            hiprtc.ocgpuHiprtcDestroyProgram.is_some(),
            hiprtc.ocgpuHiprtcCompileProgram.is_some(),
            hiprtc.ocgpuHiprtcGetProgramLogSize.is_some(),
            hiprtc.ocgpuHiprtcGetProgramLog.is_some(),
            hiprtc.ocgpuHiprtcAddNameExpression.is_some(),
            hiprtc.ocgpuHiprtcGetLoweredName.is_some(),
            hiprtc.ocgpuHiprtcGetCodeSize.is_some(),
            hiprtc.ocgpuHiprtcGetCode.is_some(),
            hiprtc.ocgpuHiprtcGetBitcodeSize.is_some(),
            hiprtc.ocgpuHiprtcGetBitcode.is_some(),
            hiprtc.ocgpuHiprtcLinkCreate.is_some(),
            hiprtc.ocgpuHiprtcLinkAddFile.is_some(),
            hiprtc.ocgpuHiprtcLinkAddData.is_some(),
            hiprtc.ocgpuHiprtcLinkComplete.is_some(),
            hiprtc.ocgpuHiprtcLinkDestroy.is_some(),
        ];
        assert_eq!(
            hiprtc_present
                .into_iter()
                .filter(|present| *present)
                .count(),
            OCGPU_HIPRTC_API_V1_SLOT_COUNT as usize
        );

        let mut optional = ocgpuNvrtcApi_v1::default();
        resolve_nvrtc_raw(&MockProvider::missing(b"nvrtcGetCUBIN"), &mut optional)
            .expect("optional missing symbol is not an error");
        assert!(optional.ocgpuNvrtcGetCUBIN.is_none());
        assert!(optional.ocgpuNvrtcGetPTX.is_some());
    }

    #[test]
    fn built_tables_publish_abi_version_code_flags_and_all_or_nothing_common() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset_mocks();
        let nvrtc = build_nvrtc_state_from(&MockProvider::all(), Path::new("mock-nvrtc"))
            .expect("complete NVRTC provider");
        assert_eq!(nvrtc.common_table.abi_version, OCGPU_ABI_VERSION_1);
        assert_eq!(nvrtc.raw_table.abi_version, OCGPU_ABI_VERSION_1);
        assert_eq!(nvrtc.common_table.flags, OCGPU_RTC_API_FLAG_CODE_IS_PTX);
        assert_eq!(nvrtc.raw_table.flags, OCGPU_RTC_API_FLAG_CODE_IS_PTX);

        let hiprtc = build_hiprtc_state_from(&MockProvider::all(), Path::new("mock-hiprtc"))
            .expect("complete HIPRTC provider");
        assert_eq!(hiprtc.common_table.abi_version, OCGPU_ABI_VERSION_1);
        assert_eq!(hiprtc.raw_table.abi_version, OCGPU_ABI_VERSION_1);
        assert_eq!(
            hiprtc.common_table.flags,
            OCGPU_RTC_API_FLAG_CODE_IS_HIP_CODE_OBJECT
        );
        assert_eq!(
            hiprtc.raw_table.flags,
            OCGPU_RTC_API_FLAG_CODE_IS_HIP_CODE_OBJECT
        );
        assert_eq!(
            hiprtc
                .common_table
                .ocgpuRtcCreateProgram
                .expect("common create shim") as *const () as usize,
            hiprtc_common_create_program as *const () as usize
        );
        assert_eq!(
            hiprtc
                .common_table
                .ocgpuRtcCompileProgram
                .expect("common compile shim") as *const () as usize,
            hiprtc_common_compile_program as *const () as usize
        );
        assert_eq!(
            hiprtc
                .raw_table
                .ocgpuHiprtcCreateProgram
                .expect("exact raw create") as *const () as usize,
            mock_hip_create_program as *const () as usize
        );
        assert_eq!(
            hiprtc
                .raw_table
                .ocgpuHiprtcCompileProgram
                .expect("exact raw compile") as *const () as usize,
            mock_hip_compile_program as *const () as usize
        );

        let Err(failure) = build_hiprtc_state_from(
            &MockProvider::missing(b"hiprtcGetCode"),
            Path::new("incomplete"),
        ) else {
            panic!("mandatory missing symbol must fail the whole common table")
        };
        assert_eq!(failure.as_ocgpu_result(), OCGPU_ERROR_SYMBOL_UNAVAILABLE);
    }

    #[test]
    fn hip_const_qualification_adapters_forward_arrays_unchanged() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset_mocks();
        let state = build_hiprtc_state_from(&MockProvider::all(), Path::new("mock-hiprtc"))
            .expect("complete HIPRTC provider");
        let headers = [c"#define VALUE 7".as_ptr()];
        let include_names = [c"value.h".as_ptr()];
        let options = [c"--gpu-architecture=gfx90c".as_ptr()];
        let mut program = std::ptr::null_mut();
        // SAFETY: every C string, pointer array, and output slot is live for
        // these exact mock calls.
        let create_result = unsafe {
            hiprtc_create_program_from_state(
                &state,
                &raw mut program,
                c"source".as_ptr(),
                c"mock.hip".as_ptr(),
                1,
                headers.as_ptr(),
                include_names.as_ptr(),
            )
        };
        assert_eq!(create_result, OCGPU_RTC_SUCCESS);
        assert_eq!(HIP_HEADER_COUNT.load(Ordering::SeqCst), 1);
        assert_eq!(
            HIP_HEADERS_POINTER.load(Ordering::SeqCst),
            headers.as_ptr().addr()
        );
        assert_eq!(
            HIP_INCLUDE_NAMES_POINTER.load(Ordering::SeqCst),
            include_names.as_ptr().addr()
        );
        // SAFETY: the mock program token and option pointer remain valid.
        let compile_result =
            unsafe { hiprtc_compile_program_from_state(&state, program, 1, options.as_ptr()) };
        assert_eq!(compile_result, OCGPU_RTC_SUCCESS);
        assert_eq!(HIP_OPTION_COUNT.load(Ordering::SeqCst), 1);
        assert_eq!(
            HIP_OPTIONS_POINTER.load(Ordering::SeqCst),
            options.as_ptr().addr()
        );
    }

    #[test]
    fn safe_success_path_exercises_all_eleven_common_calls() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset_mocks();
        let compiler = mock_compiler();
        assert_eq!(compiler.version().expect("version"), (12, 4));
        assert_eq!(
            compiler
                .error_string(OCGPU_RTC_ERROR_COMPILATION)
                .as_deref(),
            Some("mock RTC error")
        );
        let header = Header::new(c"#define VALUE 7", c"value.h");
        let mut program = compiler
            .create_program(c"#include <value.h>", Some(c"mock.cu"), &[header])
            .expect("create");
        program
            .add_name_expression(c"mock_kernel")
            .expect("name expression");
        program.compile(&[c"--std=c++17"]).expect("compile");
        assert_eq!(
            program
                .lowered_name(c"mock_kernel")
                .expect("lowered name")
                .as_c_str(),
            c"_Z11mock_kernelv"
        );
        assert_eq!(program.code().expect("code"), MOCK_CODE);
        assert_eq!(program.log().expect("log"), MOCK_LOG);
        program.destroy().expect("destroy");
        assert_eq!(CALLS.load(Ordering::SeqCst), ALL_COMMON_CALLS);
        assert_eq!(DESTROY_CALLS.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn compile_failure_contains_raw_result_and_bounded_log() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset_mocks();
        COMPILE_RESULT.store(OCGPU_RTC_ERROR_COMPILATION, Ordering::SeqCst);
        let mut program = mock_compiler()
            .create_program(c"invalid source", None, &[])
            .expect("create");
        let error = program.compile(&[]).expect_err("compile must fail");
        let Error::Compile(failure) = error else {
            panic!("expected compilation failure")
        };
        assert_eq!(failure.rtc.result, OCGPU_RTC_ERROR_COMPILATION);
        assert_eq!(failure.log, MOCK_LOG);
        assert!(failure.log_error.is_none());
        assert!(program.compile_attempted());
        assert!(!program.is_compiled());
        assert!(matches!(
            program.add_name_expression(c"too_late"),
            Err(Error::InvalidState {
                operation: "AddNameExpression",
                ..
            })
        ));
        program.destroy().expect("destroy");
    }

    #[test]
    fn oversized_compile_log_is_rejected_without_copying() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset_mocks();
        COMPILE_RESULT.store(OCGPU_RTC_ERROR_COMPILATION, Ordering::SeqCst);
        LOG_SIZE.store(MAX_LOG_BYTES + 1, Ordering::SeqCst);
        let mut program = mock_compiler()
            .create_program(c"invalid source", None, &[])
            .expect("create");
        let Error::Compile(failure) = program.compile(&[]).expect_err("compile must fail") else {
            panic!("expected compilation failure")
        };
        assert!(failure.log.is_empty());
        assert!(matches!(
            failure.log_error.as_deref(),
            Some(Error::OutputTooLarge {
                kind: "program log",
                bytes,
                limit: MAX_LOG_BYTES,
            }) if *bytes == MAX_LOG_BYTES + 1
        ));
        assert_eq!(CALLS.load(Ordering::SeqCst) & CALL_LOG, 0);
        program.destroy().expect("destroy");
    }

    #[test]
    fn state_machine_and_output_limit_prevent_invalid_vendor_calls() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset_mocks();
        let mut program = mock_compiler()
            .create_program(c"source", None, &[])
            .expect("create");
        assert!(matches!(program.code(), Err(Error::InvalidState { .. })));
        assert!(matches!(
            program.lowered_name(c"kernel"),
            Err(Error::InvalidState { .. })
        ));
        program.compile(&[]).expect("compile");
        assert!(matches!(
            program.add_name_expression(c"late"),
            Err(Error::InvalidState { .. })
        ));
        CODE_SIZE.store(MAX_CODE_BYTES + 1, Ordering::SeqCst);
        assert!(matches!(
            program.code(),
            Err(Error::OutputTooLarge {
                kind: "loadable code",
                bytes,
                limit: MAX_CODE_BYTES,
            }) if bytes == MAX_CODE_BYTES + 1
        ));
        assert_eq!(CALLS.load(Ordering::SeqCst) & CALL_CODE, 0);
        program.destroy().expect("destroy");
    }

    #[test]
    fn default_limits_are_conservative_but_explicit_policy_can_scale() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset_mocks();
        let compiler = mock_compiler();
        let oversized_source =
            CString::new(vec![b'x'; MAX_SOURCE_BYTES + 1]).expect("source contains no NUL");
        assert!(matches!(
            compiler.create_program(&oversized_source, None, &[]),
            Err(Error::InputTooLarge { kind: "source", .. })
        ));

        let oversized_name =
            CString::new(vec![b'n'; MAX_SOURCE_BYTES + 1]).expect("name contains no NUL");
        assert!(matches!(
            compiler.create_program(c"source", Some(&oversized_name), &[]),
            Err(Error::InputTooLarge {
                kind: "program name",
                ..
            })
        ));

        let headers = vec![Header::new(c"", c"h"); MAX_HEADERS + 1];
        assert!(matches!(
            compiler.create_program(c"source", None, &headers),
            Err(Error::TooManyItems {
                kind: "headers",
                ..
            })
        ));

        let custom = Limits {
            max_source_bytes: MAX_SOURCE_BYTES + 1,
            max_options: MAX_OPTIONS + 1,
            max_headers: MAX_HEADERS + 1,
            ..Limits::default()
        };
        let options = vec![c"-DVALUE=1"; MAX_OPTIONS + 1];
        let mut default_program = compiler
            .create_program(c"source", None, &[])
            .expect("default program");
        assert!(matches!(
            default_program.compile(&options),
            Err(Error::TooManyItems {
                kind: "options",
                ..
            })
        ));
        default_program
            .compile_with_limits(&options, custom)
            .expect("one-call larger option policy");
        default_program.destroy().expect("destroy default program");

        let mut program = compiler
            .create_program_with_limits(&oversized_source, None, &headers, custom)
            .expect("explicit larger policy");
        program.compile(&options).expect("attached larger policy");
        program.destroy().expect("destroy");
    }

    #[test]
    fn explicit_destroy_and_drop_each_invoke_destroy_exactly_once() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset_mocks();
        let program = mock_compiler()
            .create_program(c"source", None, &[])
            .expect("create");
        program.destroy().expect("explicit destroy");
        assert_eq!(DESTROY_CALLS.load(Ordering::SeqCst), 1);

        let program = mock_compiler()
            .create_program(c"source", None, &[])
            .expect("create");
        drop(program);
        assert_eq!(DESTROY_CALLS.load(Ordering::SeqCst), 2);
    }
}
