// SPDX-License-Identifier: CC0-1.0

//! SDK-independent runtime-compilation ABI declarations.
//!
//! Function-table entries are nullable for C ABI size negotiation. A producer
//! must populate all eleven entries in [`ocgpuRtcApi_v1`] before reporting a
//! successful common-table negotiation.

use core::ffi::{c_char, c_void};

use crate::{ocgpuBackend, ocgpuResult};

/// Backend-neutral runtime-compilation result code.
pub type ocgpuRtcResult = i32;

/// Opaque backend-neutral runtime-compilation program handle.
pub type ocgpuRtcProgram = *mut c_void;

/// Opaque backend-neutral runtime-compilation link-state handle.
pub type ocgpuRtcLinkState = *mut c_void;

/// Backend-neutral runtime-compilation linker input kind.
pub type ocgpuRtcJitInputType = i32;

/// Backend-neutral runtime-compilation linker option kind.
pub type ocgpuRtcJitOption = i32;

/// Raw NVRTC result code.
pub type ocgpuNvrtcResult = ocgpuRtcResult;

/// Raw NVRTC program handle.
pub type ocgpuNvrtcProgram = ocgpuRtcProgram;

/// Raw HIPRTC result code.
pub type ocgpuHiprtcResult = ocgpuRtcResult;

/// Raw HIPRTC program handle.
pub type ocgpuHiprtcProgram = ocgpuRtcProgram;

/// Raw HIPRTC link-state handle.
pub type ocgpuHiprtcLinkState = ocgpuRtcLinkState;

/// Raw HIPRTC linker input kind.
pub type ocgpuHiprtcJitInputType = ocgpuRtcJitInputType;

/// Raw HIPRTC linker option kind.
pub type ocgpuHiprtcJitOption = ocgpuRtcJitOption;

/// Size shared by the metadata prefix of every RTC ABI v1 table.
pub const OCGPU_RTC_TABLE_PREFIX_SIZE: u32 = 24;

/// Number of mandatory backend-neutral RTC ABI v1 function slots.
pub const OCGPU_RTC_API_V1_SLOT_COUNT: u32 = 11;

/// Number of public NVRTC 12.4 function slots.
pub const OCGPU_NVRTC_API_V1_SLOT_COUNT: u32 = 21;

/// Number of public HIPRTC 5.7.1 function slots.
pub const OCGPU_HIPRTC_API_V1_SLOT_COUNT: u32 = 18;

/// Common code getters return NVIDIA PTX text when this bit is set.
///
/// Mutually exclusive with [`OCGPU_RTC_API_FLAG_CODE_IS_HIP_CODE_OBJECT`].
pub const OCGPU_RTC_API_FLAG_CODE_IS_PTX: u32 = 1 << 0;

/// Common code getters return a HIP loadable code-object image when this bit is set.
///
/// Mutually exclusive with [`OCGPU_RTC_API_FLAG_CODE_IS_PTX`].
pub const OCGPU_RTC_API_FLAG_CODE_IS_HIP_CODE_OBJECT: u32 = 1 << 1;

/// Successful runtime compilation.
pub const OCGPU_RTC_SUCCESS: ocgpuRtcResult = 0;
/// Runtime compiler allocation failed.
pub const OCGPU_RTC_ERROR_OUT_OF_MEMORY: ocgpuRtcResult = 1;
/// Runtime compiler program creation failed.
pub const OCGPU_RTC_ERROR_PROGRAM_CREATION_FAILURE: ocgpuRtcResult = 2;
/// A runtime compiler input was invalid.
pub const OCGPU_RTC_ERROR_INVALID_INPUT: ocgpuRtcResult = 3;
/// A runtime compiler program handle was invalid.
pub const OCGPU_RTC_ERROR_INVALID_PROGRAM: ocgpuRtcResult = 4;
/// A runtime compiler option was invalid.
pub const OCGPU_RTC_ERROR_INVALID_OPTION: ocgpuRtcResult = 5;
/// Device-code compilation failed.
pub const OCGPU_RTC_ERROR_COMPILATION: ocgpuRtcResult = 6;
/// A runtime compiler builtin operation failed.
pub const OCGPU_RTC_ERROR_BUILTIN_OPERATION_FAILURE: ocgpuRtcResult = 7;
/// A name expression was added after compilation.
pub const OCGPU_RTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION: ocgpuRtcResult = 8;
/// A lowered name was requested before compilation.
pub const OCGPU_RTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION: ocgpuRtcResult = 9;
/// A name expression was not valid.
pub const OCGPU_RTC_ERROR_NAME_EXPRESSION_NOT_VALID: ocgpuRtcResult = 10;
/// An internal runtime compiler error occurred.
pub const OCGPU_RTC_ERROR_INTERNAL_ERROR: ocgpuRtcResult = 11;

/// Exact NVRTC 12.4 success code.
pub const OCGPU_NVRTC_SUCCESS: ocgpuNvrtcResult = 0;
/// Exact NVRTC 12.4 out-of-memory code.
pub const OCGPU_NVRTC_ERROR_OUT_OF_MEMORY: ocgpuNvrtcResult = 1;
/// Exact NVRTC 12.4 program-creation-failure code.
pub const OCGPU_NVRTC_ERROR_PROGRAM_CREATION_FAILURE: ocgpuNvrtcResult = 2;
/// Exact NVRTC 12.4 invalid-input code.
pub const OCGPU_NVRTC_ERROR_INVALID_INPUT: ocgpuNvrtcResult = 3;
/// Exact NVRTC 12.4 invalid-program code.
pub const OCGPU_NVRTC_ERROR_INVALID_PROGRAM: ocgpuNvrtcResult = 4;
/// Exact NVRTC 12.4 invalid-option code.
pub const OCGPU_NVRTC_ERROR_INVALID_OPTION: ocgpuNvrtcResult = 5;
/// Exact NVRTC 12.4 compilation-failure code.
pub const OCGPU_NVRTC_ERROR_COMPILATION: ocgpuNvrtcResult = 6;
/// Exact NVRTC 12.4 builtin-operation-failure code.
pub const OCGPU_NVRTC_ERROR_BUILTIN_OPERATION_FAILURE: ocgpuNvrtcResult = 7;
/// Exact NVRTC 12.4 late-name-expression code.
pub const OCGPU_NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION: ocgpuNvrtcResult = 8;
/// Exact NVRTC 12.4 early-lowered-name code.
pub const OCGPU_NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION: ocgpuNvrtcResult = 9;
/// Exact NVRTC 12.4 invalid-name-expression code.
pub const OCGPU_NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID: ocgpuNvrtcResult = 10;
/// Exact NVRTC 12.4 internal-error code.
pub const OCGPU_NVRTC_ERROR_INTERNAL_ERROR: ocgpuNvrtcResult = 11;
/// Exact NVRTC 12.4 time-file-write-failure code.
pub const OCGPU_NVRTC_ERROR_TIME_FILE_WRITE_FAILED: ocgpuNvrtcResult = 12;

/// Exact HIPRTC 5.7.1 success code.
pub const OCGPU_HIPRTC_SUCCESS: ocgpuHiprtcResult = 0;
/// Exact HIPRTC 5.7.1 out-of-memory code.
pub const OCGPU_HIPRTC_ERROR_OUT_OF_MEMORY: ocgpuHiprtcResult = 1;
/// Exact HIPRTC 5.7.1 program-creation-failure code.
pub const OCGPU_HIPRTC_ERROR_PROGRAM_CREATION_FAILURE: ocgpuHiprtcResult = 2;
/// Exact HIPRTC 5.7.1 invalid-input code.
pub const OCGPU_HIPRTC_ERROR_INVALID_INPUT: ocgpuHiprtcResult = 3;
/// Exact HIPRTC 5.7.1 invalid-program code.
pub const OCGPU_HIPRTC_ERROR_INVALID_PROGRAM: ocgpuHiprtcResult = 4;
/// Exact HIPRTC 5.7.1 invalid-option code.
pub const OCGPU_HIPRTC_ERROR_INVALID_OPTION: ocgpuHiprtcResult = 5;
/// Exact HIPRTC 5.7.1 compilation-failure code.
pub const OCGPU_HIPRTC_ERROR_COMPILATION: ocgpuHiprtcResult = 6;
/// Exact HIPRTC 5.7.1 builtin-operation-failure code.
pub const OCGPU_HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE: ocgpuHiprtcResult = 7;
/// Exact HIPRTC 5.7.1 late-name-expression code.
pub const OCGPU_HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION: ocgpuHiprtcResult = 8;
/// Exact HIPRTC 5.7.1 early-lowered-name code.
pub const OCGPU_HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION: ocgpuHiprtcResult = 9;
/// Exact HIPRTC 5.7.1 invalid-name-expression code.
pub const OCGPU_HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID: ocgpuHiprtcResult = 10;
/// Exact HIPRTC 5.7.1 internal-error code.
pub const OCGPU_HIPRTC_ERROR_INTERNAL_ERROR: ocgpuHiprtcResult = 11;
/// Exact HIPRTC 5.7.1 linking-failure code.
pub const OCGPU_HIPRTC_ERROR_LINKING: ocgpuHiprtcResult = 100;

/// Exact HIPRTC 5.7.1 maximum-registers linker option.
pub const OCGPU_HIPRTC_JIT_MAX_REGISTERS: ocgpuHiprtcJitOption = 0;
/// Exact HIPRTC 5.7.1 threads-per-block linker option.
pub const OCGPU_HIPRTC_JIT_THREADS_PER_BLOCK: ocgpuHiprtcJitOption = 1;
/// Exact HIPRTC 5.7.1 wall-time linker option.
pub const OCGPU_HIPRTC_JIT_WALL_TIME: ocgpuHiprtcJitOption = 2;
/// Exact HIPRTC 5.7.1 information-log-buffer linker option.
pub const OCGPU_HIPRTC_JIT_INFO_LOG_BUFFER: ocgpuHiprtcJitOption = 3;
/// Exact HIPRTC 5.7.1 information-log-size linker option.
pub const OCGPU_HIPRTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES: ocgpuHiprtcJitOption = 4;
/// Exact HIPRTC 5.7.1 error-log-buffer linker option.
pub const OCGPU_HIPRTC_JIT_ERROR_LOG_BUFFER: ocgpuHiprtcJitOption = 5;
/// Exact HIPRTC 5.7.1 error-log-size linker option.
pub const OCGPU_HIPRTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: ocgpuHiprtcJitOption = 6;
/// Exact HIPRTC 5.7.1 optimization-level linker option.
pub const OCGPU_HIPRTC_JIT_OPTIMIZATION_LEVEL: ocgpuHiprtcJitOption = 7;
/// Exact HIPRTC 5.7.1 current-context-target linker option.
pub const OCGPU_HIPRTC_JIT_TARGET_FROM_HIPCONTEXT: ocgpuHiprtcJitOption = 8;
/// Exact HIPRTC 5.7.1 explicit-target linker option.
pub const OCGPU_HIPRTC_JIT_TARGET: ocgpuHiprtcJitOption = 9;
/// Exact HIPRTC 5.7.1 fallback-strategy linker option.
pub const OCGPU_HIPRTC_JIT_FALLBACK_STRATEGY: ocgpuHiprtcJitOption = 10;
/// Exact HIPRTC 5.7.1 debug-information linker option.
pub const OCGPU_HIPRTC_JIT_GENERATE_DEBUG_INFO: ocgpuHiprtcJitOption = 11;
/// Exact HIPRTC 5.7.1 verbose-log linker option.
pub const OCGPU_HIPRTC_JIT_LOG_VERBOSE: ocgpuHiprtcJitOption = 12;
/// Exact HIPRTC 5.7.1 line-information linker option.
pub const OCGPU_HIPRTC_JIT_GENERATE_LINE_INFO: ocgpuHiprtcJitOption = 13;
/// Exact HIPRTC 5.7.1 cache-mode linker option.
pub const OCGPU_HIPRTC_JIT_CACHE_MODE: ocgpuHiprtcJitOption = 14;
/// Exact HIPRTC 5.7.1 new-SM3X linker option.
pub const OCGPU_HIPRTC_JIT_NEW_SM3X_OPT: ocgpuHiprtcJitOption = 15;
/// Exact HIPRTC 5.7.1 fast-compile linker option.
pub const OCGPU_HIPRTC_JIT_FAST_COMPILE: ocgpuHiprtcJitOption = 16;
/// Exact HIPRTC 5.7.1 global-symbol-names linker option.
pub const OCGPU_HIPRTC_JIT_GLOBAL_SYMBOL_NAMES: ocgpuHiprtcJitOption = 17;
/// Exact HIPRTC 5.7.1 global-symbol-address linker option.
pub const OCGPU_HIPRTC_JIT_GLOBAL_SYMBOL_ADDRESS: ocgpuHiprtcJitOption = 18;
/// Exact HIPRTC 5.7.1 global-symbol-count linker option.
pub const OCGPU_HIPRTC_JIT_GLOBAL_SYMBOL_COUNT: ocgpuHiprtcJitOption = 19;
/// Exact HIPRTC 5.7.1 link-time-optimization linker option.
pub const OCGPU_HIPRTC_JIT_LTO: ocgpuHiprtcJitOption = 20;
/// Exact HIPRTC 5.7.1 flush-to-zero linker option.
pub const OCGPU_HIPRTC_JIT_FTZ: ocgpuHiprtcJitOption = 21;
/// Exact HIPRTC 5.7.1 precise-division linker option.
pub const OCGPU_HIPRTC_JIT_PREC_DIV: ocgpuHiprtcJitOption = 22;
/// Exact HIPRTC 5.7.1 precise-square-root linker option.
pub const OCGPU_HIPRTC_JIT_PREC_SQRT: ocgpuHiprtcJitOption = 23;
/// Exact HIPRTC 5.7.1 fused-multiply-add linker option.
pub const OCGPU_HIPRTC_JIT_FMA: ocgpuHiprtcJitOption = 24;
/// Exact HIPRTC 5.7.1 ordinary linker-option count.
pub const OCGPU_HIPRTC_JIT_NUM_OPTIONS: ocgpuHiprtcJitOption = 25;
/// Exact HIPRTC 5.7.1 AMD IR-to-ISA option-list extension.
pub const OCGPU_HIPRTC_JIT_IR_TO_ISA_OPT_EXT: ocgpuHiprtcJitOption = 10_000;
/// Exact HIPRTC 5.7.1 AMD IR-to-ISA option-count extension.
pub const OCGPU_HIPRTC_JIT_IR_TO_ISA_OPT_COUNT_EXT: ocgpuHiprtcJitOption = 10_001;

/// Exact HIPRTC 5.7.1 cubin input kind.
pub const OCGPU_HIPRTC_JIT_INPUT_CUBIN: ocgpuHiprtcJitInputType = 0;
/// Exact HIPRTC 5.7.1 PTX input kind.
pub const OCGPU_HIPRTC_JIT_INPUT_PTX: ocgpuHiprtcJitInputType = 1;
/// Exact HIPRTC 5.7.1 fat-binary input kind.
pub const OCGPU_HIPRTC_JIT_INPUT_FATBINARY: ocgpuHiprtcJitInputType = 2;
/// Exact HIPRTC 5.7.1 object input kind.
pub const OCGPU_HIPRTC_JIT_INPUT_OBJECT: ocgpuHiprtcJitInputType = 3;
/// Exact HIPRTC 5.7.1 library input kind.
pub const OCGPU_HIPRTC_JIT_INPUT_LIBRARY: ocgpuHiprtcJitInputType = 4;
/// Exact HIPRTC 5.7.1 NVVM input kind.
pub const OCGPU_HIPRTC_JIT_INPUT_NVVM: ocgpuHiprtcJitInputType = 5;
/// Exact HIPRTC 5.7.1 legacy input-kind count.
pub const OCGPU_HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES: ocgpuHiprtcJitInputType = 6;
/// Exact HIPRTC 5.7.1 LLVM-bitcode input kind.
pub const OCGPU_HIPRTC_JIT_INPUT_LLVM_BITCODE: ocgpuHiprtcJitInputType = 100;
/// Exact HIPRTC 5.7.1 bundled-LLVM-bitcode input kind.
pub const OCGPU_HIPRTC_JIT_INPUT_LLVM_BUNDLED_BITCODE: ocgpuHiprtcJitInputType = 101;
/// Exact HIPRTC 5.7.1 archive-of-bundled-bitcode input kind.
pub const OCGPU_HIPRTC_JIT_INPUT_LLVM_ARCHIVES_OF_BUNDLED_BITCODE: ocgpuHiprtcJitInputType = 102;
/// Exact HIPRTC 5.7.1 total input-kind count.
pub const OCGPU_HIPRTC_JIT_NUM_INPUT_TYPES: ocgpuHiprtcJitInputType = 9;

/// Mandatory-on-success backend-neutral runtime-compilation ABI table.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ocgpuRtcApi_v1 {
    /// Bytes understood by the producer.
    pub struct_size: u32,
    /// Negotiated ABI version.
    pub abi_version: u32,
    /// Backend bound to every entry.
    pub backend: ocgpuBackend,
    /// Negotiated RTC capability bits.
    pub flags: u32,
    /// Runtime-compiler major version.
    pub rtc_version_major: i32,
    /// Runtime-compiler minor version.
    pub rtc_version_minor: i32,
    /// Returns the stable string for a backend-compatible RTC result.
    pub ocgpuRtcGetErrorString:
        Option<unsafe extern "C" fn(result: ocgpuRtcResult) -> *const c_char>,
    /// Queries the runtime-compiler version.
    pub ocgpuRtcVersion:
        Option<unsafe extern "C" fn(major: *mut i32, minor: *mut i32) -> ocgpuRtcResult>,
    /// Creates a runtime-compilation program.
    #[allow(clippy::type_complexity)]
    pub ocgpuRtcCreateProgram: Option<
        unsafe extern "C" fn(
            program: *mut ocgpuRtcProgram,
            source: *const c_char,
            name: *const c_char,
            header_count: i32,
            headers: *const *const c_char,
            include_names: *const *const c_char,
        ) -> ocgpuRtcResult,
    >,
    /// Destroys a runtime-compilation program.
    pub ocgpuRtcDestroyProgram:
        Option<unsafe extern "C" fn(program: *mut ocgpuRtcProgram) -> ocgpuRtcResult>,
    /// Compiles a runtime-compilation program.
    pub ocgpuRtcCompileProgram: Option<
        unsafe extern "C" fn(
            program: ocgpuRtcProgram,
            option_count: i32,
            options: *const *const c_char,
        ) -> ocgpuRtcResult,
    >,
    /// Queries the compilation-log byte count.
    pub ocgpuRtcGetProgramLogSize: Option<
        unsafe extern "C" fn(program: ocgpuRtcProgram, log_size: *mut usize) -> ocgpuRtcResult,
    >,
    /// Copies the compilation log.
    pub ocgpuRtcGetProgramLog:
        Option<unsafe extern "C" fn(program: ocgpuRtcProgram, log: *mut c_char) -> ocgpuRtcResult>,
    /// Registers a name expression before compilation.
    pub ocgpuRtcAddNameExpression: Option<
        unsafe extern "C" fn(
            program: ocgpuRtcProgram,
            name_expression: *const c_char,
        ) -> ocgpuRtcResult,
    >,
    /// Resolves a registered expression to its lowered name.
    pub ocgpuRtcGetLoweredName: Option<
        unsafe extern "C" fn(
            program: ocgpuRtcProgram,
            name_expression: *const c_char,
            lowered_name: *mut *const c_char,
        ) -> ocgpuRtcResult,
    >,
    /// Queries the backend-native loadable-code byte count.
    pub ocgpuRtcGetCodeSize: Option<
        unsafe extern "C" fn(program: ocgpuRtcProgram, code_size: *mut usize) -> ocgpuRtcResult,
    >,
    /// Copies backend-native loadable code.
    pub ocgpuRtcGetCode:
        Option<unsafe extern "C" fn(program: ocgpuRtcProgram, code: *mut c_char) -> ocgpuRtcResult>,
}

/// Complete public NVRTC 12.4 function table.
///
/// The private `__nvrtcCPEx` PE export is intentionally not part of this
/// public ABI table.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ocgpuNvrtcApi_v1 {
    /// Bytes understood by the producer.
    pub struct_size: u32,
    /// Negotiated ABI version.
    pub abi_version: u32,
    /// Backend bound to every entry.
    pub backend: ocgpuBackend,
    /// Negotiated RTC capability bits.
    pub flags: u32,
    /// NVRTC major version.
    pub rtc_version_major: i32,
    /// NVRTC minor version.
    pub rtc_version_minor: i32,
    /// Raw `nvrtcGetErrorString`.
    pub ocgpuNvrtcGetErrorString:
        Option<unsafe extern "C" fn(result: ocgpuNvrtcResult) -> *const c_char>,
    /// Raw `nvrtcVersion`.
    pub ocgpuNvrtcVersion:
        Option<unsafe extern "C" fn(major: *mut i32, minor: *mut i32) -> ocgpuNvrtcResult>,
    /// Raw `nvrtcCreateProgram`.
    #[allow(clippy::type_complexity)]
    pub ocgpuNvrtcCreateProgram: Option<
        unsafe extern "C" fn(
            program: *mut ocgpuNvrtcProgram,
            source: *const c_char,
            name: *const c_char,
            header_count: i32,
            headers: *const *const c_char,
            include_names: *const *const c_char,
        ) -> ocgpuNvrtcResult,
    >,
    /// Raw `nvrtcDestroyProgram`.
    pub ocgpuNvrtcDestroyProgram:
        Option<unsafe extern "C" fn(program: *mut ocgpuNvrtcProgram) -> ocgpuNvrtcResult>,
    /// Raw `nvrtcCompileProgram`.
    pub ocgpuNvrtcCompileProgram: Option<
        unsafe extern "C" fn(
            program: ocgpuNvrtcProgram,
            option_count: i32,
            options: *const *const c_char,
        ) -> ocgpuNvrtcResult,
    >,
    /// Raw `nvrtcGetProgramLogSize`.
    pub ocgpuNvrtcGetProgramLogSize: Option<
        unsafe extern "C" fn(program: ocgpuNvrtcProgram, log_size: *mut usize) -> ocgpuNvrtcResult,
    >,
    /// Raw `nvrtcGetProgramLog`.
    pub ocgpuNvrtcGetProgramLog: Option<
        unsafe extern "C" fn(program: ocgpuNvrtcProgram, log: *mut c_char) -> ocgpuNvrtcResult,
    >,
    /// Raw `nvrtcAddNameExpression`.
    pub ocgpuNvrtcAddNameExpression: Option<
        unsafe extern "C" fn(
            program: ocgpuNvrtcProgram,
            name_expression: *const c_char,
        ) -> ocgpuNvrtcResult,
    >,
    /// Raw `nvrtcGetLoweredName`.
    pub ocgpuNvrtcGetLoweredName: Option<
        unsafe extern "C" fn(
            program: ocgpuNvrtcProgram,
            name_expression: *const c_char,
            lowered_name: *mut *const c_char,
        ) -> ocgpuNvrtcResult,
    >,
    /// Raw `nvrtcGetPTXSize`.
    pub ocgpuNvrtcGetPTXSize: Option<
        unsafe extern "C" fn(program: ocgpuNvrtcProgram, ptx_size: *mut usize) -> ocgpuNvrtcResult,
    >,
    /// Raw `nvrtcGetPTX`.
    pub ocgpuNvrtcGetPTX: Option<
        unsafe extern "C" fn(program: ocgpuNvrtcProgram, ptx: *mut c_char) -> ocgpuNvrtcResult,
    >,
    /// Raw `nvrtcGetNumSupportedArchs`.
    pub ocgpuNvrtcGetNumSupportedArchs:
        Option<unsafe extern "C" fn(architecture_count: *mut i32) -> ocgpuNvrtcResult>,
    /// Raw `nvrtcGetSupportedArchs`.
    pub ocgpuNvrtcGetSupportedArchs:
        Option<unsafe extern "C" fn(architectures: *mut i32) -> ocgpuNvrtcResult>,
    /// Raw `nvrtcGetCUBINSize`.
    pub ocgpuNvrtcGetCUBINSize: Option<
        unsafe extern "C" fn(
            program: ocgpuNvrtcProgram,
            cubin_size: *mut usize,
        ) -> ocgpuNvrtcResult,
    >,
    /// Raw `nvrtcGetCUBIN`.
    pub ocgpuNvrtcGetCUBIN: Option<
        unsafe extern "C" fn(program: ocgpuNvrtcProgram, cubin: *mut c_char) -> ocgpuNvrtcResult,
    >,
    /// Raw deprecated `nvrtcGetNVVMSize`.
    pub ocgpuNvrtcGetNVVMSize: Option<
        unsafe extern "C" fn(program: ocgpuNvrtcProgram, nvvm_size: *mut usize) -> ocgpuNvrtcResult,
    >,
    /// Raw deprecated `nvrtcGetNVVM`.
    pub ocgpuNvrtcGetNVVM: Option<
        unsafe extern "C" fn(program: ocgpuNvrtcProgram, nvvm: *mut c_char) -> ocgpuNvrtcResult,
    >,
    /// Raw `nvrtcGetLTOIRSize`.
    pub ocgpuNvrtcGetLTOIRSize: Option<
        unsafe extern "C" fn(
            program: ocgpuNvrtcProgram,
            lto_ir_size: *mut usize,
        ) -> ocgpuNvrtcResult,
    >,
    /// Raw `nvrtcGetLTOIR`.
    pub ocgpuNvrtcGetLTOIR: Option<
        unsafe extern "C" fn(program: ocgpuNvrtcProgram, lto_ir: *mut c_char) -> ocgpuNvrtcResult,
    >,
    /// Raw `nvrtcGetOptiXIRSize`.
    pub ocgpuNvrtcGetOptiXIRSize: Option<
        unsafe extern "C" fn(
            program: ocgpuNvrtcProgram,
            optix_ir_size: *mut usize,
        ) -> ocgpuNvrtcResult,
    >,
    /// Raw `nvrtcGetOptiXIR`.
    pub ocgpuNvrtcGetOptiXIR: Option<
        unsafe extern "C" fn(program: ocgpuNvrtcProgram, optix_ir: *mut c_char) -> ocgpuNvrtcResult,
    >,
}

/// Complete public HIPRTC 5.7.1 function table.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ocgpuHiprtcApi_v1 {
    /// Bytes understood by the producer.
    pub struct_size: u32,
    /// Negotiated ABI version.
    pub abi_version: u32,
    /// Backend bound to every entry.
    pub backend: ocgpuBackend,
    /// Negotiated RTC capability bits.
    pub flags: u32,
    /// HIPRTC major version.
    pub rtc_version_major: i32,
    /// HIPRTC minor version.
    pub rtc_version_minor: i32,
    /// Raw `hiprtcGetErrorString`.
    pub ocgpuHiprtcGetErrorString:
        Option<unsafe extern "C" fn(result: ocgpuHiprtcResult) -> *const c_char>,
    /// Raw `hiprtcVersion`.
    pub ocgpuHiprtcVersion:
        Option<unsafe extern "C" fn(major: *mut i32, minor: *mut i32) -> ocgpuHiprtcResult>,
    /// Raw `hiprtcCreateProgram`.
    #[allow(clippy::type_complexity)]
    pub ocgpuHiprtcCreateProgram: Option<
        unsafe extern "C" fn(
            program: *mut ocgpuHiprtcProgram,
            source: *const c_char,
            name: *const c_char,
            header_count: i32,
            headers: *mut *const c_char,
            include_names: *mut *const c_char,
        ) -> ocgpuHiprtcResult,
    >,
    /// Raw `hiprtcDestroyProgram`.
    pub ocgpuHiprtcDestroyProgram:
        Option<unsafe extern "C" fn(program: *mut ocgpuHiprtcProgram) -> ocgpuHiprtcResult>,
    /// Raw `hiprtcCompileProgram`.
    pub ocgpuHiprtcCompileProgram: Option<
        unsafe extern "C" fn(
            program: ocgpuHiprtcProgram,
            option_count: i32,
            options: *mut *const c_char,
        ) -> ocgpuHiprtcResult,
    >,
    /// Raw `hiprtcGetProgramLogSize`.
    pub ocgpuHiprtcGetProgramLogSize: Option<
        unsafe extern "C" fn(
            program: ocgpuHiprtcProgram,
            log_size: *mut usize,
        ) -> ocgpuHiprtcResult,
    >,
    /// Raw `hiprtcGetProgramLog`.
    pub ocgpuHiprtcGetProgramLog: Option<
        unsafe extern "C" fn(program: ocgpuHiprtcProgram, log: *mut c_char) -> ocgpuHiprtcResult,
    >,
    /// Raw `hiprtcAddNameExpression`.
    pub ocgpuHiprtcAddNameExpression: Option<
        unsafe extern "C" fn(
            program: ocgpuHiprtcProgram,
            name_expression: *const c_char,
        ) -> ocgpuHiprtcResult,
    >,
    /// Raw `hiprtcGetLoweredName`.
    pub ocgpuHiprtcGetLoweredName: Option<
        unsafe extern "C" fn(
            program: ocgpuHiprtcProgram,
            name_expression: *const c_char,
            lowered_name: *mut *const c_char,
        ) -> ocgpuHiprtcResult,
    >,
    /// Raw `hiprtcGetCodeSize`.
    pub ocgpuHiprtcGetCodeSize: Option<
        unsafe extern "C" fn(
            program: ocgpuHiprtcProgram,
            code_size: *mut usize,
        ) -> ocgpuHiprtcResult,
    >,
    /// Raw `hiprtcGetCode`.
    pub ocgpuHiprtcGetCode: Option<
        unsafe extern "C" fn(program: ocgpuHiprtcProgram, code: *mut c_char) -> ocgpuHiprtcResult,
    >,
    /// Raw `hiprtcGetBitcodeSize`.
    pub ocgpuHiprtcGetBitcodeSize: Option<
        unsafe extern "C" fn(
            program: ocgpuHiprtcProgram,
            bitcode_size: *mut usize,
        ) -> ocgpuHiprtcResult,
    >,
    /// Raw `hiprtcGetBitcode`.
    pub ocgpuHiprtcGetBitcode: Option<
        unsafe extern "C" fn(
            program: ocgpuHiprtcProgram,
            bitcode: *mut c_char,
        ) -> ocgpuHiprtcResult,
    >,
    /// Raw `hiprtcLinkCreate`.
    pub ocgpuHiprtcLinkCreate: Option<
        unsafe extern "C" fn(
            option_count: u32,
            options: *mut ocgpuHiprtcJitOption,
            option_values: *mut *mut c_void,
            link_state: *mut ocgpuHiprtcLinkState,
        ) -> ocgpuHiprtcResult,
    >,
    /// Raw `hiprtcLinkAddFile`.
    pub ocgpuHiprtcLinkAddFile: Option<
        unsafe extern "C" fn(
            link_state: ocgpuHiprtcLinkState,
            input_type: ocgpuHiprtcJitInputType,
            file_path: *const c_char,
            option_count: u32,
            options: *mut ocgpuHiprtcJitOption,
            option_values: *mut *mut c_void,
        ) -> ocgpuHiprtcResult,
    >,
    /// Raw `hiprtcLinkAddData`.
    #[allow(clippy::too_many_arguments)]
    pub ocgpuHiprtcLinkAddData: Option<
        unsafe extern "C" fn(
            link_state: ocgpuHiprtcLinkState,
            input_type: ocgpuHiprtcJitInputType,
            image: *mut c_void,
            image_size: usize,
            name: *const c_char,
            option_count: u32,
            options: *mut ocgpuHiprtcJitOption,
            option_values: *mut *mut c_void,
        ) -> ocgpuHiprtcResult,
    >,
    /// Raw `hiprtcLinkComplete`.
    pub ocgpuHiprtcLinkComplete: Option<
        unsafe extern "C" fn(
            link_state: ocgpuHiprtcLinkState,
            binary: *mut *mut c_void,
            binary_size: *mut usize,
        ) -> ocgpuHiprtcResult,
    >,
    /// Raw `hiprtcLinkDestroy`.
    pub ocgpuHiprtcLinkDestroy:
        Option<unsafe extern "C" fn(link_state: ocgpuHiprtcLinkState) -> ocgpuHiprtcResult>,
}

unsafe extern "C" {
    /// Negotiates a backend-bound common RTC ABI table.
    pub fn ocgpuRtcGetApi(
        backend: ocgpuBackend,
        requested_abi: u32,
        output_size: usize,
        output: *mut ocgpuRtcApi_v1,
    ) -> ocgpuResult;

    /// Negotiates a raw NVRTC ABI table.
    pub fn ocgpuNvrtcGetApi(
        requested_abi: u32,
        output_size: usize,
        output: *mut ocgpuNvrtcApi_v1,
    ) -> ocgpuResult;

    /// Negotiates a raw HIPRTC ABI table.
    pub fn ocgpuHiprtcGetApi(
        requested_abi: u32,
        output_size: usize,
        output: *mut ocgpuHiprtcApi_v1,
    ) -> ocgpuResult;
}
