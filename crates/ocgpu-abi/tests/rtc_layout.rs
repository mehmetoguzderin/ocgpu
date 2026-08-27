// SPDX-License-Identifier: CC0-1.0

//! RTC ABI constants and 64-bit layout conformance tests.

#![allow(non_snake_case)]

use core::mem::{align_of, offset_of, size_of};
use ocgpu_abi::*;

macro_rules! assert_prefix_layout {
    ($table:ty) => {
        assert_eq!(offset_of!($table, struct_size), 0);
        assert_eq!(offset_of!($table, abi_version), 4);
        assert_eq!(offset_of!($table, backend), 8);
        assert_eq!(offset_of!($table, flags), 12);
        assert_eq!(offset_of!($table, rtc_version_major), 16);
        assert_eq!(offset_of!($table, rtc_version_minor), 20);
    };
}

macro_rules! assert_slot_offsets {
    ($table:ty, $($field:ident => $offset:expr),+ $(,)?) => {
        $(assert_eq!(offset_of!($table, $field), $offset);)+
    };
}

#[test]
fn primitive_layouts_are_c_compatible() {
    assert_eq!(size_of::<ocgpuRtcResult>(), 4);
    assert_eq!(align_of::<ocgpuRtcResult>(), 4);
    assert_eq!(size_of::<ocgpuRtcProgram>(), 8);
    assert_eq!(align_of::<ocgpuRtcProgram>(), 8);
    assert_eq!(size_of::<ocgpuRtcLinkState>(), 8);
    assert_eq!(align_of::<ocgpuRtcLinkState>(), 8);
    assert_eq!(size_of::<ocgpuRtcJitInputType>(), 4);
    assert_eq!(align_of::<ocgpuRtcJitInputType>(), 4);
    assert_eq!(size_of::<ocgpuRtcJitOption>(), 4);
    assert_eq!(align_of::<ocgpuRtcJitOption>(), 4);

    assert_eq!(size_of::<ocgpuNvrtcResult>(), 4);
    assert_eq!(align_of::<ocgpuNvrtcResult>(), 4);
    assert_eq!(size_of::<ocgpuNvrtcProgram>(), 8);
    assert_eq!(align_of::<ocgpuNvrtcProgram>(), 8);
    assert_eq!(size_of::<ocgpuHiprtcResult>(), 4);
    assert_eq!(align_of::<ocgpuHiprtcResult>(), 4);
    assert_eq!(size_of::<ocgpuHiprtcProgram>(), 8);
    assert_eq!(align_of::<ocgpuHiprtcProgram>(), 8);
    assert_eq!(size_of::<ocgpuHiprtcLinkState>(), 8);
    assert_eq!(align_of::<ocgpuHiprtcLinkState>(), 8);
    assert_eq!(size_of::<ocgpuHiprtcJitInputType>(), 4);
    assert_eq!(align_of::<ocgpuHiprtcJitInputType>(), 4);
    assert_eq!(size_of::<ocgpuHiprtcJitOption>(), 4);
    assert_eq!(align_of::<ocgpuHiprtcJitOption>(), 4);
}

#[test]
fn common_result_values_are_exact() {
    assert_eq!(OCGPU_RTC_API_FLAG_CODE_IS_PTX, 1);
    assert_eq!(OCGPU_RTC_API_FLAG_CODE_IS_HIP_CODE_OBJECT, 2);
    assert_eq!(
        OCGPU_RTC_API_FLAG_CODE_IS_PTX & OCGPU_RTC_API_FLAG_CODE_IS_HIP_CODE_OBJECT,
        0
    );
    assert_eq!(OCGPU_RTC_SUCCESS, 0);
    assert_eq!(OCGPU_RTC_ERROR_OUT_OF_MEMORY, 1);
    assert_eq!(OCGPU_RTC_ERROR_PROGRAM_CREATION_FAILURE, 2);
    assert_eq!(OCGPU_RTC_ERROR_INVALID_INPUT, 3);
    assert_eq!(OCGPU_RTC_ERROR_INVALID_PROGRAM, 4);
    assert_eq!(OCGPU_RTC_ERROR_INVALID_OPTION, 5);
    assert_eq!(OCGPU_RTC_ERROR_COMPILATION, 6);
    assert_eq!(OCGPU_RTC_ERROR_BUILTIN_OPERATION_FAILURE, 7);
    assert_eq!(OCGPU_RTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION, 8);
    assert_eq!(OCGPU_RTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION, 9);
    assert_eq!(OCGPU_RTC_ERROR_NAME_EXPRESSION_NOT_VALID, 10);
    assert_eq!(OCGPU_RTC_ERROR_INTERNAL_ERROR, 11);
}

#[test]
fn nvrtc_12_4_result_values_are_exact() {
    assert_eq!(OCGPU_NVRTC_SUCCESS, 0);
    assert_eq!(OCGPU_NVRTC_ERROR_OUT_OF_MEMORY, 1);
    assert_eq!(OCGPU_NVRTC_ERROR_PROGRAM_CREATION_FAILURE, 2);
    assert_eq!(OCGPU_NVRTC_ERROR_INVALID_INPUT, 3);
    assert_eq!(OCGPU_NVRTC_ERROR_INVALID_PROGRAM, 4);
    assert_eq!(OCGPU_NVRTC_ERROR_INVALID_OPTION, 5);
    assert_eq!(OCGPU_NVRTC_ERROR_COMPILATION, 6);
    assert_eq!(OCGPU_NVRTC_ERROR_BUILTIN_OPERATION_FAILURE, 7);
    assert_eq!(OCGPU_NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION, 8);
    assert_eq!(OCGPU_NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION, 9);
    assert_eq!(OCGPU_NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID, 10);
    assert_eq!(OCGPU_NVRTC_ERROR_INTERNAL_ERROR, 11);
    assert_eq!(OCGPU_NVRTC_ERROR_TIME_FILE_WRITE_FAILED, 12);
}

#[test]
fn hiprtc_5_7_1_result_values_are_exact() {
    assert_eq!(OCGPU_HIPRTC_SUCCESS, 0);
    assert_eq!(OCGPU_HIPRTC_ERROR_OUT_OF_MEMORY, 1);
    assert_eq!(OCGPU_HIPRTC_ERROR_PROGRAM_CREATION_FAILURE, 2);
    assert_eq!(OCGPU_HIPRTC_ERROR_INVALID_INPUT, 3);
    assert_eq!(OCGPU_HIPRTC_ERROR_INVALID_PROGRAM, 4);
    assert_eq!(OCGPU_HIPRTC_ERROR_INVALID_OPTION, 5);
    assert_eq!(OCGPU_HIPRTC_ERROR_COMPILATION, 6);
    assert_eq!(OCGPU_HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE, 7);
    assert_eq!(OCGPU_HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION, 8);
    assert_eq!(OCGPU_HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION, 9);
    assert_eq!(OCGPU_HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID, 10);
    assert_eq!(OCGPU_HIPRTC_ERROR_INTERNAL_ERROR, 11);
    assert_eq!(OCGPU_HIPRTC_ERROR_LINKING, 100);
}

#[test]
fn hiprtc_5_7_1_jit_option_values_are_exact() {
    let ordinary_options = [
        OCGPU_HIPRTC_JIT_MAX_REGISTERS,
        OCGPU_HIPRTC_JIT_THREADS_PER_BLOCK,
        OCGPU_HIPRTC_JIT_WALL_TIME,
        OCGPU_HIPRTC_JIT_INFO_LOG_BUFFER,
        OCGPU_HIPRTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        OCGPU_HIPRTC_JIT_ERROR_LOG_BUFFER,
        OCGPU_HIPRTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        OCGPU_HIPRTC_JIT_OPTIMIZATION_LEVEL,
        OCGPU_HIPRTC_JIT_TARGET_FROM_HIPCONTEXT,
        OCGPU_HIPRTC_JIT_TARGET,
        OCGPU_HIPRTC_JIT_FALLBACK_STRATEGY,
        OCGPU_HIPRTC_JIT_GENERATE_DEBUG_INFO,
        OCGPU_HIPRTC_JIT_LOG_VERBOSE,
        OCGPU_HIPRTC_JIT_GENERATE_LINE_INFO,
        OCGPU_HIPRTC_JIT_CACHE_MODE,
        OCGPU_HIPRTC_JIT_NEW_SM3X_OPT,
        OCGPU_HIPRTC_JIT_FAST_COMPILE,
        OCGPU_HIPRTC_JIT_GLOBAL_SYMBOL_NAMES,
        OCGPU_HIPRTC_JIT_GLOBAL_SYMBOL_ADDRESS,
        OCGPU_HIPRTC_JIT_GLOBAL_SYMBOL_COUNT,
        OCGPU_HIPRTC_JIT_LTO,
        OCGPU_HIPRTC_JIT_FTZ,
        OCGPU_HIPRTC_JIT_PREC_DIV,
        OCGPU_HIPRTC_JIT_PREC_SQRT,
        OCGPU_HIPRTC_JIT_FMA,
        OCGPU_HIPRTC_JIT_NUM_OPTIONS,
    ];
    assert_eq!(
        ordinary_options,
        core::array::from_fn(|index| i32::try_from(index).expect("25 options fit i32"))
    );
    assert_eq!(OCGPU_HIPRTC_JIT_IR_TO_ISA_OPT_EXT, 10_000);
    assert_eq!(OCGPU_HIPRTC_JIT_IR_TO_ISA_OPT_COUNT_EXT, 10_001);
}

#[test]
fn hiprtc_5_7_1_jit_input_values_are_exact() {
    assert_eq!(OCGPU_HIPRTC_JIT_INPUT_CUBIN, 0);
    assert_eq!(OCGPU_HIPRTC_JIT_INPUT_PTX, 1);
    assert_eq!(OCGPU_HIPRTC_JIT_INPUT_FATBINARY, 2);
    assert_eq!(OCGPU_HIPRTC_JIT_INPUT_OBJECT, 3);
    assert_eq!(OCGPU_HIPRTC_JIT_INPUT_LIBRARY, 4);
    assert_eq!(OCGPU_HIPRTC_JIT_INPUT_NVVM, 5);
    assert_eq!(OCGPU_HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES, 6);
    assert_eq!(OCGPU_HIPRTC_JIT_INPUT_LLVM_BITCODE, 100);
    assert_eq!(OCGPU_HIPRTC_JIT_INPUT_LLVM_BUNDLED_BITCODE, 101);
    assert_eq!(OCGPU_HIPRTC_JIT_INPUT_LLVM_ARCHIVES_OF_BUNDLED_BITCODE, 102);
    assert_eq!(OCGPU_HIPRTC_JIT_NUM_INPUT_TYPES, 9);
}

#[test]
fn common_table_layout_is_exact() {
    assert_eq!(OCGPU_RTC_TABLE_PREFIX_SIZE, 24);
    assert_eq!(OCGPU_RTC_API_V1_SLOT_COUNT, 11);
    assert_eq!(size_of::<ocgpuRtcApi_v1>(), 112);
    assert_eq!(align_of::<ocgpuRtcApi_v1>(), 8);
    assert_prefix_layout!(ocgpuRtcApi_v1);
    assert_slot_offsets!(
        ocgpuRtcApi_v1,
        ocgpuRtcGetErrorString => 24,
        ocgpuRtcVersion => 32,
        ocgpuRtcCreateProgram => 40,
        ocgpuRtcDestroyProgram => 48,
        ocgpuRtcCompileProgram => 56,
        ocgpuRtcGetProgramLogSize => 64,
        ocgpuRtcGetProgramLog => 72,
        ocgpuRtcAddNameExpression => 80,
        ocgpuRtcGetLoweredName => 88,
        ocgpuRtcGetCodeSize => 96,
        ocgpuRtcGetCode => 104,
    );
}

#[test]
fn nvrtc_table_layout_is_exact() {
    assert_eq!(OCGPU_NVRTC_API_V1_SLOT_COUNT, 21);
    assert_eq!(size_of::<ocgpuNvrtcApi_v1>(), 192);
    assert_eq!(align_of::<ocgpuNvrtcApi_v1>(), 8);
    assert_prefix_layout!(ocgpuNvrtcApi_v1);
    assert_slot_offsets!(
        ocgpuNvrtcApi_v1,
        ocgpuNvrtcGetErrorString => 24,
        ocgpuNvrtcVersion => 32,
        ocgpuNvrtcCreateProgram => 40,
        ocgpuNvrtcDestroyProgram => 48,
        ocgpuNvrtcCompileProgram => 56,
        ocgpuNvrtcGetProgramLogSize => 64,
        ocgpuNvrtcGetProgramLog => 72,
        ocgpuNvrtcAddNameExpression => 80,
        ocgpuNvrtcGetLoweredName => 88,
        ocgpuNvrtcGetPTXSize => 96,
        ocgpuNvrtcGetPTX => 104,
        ocgpuNvrtcGetNumSupportedArchs => 112,
        ocgpuNvrtcGetSupportedArchs => 120,
        ocgpuNvrtcGetCUBINSize => 128,
        ocgpuNvrtcGetCUBIN => 136,
        ocgpuNvrtcGetNVVMSize => 144,
        ocgpuNvrtcGetNVVM => 152,
        ocgpuNvrtcGetLTOIRSize => 160,
        ocgpuNvrtcGetLTOIR => 168,
        ocgpuNvrtcGetOptiXIRSize => 176,
        ocgpuNvrtcGetOptiXIR => 184,
    );
}

#[test]
fn hiprtc_table_layout_is_exact() {
    assert_eq!(OCGPU_HIPRTC_API_V1_SLOT_COUNT, 18);
    assert_eq!(size_of::<ocgpuHiprtcApi_v1>(), 168);
    assert_eq!(align_of::<ocgpuHiprtcApi_v1>(), 8);
    assert_prefix_layout!(ocgpuHiprtcApi_v1);
    assert_slot_offsets!(
        ocgpuHiprtcApi_v1,
        ocgpuHiprtcGetErrorString => 24,
        ocgpuHiprtcVersion => 32,
        ocgpuHiprtcCreateProgram => 40,
        ocgpuHiprtcDestroyProgram => 48,
        ocgpuHiprtcCompileProgram => 56,
        ocgpuHiprtcGetProgramLogSize => 64,
        ocgpuHiprtcGetProgramLog => 72,
        ocgpuHiprtcAddNameExpression => 80,
        ocgpuHiprtcGetLoweredName => 88,
        ocgpuHiprtcGetCodeSize => 96,
        ocgpuHiprtcGetCode => 104,
        ocgpuHiprtcGetBitcodeSize => 112,
        ocgpuHiprtcGetBitcode => 120,
        ocgpuHiprtcLinkCreate => 128,
        ocgpuHiprtcLinkAddFile => 136,
        ocgpuHiprtcLinkAddData => 144,
        ocgpuHiprtcLinkComplete => 152,
        ocgpuHiprtcLinkDestroy => 160,
    );
}

#[test]
fn tables_are_copy_clone_default_and_slots_default_to_null() {
    fn assert_traits<T: Copy + Clone + Default>() {}

    assert_traits::<ocgpuRtcApi_v1>();
    assert_traits::<ocgpuNvrtcApi_v1>();
    assert_traits::<ocgpuHiprtcApi_v1>();

    let common = ocgpuRtcApi_v1::default();
    assert_eq!(common.struct_size, 0);
    assert!(common.ocgpuRtcGetErrorString.is_none());
    assert!(common.ocgpuRtcVersion.is_none());
    assert!(common.ocgpuRtcCreateProgram.is_none());
    assert!(common.ocgpuRtcDestroyProgram.is_none());
    assert!(common.ocgpuRtcCompileProgram.is_none());
    assert!(common.ocgpuRtcGetProgramLogSize.is_none());
    assert!(common.ocgpuRtcGetProgramLog.is_none());
    assert!(common.ocgpuRtcAddNameExpression.is_none());
    assert!(common.ocgpuRtcGetLoweredName.is_none());
    assert!(common.ocgpuRtcGetCodeSize.is_none());
    assert!(common.ocgpuRtcGetCode.is_none());
}
