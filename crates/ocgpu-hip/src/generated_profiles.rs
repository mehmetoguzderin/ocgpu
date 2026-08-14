// SPDX-License-Identifier: CC0-1.0

//! Generated from `oracle/vendor/hip/runtime-profiles.json`; do not edit.

/// One platform-specific interval in a reviewed HIP runtime ABI profile.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct HipPlatformRuntimeProfile {
    /// Library basenames tried in fail-closed preference order.
    pub library_candidates: &'static [&'static str],
    /// Smallest accepted `hipRuntimeGetVersion` value.
    pub runtime_version_min_inclusive: i32,
    /// Largest accepted `hipRuntimeGetVersion` value.
    pub runtime_version_max_inclusive: i32,
    /// Smallest reviewed version with `hipGetProcAddress`, if any.
    pub proc_address_min_inclusive: Option<i32>,
    /// Smallest version allowed to populate the exhaustive raw inventory.
    pub raw_inventory_min_inclusive: Option<i32>,
}

/// Reviewed compatibility facts for one HIP runtime ABI major.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct HipRuntimeProfileDescriptor {
    /// Major decoded from `hipRuntimeGetVersion`.
    pub runtime_major: i32,
    /// Bits advertised through common and raw table flags.
    pub table_flag: u32,
    /// Symbols available through the common profile.
    pub common_symbols: &'static [&'static str],
    /// Common symbols with identical normalized declarations.
    pub raw_exact_symbols: &'static [&'static str],
    /// Common symbols reached through a reviewed adapter.
    pub common_adapter_symbols: &'static [&'static str],
    /// Symbols callable before profile selection.
    pub bootstrap_symbols: &'static [&'static str],
    /// Windows compatibility facts.
    pub windows: HipPlatformRuntimeProfile,
    /// Linux compatibility facts.
    pub linux: HipPlatformRuntimeProfile,
}

/// Bootstrap calls made before profile selection.
pub(crate) const HIP_BOOTSTRAP_SYMBOLS: &[&str] = &["hipRuntimeGetVersion"];

/// Common subset available in every profile.
pub(crate) const HIP_COMMON_PROFILE_SYMBOLS: &[&str] = &[
    "hipInit",
    "hipDriverGetVersion",
    "hipGetDeviceCount",
    "hipDeviceGet",
    "hipDeviceGetName",
    "hipDeviceGetAttribute",
    "hipCtxCreate",
    "hipCtxDestroy",
    "hipCtxSetCurrent",
    "hipCtxGetCurrent",
    "hipCtxSynchronize",
    "hipMalloc",
    "hipFree",
    "hipMemcpyHtoD",
    "hipMemcpyDtoH",
    "hipStreamCreateWithFlags",
    "hipStreamDestroy",
    "hipStreamSynchronize",
    "hipEventCreateWithFlags",
    "hipEventDestroy",
    "hipEventRecord",
    "hipEventSynchronize",
    "hipModuleLoadData",
    "hipModuleUnload",
    "hipModuleGetFunction",
    "hipModuleLaunchKernel",
];

/// Legacy common subset with identical declarations.
pub(crate) const HIP_LEGACY_COMMON_RAW_EXACT_SYMBOLS: &[&str] = &[
    "hipInit",
    "hipDriverGetVersion",
    "hipGetDeviceCount",
    "hipDeviceGet",
    "hipDeviceGetName",
    "hipDeviceGetAttribute",
    "hipCtxCreate",
    "hipCtxDestroy",
    "hipCtxSetCurrent",
    "hipCtxGetCurrent",
    "hipCtxSynchronize",
    "hipMalloc",
    "hipFree",
    "hipMemcpyDtoH",
    "hipStreamCreateWithFlags",
    "hipStreamDestroy",
    "hipStreamSynchronize",
    "hipEventCreateWithFlags",
    "hipEventDestroy",
    "hipEventRecord",
    "hipEventSynchronize",
    "hipModuleLoadData",
    "hipModuleUnload",
    "hipModuleGetFunction",
    "hipModuleLaunchKernel",
];

/// Legacy calls requiring an explicit adapter.
pub(crate) const HIP_COMMON_ADAPTER_SYMBOLS: &[&str] = &["hipMemcpyHtoD"];

/// HIP 7 uses current declarations for the complete common subset.
pub(crate) const HIP7_COMMON_RAW_EXACT_SYMBOLS: &[&str] = HIP_COMMON_PROFILE_SYMBOLS;

/// HIP 7 needs no common-call adapter.
pub(crate) const HIP7_COMMON_ADAPTER_SYMBOLS: &[&str] = &[];

/// Closed set of supported profiles, newest first.
pub(crate) const HIP_RUNTIME_PROFILES: &[HipRuntimeProfileDescriptor] = &[
    HipRuntimeProfileDescriptor {
        runtime_major: 7,
        table_flag: ocgpu_abi::OCGPU_API_FLAG_HIP_RUNTIME_PROFILE_7,
        common_symbols: HIP_COMMON_PROFILE_SYMBOLS,
        raw_exact_symbols: HIP7_COMMON_RAW_EXACT_SYMBOLS,
        common_adapter_symbols: HIP7_COMMON_ADAPTER_SYMBOLS,
        bootstrap_symbols: HIP_BOOTSTRAP_SYMBOLS,
        windows: HipPlatformRuntimeProfile {
            library_candidates: &["amdhip64_7.dll"],
            runtime_version_min_inclusive: 70_253_210,
            runtime_version_max_inclusive: 79_999_999,
            proc_address_min_inclusive: Some(70_253_210),
            raw_inventory_min_inclusive: Some(70_253_210),
        },
        linux: HipPlatformRuntimeProfile {
            library_candidates: &["libamdhip64.so.7", "libamdhip64.so"],
            runtime_version_min_inclusive: 70_253_210,
            runtime_version_max_inclusive: 79_999_999,
            proc_address_min_inclusive: Some(70_253_210),
            raw_inventory_min_inclusive: Some(71_460_850),
        },
    },
    HipRuntimeProfileDescriptor {
        runtime_major: 6,
        table_flag: ocgpu_abi::OCGPU_API_FLAG_HIP_RUNTIME_PROFILE_6,
        common_symbols: HIP_COMMON_PROFILE_SYMBOLS,
        raw_exact_symbols: HIP_LEGACY_COMMON_RAW_EXACT_SYMBOLS,
        common_adapter_symbols: HIP_COMMON_ADAPTER_SYMBOLS,
        bootstrap_symbols: HIP_BOOTSTRAP_SYMBOLS,
        windows: HipPlatformRuntimeProfile {
            library_candidates: &["amdhip64_6.dll"],
            runtime_version_min_inclusive: 60_140_093,
            runtime_version_max_inclusive: 69_999_999,
            proc_address_min_inclusive: Some(60_241_134),
            raw_inventory_min_inclusive: None,
        },
        linux: HipPlatformRuntimeProfile {
            library_candidates: &["libamdhip64.so.6", "libamdhip64.so"],
            runtime_version_min_inclusive: 60_140_093,
            runtime_version_max_inclusive: 69_999_999,
            proc_address_min_inclusive: Some(60_241_134),
            raw_inventory_min_inclusive: None,
        },
    },
    HipRuntimeProfileDescriptor {
        runtime_major: 5,
        table_flag: ocgpu_abi::OCGPU_API_FLAG_HIP_RUNTIME_PROFILE_5,
        common_symbols: HIP_COMMON_PROFILE_SYMBOLS,
        raw_exact_symbols: HIP_LEGACY_COMMON_RAW_EXACT_SYMBOLS,
        common_adapter_symbols: HIP_COMMON_ADAPTER_SYMBOLS,
        bootstrap_symbols: HIP_BOOTSTRAP_SYMBOLS,
        windows: HipPlatformRuntimeProfile {
            library_candidates: &["amdhip64.dll"],
            runtime_version_min_inclusive: 50_731_541,
            runtime_version_max_inclusive: 59_999_999,
            proc_address_min_inclusive: None,
            raw_inventory_min_inclusive: None,
        },
        linux: HipPlatformRuntimeProfile {
            library_candidates: &["libamdhip64.so.5", "libamdhip64.so"],
            runtime_version_min_inclusive: 50_731_541,
            runtime_version_max_inclusive: 59_999_999,
            proc_address_min_inclusive: None,
            raw_inventory_min_inclusive: None,
        },
    },
];
