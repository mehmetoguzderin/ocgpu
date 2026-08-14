// SPDX-License-Identifier: CC0-1.0

//! SDK-free HIP runtime discovery and validated driver-shaped core dispatch.

use ocgpu_abi::{
    OCGPU_ABI_VERSION_1, OCGPU_BACKEND_HIP, ocgpuApi_v1, ocgpuContext, ocgpuDevice,
    ocgpuDeviceAttribute, ocgpuDeviceptr, ocgpuEvent, ocgpuFunction, ocgpuHipApi_v1, ocgpuModule,
    ocgpuResult, ocgpuStream,
};
use ocgpu_loader::{Backend, Library, LoadError, TableSlotError, write_function_slot};
use std::error::Error as StdError;
use std::ffi::{CString, c_char, c_void};
use std::fmt;
use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::OnceLock;

#[allow(clippy::doc_markdown)]
mod raw_symbols {
    include!("generated_symbols.rs");
}
mod generated_profiles;

use generated_profiles::{
    HIP_BOOTSTRAP_SYMBOLS, HIP_RUNTIME_PROFILES, HipPlatformRuntimeProfile,
    HipRuntimeProfileDescriptor,
};
use raw_symbols::{
    HIP_RAW_INVENTORY, HIP_RAW_SYMBOLS, RawInventoryDescriptor, RawSymbolDescriptor,
};

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
const CURRENT_PLATFORM_MASK: u8 = 1;
#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
const CURRENT_PLATFORM_MASK: u8 = 2;
#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
const CURRENT_PLATFORM_MASK: u8 = 4;
#[cfg(not(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "windows", target_arch = "x86_64"),
    all(target_os = "linux", target_arch = "aarch64")
)))]
const CURRENT_PLATFORM_MASK: u8 = 0;

/// General HIP inventory baseline encoded as `major.minor.patch` digits.
#[cfg(target_os = "linux")]
pub const COMPILED_API_VERSION: i32 = 71_460_850;
/// Windows HIP SDK inventory baseline encoded as `major.minor.patch` digits.
#[cfg(target_os = "windows")]
pub const COMPILED_API_VERSION: i32 = 70_200_000;
/// Fallback baseline for documentation and unsupported-target builds.
#[cfg(not(any(target_os = "linux", target_os = "windows")))]
pub const COMPILED_API_VERSION: i32 = 71_460_850;

const HIP_SUCCESS: ocgpuResult = 0;
const HIP_VERSION_MAJOR_SCALE: i32 = 10_000_000;

/// Runtime-major ABI profile selected after a direct version bootstrap.
///
/// Profiles are deliberately closed: an unrecognized runtime major is never
/// treated as if it implemented the newest ABI known to this build.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RuntimeProfile {
    /// HIP 5.x runtime ABI.
    Hip5,
    /// HIP 6.x runtime ABI.
    Hip6,
    /// HIP 7.x runtime ABI.
    Hip7,
}

impl RuntimeProfile {
    /// Runtime major represented by this profile.
    #[must_use]
    pub const fn runtime_major(self) -> i32 {
        match self {
            Self::Hip5 => 5,
            Self::Hip6 => 6,
            Self::Hip7 => 7,
        }
    }

    /// ABI-table flag bits encoding this selected runtime profile.
    #[must_use]
    pub const fn api_flags(self) -> u32 {
        match self {
            Self::Hip5 => ocgpu_abi::OCGPU_API_FLAG_HIP_RUNTIME_PROFILE_5,
            Self::Hip6 => ocgpu_abi::OCGPU_API_FLAG_HIP_RUNTIME_PROFILE_6,
            Self::Hip7 => ocgpu_abi::OCGPU_API_FLAG_HIP_RUNTIME_PROFILE_7,
        }
    }
}

fn runtime_profile_descriptor(
    profile: RuntimeProfile,
) -> Option<&'static HipRuntimeProfileDescriptor> {
    HIP_RUNTIME_PROFILES
        .iter()
        .find(|descriptor| descriptor.runtime_major == profile.runtime_major())
}

#[cfg(target_os = "windows")]
const fn platform_runtime_profile(
    descriptor: &HipRuntimeProfileDescriptor,
) -> &HipPlatformRuntimeProfile {
    &descriptor.windows
}

#[cfg(target_os = "linux")]
const fn platform_runtime_profile(
    descriptor: &HipRuntimeProfileDescriptor,
) -> &HipPlatformRuntimeProfile {
    &descriptor.linux
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
const fn platform_runtime_profile(
    descriptor: &HipRuntimeProfileDescriptor,
) -> &HipPlatformRuntimeProfile {
    // Backend loading rejects this target before profile selection. Retaining a
    // concrete reference keeps generated metadata inspectable in documentation.
    &descriptor.linux
}

/// `hipInit` ABI.
pub type InitFn = unsafe extern "C" fn(u32) -> ocgpuResult;
/// `hipDriverGetVersion` ABI.
pub type DriverGetVersionFn = unsafe extern "C" fn(*mut i32) -> ocgpuResult;
/// `hipRuntimeGetVersion` ABI.
pub type RuntimeGetVersionFn = unsafe extern "C" fn(*mut i32) -> ocgpuResult;
/// `hipGetDeviceCount` ABI.
pub type DeviceGetCountFn = unsafe extern "C" fn(*mut i32) -> ocgpuResult;
/// `hipDeviceGet` ABI.
pub type DeviceGetFn = unsafe extern "C" fn(*mut ocgpuDevice, i32) -> ocgpuResult;
/// `hipDeviceGetName` ABI.
pub type DeviceGetNameFn = unsafe extern "C" fn(*mut c_char, i32, ocgpuDevice) -> ocgpuResult;
/// `hipDeviceGetAttribute` ABI.
pub type DeviceGetAttributeFn =
    unsafe extern "C" fn(*mut i32, ocgpuDeviceAttribute, ocgpuDevice) -> ocgpuResult;

static DEVICE_GET_ATTRIBUTE_TARGET: OnceLock<DeviceGetAttributeFn> = OnceLock::new();

fn hip_device_attribute(attribute: ocgpuDeviceAttribute) -> Option<ocgpuDeviceAttribute> {
    macro_rules! map_attributes {
        ($($common:ident => $native:ident),+ $(,)?) => {
            match attribute {
                $(ocgpu_abi::$common => Some(ocgpu_abi::$native),)+
                _ => None,
            }
        };
    }
    map_attributes!(
        OCGPU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK => OCGPU_HIP_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        OCGPU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X => OCGPU_HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
        OCGPU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y => OCGPU_HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
        OCGPU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z => OCGPU_HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
        OCGPU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X => OCGPU_HIP_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
        OCGPU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y => OCGPU_HIP_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
        OCGPU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z => OCGPU_HIP_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
        OCGPU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK => OCGPU_HIP_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
        OCGPU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY => OCGPU_HIP_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY,
        OCGPU_DEVICE_ATTRIBUTE_WARP_SIZE => OCGPU_HIP_DEVICE_ATTRIBUTE_WARP_SIZE,
        OCGPU_DEVICE_ATTRIBUTE_MAX_PITCH => OCGPU_HIP_DEVICE_ATTRIBUTE_MAX_PITCH,
        OCGPU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK => OCGPU_HIP_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
        OCGPU_DEVICE_ATTRIBUTE_CLOCK_RATE => OCGPU_HIP_DEVICE_ATTRIBUTE_CLOCK_RATE,
        OCGPU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT => OCGPU_HIP_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT,
        OCGPU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT => OCGPU_HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        OCGPU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT => OCGPU_HIP_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT,
        OCGPU_DEVICE_ATTRIBUTE_INTEGRATED => OCGPU_HIP_DEVICE_ATTRIBUTE_INTEGRATED,
        OCGPU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY => OCGPU_HIP_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY,
        OCGPU_DEVICE_ATTRIBUTE_COMPUTE_MODE => OCGPU_HIP_DEVICE_ATTRIBUTE_COMPUTE_MODE,
        OCGPU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS => OCGPU_HIP_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS,
        OCGPU_DEVICE_ATTRIBUTE_PCI_BUS_ID => OCGPU_HIP_DEVICE_ATTRIBUTE_PCI_BUS_ID,
        OCGPU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID => OCGPU_HIP_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
        OCGPU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE => OCGPU_HIP_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
        OCGPU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH => OCGPU_HIP_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
        OCGPU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE => OCGPU_HIP_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
        OCGPU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR => OCGPU_HIP_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
        OCGPU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING => OCGPU_HIP_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
        OCGPU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR => OCGPU_HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        OCGPU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR => OCGPU_HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        OCGPU_DEVICE_ATTRIBUTE_MANAGED_MEMORY => OCGPU_HIP_DEVICE_ATTRIBUTE_MANAGED_MEMORY,
        OCGPU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS => OCGPU_HIP_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS,
        OCGPU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS => OCGPU_HIP_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS,
    )
}

unsafe extern "C" fn common_device_get_attribute(
    value: *mut i32,
    attribute: ocgpuDeviceAttribute,
    device: ocgpuDevice,
) -> ocgpuResult {
    let Some(native_attribute) = hip_device_attribute(attribute) else {
        return ocgpu_abi::OCGPU_ERROR_INVALID_ARGUMENT;
    };
    let Some(target) = DEVICE_GET_ATTRIBUTE_TARGET.get() else {
        return ocgpu_abi::OCGPU_ERROR_INTERNAL;
    };
    // SAFETY: the validated process-lifetime target has HIP's exact raw ABI;
    // only its backend-native integer attribute value is substituted here.
    unsafe { target(value, native_attribute, device) }
}
/// `hipCtxCreate` ABI.
pub type CtxCreateFn = unsafe extern "C" fn(*mut ocgpuContext, u32, ocgpuDevice) -> ocgpuResult;
/// `hipCtxDestroy` ABI.
pub type CtxDestroyFn = unsafe extern "C" fn(ocgpuContext) -> ocgpuResult;
/// `hipCtxSetCurrent` ABI.
pub type CtxSetCurrentFn = unsafe extern "C" fn(ocgpuContext) -> ocgpuResult;
/// `hipCtxGetCurrent` ABI.
pub type CtxGetCurrentFn = unsafe extern "C" fn(*mut ocgpuContext) -> ocgpuResult;
/// `hipCtxSynchronize` ABI.
pub type CtxSynchronizeFn = unsafe extern "C" fn() -> ocgpuResult;
/// Unified allocation ABI adapted over `hipMalloc`.
pub type MemAllocFn = unsafe extern "C" fn(*mut ocgpuDeviceptr, usize) -> ocgpuResult;
/// Unified free ABI adapted over `hipFree`.
pub type MemFreeFn = unsafe extern "C" fn(ocgpuDeviceptr) -> ocgpuResult;
/// `hipMalloc` ABI used by the raw runtime table.
pub type MallocFn = unsafe extern "C" fn(*mut *mut c_void, usize) -> ocgpuResult;
/// `hipFree` ABI used by the raw runtime table.
pub type FreeFn = unsafe extern "C" fn(*mut c_void) -> ocgpuResult;

static MALLOC_TARGET: OnceLock<MallocFn> = OnceLock::new();
static FREE_TARGET: OnceLock<FreeFn> = OnceLock::new();

unsafe extern "C" fn common_mem_alloc(
    device_ptr: *mut ocgpuDeviceptr,
    bytes: usize,
) -> ocgpuResult {
    if device_ptr.is_null() {
        return ocgpu_abi::OCGPU_ERROR_INVALID_ARGUMENT;
    }
    let Some(target) = MALLOC_TARGET.get() else {
        return ocgpu_abi::OCGPU_ERROR_INTERNAL;
    };
    let mut native_ptr = std::ptr::null_mut();
    // SAFETY: output storage is local and writable, and the process-lifetime
    // target has the exact raw `hipMalloc` ABI.
    let result = unsafe { target(&raw mut native_ptr, bytes) };
    if result == HIP_SUCCESS {
        // SAFETY: the caller supplied non-null writable output for the unified
        // pointer-sized integer. Every bit pattern is a valid `usize`.
        unsafe { device_ptr.write(native_ptr as usize) };
    }
    result
}

unsafe extern "C" fn common_mem_free(device_ptr: ocgpuDeviceptr) -> ocgpuResult {
    let Some(target) = FREE_TARGET.get() else {
        return ocgpu_abi::OCGPU_ERROR_INTERNAL;
    };
    // SAFETY: this reverses the lossless pointer-to-`usize` conversion made by
    // `common_mem_alloc`; the raw target consumes the opaque value only.
    unsafe { target(device_ptr as *mut c_void) }
}
/// Unified host-to-device copy ABI adapted over HIP's pointer-typed device ABI.
pub type MemcpyHtoDFn = unsafe extern "C" fn(ocgpuDeviceptr, *const c_void, usize) -> ocgpuResult;
type NativeMemcpyHtoDFn = unsafe extern "C" fn(*mut c_void, *const c_void, usize) -> ocgpuResult;
/// HIP 5/6 `hipMemcpyHtoD` ABI, whose source parameter predates const-correctness.
type LegacyMemcpyHtoDFn = unsafe extern "C" fn(*mut c_void, *mut c_void, usize) -> ocgpuResult;
static NATIVE_MEMCPY_HTOD_TARGET: OnceLock<NativeMemcpyHtoDFn> = OnceLock::new();
static LEGACY_MEMCPY_HTOD_TARGET: OnceLock<LegacyMemcpyHtoDFn> = OnceLock::new();

unsafe extern "C" fn common_native_memcpy_htod(
    destination: ocgpuDeviceptr,
    source: *const c_void,
    bytes: usize,
) -> ocgpuResult {
    let Some(target) = NATIVE_MEMCPY_HTOD_TARGET.get() else {
        return ocgpu_abi::OCGPU_ERROR_INTERNAL;
    };
    // SAFETY: the unified integer is the lossless representation of HIP's
    // opaque pointer returned by `hipMalloc`; the remaining arguments match.
    unsafe { target(destination as *mut c_void, source, bytes) }
}

unsafe extern "C" fn common_legacy_memcpy_htod(
    destination: ocgpuDeviceptr,
    source: *const c_void,
    bytes: usize,
) -> ocgpuResult {
    let Some(target) = LEGACY_MEMCPY_HTOD_TARGET.get() else {
        return ocgpu_abi::OCGPU_ERROR_INTERNAL;
    };
    // SAFETY: HIP 5/6 declared the input-only host source as `void *` before
    // HIP 7 made that contract const-correct. The operation reads `bytes` from
    // the source and does not acquire permission to mutate it; only the raw
    // pointer qualification is adapted for the exact legacy function type.
    unsafe { target(destination as *mut c_void, source.cast_mut(), bytes) }
}
/// Unified device-to-host copy ABI adapted over HIP's pointer-typed device ABI.
pub type MemcpyDtoHFn = unsafe extern "C" fn(*mut c_void, ocgpuDeviceptr, usize) -> ocgpuResult;
type NativeMemcpyDtoHFn = unsafe extern "C" fn(*mut c_void, *mut c_void, usize) -> ocgpuResult;
static NATIVE_MEMCPY_DTOH_TARGET: OnceLock<NativeMemcpyDtoHFn> = OnceLock::new();

unsafe extern "C" fn common_native_memcpy_dtoh(
    destination: *mut c_void,
    source: ocgpuDeviceptr,
    bytes: usize,
) -> ocgpuResult {
    let Some(target) = NATIVE_MEMCPY_DTOH_TARGET.get() else {
        return ocgpu_abi::OCGPU_ERROR_INTERNAL;
    };
    // SAFETY: the unified integer is the lossless representation of HIP's
    // opaque pointer returned by `hipMalloc`; the remaining arguments match.
    unsafe { target(destination, source as *mut c_void, bytes) }
}
/// `hipStreamCreateWithFlags` ABI.
pub type StreamCreateFn = unsafe extern "C" fn(*mut ocgpuStream, u32) -> ocgpuResult;
/// `hipStreamDestroy` ABI.
pub type StreamDestroyFn = unsafe extern "C" fn(ocgpuStream) -> ocgpuResult;
/// `hipStreamSynchronize` ABI.
pub type StreamSynchronizeFn = unsafe extern "C" fn(ocgpuStream) -> ocgpuResult;
/// `hipEventCreateWithFlags` ABI.
pub type EventCreateFn = unsafe extern "C" fn(*mut ocgpuEvent, u32) -> ocgpuResult;
/// `hipEventDestroy` ABI.
pub type EventDestroyFn = unsafe extern "C" fn(ocgpuEvent) -> ocgpuResult;
/// `hipEventRecord` ABI.
pub type EventRecordFn = unsafe extern "C" fn(ocgpuEvent, ocgpuStream) -> ocgpuResult;
/// `hipEventSynchronize` ABI.
pub type EventSynchronizeFn = unsafe extern "C" fn(ocgpuEvent) -> ocgpuResult;
/// `hipModuleLoadData` ABI.
pub type ModuleLoadDataFn = unsafe extern "C" fn(*mut ocgpuModule, *const c_void) -> ocgpuResult;
/// `hipModuleUnload` ABI.
pub type ModuleUnloadFn = unsafe extern "C" fn(ocgpuModule) -> ocgpuResult;
/// `hipModuleGetFunction` ABI.
pub type ModuleGetFunctionFn =
    unsafe extern "C" fn(*mut ocgpuFunction, ocgpuModule, *const c_char) -> ocgpuResult;
/// `hipModuleLaunchKernel` ABI.
pub type LaunchKernelFn = unsafe extern "C" fn(
    ocgpuFunction,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    ocgpuStream,
    *mut *mut c_void,
    *mut *mut c_void,
) -> ocgpuResult;

type GetProcAddressFn =
    unsafe extern "C" fn(*const c_char, *mut *mut c_void, i32, u64, *mut i32) -> ocgpuResult;

/// Interpreted HIP proc-address query status.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QueryStatus {
    /// A compatible entry point was found.
    Success,
    /// The runtime did not recognize the symbol.
    SymbolNotFound,
    /// The symbol exists but not for the requested API version.
    VersionInsufficient,
    /// A newer runtime returned a status unknown to this build.
    Unknown(i32),
}

impl From<i32> for QueryStatus {
    fn from(value: i32) -> Self {
        match value {
            0 => Self::Success,
            1 => Self::SymbolNotFound,
            2 => Self::VersionInsufficient,
            other => Self::Unknown(other),
        }
    }
}

/// One version-aware lookup attempt.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProcAttempt {
    /// Requested HIP version encoding.
    pub requested_version: i32,
    /// `hipError_t` returned by `hipGetProcAddress`.
    pub call_result: ocgpuResult,
    /// Query status returned separately by the runtime.
    pub query_status: QueryStatus,
    /// Whether the call returned a non-null pointer.
    pub returned_pointer: bool,
}

/// How a symbol was ultimately resolved.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResolutionKind {
    /// Returned by `hipGetProcAddress`.
    ProcAddress,
    /// Found directly in the shared library export table.
    Direct,
    /// Found directly with an exact legacy ABI and exposed through a typed
    /// common-profile adapter; the newer-shaped raw table slot remains null.
    DirectAdapter,
    /// Neither version-aware nor direct lookup found the symbol.
    Missing,
    /// The compiled manifest excludes this symbol on the current platform.
    PlatformUnavailable,
    /// The selected runtime profile deliberately excludes this ABI slot.
    ProfileUnavailable,
}

/// Resolution diagnostics for one manifest entry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SymbolReport {
    /// Canonical vendor entry-point name.
    pub canonical_name: &'static str,
    /// Export or proc-address name that succeeded.
    pub resolved_name: Option<&'static str>,
    /// Successful resolution path, or `Missing`.
    pub resolution: ResolutionKind,
    /// Whether the entry point was resolved.
    pub available: bool,
    /// Whether the manifest marks this entry point applicable to this platform.
    pub applicable: bool,
    /// Whether ABI v1 common-core validation requires this entry point.
    pub required: bool,
    /// Exact reviewed proc-address attempt made before direct fallback.
    pub proc_attempts: Vec<ProcAttempt>,
}

/// Immutable backend-load diagnostics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Diagnostics {
    /// Loaded library path or Linux system-loader identity.
    pub library_path: PathBuf,
    /// Exact runtime-major ABI profile selected during version bootstrap.
    pub runtime_profile: RuntimeProfile,
    /// Runtime version from `hipRuntimeGetVersion`, when available.
    pub runtime_version: Option<i32>,
    /// Driver version from `hipDriverGetVersion`, when available.
    pub driver_version: Option<i32>,
    /// Whether proc-address resolution is enabled and usable for the selected
    /// profile. HIP 5/6 remain direct-only even if an export happens to exist.
    pub proc_address_support: bool,
    /// Architecture accepted by the OS loader for this process.
    pub loaded_architecture: &'static str,
    /// Resolution result for every emitted vendor/runtime manifest symbol.
    pub symbols: Vec<SymbolReport>,
}

impl Diagnostics {
    /// Finds the diagnostic record for a canonical symbol.
    #[must_use]
    pub fn symbol(&self, canonical_name: &str) -> Option<&SymbolReport> {
        self.symbols
            .iter()
            .find(|report| report.canonical_name == canonical_name)
    }
}

/// Backend construction or validation failure.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// Secure OS-loader failure.
    Loader(LoadError),
    /// The library loaded, but version-aware lookup proved the common profile
    /// requires a newer runtime.
    BackendTooOld {
        /// Loaded library identity.
        library: PathBuf,
        /// Canonical names of unavailable required entry points.
        symbols: Vec<&'static str>,
    },
    /// The library loaded, but it did not satisfy the common core profile.
    MissingCoreSymbols {
        /// Loaded library identity.
        library: PathBuf,
        /// Canonical names of all missing required entry points.
        symbols: Vec<&'static str>,
    },
    /// The runtime major is newer than, or otherwise outside, the closed set of
    /// ABI profiles understood by this build.
    UnsupportedRuntimeProfile {
        /// Loaded library identity.
        library: PathBuf,
        /// Full integer returned by `hipRuntimeGetVersion`.
        runtime_version: i32,
        /// Major decoded from `runtime_version`.
        runtime_major: i32,
    },
    /// A versioned library name did not contain its claimed runtime major.
    RuntimeProfileMismatch {
        /// Loaded library identity.
        library: PathBuf,
        /// Profile claimed by the versioned library name.
        expected: RuntimeProfile,
        /// Profile reported by `hipRuntimeGetVersion`.
        detected: RuntimeProfile,
        /// Full integer returned by `hipRuntimeGetVersion`.
        runtime_version: i32,
    },
    /// The runtime major is known, but the exact version is outside the
    /// compatibility interval reviewed for that profile.
    UnsupportedRuntimeVersion {
        /// Loaded library identity.
        library: PathBuf,
        /// Profile family selected from the reported major.
        runtime_profile: RuntimeProfile,
        /// Full integer returned by `hipRuntimeGetVersion`.
        runtime_version: i32,
        /// Lowest reviewed version, inclusive.
        minimum_supported: i32,
        /// Highest reviewed version, inclusive.
        maximum_supported: i32,
    },
    /// Generated raw-table metadata did not identify a valid function slot.
    InvalidRawTableDescriptor {
        /// Manifest symbol whose field could not be populated.
        symbol: &'static str,
        /// Bounds or alignment failure for its generated field offset.
        source: TableSlotError,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Loader(error) => write!(formatter, "HIP loader error: {error}"),
            Self::BackendTooOld { library, symbols } => write!(
                formatter,
                "HIP library {} is too old for core symbols: {}",
                library.display(),
                symbols.join(", ")
            ),
            Self::MissingCoreSymbols { library, symbols } => write!(
                formatter,
                "HIP library {} is missing core symbols: {}",
                library.display(),
                symbols.join(", ")
            ),
            Self::UnsupportedRuntimeProfile {
                library,
                runtime_version,
                runtime_major,
            } => write!(
                formatter,
                "HIP library {} reports unsupported runtime version {runtime_version} (major {runtime_major})",
                library.display()
            ),
            Self::RuntimeProfileMismatch {
                library,
                expected,
                detected,
                runtime_version,
            } => write!(
                formatter,
                "HIP library {} claims the {expected:?} profile but reports {detected:?} runtime version {runtime_version}",
                library.display()
            ),
            Self::UnsupportedRuntimeVersion {
                library,
                runtime_profile,
                runtime_version,
                minimum_supported,
                maximum_supported,
            } => write!(
                formatter,
                "HIP library {} reports {runtime_profile:?} version {runtime_version}, outside the reviewed interval {minimum_supported}..={maximum_supported}",
                library.display()
            ),
            Self::InvalidRawTableDescriptor { symbol, source } => {
                write!(
                    formatter,
                    "invalid HIP raw-table descriptor for {symbol}: {source}"
                )
            }
        }
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Loader(error) => Some(error),
            Self::BackendTooOld { .. }
            | Self::MissingCoreSymbols { .. }
            | Self::UnsupportedRuntimeProfile { .. }
            | Self::RuntimeProfileMismatch { .. }
            | Self::UnsupportedRuntimeVersion { .. } => None,
            Self::InvalidRawTableDescriptor { source, .. } => Some(source),
        }
    }
}

impl From<LoadError> for Error {
    fn from(value: LoadError) -> Self {
        Self::Loader(value)
    }
}

impl Error {
    /// Whether this error reports a known profile older than its reviewed floor.
    #[must_use]
    pub const fn below_minimum_runtime_version(&self) -> bool {
        match self {
            Self::UnsupportedRuntimeVersion {
                runtime_version,
                minimum_supported,
                ..
            } => *runtime_version < *minimum_supported,
            _ => false,
        }
    }
}

/// Optional raw HIP driver-shaped entry points before core validation.
pub struct UnvalidatedApi {
    /// Load and symbol-resolution diagnostics.
    pub diagnostics: Diagnostics,
    /// Optional `hipRuntimeGetVersion` helper.
    pub runtime_get_version: Option<RuntimeGetVersionFn>,
    /// Optional `hipInit`.
    pub init: Option<InitFn>,
    /// Optional `hipDriverGetVersion`.
    pub driver_get_version: Option<DriverGetVersionFn>,
    /// Optional `hipGetDeviceCount`.
    pub device_get_count: Option<DeviceGetCountFn>,
    /// Optional `hipDeviceGet`.
    pub device_get: Option<DeviceGetFn>,
    /// Optional `hipDeviceGetName`.
    pub device_get_name: Option<DeviceGetNameFn>,
    /// Optional `hipDeviceGetAttribute`.
    pub device_get_attribute: Option<DeviceGetAttributeFn>,
    /// Optional `hipCtxCreate`.
    pub ctx_create: Option<CtxCreateFn>,
    /// Optional `hipCtxDestroy`.
    pub ctx_destroy: Option<CtxDestroyFn>,
    /// Optional `hipCtxSetCurrent`.
    pub ctx_set_current: Option<CtxSetCurrentFn>,
    /// Optional `hipCtxGetCurrent`.
    pub ctx_get_current: Option<CtxGetCurrentFn>,
    /// Optional `hipCtxSynchronize`.
    pub ctx_synchronize: Option<CtxSynchronizeFn>,
    /// Optional unified adapter over `hipMalloc`.
    pub mem_alloc: Option<MemAllocFn>,
    /// Optional unified adapter over `hipFree`.
    pub mem_free: Option<MemFreeFn>,
    /// Optional raw-runtime `hipMalloc`.
    pub malloc: Option<MallocFn>,
    /// Optional raw-runtime `hipFree`.
    pub free: Option<FreeFn>,
    /// Optional `hipMemcpyHtoD`.
    pub memcpy_htod: Option<MemcpyHtoDFn>,
    /// Optional `hipMemcpyDtoH`.
    pub memcpy_dtoh: Option<MemcpyDtoHFn>,
    /// Optional `hipStreamCreateWithFlags`.
    pub stream_create: Option<StreamCreateFn>,
    /// Optional `hipStreamDestroy`.
    pub stream_destroy: Option<StreamDestroyFn>,
    /// Optional `hipStreamSynchronize`.
    pub stream_synchronize: Option<StreamSynchronizeFn>,
    /// Optional `hipEventCreateWithFlags`.
    pub event_create: Option<EventCreateFn>,
    /// Optional `hipEventDestroy`.
    pub event_destroy: Option<EventDestroyFn>,
    /// Optional `hipEventRecord`.
    pub event_record: Option<EventRecordFn>,
    /// Optional `hipEventSynchronize`.
    pub event_synchronize: Option<EventSynchronizeFn>,
    /// Optional `hipModuleLoadData`.
    pub module_load_data: Option<ModuleLoadDataFn>,
    /// Optional `hipModuleUnload`.
    pub module_unload: Option<ModuleUnloadFn>,
    /// Optional `hipModuleGetFunction`.
    pub module_get_function: Option<ModuleGetFunctionFn>,
    /// Optional `hipModuleLaunchKernel`.
    pub launch_kernel: Option<LaunchKernelFn>,
    raw_table: ocgpuHipApi_v1,
}

impl fmt::Debug for UnvalidatedApi {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("UnvalidatedApi")
            .field("diagnostics", &self.diagnostics)
            .finish_non_exhaustive()
    }
}

impl UnvalidatedApi {
    /// Produces a raw HIP ABI table with null entries for unavailable symbols.
    #[must_use]
    pub fn raw_table(&self) -> ocgpuHipApi_v1 {
        self.raw_table
    }

    /// Borrows the process-lifetime raw HIP table without copying it.
    #[must_use]
    pub const fn raw_table_ref(&self) -> &ocgpuHipApi_v1 {
        &self.raw_table
    }

    fn metadata_prefix(&self) -> (u32, u32, u32, u32, i32) {
        (
            u32::try_from(size_of::<ocgpuApi_v1>()).unwrap_or(u32::MAX),
            OCGPU_ABI_VERSION_1,
            OCGPU_BACKEND_HIP,
            self.diagnostics.runtime_profile.api_flags(),
            self.diagnostics.driver_version.unwrap_or(0),
        )
    }

    fn common_layout_table(&self) -> ocgpuApi_v1 {
        let (struct_size, abi_version, backend, flags, driver_version) = self.metadata_prefix();
        ocgpuApi_v1 {
            struct_size,
            abi_version,
            backend,
            flags,
            driver_version,
            reserved0: 0,
            ocgpuInit: self.init,
            ocgpuDriverGetVersion: self.driver_get_version,
            ocgpuDeviceGetCount: self.device_get_count,
            ocgpuDeviceGet: self.device_get,
            ocgpuDeviceGetName: self.device_get_name,
            ocgpuDeviceGetAttribute: self.device_get_attribute,
            ocgpuCtxCreate: self.ctx_create,
            ocgpuCtxDestroy: self.ctx_destroy,
            ocgpuCtxSetCurrent: self.ctx_set_current,
            ocgpuCtxGetCurrent: self.ctx_get_current,
            ocgpuCtxSynchronize: self.ctx_synchronize,
            ocgpuMemAlloc: self.mem_alloc,
            ocgpuMemFree: self.mem_free,
            ocgpuMemcpyHtoD: self.memcpy_htod,
            ocgpuMemcpyDtoH: self.memcpy_dtoh,
            ocgpuStreamCreate: self.stream_create,
            ocgpuStreamDestroy: self.stream_destroy,
            ocgpuStreamSynchronize: self.stream_synchronize,
            ocgpuEventCreate: self.event_create,
            ocgpuEventDestroy: self.event_destroy,
            ocgpuEventRecord: self.event_record,
            ocgpuEventSynchronize: self.event_synchronize,
            ocgpuModuleLoadData: self.module_load_data,
            ocgpuModuleUnload: self.module_unload,
            ocgpuModuleGetFunction: self.module_get_function,
            ocgpuLaunchKernel: self.launch_kernel,
        }
    }
}

/// Non-null HIP common-core dispatch.
#[derive(Debug)]
pub struct ValidatedCoreApi {
    raw: &'static UnvalidatedApi,
    /// Validated `hipInit`.
    pub init: InitFn,
    /// Validated `hipDriverGetVersion`.
    pub driver_get_version: DriverGetVersionFn,
    /// Validated `hipGetDeviceCount`.
    pub device_get_count: DeviceGetCountFn,
    /// Validated `hipDeviceGet`.
    pub device_get: DeviceGetFn,
    /// Validated `hipDeviceGetName`.
    pub device_get_name: DeviceGetNameFn,
    /// Validated unified-attribute adapter over `hipDeviceGetAttribute`.
    pub device_get_attribute: DeviceGetAttributeFn,
    /// Validated `hipCtxCreate`.
    pub ctx_create: CtxCreateFn,
    /// Validated `hipCtxDestroy`.
    pub ctx_destroy: CtxDestroyFn,
    /// Validated `hipCtxSetCurrent`.
    pub ctx_set_current: CtxSetCurrentFn,
    /// Validated `hipCtxGetCurrent`.
    pub ctx_get_current: CtxGetCurrentFn,
    /// Validated `hipCtxSynchronize`.
    pub ctx_synchronize: CtxSynchronizeFn,
    /// Validated unified allocation adapter over `hipMalloc`.
    pub mem_alloc: MemAllocFn,
    /// Validated unified free adapter over `hipFree`.
    pub mem_free: MemFreeFn,
    /// Validated `hipMemcpyHtoD`.
    pub memcpy_htod: MemcpyHtoDFn,
    /// Validated `hipMemcpyDtoH`.
    pub memcpy_dtoh: MemcpyDtoHFn,
    /// Validated `hipStreamCreateWithFlags`.
    pub stream_create: StreamCreateFn,
    /// Validated `hipStreamDestroy`.
    pub stream_destroy: StreamDestroyFn,
    /// Validated `hipStreamSynchronize`.
    pub stream_synchronize: StreamSynchronizeFn,
    /// Validated `hipEventCreateWithFlags`.
    pub event_create: EventCreateFn,
    /// Validated `hipEventDestroy`.
    pub event_destroy: EventDestroyFn,
    /// Validated `hipEventRecord`.
    pub event_record: EventRecordFn,
    /// Validated `hipEventSynchronize`.
    pub event_synchronize: EventSynchronizeFn,
    /// Validated `hipModuleLoadData`.
    pub module_load_data: ModuleLoadDataFn,
    /// Validated `hipModuleUnload`.
    pub module_unload: ModuleUnloadFn,
    /// Validated `hipModuleGetFunction`.
    pub module_get_function: ModuleGetFunctionFn,
    /// Validated `hipModuleLaunchKernel`.
    pub launch_kernel: LaunchKernelFn,
}

impl ValidatedCoreApi {
    /// Optional/raw table used to construct this validated table.
    #[must_use]
    pub const fn raw(&self) -> &'static UnvalidatedApi {
        self.raw
    }

    /// Backend-load and symbol-resolution diagnostics.
    #[must_use]
    pub fn diagnostics(&self) -> &Diagnostics {
        &self.raw.diagnostics
    }

    /// Exact HIP runtime-major profile selected for this dispatch table.
    #[must_use]
    pub const fn runtime_profile(&self) -> RuntimeProfile {
        self.raw.diagnostics.runtime_profile
    }

    /// Produces the immutable backend-bound unified ABI table.
    #[must_use]
    pub fn common_table(&self) -> ocgpuApi_v1 {
        let mut table = self.raw.common_layout_table();
        table.ocgpuDeviceGetAttribute = Some(self.device_get_attribute);
        table.ocgpuMemAlloc = Some(self.mem_alloc);
        table.ocgpuMemFree = Some(self.mem_free);
        table
    }

    /// Produces the raw HIP ABI table.
    #[must_use]
    pub fn raw_table(&self) -> ocgpuHipApi_v1 {
        self.raw.raw_table()
    }

    /// Borrows the process-lifetime raw HIP table without copying it.
    #[must_use]
    pub const fn raw_table_ref(&self) -> &ocgpuHipApi_v1 {
        self.raw.raw_table_ref()
    }
}

static UNVALIDATED: OnceLock<Result<UnvalidatedApi, Error>> = OnceLock::new();
static VALIDATED: OnceLock<Result<ValidatedCoreApi, Error>> = OnceLock::new();

/// Loads HIP and resolves every platform-applicable manifest entry point.
pub fn load_unvalidated() -> Result<&'static UnvalidatedApi, Error> {
    let result = UNVALIDATED.get_or_init(|| {
        let library = ocgpu_loader::load(Backend::Hip)?;
        build_from_library(library)
    });
    clone_result_ref(result)
}

/// Loads HIP and validates the complete ABI v1 common core exactly once.
pub fn load() -> Result<&'static ValidatedCoreApi, Error> {
    let result = VALIDATED.get_or_init(|| validate(load_unvalidated()?));
    clone_result_ref(result)
}

/// Loads HIP from an explicit canonicalized absolute path and validates it.
///
/// # Safety
///
/// The selected library and all of its dependencies must be trusted native code
/// that is safe to initialize and implements the exact HIP ABIs represented by
/// this crate's generated table. The library must not be replaced with
/// incompatible code between path validation and operating-system loading.
#[cfg(feature = "explicit-library-path")]
pub unsafe fn load_from_absolute(path: &Path) -> Result<&'static ValidatedCoreApi, Error> {
    // SAFETY: this function has the same trusted-library and exact-ABI contract.
    let unvalidated = unsafe { load_unvalidated_from_absolute(path) }?;
    let result = VALIDATED.get_or_init(|| validate(unvalidated));
    clone_result_ref(result)
}

/// Loads HIP's optional raw inventory from an explicit absolute path.
///
/// # Safety
///
/// The selected library and all of its dependencies must be trusted native code
/// that is safe to initialize and implements the exact HIP ABIs represented by
/// this crate's generated table. The library must not be replaced with
/// incompatible code between path validation and operating-system loading.
#[cfg(feature = "explicit-library-path")]
pub unsafe fn load_unvalidated_from_absolute(
    path: &Path,
) -> Result<&'static UnvalidatedApi, Error> {
    // SAFETY: the caller guarantees the library trust and ABI requirements.
    let library = unsafe { ocgpu_loader::load_from_absolute(Backend::Hip, path) }?;
    let unvalidated = UNVALIDATED.get_or_init(|| build_from_library(library));
    clone_result_ref(unvalidated)
}

/// Returns diagnostics after initializing the unvalidated table.
pub fn diagnostics() -> Result<&'static Diagnostics, Error> {
    load_unvalidated().map(|api| &api.diagnostics)
}

fn clone_result_ref<T>(result: &'static Result<T, Error>) -> Result<&'static T, Error> {
    match result {
        Ok(value) => Ok(value),
        Err(error) => Err(error.clone()),
    }
}

#[allow(clippy::similar_names, clippy::too_many_lines)]
fn validate(raw: &'static UnvalidatedApi) -> Result<ValidatedCoreApi, Error> {
    let mut missing = Vec::new();
    macro_rules! check {
        ($field:ident, $name:literal) => {
            if raw.$field.is_none() {
                missing.push($name);
            }
        };
    }
    check!(init, "hipInit");
    check!(driver_get_version, "hipDriverGetVersion");
    check!(device_get_count, "hipGetDeviceCount");
    check!(device_get, "hipDeviceGet");
    check!(device_get_name, "hipDeviceGetName");
    check!(device_get_attribute, "hipDeviceGetAttribute");
    check!(ctx_create, "hipCtxCreate");
    check!(ctx_destroy, "hipCtxDestroy");
    check!(ctx_set_current, "hipCtxSetCurrent");
    check!(ctx_get_current, "hipCtxGetCurrent");
    check!(ctx_synchronize, "hipCtxSynchronize");
    check!(mem_alloc, "hipMalloc");
    check!(mem_free, "hipFree");
    check!(memcpy_htod, "hipMemcpyHtoD");
    check!(memcpy_dtoh, "hipMemcpyDtoH");
    check!(stream_create, "hipStreamCreateWithFlags");
    check!(stream_destroy, "hipStreamDestroy");
    check!(stream_synchronize, "hipStreamSynchronize");
    check!(event_create, "hipEventCreateWithFlags");
    check!(event_destroy, "hipEventDestroy");
    check!(event_record, "hipEventRecord");
    check!(event_synchronize, "hipEventSynchronize");
    check!(module_load_data, "hipModuleLoadData");
    check!(module_unload, "hipModuleUnload");
    check!(module_get_function, "hipModuleGetFunction");
    check!(launch_kernel, "hipModuleLaunchKernel");
    if !missing.is_empty() {
        let version_insufficient = missing.iter().any(|name| {
            raw.diagnostics.symbol(name).is_some_and(|report| {
                report
                    .proc_attempts
                    .iter()
                    .any(|attempt| attempt.query_status == QueryStatus::VersionInsufficient)
            })
        });
        let library = raw.diagnostics.library_path.clone();
        return Err(if version_insufficient {
            Error::BackendTooOld {
                library,
                symbols: missing,
            }
        } else {
            Error::MissingCoreSymbols {
                library,
                symbols: missing,
            }
        });
    }

    let (
        Some(init),
        Some(driver_get_version),
        Some(device_get_count),
        Some(device_get),
        Some(device_get_name),
        Some(device_get_attribute),
        Some(ctx_create),
        Some(ctx_destroy),
        Some(ctx_set_current),
        Some(ctx_get_current),
        Some(ctx_synchronize),
        Some(mem_alloc),
        Some(mem_free),
        Some(memcpy_htod),
        Some(memcpy_dtoh),
        Some(stream_create),
        Some(stream_destroy),
        Some(stream_synchronize),
        Some(event_create),
        Some(event_destroy),
        Some(event_record),
        Some(event_synchronize),
        Some(module_load_data),
        Some(module_unload),
        Some(module_get_function),
        Some(launch_kernel),
    ) = (
        raw.init,
        raw.driver_get_version,
        raw.device_get_count,
        raw.device_get,
        raw.device_get_name,
        raw.device_get_attribute,
        raw.ctx_create,
        raw.ctx_destroy,
        raw.ctx_set_current,
        raw.ctx_get_current,
        raw.ctx_synchronize,
        raw.mem_alloc,
        raw.mem_free,
        raw.memcpy_htod,
        raw.memcpy_dtoh,
        raw.stream_create,
        raw.stream_destroy,
        raw.stream_synchronize,
        raw.event_create,
        raw.event_destroy,
        raw.event_record,
        raw.event_synchronize,
        raw.module_load_data,
        raw.module_unload,
        raw.module_get_function,
        raw.launch_kernel,
    )
    else {
        return Err(Error::MissingCoreSymbols {
            library: raw.diagnostics.library_path.clone(),
            symbols: Vec::new(),
        });
    };

    DEVICE_GET_ATTRIBUTE_TARGET.get_or_init(|| device_get_attribute);

    Ok(ValidatedCoreApi {
        raw,
        init,
        driver_get_version,
        device_get_count,
        device_get,
        device_get_name,
        device_get_attribute: common_device_get_attribute,
        ctx_create,
        ctx_destroy,
        ctx_set_current,
        ctx_get_current,
        ctx_synchronize,
        mem_alloc,
        mem_free,
        memcpy_htod,
        memcpy_dtoh,
        stream_create,
        stream_destroy,
        stream_synchronize,
        event_create,
        event_destroy,
        event_record,
        event_synchronize,
        module_load_data,
        module_unload,
        module_get_function,
        launch_kernel,
    })
}

#[derive(Clone, Copy)]
struct SymbolSpec {
    canonical: &'static str,
    proc_name: &'static str,
    direct_names: &'static [&'static str],
    proc_flags: u64,
    required: bool,
}

struct ProcLookup {
    address: Option<NonNull<c_void>>,
    attempt: ProcAttempt,
}

trait SymbolSource {
    fn proc_supported(&self) -> bool;
    fn proc_lookup(&self, name: &str, api_version: i32, flags: u64) -> ProcLookup;
    fn direct_lookup(&self, name: &str) -> Option<NonNull<c_void>>;
}

#[derive(Clone, Copy, Debug)]
struct RuntimeBootstrap {
    runtime_get_version: RuntimeGetVersionFn,
    runtime_version: i32,
    runtime_profile: RuntimeProfile,
    proc_address_permitted: bool,
    full_raw_inventory: bool,
    address: NonNull<c_void>,
}

fn bootstrap_runtime<S: SymbolSource>(
    source: &S,
    library: &Path,
    expected: Option<RuntimeProfile>,
) -> Result<RuntimeBootstrap, Error> {
    let runtime_version_symbol = "hipRuntimeGetVersion";
    if !HIP_BOOTSTRAP_SYMBOLS.contains(&runtime_version_symbol) {
        return Err(Error::MissingCoreSymbols {
            library: library.to_path_buf(),
            symbols: vec![runtime_version_symbol],
        });
    }
    let Some(address) = source.direct_lookup(runtime_version_symbol) else {
        return Err(Error::MissingCoreSymbols {
            library: library.to_path_buf(),
            symbols: vec![runtime_version_symbol],
        });
    };
    // SAFETY: `hipRuntimeGetVersion` has retained this output-only C ABI across
    // every supported profile and is the sole callable used for classification.
    let runtime_get_version =
        unsafe { std::mem::transmute::<*mut c_void, RuntimeGetVersionFn>(address.as_ptr()) };
    let mut runtime_version = 0;
    // SAFETY: the output points to initialized writable storage and the exact
    // direct export was converted to its stable bootstrap ABI above.
    let result = unsafe { runtime_get_version(&raw mut runtime_version) };
    if result != HIP_SUCCESS || runtime_version <= 0 {
        return Err(Error::MissingCoreSymbols {
            library: library.to_path_buf(),
            symbols: vec![runtime_version_symbol],
        });
    }

    let runtime_major = runtime_version / HIP_VERSION_MAJOR_SCALE;
    let descriptor = HIP_RUNTIME_PROFILES
        .iter()
        .find(|descriptor| descriptor.runtime_major == runtime_major);
    let (descriptor, runtime_profile) = match descriptor {
        Some(descriptor) if descriptor.runtime_major == 5 => (descriptor, RuntimeProfile::Hip5),
        Some(descriptor) if descriptor.runtime_major == 6 => (descriptor, RuntimeProfile::Hip6),
        Some(descriptor) if descriptor.runtime_major == 7 => (descriptor, RuntimeProfile::Hip7),
        None if runtime_major < RuntimeProfile::Hip5.runtime_major() => {
            return Err(Error::BackendTooOld {
                library: library.to_path_buf(),
                symbols: vec![runtime_version_symbol],
            });
        }
        Some(_) | None => {
            return Err(Error::UnsupportedRuntimeProfile {
                library: library.to_path_buf(),
                runtime_version,
                runtime_major,
            });
        }
    };
    if let Some(expected) = expected {
        if expected != runtime_profile {
            return Err(Error::RuntimeProfileMismatch {
                library: library.to_path_buf(),
                expected,
                detected: runtime_profile,
                runtime_version,
            });
        }
    }
    if descriptor.table_flag != runtime_profile.api_flags()
        || !descriptor
            .bootstrap_symbols
            .contains(&runtime_version_symbol)
    {
        return Err(Error::UnsupportedRuntimeProfile {
            library: library.to_path_buf(),
            runtime_version,
            runtime_major,
        });
    }
    let platform_profile = platform_runtime_profile(descriptor);
    if !(platform_profile.runtime_version_min_inclusive
        ..=platform_profile.runtime_version_max_inclusive)
        .contains(&runtime_version)
    {
        return Err(Error::UnsupportedRuntimeVersion {
            library: library.to_path_buf(),
            runtime_profile,
            runtime_version,
            minimum_supported: platform_profile.runtime_version_min_inclusive,
            maximum_supported: platform_profile.runtime_version_max_inclusive,
        });
    }
    Ok(RuntimeBootstrap {
        runtime_get_version,
        runtime_version,
        runtime_profile,
        proc_address_permitted: runtime_profile == RuntimeProfile::Hip7
            && platform_profile
                .proc_address_min_inclusive
                .is_some_and(|minimum| runtime_version >= minimum),
        full_raw_inventory: platform_profile
            .raw_inventory_min_inclusive
            .is_some_and(|minimum| runtime_version >= minimum),
        address,
    })
}

fn expected_runtime_profile(library: &Path) -> Option<RuntimeProfile> {
    let file_name = library.file_name()?.to_str()?;
    let mut matched = None;
    for descriptor in HIP_RUNTIME_PROFILES {
        let platform_profile = platform_runtime_profile(descriptor);
        if platform_profile
            .library_candidates
            .iter()
            .any(|candidate| library_name_matches(file_name, candidate))
        {
            let profile = match descriptor.runtime_major {
                5 => RuntimeProfile::Hip5,
                6 => RuntimeProfile::Hip6,
                7 => RuntimeProfile::Hip7,
                _ => return None,
            };
            if matched.is_some_and(|previous| previous != profile) {
                // An unversioned SONAME shared by profiles claims no major.
                return None;
            }
            matched = Some(profile);
        }
    }
    matched
}

fn library_name_matches(file_name: &str, candidate: &str) -> bool {
    #[cfg(target_os = "windows")]
    {
        file_name.eq_ignore_ascii_case(candidate)
    }
    #[cfg(target_os = "linux")]
    {
        file_name == candidate
            || (candidate != "libamdhip64.so"
                && file_name
                    .strip_prefix(candidate)
                    .is_some_and(|suffix| suffix.starts_with('.')))
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        let _ = (file_name, candidate);
        false
    }
}

struct LibrarySource {
    library: &'static Library,
    get_proc_address: Option<GetProcAddressFn>,
}

impl LibrarySource {
    fn new(library: &'static Library, proc_address_permitted: bool) -> Self {
        let get_proc_address = (proc_address_permitted
            && inventory_platform_applicable("hipGetProcAddress"))
        .then(|| library.find(b"hipGetProcAddress").ok().flatten())
        .flatten()
        .map(|address| {
            // SAFETY: this direct export is defined by HIP with exactly the
            // `GetProcAddressFn` ABI declared above.
            unsafe { std::mem::transmute::<*mut c_void, GetProcAddressFn>(address.as_ptr()) }
        });
        Self {
            library,
            get_proc_address,
        }
    }
}

fn build_from_library(library: &'static Library) -> Result<UnvalidatedApi, Error> {
    let library_path = library.loaded_path().to_path_buf();
    let bootstrap_source = LibrarySource {
        library,
        get_proc_address: None,
    };
    let bootstrap = bootstrap_runtime(
        &bootstrap_source,
        &library_path,
        expected_runtime_profile(&library_path),
    )?;
    let source = LibrarySource::new(library, bootstrap.proc_address_permitted);
    build_from_profile_source(&source, library_path, bootstrap)
}

impl SymbolSource for LibrarySource {
    fn proc_supported(&self) -> bool {
        self.get_proc_address.is_some()
    }

    fn proc_lookup(&self, name: &str, api_version: i32, flags: u64) -> ProcLookup {
        let Some(get_proc_address) = self.get_proc_address else {
            return failed_proc_lookup(api_version);
        };
        let Ok(name_c) = CString::new(name) else {
            return failed_proc_lookup(api_version);
        };
        let mut address = std::ptr::null_mut();
        let mut status = -1;
        // SAFETY: all outputs point to initialized writable storage and the
        // requested name is NUL-terminated for the duration of the call.
        let call_result = unsafe {
            get_proc_address(
                name_c.as_ptr(),
                &raw mut address,
                api_version,
                flags,
                &raw mut status,
            )
        };
        let returned_pointer = NonNull::new(address);
        let query_status = status.into();
        let address = if call_result == HIP_SUCCESS && query_status == QueryStatus::Success {
            returned_pointer
        } else {
            None
        };
        ProcLookup {
            attempt: ProcAttempt {
                requested_version: api_version,
                call_result,
                query_status,
                returned_pointer: returned_pointer.is_some(),
            },
            address,
        }
    }

    fn direct_lookup(&self, name: &str) -> Option<NonNull<c_void>> {
        self.library.find(name.as_bytes()).ok().flatten()
    }
}

fn failed_proc_lookup(api_version: i32) -> ProcLookup {
    ProcLookup {
        address: None,
        attempt: ProcAttempt {
            requested_version: api_version,
            call_result: -1,
            query_status: QueryStatus::SymbolNotFound,
            returned_pointer: false,
        },
    }
}

struct Resolved {
    address: Option<NonNull<c_void>>,
    report: SymbolReport,
}

fn resolve<S: SymbolSource>(source: &S, spec: SymbolSpec, proc_versions: &[i32]) -> Resolved {
    let mut proc_attempts = Vec::new();
    if source.proc_supported() {
        for &version in proc_versions {
            let lookup = source.proc_lookup(spec.proc_name, version, spec.proc_flags);
            proc_attempts.push(lookup.attempt);
            if let Some(address) = lookup.address {
                return Resolved {
                    address: Some(address),
                    report: SymbolReport {
                        canonical_name: spec.canonical,
                        resolved_name: Some(spec.proc_name),
                        resolution: ResolutionKind::ProcAddress,
                        available: true,
                        applicable: true,
                        required: spec.required,
                        proc_attempts,
                    },
                };
            }
        }
    }
    for direct_name in spec.direct_names {
        if let Some(address) = source.direct_lookup(direct_name) {
            return Resolved {
                address: Some(address),
                report: SymbolReport {
                    canonical_name: spec.canonical,
                    resolved_name: Some(direct_name),
                    resolution: ResolutionKind::Direct,
                    available: true,
                    applicable: true,
                    required: spec.required,
                    proc_attempts,
                },
            };
        }
    }
    Resolved {
        address: None,
        report: SymbolReport {
            canonical_name: spec.canonical,
            resolved_name: None,
            resolution: ResolutionKind::Missing,
            available: false,
            applicable: true,
            required: spec.required,
            proc_attempts,
        },
    }
}

fn platform_is_applicable(platform_mask: u8) -> bool {
    platform_mask & CURRENT_PLATFORM_MASK != 0
}

fn platform_unavailable_report(canonical: &'static str, required: bool) -> SymbolReport {
    SymbolReport {
        canonical_name: canonical,
        resolved_name: None,
        resolution: ResolutionKind::PlatformUnavailable,
        available: false,
        applicable: false,
        required,
        proc_attempts: Vec::new(),
    }
}

fn profile_unavailable_report(canonical: &'static str, required: bool) -> SymbolReport {
    SymbolReport {
        canonical_name: canonical,
        resolved_name: None,
        resolution: ResolutionKind::ProfileUnavailable,
        available: false,
        applicable: true,
        required,
        proc_attempts: Vec::new(),
    }
}

fn profile_common_symbol(runtime_profile: RuntimeProfile, canonical: &str) -> bool {
    runtime_profile_descriptor(runtime_profile)
        .is_some_and(|descriptor| descriptor.common_symbols.contains(&canonical))
}

fn profile_raw_exact_symbol(runtime_profile: RuntimeProfile, canonical: &str) -> bool {
    runtime_profile_descriptor(runtime_profile)
        .is_some_and(|descriptor| descriptor.raw_exact_symbols.contains(&canonical))
}

fn common_adapter_symbol(runtime_profile: RuntimeProfile, canonical: &str) -> bool {
    runtime_profile_descriptor(runtime_profile)
        .is_some_and(|descriptor| descriptor.common_adapter_symbols.contains(&canonical))
}

fn inventory_descriptor(canonical: &str) -> Option<&'static RawInventoryDescriptor> {
    HIP_RAW_INVENTORY
        .iter()
        .find(|descriptor| descriptor.canonical == canonical)
}

fn inventory_platform_applicable(canonical: &str) -> bool {
    inventory_descriptor(canonical)
        .is_none_or(|descriptor| platform_is_applicable(descriptor.platform_mask))
}

#[cfg(target_os = "linux")]
const fn target_proc_version(descriptor: &RawInventoryDescriptor) -> i32 {
    descriptor.proc_version_linux
}

#[cfg(target_os = "windows")]
const fn target_proc_version(descriptor: &RawInventoryDescriptor) -> i32 {
    descriptor.proc_version_windows
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
const fn target_proc_version(_descriptor: &RawInventoryDescriptor) -> i32 {
    0
}

fn write_raw_inventory_descriptor(
    table: &mut ocgpuHipApi_v1,
    descriptor: &RawInventoryDescriptor,
    address: NonNull<c_void>,
) -> Result<(), Error> {
    let Some(table_offset) = descriptor.table_offset else {
        return Ok(());
    };
    // SAFETY: emitted inventory offsets are generated from the exact typed ABI
    // field represented by this same manifest row.
    unsafe { write_function_slot(table, table_offset, address) }.map_err(|source| {
        Error::InvalidRawTableDescriptor {
            symbol: descriptor.canonical,
            source,
        }
    })
}

fn write_raw_descriptor(
    table: &mut ocgpuHipApi_v1,
    descriptor: &RawSymbolDescriptor,
    address: NonNull<c_void>,
) -> Result<(), Error> {
    // SAFETY: the descriptor is generated beside the typed ABI table from the
    // same manifest row, so its offset and resolved symbol signature agree.
    unsafe { write_function_slot(table, descriptor.table_offset, address) }.map_err(|source| {
        Error::InvalidRawTableDescriptor {
            symbol: descriptor.canonical,
            source,
        }
    })
}

fn write_raw_symbol(
    table: &mut ocgpuHipApi_v1,
    canonical: &str,
    address: NonNull<c_void>,
) -> Result<(), Error> {
    if let Some(descriptor) = HIP_RAW_SYMBOLS
        .iter()
        .find(|descriptor| descriptor.canonical == canonical)
    {
        write_raw_descriptor(table, descriptor, address)?;
    }
    Ok(())
}

fn descriptor_proc_version(descriptor: &RawInventoryDescriptor) -> Option<i32> {
    let proc_version = target_proc_version(descriptor);
    (proc_version > 0).then_some(proc_version)
}

fn resolve_inventory_descriptor<S: SymbolSource>(
    source: &S,
    table: &mut ocgpuHipApi_v1,
    descriptor: &RawInventoryDescriptor,
    full_raw_inventory: bool,
) -> Result<SymbolReport, Error> {
    if !platform_is_applicable(descriptor.platform_mask) {
        return Ok(platform_unavailable_report(descriptor.canonical, false));
    }
    if !full_raw_inventory {
        return Ok(profile_unavailable_report(descriptor.canonical, false));
    }
    let descriptor_version = descriptor_proc_version(descriptor);
    let resolved = resolve(
        source,
        SymbolSpec {
            canonical: descriptor.canonical,
            proc_name: descriptor.proc_name,
            direct_names: descriptor.direct_names,
            proc_flags: descriptor.proc_flags,
            required: false,
        },
        descriptor_version.as_slice(),
    );
    if let Some(address) = resolved.address {
        write_raw_inventory_descriptor(table, descriptor, address)?;
    }
    Ok(resolved.report)
}

#[allow(clippy::similar_names, clippy::too_many_lines)]
#[cfg(test)]
fn build_from_source<S: SymbolSource>(
    source: &S,
    library_path: PathBuf,
) -> Result<UnvalidatedApi, Error> {
    let bootstrap = bootstrap_runtime(source, &library_path, None)?;
    build_from_profile_source(source, library_path, bootstrap)
}

#[allow(clippy::similar_names, clippy::too_many_lines)]
fn build_from_profile_source<S: SymbolSource>(
    source: &S,
    library_path: PathBuf,
    bootstrap: RuntimeBootstrap,
) -> Result<UnvalidatedApi, Error> {
    let mut reports = Vec::with_capacity(HIP_RAW_INVENTORY.len());
    let RuntimeBootstrap {
        runtime_get_version,
        runtime_version,
        runtime_profile,
        proc_address_permitted,
        full_raw_inventory,
        address: runtime_address,
    } = bootstrap;
    let Some(profile_descriptor) = runtime_profile_descriptor(runtime_profile) else {
        return Err(Error::UnsupportedRuntimeProfile {
            library: library_path,
            runtime_version,
            runtime_major: runtime_profile.runtime_major(),
        });
    };
    let table_flag = profile_descriptor.table_flag;
    let mut raw_table = ocgpuHipApi_v1 {
        struct_size: u32::try_from(size_of::<ocgpuHipApi_v1>()).unwrap_or(u32::MAX),
        abi_version: OCGPU_ABI_VERSION_1,
        backend: OCGPU_BACKEND_HIP,
        flags: table_flag,
        driver_version: 0,
        reserved0: 0,
        ..ocgpuHipApi_v1::default()
    };
    if let Some(descriptor) = inventory_descriptor("hipRuntimeGetVersion") {
        if platform_is_applicable(descriptor.platform_mask) {
            write_raw_inventory_descriptor(&mut raw_table, descriptor, runtime_address)?;
            reports.push(SymbolReport {
                canonical_name: descriptor.canonical,
                resolved_name: Some("hipRuntimeGetVersion"),
                resolution: ResolutionKind::Direct,
                available: true,
                applicable: true,
                required: false,
                proc_attempts: Vec::new(),
            });
        }
    }

    macro_rules! resolve_field {
        ($type:ty, $canonical:literal, [$($direct:literal),+ $(,)?]) => {{
            let descriptor = inventory_descriptor($canonical);
            let proc_version = descriptor.and_then(descriptor_proc_version);
            let fallback_direct_names: &[&str] = &[$($direct),+];
            let platform_applicable = descriptor
                .is_none_or(|descriptor| platform_is_applicable(descriptor.platform_mask));
            let resolved = if !platform_applicable {
                Resolved {
                    address: None,
                    report: platform_unavailable_report($canonical, true),
                }
            } else if !full_raw_inventory {
                if profile_common_symbol(runtime_profile, $canonical)
                    && profile_raw_exact_symbol(runtime_profile, $canonical)
                {
                    resolve(
                        source,
                        SymbolSpec {
                            canonical: $canonical,
                            proc_name: $canonical,
                            direct_names: fallback_direct_names,
                            proc_flags: 0,
                            required: true,
                        },
                        &[],
                    )
                } else {
                    Resolved {
                        address: None,
                        report: profile_unavailable_report($canonical, true),
                    }
                }
            } else {
                resolve(
                    source,
                    SymbolSpec {
                        canonical: $canonical,
                        proc_name: descriptor.map_or($canonical, |descriptor| descriptor.proc_name),
                        direct_names: descriptor
                            .map_or(fallback_direct_names, |descriptor| descriptor.direct_names),
                        proc_flags: descriptor.map_or(0, |descriptor| descriptor.proc_flags),
                        required: true,
                    },
                    proc_version.as_slice(),
                )
            };
            if let Some(address) = resolved.address {
                write_raw_symbol(&mut raw_table, $canonical, address)?;
            }
            reports.push(resolved.report);
            resolved.address.map(|address| {
                // SAFETY: each manifest row pairs the vendor symbol with its exact
                // ABI-compatible function-pointer declaration.
                unsafe { std::mem::transmute::<*mut c_void, $type>(address.as_ptr()) }
            })
        }};
    }

    let init = resolve_field!(InitFn, "hipInit", ["hipInit"]);
    let driver_get_version = resolve_field!(
        DriverGetVersionFn,
        "hipDriverGetVersion",
        ["hipDriverGetVersion"]
    );
    let device_get_count =
        resolve_field!(DeviceGetCountFn, "hipGetDeviceCount", ["hipGetDeviceCount"]);
    let device_get = resolve_field!(DeviceGetFn, "hipDeviceGet", ["hipDeviceGet"]);
    let device_get_name = resolve_field!(DeviceGetNameFn, "hipDeviceGetName", ["hipDeviceGetName"]);
    let device_get_attribute = resolve_field!(
        DeviceGetAttributeFn,
        "hipDeviceGetAttribute",
        ["hipDeviceGetAttribute"]
    );
    let ctx_create = resolve_field!(CtxCreateFn, "hipCtxCreate", ["hipCtxCreate"]);
    let ctx_destroy = resolve_field!(CtxDestroyFn, "hipCtxDestroy", ["hipCtxDestroy"]);
    let ctx_set_current = resolve_field!(CtxSetCurrentFn, "hipCtxSetCurrent", ["hipCtxSetCurrent"]);
    let ctx_get_current = resolve_field!(CtxGetCurrentFn, "hipCtxGetCurrent", ["hipCtxGetCurrent"]);
    let ctx_synchronize =
        resolve_field!(CtxSynchronizeFn, "hipCtxSynchronize", ["hipCtxSynchronize"]);
    let malloc = resolve_field!(MallocFn, "hipMalloc", ["hipMalloc"]);
    let free = resolve_field!(FreeFn, "hipFree", ["hipFree"]);
    if let Some(target) = malloc {
        MALLOC_TARGET.get_or_init(|| target);
    }
    if let Some(target) = free {
        FREE_TARGET.get_or_init(|| target);
    }
    let mem_alloc = malloc.map(|_| common_mem_alloc as MemAllocFn);
    let mem_free = free.map(|_| common_mem_free as MemFreeFn);
    let memcpy_htod = if common_adapter_symbol(runtime_profile, "hipMemcpyHtoD") {
        let descriptor = inventory_descriptor("hipMemcpyHtoD");
        let platform_applicable =
            descriptor.is_none_or(|descriptor| platform_is_applicable(descriptor.platform_mask));
        let mut resolved = if !platform_applicable {
            Resolved {
                address: None,
                report: platform_unavailable_report("hipMemcpyHtoD", true),
            }
        } else if profile_common_symbol(runtime_profile, "hipMemcpyHtoD") {
            resolve(
                source,
                SymbolSpec {
                    canonical: "hipMemcpyHtoD",
                    proc_name: "hipMemcpyHtoD",
                    direct_names: &["hipMemcpyHtoD"],
                    proc_flags: 0,
                    required: true,
                },
                &[],
            )
        } else {
            Resolved {
                address: None,
                report: profile_unavailable_report("hipMemcpyHtoD", true),
            }
        };
        if resolved.address.is_some() {
            resolved.report.resolution = ResolutionKind::DirectAdapter;
        }
        reports.push(resolved.report);
        resolved.address.map(|address| {
            // SAFETY: generated compatibility evidence selects this branch only
            // for HIP 5/6, whose exact declaration uses a mutable raw source.
            let target =
                unsafe { std::mem::transmute::<*mut c_void, LegacyMemcpyHtoDFn>(address.as_ptr()) };
            LEGACY_MEMCPY_HTOD_TARGET.get_or_init(|| target);
            common_legacy_memcpy_htod as MemcpyHtoDFn
        })
    } else {
        let native = resolve_field!(NativeMemcpyHtoDFn, "hipMemcpyHtoD", ["hipMemcpyHtoD"]);
        if let Some(target) = native {
            NATIVE_MEMCPY_HTOD_TARGET.get_or_init(|| target);
        }
        native.map(|_| common_native_memcpy_htod as MemcpyHtoDFn)
    };
    let native_memcpy_dtoh = resolve_field!(NativeMemcpyDtoHFn, "hipMemcpyDtoH", ["hipMemcpyDtoH"]);
    if let Some(target) = native_memcpy_dtoh {
        NATIVE_MEMCPY_DTOH_TARGET.get_or_init(|| target);
    }
    let memcpy_dtoh = native_memcpy_dtoh.map(|_| common_native_memcpy_dtoh as MemcpyDtoHFn);
    let stream_create = resolve_field!(
        StreamCreateFn,
        "hipStreamCreateWithFlags",
        ["hipStreamCreateWithFlags"]
    );
    let stream_destroy = resolve_field!(StreamDestroyFn, "hipStreamDestroy", ["hipStreamDestroy"]);
    let stream_synchronize = resolve_field!(
        StreamSynchronizeFn,
        "hipStreamSynchronize",
        ["hipStreamSynchronize"]
    );
    let event_create = resolve_field!(
        EventCreateFn,
        "hipEventCreateWithFlags",
        ["hipEventCreateWithFlags"]
    );
    let event_destroy = resolve_field!(EventDestroyFn, "hipEventDestroy", ["hipEventDestroy"]);
    let event_record = resolve_field!(EventRecordFn, "hipEventRecord", ["hipEventRecord"]);
    let event_synchronize = resolve_field!(
        EventSynchronizeFn,
        "hipEventSynchronize",
        ["hipEventSynchronize"]
    );
    let module_load_data =
        resolve_field!(ModuleLoadDataFn, "hipModuleLoadData", ["hipModuleLoadData"]);
    let module_unload = resolve_field!(ModuleUnloadFn, "hipModuleUnload", ["hipModuleUnload"]);
    let module_get_function = resolve_field!(
        ModuleGetFunctionFn,
        "hipModuleGetFunction",
        ["hipModuleGetFunction"]
    );
    let launch_kernel = resolve_field!(
        LaunchKernelFn,
        "hipModuleLaunchKernel",
        ["hipModuleLaunchKernel"]
    );

    let driver_version = driver_get_version.and_then(|query| {
        let mut version = 0;
        // SAFETY: output points to initialized writable storage and the function
        // pointer was resolved using the exact vendor ABI.
        let result = unsafe { query(&raw mut version) };
        (result == HIP_SUCCESS).then_some(version)
    });

    raw_table.driver_version = driver_version.unwrap_or(0);
    for descriptor in HIP_RAW_INVENTORY {
        if reports
            .iter()
            .any(|report| report.canonical_name == descriptor.canonical)
        {
            continue;
        }
        reports.push(resolve_inventory_descriptor(
            source,
            &mut raw_table,
            descriptor,
            full_raw_inventory,
        )?);
    }

    Ok(UnvalidatedApi {
        diagnostics: Diagnostics {
            library_path,
            runtime_profile,
            runtime_version: Some(runtime_version),
            driver_version,
            proc_address_support: proc_address_permitted && source.proc_supported(),
            loaded_architecture: std::env::consts::ARCH,
            symbols: reports,
        },
        runtime_get_version: Some(runtime_get_version),
        init,
        driver_get_version,
        device_get_count,
        device_get,
        device_get_name,
        device_get_attribute,
        ctx_create,
        ctx_destroy,
        ctx_set_current,
        ctx_get_current,
        ctx_synchronize,
        mem_alloc,
        mem_free,
        malloc,
        free,
        memcpy_htod,
        memcpy_dtoh,
        stream_create,
        stream_destroy,
        stream_synchronize,
        event_create,
        event_destroy,
        event_record,
        event_synchronize,
        module_load_data,
        module_unload,
        module_get_function,
        launch_kernel,
        raw_table,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        COMPILED_API_VERSION, HIP_SUCCESS, ProcAttempt, ProcLookup, QueryStatus, ResolutionKind,
        SymbolSource, build_from_source, validate,
    };
    use std::cell::Cell;
    use std::collections::BTreeSet;
    use std::ffi::c_void;
    use std::ptr::NonNull;

    const MOCK_DEVICE_POINTER: usize = 0x1234_5000;
    #[cfg(target_os = "windows")]
    const EXPECTED_PROC_VERSION: i32 = 702;
    #[cfg(not(target_os = "windows"))]
    const EXPECTED_PROC_VERSION: i32 = 714;

    const COMMON_ATTRIBUTE_MAP: &[(i32, i32)] = &[
        (
            ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
            ocgpu_abi::OCGPU_HIP_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        ),
        (
            ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
            ocgpu_abi::OCGPU_HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
        ),
        (
            ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
            ocgpu_abi::OCGPU_HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
        ),
        (
            ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
            ocgpu_abi::OCGPU_HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
        ),
        (
            ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
            ocgpu_abi::OCGPU_HIP_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
        ),
        (
            ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
            ocgpu_abi::OCGPU_HIP_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
        ),
        (
            ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
            ocgpu_abi::OCGPU_HIP_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
        ),
        (
            ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
            ocgpu_abi::OCGPU_HIP_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
        ),
        (
            ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY,
            ocgpu_abi::OCGPU_HIP_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY,
        ),
        (
            ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_WARP_SIZE,
            ocgpu_abi::OCGPU_HIP_DEVICE_ATTRIBUTE_WARP_SIZE,
        ),
        (
            ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MAX_PITCH,
            ocgpu_abi::OCGPU_HIP_DEVICE_ATTRIBUTE_MAX_PITCH,
        ),
        (
            ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
            ocgpu_abi::OCGPU_HIP_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
        ),
        (
            ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_CLOCK_RATE,
            ocgpu_abi::OCGPU_HIP_DEVICE_ATTRIBUTE_CLOCK_RATE,
        ),
        (
            ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT,
            ocgpu_abi::OCGPU_HIP_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT,
        ),
        (
            ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            ocgpu_abi::OCGPU_HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        ),
        (
            ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT,
            ocgpu_abi::OCGPU_HIP_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT,
        ),
        (
            ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_INTEGRATED,
            ocgpu_abi::OCGPU_HIP_DEVICE_ATTRIBUTE_INTEGRATED,
        ),
        (
            ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY,
            ocgpu_abi::OCGPU_HIP_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY,
        ),
        (
            ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_COMPUTE_MODE,
            ocgpu_abi::OCGPU_HIP_DEVICE_ATTRIBUTE_COMPUTE_MODE,
        ),
        (
            ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS,
            ocgpu_abi::OCGPU_HIP_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS,
        ),
        (
            ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
            ocgpu_abi::OCGPU_HIP_DEVICE_ATTRIBUTE_PCI_BUS_ID,
        ),
        (
            ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
            ocgpu_abi::OCGPU_HIP_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
        ),
        (
            ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
            ocgpu_abi::OCGPU_HIP_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
        ),
        (
            ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
            ocgpu_abi::OCGPU_HIP_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
        ),
        (
            ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
            ocgpu_abi::OCGPU_HIP_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
        ),
        (
            ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
            ocgpu_abi::OCGPU_HIP_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
        ),
        (
            ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
            ocgpu_abi::OCGPU_HIP_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
        ),
        (
            ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            ocgpu_abi::OCGPU_HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        ),
        (
            ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            ocgpu_abi::OCGPU_HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        ),
        (
            ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MANAGED_MEMORY,
            ocgpu_abi::OCGPU_HIP_DEVICE_ATTRIBUTE_MANAGED_MEMORY,
        ),
        (
            ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS,
            ocgpu_abi::OCGPU_HIP_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS,
        ),
        (
            ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS,
            ocgpu_abi::OCGPU_HIP_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS,
        ),
    ];

    #[cfg(feature = "explicit-library-path")]
    #[test]
    fn explicit_path_loading_requires_an_unsafe_caller() {
        let _: unsafe fn(
            &std::path::Path,
        ) -> Result<&'static super::ValidatedCoreApi, super::Error> = super::load_from_absolute;
        let _: unsafe fn(&std::path::Path) -> Result<&'static super::UnvalidatedApi, super::Error> =
            super::load_unvalidated_from_absolute;
    }

    unsafe extern "C" fn mock_function() -> i32 {
        HIP_SUCCESS
    }

    unsafe extern "C" fn mock_malloc(output: *mut *mut c_void, _bytes: usize) -> i32 {
        // SAFETY: the mock ABI requires callers to supply writable output storage.
        if let Some(output) = unsafe { output.as_mut() } {
            *output = MOCK_DEVICE_POINTER as *mut c_void;
            HIP_SUCCESS
        } else {
            1
        }
    }

    unsafe extern "C" fn mock_free(device_ptr: *mut c_void) -> i32 {
        if device_ptr as usize == MOCK_DEVICE_POINTER {
            HIP_SUCCESS
        } else {
            1
        }
    }

    unsafe extern "C" fn mock_memcpy_htod(
        _destination: *mut c_void,
        _source: *const c_void,
        _bytes: usize,
    ) -> i32 {
        HIP_SUCCESS
    }

    unsafe extern "C" fn mock_legacy_memcpy_htod(
        _destination: *mut c_void,
        _source: *mut c_void,
        _bytes: usize,
    ) -> i32 {
        HIP_SUCCESS
    }

    unsafe extern "C" fn mock_memcpy_dtoh(
        _destination: *mut c_void,
        _source: *mut c_void,
        _bytes: usize,
    ) -> i32 {
        HIP_SUCCESS
    }

    unsafe extern "C" fn mock_runtime_version(output: *mut i32) -> i32 {
        // SAFETY: the mock ABI requires callers to supply writable output storage.
        if let Some(output) = unsafe { output.as_mut() } {
            *output = reviewed_raw_inventory_floor(super::RuntimeProfile::Hip7);
            HIP_SUCCESS
        } else {
            1
        }
    }

    fn write_mock_runtime_version(output: *mut i32, version: i32) -> i32 {
        // SAFETY: mock callers use the exact `hipRuntimeGetVersion` output ABI.
        if let Some(output) = unsafe { output.as_mut() } {
            *output = version;
            HIP_SUCCESS
        } else {
            1
        }
    }

    fn reviewed_profile_floor(profile: super::RuntimeProfile) -> i32 {
        let descriptor = super::runtime_profile_descriptor(profile)
            .expect("test profile has generated metadata");
        super::platform_runtime_profile(descriptor).runtime_version_min_inclusive
    }

    fn reviewed_raw_inventory_floor(profile: super::RuntimeProfile) -> i32 {
        let descriptor = super::runtime_profile_descriptor(profile)
            .expect("test profile has generated metadata");
        super::platform_runtime_profile(descriptor)
            .raw_inventory_min_inclusive
            .expect("test profile has a reviewed raw inventory")
    }

    unsafe extern "C" fn mock_runtime_version_5(output: *mut i32) -> i32 {
        write_mock_runtime_version(output, reviewed_profile_floor(super::RuntimeProfile::Hip5))
    }

    unsafe extern "C" fn mock_runtime_version_5_below_floor(output: *mut i32) -> i32 {
        write_mock_runtime_version(
            output,
            reviewed_profile_floor(super::RuntimeProfile::Hip5) - 1,
        )
    }

    unsafe extern "C" fn mock_runtime_version_6(output: *mut i32) -> i32 {
        write_mock_runtime_version(output, 60_400_000)
    }

    unsafe extern "C" fn mock_runtime_version_6_below_floor(output: *mut i32) -> i32 {
        write_mock_runtime_version(output, 60_100_000)
    }

    unsafe extern "C" fn mock_runtime_version_8(output: *mut i32) -> i32 {
        write_mock_runtime_version(output, 80_000_000)
    }

    unsafe extern "C" fn mock_runtime_version_4(output: *mut i32) -> i32 {
        write_mock_runtime_version(output, 40_000_000)
    }

    unsafe extern "C" fn mock_driver_version(output: *mut i32) -> i32 {
        // SAFETY: the mock ABI requires callers to supply writable output storage.
        if let Some(output) = unsafe { output.as_mut() } {
            *output = 7_000;
            HIP_SUCCESS
        } else {
            1
        }
    }

    unsafe extern "C" fn mock_device_get_attribute(
        output: *mut i32,
        attribute: i32,
        _device: i32,
    ) -> i32 {
        // SAFETY: the mock ABI requires callers to supply writable output storage.
        if let Some(output) = unsafe { output.as_mut() } {
            *output = attribute;
            HIP_SUCCESS
        } else {
            1
        }
    }

    struct MockLibrary {
        proc_success_version: Option<i32>,
        missing: BTreeSet<&'static str>,
    }

    impl Default for MockLibrary {
        fn default() -> Self {
            Self {
                proc_success_version: Some(EXPECTED_PROC_VERSION),
                missing: BTreeSet::new(),
            }
        }
    }

    impl MockLibrary {
        fn address_for(name: &str) -> NonNull<c_void> {
            let address = match name {
                "hipRuntimeGetVersion" => mock_runtime_version as *const () as *mut c_void,
                "hipDriverGetVersion" => mock_driver_version as *const () as *mut c_void,
                "hipDeviceGetAttribute" => mock_device_get_attribute as *const () as *mut c_void,
                "hipMalloc" => mock_malloc as *const () as *mut c_void,
                "hipFree" => mock_free as *const () as *mut c_void,
                "hipMemcpyHtoD" => mock_memcpy_htod as *const () as *mut c_void,
                "hipMemcpyDtoH" => mock_memcpy_dtoh as *const () as *mut c_void,
                _ => mock_function as *const () as *mut c_void,
            };
            NonNull::new(address).expect("function addresses are non-null")
        }
    }

    impl SymbolSource for MockLibrary {
        fn proc_supported(&self) -> bool {
            self.proc_success_version.is_some()
        }

        fn proc_lookup(&self, name: &str, api_version: i32, _flags: u64) -> ProcLookup {
            let found =
                !self.missing.contains(name) && self.proc_success_version == Some(api_version);
            let address = found.then(|| Self::address_for(name));
            ProcLookup {
                attempt: ProcAttempt {
                    requested_version: api_version,
                    call_result: HIP_SUCCESS,
                    query_status: if found {
                        QueryStatus::Success
                    } else {
                        QueryStatus::VersionInsufficient
                    },
                    returned_pointer: address.is_some(),
                },
                address,
            }
        }

        fn direct_lookup(&self, name: &str) -> Option<NonNull<c_void>> {
            let bootstrap_missing =
                name == "hipGetProcAddress" && self.proc_success_version.is_none();
            (!bootstrap_missing && !self.missing.contains(name)).then(|| Self::address_for(name))
        }
    }

    #[derive(Clone, Copy)]
    enum MockRuntime {
        Hip7,
        Hip5,
        Hip5BelowFloor,
        Hip6,
        Hip6BelowFloor,
        Hip8,
        Hip4,
    }

    struct ProfileSource {
        runtime: MockRuntime,
        proc_queries: Cell<usize>,
        direct_queries: Cell<usize>,
    }

    impl ProfileSource {
        fn new(runtime: MockRuntime) -> Self {
            Self {
                runtime,
                proc_queries: Cell::new(0),
                direct_queries: Cell::new(0),
            }
        }

        fn runtime_address(&self) -> NonNull<c_void> {
            let address = match self.runtime {
                MockRuntime::Hip7 => mock_runtime_version as *const () as *mut c_void,
                MockRuntime::Hip5 => mock_runtime_version_5 as *const () as *mut c_void,
                MockRuntime::Hip5BelowFloor => {
                    mock_runtime_version_5_below_floor as *const () as *mut c_void
                }
                MockRuntime::Hip6 => mock_runtime_version_6 as *const () as *mut c_void,
                MockRuntime::Hip6BelowFloor => {
                    mock_runtime_version_6_below_floor as *const () as *mut c_void
                }
                MockRuntime::Hip8 => mock_runtime_version_8 as *const () as *mut c_void,
                MockRuntime::Hip4 => mock_runtime_version_4 as *const () as *mut c_void,
            };
            NonNull::new(address).expect("function addresses are non-null")
        }
    }

    impl SymbolSource for ProfileSource {
        fn proc_supported(&self) -> bool {
            true
        }

        fn proc_lookup(&self, name: &str, api_version: i32, _flags: u64) -> ProcLookup {
            self.proc_queries.set(self.proc_queries.get() + 1);
            ProcLookup {
                address: Some(MockLibrary::address_for(name)),
                attempt: ProcAttempt {
                    requested_version: api_version,
                    call_result: HIP_SUCCESS,
                    query_status: QueryStatus::Success,
                    returned_pointer: true,
                },
            }
        }

        fn direct_lookup(&self, name: &str) -> Option<NonNull<c_void>> {
            self.direct_queries.set(self.direct_queries.get() + 1);
            Some(if name == "hipRuntimeGetVersion" {
                self.runtime_address()
            } else if name == "hipMemcpyHtoD" && !matches!(self.runtime, MockRuntime::Hip7) {
                NonNull::new(mock_legacy_memcpy_htod as *const () as *mut c_void)
                    .expect("function addresses are non-null")
            } else {
                MockLibrary::address_for(name)
            })
        }
    }

    #[allow(clippy::cast_ptr_alignment)]
    unsafe fn table_slot_address<T>(table: &T, offset: usize) -> Option<usize> {
        type ErasedFunction = unsafe extern "C" fn();
        let base = std::ptr::from_ref(table).cast::<u8>();
        // SAFETY: callers use a generated `offset_of!` for a nullable function
        // field, whose one-pointer representation is asserted by the ABI crate.
        let entry = unsafe { base.add(offset).cast::<Option<ErasedFunction>>().read() };
        entry.map(|function| function as usize)
    }

    #[test]
    fn mock_library_builds_and_validates_without_touching_a_gpu() {
        let raw = Box::leak(Box::new(
            build_from_source(&MockLibrary::default(), "mock-hip".into())
                .expect("generated descriptors must fit the mock raw table"),
        ));
        let validated = validate(raw).expect("complete mock exports satisfy core profile");
        assert_eq!(
            validated.diagnostics().runtime_version,
            Some(reviewed_raw_inventory_floor(super::RuntimeProfile::Hip7))
        );
        assert_eq!(validated.common_table().driver_version, 7_000);
        let reported_names: BTreeSet<_> = validated
            .diagnostics()
            .symbols
            .iter()
            .map(|report| report.canonical_name)
            .collect();
        assert_eq!(
            validated.diagnostics().symbols.len(),
            reported_names.len(),
            "diagnostics must not duplicate canonical inventory names"
        );
        assert_eq!(reported_names.len(), super::HIP_RAW_INVENTORY.len());
        assert!(super::HIP_RAW_INVENTORY.iter().all(|descriptor| {
            validated
                .diagnostics()
                .symbol(descriptor.canonical)
                .is_some()
        }));
        let raw_table = raw.raw_table();
        assert_eq!(raw_table.backend, ocgpu_abi::OCGPU_BACKEND_HIP);
        assert_eq!(raw_table.flags, super::RuntimeProfile::Hip7.api_flags());
        assert_eq!(
            raw_table.struct_size as usize,
            size_of::<ocgpu_abi::ocgpuHipApi_v1>()
        );
        assert_eq!(
            validated.common_table().struct_size as usize,
            size_of::<ocgpu_abi::ocgpuApi_v1>()
        );
        assert!(raw_table.ocgpuHipGetProcAddress.is_some());
        assert!(raw_table.ocgpuHipMalloc.is_some());
        assert!(raw_table.ocgpuHipStreamCreateWithFlags.is_some());
    }

    #[test]
    fn legacy_profiles_resolve_only_reviewed_direct_core_and_bootstrap() {
        for (runtime, expected_profile) in [
            (MockRuntime::Hip5, super::RuntimeProfile::Hip5),
            (MockRuntime::Hip6, super::RuntimeProfile::Hip6),
        ] {
            let source = ProfileSource::new(runtime);
            let raw = Box::leak(Box::new(
                build_from_source(&source, "mock-legacy-hip".into())
                    .expect("reviewed legacy profile must build"),
            ));
            assert_eq!(raw.diagnostics.runtime_profile, expected_profile);
            assert!(!raw.diagnostics.proc_address_support);
            assert_eq!(raw.raw_table_ref().flags, expected_profile.api_flags());
            assert_eq!(source.proc_queries.get(), 0);
            assert_eq!(source.direct_queries.get(), 27);
            let validated = validate(raw).expect("reviewed legacy common core must validate");
            assert_eq!(validated.common_table().flags, expected_profile.api_flags());
            let common_memcpy = validated
                .common_table()
                .ocgpuMemcpyHtoD
                .expect("legacy mutable-source entry has a common const-source adapter");
            let source_bytes = [1_u8, 2, 3, 4];
            // SAFETY: the mock adapter reads no bytes and retains no pointers.
            assert_eq!(
                unsafe { common_memcpy(0, source_bytes.as_ptr().cast(), source_bytes.len()) },
                HIP_SUCCESS
            );
            let common_copy_back = validated
                .common_table()
                .ocgpuMemcpyDtoH
                .expect("pointer-typed HIP source has a unified integer adapter");
            let mut destination_bytes = [0_u8; 4];
            // SAFETY: the mock adapter writes no bytes and retains no pointers.
            assert_eq!(
                unsafe {
                    common_copy_back(
                        destination_bytes.as_mut_ptr().cast(),
                        MOCK_DEVICE_POINTER,
                        destination_bytes.len(),
                    )
                },
                HIP_SUCCESS
            );

            for descriptor in super::HIP_RAW_INVENTORY {
                let report = raw
                    .diagnostics
                    .symbol(descriptor.canonical)
                    .expect("every inventory row has one diagnostic");
                assert!(report.proc_attempts.is_empty());
                let bootstrap = descriptor.canonical == "hipRuntimeGetVersion";
                let common = super::profile_common_symbol(expected_profile, descriptor.canonical);
                let raw_exact =
                    super::profile_raw_exact_symbol(expected_profile, descriptor.canonical);
                let adapter = super::common_adapter_symbol(expected_profile, descriptor.canonical);
                if !super::platform_is_applicable(descriptor.platform_mask) {
                    assert_eq!(report.resolution, ResolutionKind::PlatformUnavailable);
                } else if adapter {
                    assert_eq!(report.resolution, ResolutionKind::DirectAdapter);
                    assert!(report.available);
                } else if bootstrap || (common && raw_exact) {
                    assert_eq!(report.resolution, ResolutionKind::Direct);
                    assert!(report.available);
                } else {
                    assert_eq!(report.resolution, ResolutionKind::ProfileUnavailable);
                    assert!(!report.available);
                }
                if let Some(offset) = descriptor.table_offset {
                    // SAFETY: the generated inventory offset identifies its
                    // nullable function-pointer field in this exact raw table.
                    let address = unsafe { table_slot_address(raw.raw_table_ref(), offset) };
                    if super::platform_is_applicable(descriptor.platform_mask)
                        && (bootstrap || raw_exact)
                    {
                        assert!(address.is_some(), "{} remained null", descriptor.canonical);
                    } else {
                        assert_eq!(
                            address, None,
                            "{} was unexpectedly populated",
                            descriptor.canonical
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn hip7_below_raw_floor_resolves_only_reviewed_common_raw_slots() {
        let source = ProfileSource::new(MockRuntime::Hip7);
        let runtime_address = NonNull::new(mock_runtime_version as *const () as *mut c_void)
            .expect("function addresses are non-null");
        let runtime_version = reviewed_profile_floor(super::RuntimeProfile::Hip7);
        let raw = Box::leak(Box::new(
            super::build_from_profile_source(
                &source,
                "mock-restricted-hip7".into(),
                super::RuntimeBootstrap {
                    runtime_get_version: mock_runtime_version,
                    runtime_version,
                    runtime_profile: super::RuntimeProfile::Hip7,
                    proc_address_permitted: true,
                    full_raw_inventory: false,
                    address: runtime_address,
                },
            )
            .expect("reviewed HIP7 common subset must build below the raw floor"),
        ));
        validate(raw).expect("reviewed HIP7 common subset must validate");
        assert!(raw.diagnostics.proc_address_support);
        assert_eq!(source.proc_queries.get(), 0);
        assert_eq!(source.direct_queries.get(), 26);
        for report in &raw.diagnostics.symbols {
            if !report.applicable {
                assert_eq!(report.resolution, ResolutionKind::PlatformUnavailable);
            } else if report.canonical_name == "hipRuntimeGetVersion"
                || super::profile_raw_exact_symbol(
                    super::RuntimeProfile::Hip7,
                    report.canonical_name,
                )
            {
                assert_eq!(report.resolution, ResolutionKind::Direct);
                assert!(report.available);
            } else {
                assert_eq!(report.resolution, ResolutionKind::ProfileUnavailable);
                assert!(!report.available);
            }
            assert!(report.proc_attempts.is_empty());
        }
        assert!(raw.raw_table_ref().ocgpuHipMalloc.is_some());
        assert!(raw.raw_table_ref().ocgpuHipProfilerStart.is_none());
    }

    #[test]
    fn versioned_name_and_runtime_major_mismatch_is_rejected() {
        let source = ProfileSource::new(MockRuntime::Hip6);
        let error = super::bootstrap_runtime(
            &source,
            std::path::Path::new("amdhip64_7.dll"),
            Some(super::RuntimeProfile::Hip7),
        )
        .expect_err("a versioned library name cannot select another ABI profile");
        assert!(matches!(
            error,
            super::Error::RuntimeProfileMismatch {
                expected: super::RuntimeProfile::Hip7,
                detected: super::RuntimeProfile::Hip6,
                runtime_version: 60_400_000,
                ..
            }
        ));

        let below_floor = ProfileSource::new(MockRuntime::Hip6BelowFloor);
        let error = super::bootstrap_runtime(
            &below_floor,
            std::path::Path::new("amdhip64_7.dll"),
            Some(super::RuntimeProfile::Hip7),
        )
        .expect_err("filename-major mismatch precedes detected-profile interval checks");
        assert!(matches!(
            error,
            super::Error::RuntimeProfileMismatch {
                expected: super::RuntimeProfile::Hip7,
                detected: super::RuntimeProfile::Hip6,
                runtime_version: 60_100_000,
                ..
            }
        ));
    }

    #[test]
    fn known_major_below_generated_profile_floor_is_structured() {
        let minimum_supported = reviewed_profile_floor(super::RuntimeProfile::Hip5);
        let source = ProfileSource::new(MockRuntime::Hip5BelowFloor);
        let error = super::bootstrap_runtime(&source, std::path::Path::new("old-hip5"), None)
            .expect_err("a known major outside its reviewed interval must fail closed");
        assert!(error.below_minimum_runtime_version());
        let super::Error::UnsupportedRuntimeVersion {
            runtime_profile,
            runtime_version,
            minimum_supported: reported_minimum,
            maximum_supported,
            ..
        } = error
        else {
            panic!("unexpected error category")
        };
        assert_eq!(runtime_profile, super::RuntimeProfile::Hip5);
        assert_eq!(runtime_version, minimum_supported - 1);
        assert_eq!(reported_minimum, minimum_supported);
        assert_eq!(maximum_supported, 59_999_999);
    }

    #[test]
    fn unknown_future_and_pre_profile_runtime_majors_fail_closed() {
        let future = ProfileSource::new(MockRuntime::Hip8);
        let error = super::bootstrap_runtime(&future, std::path::Path::new("future"), None)
            .expect_err("unknown future ABI must not inherit the HIP 7 table");
        assert!(matches!(
            error,
            super::Error::UnsupportedRuntimeProfile {
                runtime_version: 80_000_000,
                runtime_major: 8,
                ..
            }
        ));

        let old = ProfileSource::new(MockRuntime::Hip4);
        let error = super::bootstrap_runtime(&old, std::path::Path::new("old"), None)
            .expect_err("pre-profile runtimes must be reported as too old");
        assert!(matches!(error, super::Error::BackendTooOld { .. }));
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn windows_profile_names_are_classified_case_insensitively() {
        assert_eq!(
            super::expected_runtime_profile(std::path::Path::new(
                r"C:\Windows\System32\AMDHiP64_7.DLL"
            )),
            Some(super::RuntimeProfile::Hip7)
        );
        assert_eq!(
            super::expected_runtime_profile(std::path::Path::new(
                r"C:\Windows\System32\amdhip64_6.dll"
            )),
            Some(super::RuntimeProfile::Hip6)
        );
        assert_eq!(
            super::expected_runtime_profile(std::path::Path::new(
                r"C:\Windows\System32\amdhip64.dll"
            )),
            Some(super::RuntimeProfile::Hip5)
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_versioned_sonames_claim_a_profile_but_shared_fallback_does_not() {
        assert_eq!(
            super::expected_runtime_profile(std::path::Path::new("/opt/rocm/lib/libamdhip64.so.7")),
            Some(super::RuntimeProfile::Hip7)
        );
        assert_eq!(
            super::expected_runtime_profile(std::path::Path::new(
                "/opt/rocm/lib/libamdhip64.so.6.4.2"
            )),
            Some(super::RuntimeProfile::Hip6)
        );
        assert_eq!(
            super::expected_runtime_profile(std::path::Path::new("/opt/rocm/lib/libamdhip64.so.5")),
            Some(super::RuntimeProfile::Hip5)
        );
        assert_eq!(
            super::expected_runtime_profile(std::path::Path::new("/opt/rocm/lib/libamdhip64.so")),
            None
        );
    }

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    #[test]
    fn secure_loader_candidates_and_profile_flags_match_generated_evidence() {
        let platform_profiles: Vec<_> = super::HIP_RUNTIME_PROFILES
            .iter()
            .map(super::platform_runtime_profile)
            .collect();
        let occurrence_count = |candidate: &str| {
            platform_profiles
                .iter()
                .flat_map(|profile| profile.library_candidates.iter())
                .filter(|configured| **configured == candidate)
                .count()
        };
        let mut expected_candidates = Vec::new();
        for shared in [false, true] {
            for profile in &platform_profiles {
                for &candidate in profile.library_candidates {
                    if (occurrence_count(candidate) > 1) == shared
                        && !expected_candidates.contains(&candidate)
                    {
                        expected_candidates.push(candidate);
                    }
                }
            }
        }
        assert_eq!(ocgpu_loader::Backend::Hip.candidates(), expected_candidates);

        for (descriptor, profile) in super::HIP_RUNTIME_PROFILES.iter().zip([
            super::RuntimeProfile::Hip7,
            super::RuntimeProfile::Hip6,
            super::RuntimeProfile::Hip5,
        ]) {
            assert_eq!(descriptor.runtime_major, profile.runtime_major());
            assert_eq!(descriptor.table_flag, profile.api_flags());
            assert!(
                descriptor
                    .bootstrap_symbols
                    .contains(&"hipRuntimeGetVersion")
            );
            assert_eq!(descriptor.common_symbols.len(), 26);
            let implemented_adapters: &[&str] = match profile {
                super::RuntimeProfile::Hip7 => &[],
                super::RuntimeProfile::Hip5 | super::RuntimeProfile::Hip6 => &["hipMemcpyHtoD"],
            };
            assert_eq!(descriptor.common_adapter_symbols, implemented_adapters);
            for symbol in descriptor.common_symbols {
                assert_ne!(
                    descriptor.raw_exact_symbols.contains(symbol),
                    descriptor.common_adapter_symbols.contains(symbol),
                    "{symbol} must use exactly one reviewed common-call path"
                );
            }
            assert!(
                descriptor
                    .raw_exact_symbols
                    .iter()
                    .all(|symbol| descriptor.common_symbols.contains(symbol))
            );
            assert!(
                descriptor
                    .common_adapter_symbols
                    .iter()
                    .all(|symbol| descriptor.common_symbols.contains(symbol))
            );
        }
    }

    #[test]
    fn only_the_exact_descriptor_version_is_attempted_before_direct_fallback() {
        let init_descriptor = super::HIP_RAW_INVENTORY
            .iter()
            .find(|descriptor| descriptor.canonical == "hipInit")
            .expect("HIP init has generated proc metadata");
        assert_eq!(
            super::target_proc_version(init_descriptor),
            EXPECTED_PROC_VERSION
        );
        let raw = build_from_source(
            &MockLibrary {
                proc_success_version: Some(COMPILED_API_VERSION),
                missing: BTreeSet::new(),
            },
            "mock-hip".into(),
        )
        .expect("generated descriptors must fit the mock raw table");
        let init = raw
            .diagnostics
            .symbol("hipInit")
            .expect("inventory always reports init");
        assert_eq!(init.resolution, ResolutionKind::Direct);
        assert_eq!(init.proc_attempts.len(), 1);
        assert_eq!(
            init.proc_attempts[0].requested_version,
            EXPECTED_PROC_VERSION
        );
        assert_eq!(
            init.proc_attempts[0].query_status,
            QueryStatus::VersionInsufficient
        );
        assert_eq!(init.resolved_name, Some("hipInit"));
    }

    #[test]
    fn missing_symbols_are_reported_together() {
        let raw = Box::leak(Box::new(
            build_from_source(
                &MockLibrary {
                    proc_success_version: None,
                    missing: BTreeSet::from(["hipInit", "hipMalloc", "hipModuleLaunchKernel"]),
                },
                "mock-hip".into(),
            )
            .expect("generated descriptors must fit the mock raw table"),
        ));
        let error = validate(raw).expect_err("incomplete core must fail validation");
        let super::Error::MissingCoreSymbols { symbols, .. } = error else {
            panic!("unexpected error category")
        };
        assert_eq!(symbols, ["hipInit", "hipMalloc", "hipModuleLaunchKernel"]);
    }

    #[test]
    fn version_insufficient_core_symbol_reports_old_backend() {
        let raw = Box::leak(Box::new(
            build_from_source(
                &MockLibrary {
                    proc_success_version: Some(COMPILED_API_VERSION),
                    missing: BTreeSet::from(["hipInit"]),
                },
                "mock-old-hip".into(),
            )
            .expect("generated descriptors must fit the mock raw table"),
        ));
        let error = validate(raw).expect_err("version-insufficient core must fail validation");
        let super::Error::BackendTooOld { symbols, .. } = error else {
            panic!("unexpected error category")
        };
        assert_eq!(symbols, ["hipInit"]);
    }

    #[test]
    fn missing_raw_only_entries_remain_null_without_invalidating_common_core() {
        let raw = Box::leak(Box::new(
            build_from_source(
                &MockLibrary {
                    proc_success_version: None,
                    missing: BTreeSet::from(["hipProfilerStart"]),
                },
                "mock-hip".into(),
            )
            .expect("generated descriptors must fit the mock raw table"),
        ));
        assert!(validate(raw).is_ok());
        assert!(raw.raw_table().ocgpuHipProfilerStart.is_none());
        assert!(raw.mem_alloc.is_some());
    }

    #[test]
    fn raw_memory_slots_use_exact_pointer_typed_hip_abis() {
        let table = ocgpu_abi::ocgpuHipApi_v1::default();
        let _: Option<super::MallocFn> = table.ocgpuHipMalloc;
        let _: Option<super::FreeFn> = table.ocgpuHipFree;
        let _: Option<super::NativeMemcpyHtoDFn> = table.ocgpuHipMemcpyHtoD;
        let _: Option<super::NativeMemcpyDtoHFn> = table.ocgpuHipMemcpyDtoH;
    }

    #[test]
    fn unified_memory_adapter_converts_pointer_values_without_changing_raw_abi() {
        let raw = Box::leak(Box::new(
            build_from_source(&MockLibrary::default(), "mock-hip".into())
                .expect("generated descriptors must fit the mock raw table"),
        ));
        let validated = validate(raw).expect("complete mock exports satisfy core profile");
        let common = validated.common_table();
        let common_alloc = common
            .ocgpuMemAlloc
            .expect("validated common table has an allocation adapter");
        let common_free = common
            .ocgpuMemFree
            .expect("validated common table has a free adapter");
        let mut device_ptr = 0;
        // SAFETY: the mock writes one pointer-sized result and retains nothing.
        assert_eq!(
            unsafe { common_alloc(&raw mut device_ptr, 64) },
            HIP_SUCCESS
        );
        assert_eq!(device_ptr, MOCK_DEVICE_POINTER);
        // SAFETY: this is the opaque value returned by the matching mock alloc.
        assert_eq!(unsafe { common_free(device_ptr) }, HIP_SUCCESS);

        let raw_malloc = raw
            .raw_table_ref()
            .ocgpuHipMalloc
            .expect("the raw allocation entry remains callable");
        let mut native_ptr = std::ptr::null_mut();
        // SAFETY: the raw mock accepts exact `hipMalloc` output storage.
        assert_eq!(unsafe { raw_malloc(&raw mut native_ptr, 64) }, HIP_SUCCESS);
        assert_eq!(native_ptr as usize, MOCK_DEVICE_POINTER);
    }

    #[test]
    fn unified_attribute_adapter_maps_ids_without_changing_the_raw_table() {
        let raw = Box::leak(Box::new(
            build_from_source(&MockLibrary::default(), "mock-hip".into())
                .expect("generated descriptors must fit the mock raw table"),
        ));
        let validated = validate(raw).expect("complete mock exports satisfy core profile");

        let common_query = validated
            .common_table()
            .ocgpuDeviceGetAttribute
            .expect("validated common table has an attribute adapter");
        let mut common_value = 0;
        for &(attribute, expected_native) in COMMON_ATTRIBUTE_MAP {
            // SAFETY: the mock accepts this writable output and does not retain it.
            let common_result = unsafe { common_query(&raw mut common_value, attribute, 0) };
            assert_eq!(common_result, HIP_SUCCESS);
            assert_eq!(common_value, expected_native);
        }

        let raw_query = raw
            .raw_table()
            .ocgpuHipDeviceGetAttribute
            .expect("the raw entry remains callable");
        let mut raw_value = 0;
        // SAFETY: the mock accepts this writable output and does not retain it.
        let raw_result = unsafe {
            raw_query(
                &raw mut raw_value,
                ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                0,
            )
        };
        assert_eq!(raw_result, HIP_SUCCESS);
        assert_eq!(
            raw_value,
            ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
        );

        // SAFETY: the adapter rejects this value before calling the mock target.
        let invalid_result = unsafe { common_query(&raw mut common_value, i32::MAX, 0) };
        assert_eq!(invalid_result, ocgpu_abi::OCGPU_ERROR_INVALID_ARGUMENT);
    }

    #[test]
    fn every_emitted_inventory_entry_populates_its_generated_slot() {
        let raw = build_from_source(&MockLibrary::default(), "mock-hip".into())
            .expect("generated descriptors must fit the mock raw table");
        let table = raw.raw_table();
        let mut emitted_offsets = BTreeSet::new();
        for descriptor in super::HIP_RAW_INVENTORY {
            let Some(offset) = descriptor.table_offset else {
                continue;
            };
            assert!(
                emitted_offsets.insert(offset),
                "duplicate generated slot for {}",
                descriptor.canonical
            );
            if descriptor.platform_mask & super::CURRENT_PLATFORM_MASK == 0 {
                continue;
            }
            let report = raw
                .diagnostics
                .symbol(descriptor.canonical)
                .expect("every inventory descriptor must have a report");
            assert!(
                report.available,
                "{} was not resolved",
                descriptor.canonical
            );
            // SAFETY: `offset` came from the generated typed field descriptor.
            let actual = unsafe { table_slot_address(&table, offset) };
            let expected = MockLibrary::address_for(
                report
                    .resolved_name
                    .expect("an available report records its resolved name"),
            )
            .as_ptr() as usize;
            assert_eq!(
                actual,
                Some(expected),
                "wrong slot for {}",
                descriptor.canonical
            );
        }
        assert_eq!(emitted_offsets.len(), super::HIP_RAW_SYMBOLS.len());
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn generated_resolution_metadata_is_slot_aligned_and_fail_closed() {
        const REVIEWED_PROC_SYMBOLS: &[&str] = &[
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

        assert_eq!(
            super::HIP_RAW_INVENTORY.len(),
            super::HIP_RAW_SYMBOLS.len(),
            "runtime inventory must contain only emitted vendor callables"
        );
        let raw = build_from_source(&MockLibrary::default(), "metadata-hip".into())
            .expect("generated descriptors must fit the mock raw table");
        let direct_raw = build_from_source(
            &MockLibrary {
                proc_success_version: None,
                missing: BTreeSet::new(),
            },
            "metadata-direct-hip".into(),
        )
        .expect("generated descriptors must fit the mock raw table");
        let expected_alias_reports = super::HIP_RAW_SYMBOLS
            .iter()
            .filter(|descriptor| {
                descriptor.direct_names.first().copied() != Some(descriptor.canonical)
            })
            .count();
        let mut observed_alias_reports = 0;
        for descriptor in super::HIP_RAW_SYMBOLS {
            let inventory = super::HIP_RAW_INVENTORY
                .iter()
                .find(|candidate| candidate.canonical == descriptor.canonical)
                .expect("every typed slot must have one runtime inventory row");
            assert_eq!(inventory.proc_name, descriptor.proc_name);
            assert_eq!(inventory.direct_names, descriptor.direct_names);
            assert_eq!(inventory.proc_version_linux, descriptor.proc_version_linux);
            assert_eq!(
                inventory.proc_version_windows,
                descriptor.proc_version_windows
            );
            assert_eq!(inventory.proc_flags, descriptor.proc_flags);
            assert_eq!(inventory.table_offset, Some(descriptor.table_offset));
            assert_eq!(inventory.platform_mask, descriptor.platform_mask);
            assert_eq!(descriptor.direct_names.len(), 1);

            let proc_enabled =
                descriptor.proc_version_linux > 0 || descriptor.proc_version_windows > 0;
            assert_eq!(
                proc_enabled,
                REVIEWED_PROC_SYMBOLS.contains(&descriptor.canonical),
                "unreviewed HIP raw slot enabled proc lookup: {}",
                descriptor.canonical
            );
            let report = raw
                .diagnostics
                .symbol(descriptor.canonical)
                .expect("every emitted slot must have one diagnostic row");
            assert_eq!(report.required, proc_enabled);
            let applicable = super::platform_is_applicable(descriptor.platform_mask);
            if descriptor.direct_names.first().copied() != Some(descriptor.canonical) {
                let direct_report = direct_raw
                    .diagnostics
                    .symbol(descriptor.canonical)
                    .expect("every emitted alias slot must have its own diagnostic row");
                if applicable {
                    assert_eq!(direct_report.resolution, ResolutionKind::Direct);
                    assert_eq!(
                        direct_report.resolved_name,
                        descriptor.direct_names.first().copied()
                    );
                } else {
                    assert_eq!(
                        direct_report.resolution,
                        ResolutionKind::PlatformUnavailable
                    );
                    assert_eq!(direct_report.resolved_name, None);
                }
                observed_alias_reports += 1;
            }
            if !applicable {
                assert!(!report.applicable);
                assert_eq!(report.resolution, ResolutionKind::PlatformUnavailable);
                assert!(report.proc_attempts.is_empty());
                continue;
            }
            if proc_enabled {
                assert_eq!(descriptor.proc_name, descriptor.canonical);
                assert_eq!(descriptor.direct_names, &[descriptor.canonical]);
                assert_eq!(descriptor.proc_version_linux, 714);
                assert_eq!(descriptor.proc_version_windows, 702);
                assert_eq!(descriptor.proc_flags, 0);
                assert_eq!(descriptor.platform_mask, 7);
                assert_eq!(report.resolution, ResolutionKind::ProcAddress);
                assert_eq!(report.resolved_name, Some(descriptor.proc_name));
                assert_eq!(report.proc_attempts.len(), 1);
                let direct_report = direct_raw
                    .diagnostics
                    .symbol(descriptor.canonical)
                    .expect("reviewed direct metadata must have a core diagnostic row");
                assert_eq!(direct_report.resolution, ResolutionKind::Direct);
                assert_eq!(
                    direct_report.resolved_name,
                    descriptor.direct_names.first().copied()
                );
                assert_eq!(
                    report
                        .proc_attempts
                        .first()
                        .map(|attempt| attempt.requested_version),
                    Some(super::target_proc_version(
                        super::HIP_RAW_INVENTORY
                            .iter()
                            .find(|candidate| candidate.canonical == descriptor.canonical)
                            .expect("typed descriptor has a matching inventory row")
                    ))
                );
            } else {
                assert!(report.proc_attempts.is_empty());
                assert_eq!(report.resolution, ResolutionKind::Direct);
                assert_eq!(
                    report.resolved_name,
                    descriptor.direct_names.first().copied()
                );
            }
            if descriptor.canonical.ends_with("_spt") {
                assert_eq!(descriptor.proc_flags, 2);
                assert!(descriptor.proc_name.len() < descriptor.canonical.len());
            }
        }
        assert!(expected_alias_reports > 0);
        assert_eq!(observed_alias_reports, expected_alias_reports);
        for helper in ["rocmlib", "is_rocmlib_present"] {
            assert!(
                super::HIP_RAW_INVENTORY
                    .iter()
                    .all(|descriptor| descriptor.canonical != helper),
                "Rust loader helper leaked into runtime symbol diagnostics"
            );
        }
    }

    #[test]
    fn platform_unavailable_inventory_never_queries_the_library() {
        #[derive(Default)]
        struct CountingSource {
            proc_queries: Cell<usize>,
            direct_queries: Cell<usize>,
        }

        impl SymbolSource for CountingSource {
            fn proc_supported(&self) -> bool {
                true
            }

            fn proc_lookup(&self, _name: &str, api_version: i32, _flags: u64) -> ProcLookup {
                self.proc_queries.set(self.proc_queries.get() + 1);
                ProcLookup {
                    address: None,
                    attempt: ProcAttempt {
                        requested_version: api_version,
                        call_result: 1,
                        query_status: QueryStatus::SymbolNotFound,
                        returned_pointer: false,
                    },
                }
            }

            fn direct_lookup(&self, _name: &str) -> Option<NonNull<c_void>> {
                self.direct_queries.set(self.direct_queries.get() + 1);
                None
            }
        }

        let descriptor = super::raw_symbols::RawInventoryDescriptor {
            canonical: "platformOnlyMock",
            proc_name: "platformOnlyMock",
            direct_names: &["platformOnlyMock"],
            proc_version_linux: 1,
            proc_version_windows: 1,
            proc_flags: 0,
            table_offset: None,
            platform_mask: 0,
            classification: "test_only",
        };
        let source = CountingSource::default();
        let mut table = ocgpu_abi::ocgpuHipApi_v1::default();
        let report = super::resolve_inventory_descriptor(&source, &mut table, &descriptor, true)
            .expect("an unavailable descriptor does not touch its table offset");
        assert_eq!(report.resolution, ResolutionKind::PlatformUnavailable);
        assert!(!report.applicable);
        assert!(report.proc_attempts.is_empty());
        assert_eq!(source.proc_queries.get(), 0);
        assert_eq!(source.direct_queries.get(), 0);
    }
}
