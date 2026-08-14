// SPDX-License-Identifier: CC0-1.0

//! SDK-free CUDA Driver API discovery and validated core dispatch.

use ocgpu_abi::{
    OCGPU_ABI_VERSION_1, OCGPU_BACKEND_CUDA, ocgpuApi_v1, ocgpuContext, ocgpuCuApi_v1, ocgpuDevice,
    ocgpuDeviceAttribute, ocgpuDeviceptr, ocgpuEvent, ocgpuFunction, ocgpuModule, ocgpuResult,
    ocgpuStream,
};
use ocgpu_loader::{Backend, Library, LoadError, TableSlotError, write_function_slot};
use std::error::Error as StdError;
use std::ffi::{CStr, CString, c_char, c_void};
use std::fmt;
use std::mem::size_of;
#[cfg(feature = "explicit-library-path")]
use std::path::Path;
use std::path::PathBuf;
use std::ptr::NonNull;
use std::sync::OnceLock;

#[allow(clippy::doc_markdown)]
mod raw_symbols {
    include!("generated_symbols.rs");
}

use raw_symbols::{
    CUDA_RAW_INVENTORY, CUDA_RAW_SYMBOLS, RawInventoryDescriptor, RawSymbolDescriptor,
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

/// CUDA 13.3 Update 1 interface-inventory baseline.
pub const COMPILED_API_VERSION: i32 = 13_030;

const CUDA_SUCCESS: ocgpuResult = 0;
const GET_PROC_ADDRESS_LEGACY_STREAM: u64 = 1;

/// `cuInit` ABI.
pub type InitFn = unsafe extern "C" fn(u32) -> ocgpuResult;
/// `cuDriverGetVersion` ABI.
pub type DriverGetVersionFn = unsafe extern "C" fn(*mut i32) -> ocgpuResult;
/// `cuDeviceGetCount` ABI.
pub type DeviceGetCountFn = unsafe extern "C" fn(*mut i32) -> ocgpuResult;
/// `cuDeviceGet` ABI.
pub type DeviceGetFn = unsafe extern "C" fn(*mut ocgpuDevice, i32) -> ocgpuResult;
/// `cuDeviceGetName` ABI.
pub type DeviceGetNameFn = unsafe extern "C" fn(*mut c_char, i32, ocgpuDevice) -> ocgpuResult;
/// `cuDeviceGetAttribute` ABI.
pub type DeviceGetAttributeFn =
    unsafe extern "C" fn(*mut i32, ocgpuDeviceAttribute, ocgpuDevice) -> ocgpuResult;

static DEVICE_GET_ATTRIBUTE_TARGET: OnceLock<DeviceGetAttributeFn> = OnceLock::new();

fn common_device_attribute(attribute: ocgpuDeviceAttribute) -> bool {
    matches!(
        attribute,
        ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
            | ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X
            | ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y
            | ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z
            | ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X
            | ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y
            | ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z
            | ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
            | ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY
            | ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_WARP_SIZE
            | ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MAX_PITCH
            | ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK
            | ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_CLOCK_RATE
            | ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT
            | ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
            | ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT
            | ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_INTEGRATED
            | ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY
            | ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_COMPUTE_MODE
            | ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS
            | ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_PCI_BUS_ID
            | ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID
            | ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE
            | ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH
            | ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE
            | ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR
            | ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING
            | ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
            | ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
            | ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MANAGED_MEMORY
            | ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS
            | ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS
    )
}

unsafe extern "C" fn common_device_get_attribute(
    value: *mut i32,
    attribute: ocgpuDeviceAttribute,
    device: ocgpuDevice,
) -> ocgpuResult {
    if !common_device_attribute(attribute) {
        return ocgpu_abi::OCGPU_ERROR_INVALID_ARGUMENT;
    }
    let Some(target) = DEVICE_GET_ATTRIBUTE_TARGET.get() else {
        return ocgpu_abi::OCGPU_ERROR_INTERNAL;
    };
    // SAFETY: CUDA's native attribute values are the stable common values and
    // the validated process-lifetime target has the exact raw CUDA ABI.
    unsafe { target(value, attribute, device) }
}
/// `cuCtxCreate_v2` ABI.
pub type CtxCreateFn = unsafe extern "C" fn(*mut ocgpuContext, u32, ocgpuDevice) -> ocgpuResult;
/// `cuCtxDestroy_v2` ABI.
pub type CtxDestroyFn = unsafe extern "C" fn(ocgpuContext) -> ocgpuResult;
/// `cuCtxSetCurrent` ABI.
pub type CtxSetCurrentFn = unsafe extern "C" fn(ocgpuContext) -> ocgpuResult;
/// `cuCtxGetCurrent` ABI.
pub type CtxGetCurrentFn = unsafe extern "C" fn(*mut ocgpuContext) -> ocgpuResult;
/// `cuCtxSynchronize` ABI.
pub type CtxSynchronizeFn = unsafe extern "C" fn() -> ocgpuResult;
/// `cuMemAlloc_v2` ABI.
pub type MemAllocFn = unsafe extern "C" fn(*mut ocgpuDeviceptr, usize) -> ocgpuResult;
/// `cuMemFree_v2` ABI.
pub type MemFreeFn = unsafe extern "C" fn(ocgpuDeviceptr) -> ocgpuResult;
/// `cuMemcpyHtoD_v2` ABI.
pub type MemcpyHtoDFn = unsafe extern "C" fn(ocgpuDeviceptr, *const c_void, usize) -> ocgpuResult;
/// `cuMemcpyDtoH_v2` ABI.
pub type MemcpyDtoHFn = unsafe extern "C" fn(*mut c_void, ocgpuDeviceptr, usize) -> ocgpuResult;
/// `cuStreamCreate` ABI.
pub type StreamCreateFn = unsafe extern "C" fn(*mut ocgpuStream, u32) -> ocgpuResult;
/// `cuStreamDestroy_v2` ABI.
pub type StreamDestroyFn = unsafe extern "C" fn(ocgpuStream) -> ocgpuResult;
/// `cuStreamSynchronize` ABI.
pub type StreamSynchronizeFn = unsafe extern "C" fn(ocgpuStream) -> ocgpuResult;
/// `cuEventCreate` ABI.
pub type EventCreateFn = unsafe extern "C" fn(*mut ocgpuEvent, u32) -> ocgpuResult;
/// `cuEventDestroy_v2` ABI.
pub type EventDestroyFn = unsafe extern "C" fn(ocgpuEvent) -> ocgpuResult;
/// `cuEventRecord` ABI.
pub type EventRecordFn = unsafe extern "C" fn(ocgpuEvent, ocgpuStream) -> ocgpuResult;
/// `cuEventSynchronize` ABI.
pub type EventSynchronizeFn = unsafe extern "C" fn(ocgpuEvent) -> ocgpuResult;
/// `cuModuleLoadData` ABI.
pub type ModuleLoadDataFn = unsafe extern "C" fn(*mut ocgpuModule, *const c_void) -> ocgpuResult;
/// `cuModuleUnload` ABI.
pub type ModuleUnloadFn = unsafe extern "C" fn(ocgpuModule) -> ocgpuResult;
/// `cuModuleGetFunction` ABI.
pub type ModuleGetFunctionFn =
    unsafe extern "C" fn(*mut ocgpuFunction, ocgpuModule, *const c_char) -> ocgpuResult;
/// `cuLaunchKernel` ABI.
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
    unsafe extern "C" fn(*const c_char, *mut *mut c_void, i32, u64) -> ocgpuResult;
type GetProcAddressV2Fn =
    unsafe extern "C" fn(*const c_char, *mut *mut c_void, i32, u64, *mut i32) -> ocgpuResult;

#[derive(Clone, Copy)]
enum ProcAddressEntry {
    V2(GetProcAddressV2Fn),
    Legacy(GetProcAddressFn),
}

impl ProcAddressEntry {
    const fn variant(self) -> ProcAddressVariant {
        match self {
            Self::V2(_) => ProcAddressVariant::V2,
            Self::Legacy(_) => ProcAddressVariant::Legacy,
        }
    }
}

/// CUDA proc-address bootstrap ABI selected by this loader.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProcAddressVariant {
    /// Five-argument `cuGetProcAddress_v2` with query-status output.
    V2,
    /// Four-argument legacy `cuGetProcAddress`.
    Legacy,
}

/// Interpreted CUDA proc-address query status.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QueryStatus {
    /// A compatible entry point was found.
    Success,
    /// The driver did not recognize the symbol.
    SymbolNotFound,
    /// The symbol exists but not for the requested API version.
    VersionInsufficient,
    /// A newer driver returned a status unknown to this build.
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
    /// Requested CUDA ABI version.
    pub requested_version: i32,
    /// `CUresult` returned by `cuGetProcAddress`.
    pub call_result: ocgpuResult,
    /// Query status returned separately by the driver.
    pub query_status: QueryStatus,
    /// Whether the call returned a non-null pointer.
    pub returned_pointer: bool,
}

/// How a symbol was ultimately resolved.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResolutionKind {
    /// Returned by `cuGetProcAddress`.
    ProcAddress,
    /// Found directly in the shared library export table.
    Direct,
    /// Neither version-aware nor direct lookup found the symbol.
    Missing,
    /// The compiled manifest excludes this symbol on the current platform.
    PlatformUnavailable,
}

/// Resolution diagnostics for one manifest entry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SymbolReport {
    /// Canonical vendor entry-point name.
    pub canonical_name: &'static str,
    /// Export or proc-address base name that succeeded.
    pub resolved_name: Option<&'static str>,
    /// Successful resolution path, or `Missing`.
    pub resolution: ResolutionKind,
    /// Whether the entry point was resolved.
    pub available: bool,
    /// Whether the manifest marks this entry point applicable to this platform.
    pub applicable: bool,
    /// Whether ABI v1 core validation requires this entry point.
    pub required: bool,
    /// Version-aware attempts made before direct fallback.
    pub proc_attempts: Vec<ProcAttempt>,
}

/// Immutable backend-load diagnostics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Diagnostics {
    /// Loaded library path or Linux system-loader identity.
    pub library_path: PathBuf,
    /// Runtime-reported driver version, when its query succeeded.
    pub runtime_version: Option<i32>,
    /// Whether `cuGetProcAddress` was directly exported.
    pub proc_address_support: bool,
    /// Exact proc-address export ABI selected, when one was available.
    pub proc_address_variant: Option<ProcAddressVariant>,
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
            Self::Loader(error) => write!(formatter, "CUDA loader error: {error}"),
            Self::BackendTooOld { library, symbols } => write!(
                formatter,
                "CUDA library {} is too old for core symbols: {}",
                library.display(),
                symbols.join(", ")
            ),
            Self::MissingCoreSymbols { library, symbols } => write!(
                formatter,
                "CUDA library {} is missing core symbols: {}",
                library.display(),
                symbols.join(", ")
            ),
            Self::InvalidRawTableDescriptor { symbol, source } => {
                write!(
                    formatter,
                    "invalid CUDA raw-table descriptor for {symbol}: {source}"
                )
            }
        }
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Loader(error) => Some(error),
            Self::BackendTooOld { .. } | Self::MissingCoreSymbols { .. } => None,
            Self::InvalidRawTableDescriptor { source, .. } => Some(source),
        }
    }
}

impl From<LoadError> for Error {
    fn from(value: LoadError) -> Self {
        Self::Loader(value)
    }
}

/// Optional raw CUDA entry points before core-profile validation.
pub struct UnvalidatedApi {
    /// Load and symbol-resolution diagnostics.
    pub diagnostics: Diagnostics,
    /// Optional `cuInit`.
    pub init: Option<InitFn>,
    /// Optional `cuDriverGetVersion`.
    pub driver_get_version: Option<DriverGetVersionFn>,
    /// Optional `cuDeviceGetCount`.
    pub device_get_count: Option<DeviceGetCountFn>,
    /// Optional `cuDeviceGet`.
    pub device_get: Option<DeviceGetFn>,
    /// Optional `cuDeviceGetName`.
    pub device_get_name: Option<DeviceGetNameFn>,
    /// Optional `cuDeviceGetAttribute`.
    pub device_get_attribute: Option<DeviceGetAttributeFn>,
    /// Optional `cuCtxCreate_v2`.
    pub ctx_create: Option<CtxCreateFn>,
    /// Optional `cuCtxDestroy_v2`.
    pub ctx_destroy: Option<CtxDestroyFn>,
    /// Optional `cuCtxSetCurrent`.
    pub ctx_set_current: Option<CtxSetCurrentFn>,
    /// Optional `cuCtxGetCurrent`.
    pub ctx_get_current: Option<CtxGetCurrentFn>,
    /// Optional `cuCtxSynchronize`.
    pub ctx_synchronize: Option<CtxSynchronizeFn>,
    /// Optional `cuMemAlloc_v2`.
    pub mem_alloc: Option<MemAllocFn>,
    /// Optional `cuMemFree_v2`.
    pub mem_free: Option<MemFreeFn>,
    /// Optional `cuMemcpyHtoD_v2`.
    pub memcpy_htod: Option<MemcpyHtoDFn>,
    /// Optional `cuMemcpyDtoH_v2`.
    pub memcpy_dtoh: Option<MemcpyDtoHFn>,
    /// Optional `cuStreamCreate`.
    pub stream_create: Option<StreamCreateFn>,
    /// Optional `cuStreamDestroy_v2`.
    pub stream_destroy: Option<StreamDestroyFn>,
    /// Optional `cuStreamSynchronize`.
    pub stream_synchronize: Option<StreamSynchronizeFn>,
    /// Optional `cuEventCreate`.
    pub event_create: Option<EventCreateFn>,
    /// Optional `cuEventDestroy_v2`.
    pub event_destroy: Option<EventDestroyFn>,
    /// Optional `cuEventRecord`.
    pub event_record: Option<EventRecordFn>,
    /// Optional `cuEventSynchronize`.
    pub event_synchronize: Option<EventSynchronizeFn>,
    /// Optional `cuModuleLoadData`.
    pub module_load_data: Option<ModuleLoadDataFn>,
    /// Optional `cuModuleUnload`.
    pub module_unload: Option<ModuleUnloadFn>,
    /// Optional `cuModuleGetFunction`.
    pub module_get_function: Option<ModuleGetFunctionFn>,
    /// Optional `cuLaunchKernel`.
    pub launch_kernel: Option<LaunchKernelFn>,
    raw_table: ocgpuCuApi_v1,
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
    /// Produces a raw CUDA ABI table with null entries for unavailable symbols.
    #[must_use]
    pub fn raw_table(&self) -> ocgpuCuApi_v1 {
        self.raw_table
    }

    /// Borrows the process-lifetime raw CUDA table without copying it.
    #[must_use]
    pub const fn raw_table_ref(&self) -> &ocgpuCuApi_v1 {
        &self.raw_table
    }

    fn common_layout_table(&self) -> ocgpuApi_v1 {
        ocgpuApi_v1 {
            struct_size: u32::try_from(size_of::<ocgpuApi_v1>()).unwrap_or(u32::MAX),
            abi_version: OCGPU_ABI_VERSION_1,
            backend: OCGPU_BACKEND_CUDA,
            flags: 0,
            driver_version: self.diagnostics.runtime_version.unwrap_or(0),
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

/// Non-null CUDA common-core dispatch.
///
/// Public fields are function pointers rather than `Option`s. Once this value
/// exists, hot-path calls do not repeat missing-symbol checks.
#[derive(Debug)]
pub struct ValidatedCoreApi {
    raw: &'static UnvalidatedApi,
    /// Validated `cuInit`.
    pub init: InitFn,
    /// Validated `cuDriverGetVersion`.
    pub driver_get_version: DriverGetVersionFn,
    /// Validated `cuDeviceGetCount`.
    pub device_get_count: DeviceGetCountFn,
    /// Validated `cuDeviceGet`.
    pub device_get: DeviceGetFn,
    /// Validated `cuDeviceGetName`.
    pub device_get_name: DeviceGetNameFn,
    /// Validated unified-attribute adapter over `cuDeviceGetAttribute`.
    pub device_get_attribute: DeviceGetAttributeFn,
    /// Validated `cuCtxCreate_v2`.
    pub ctx_create: CtxCreateFn,
    /// Validated `cuCtxDestroy_v2`.
    pub ctx_destroy: CtxDestroyFn,
    /// Validated `cuCtxSetCurrent`.
    pub ctx_set_current: CtxSetCurrentFn,
    /// Validated `cuCtxGetCurrent`.
    pub ctx_get_current: CtxGetCurrentFn,
    /// Validated `cuCtxSynchronize`.
    pub ctx_synchronize: CtxSynchronizeFn,
    /// Validated `cuMemAlloc_v2`.
    pub mem_alloc: MemAllocFn,
    /// Validated `cuMemFree_v2`.
    pub mem_free: MemFreeFn,
    /// Validated `cuMemcpyHtoD_v2`.
    pub memcpy_htod: MemcpyHtoDFn,
    /// Validated `cuMemcpyDtoH_v2`.
    pub memcpy_dtoh: MemcpyDtoHFn,
    /// Validated `cuStreamCreate`.
    pub stream_create: StreamCreateFn,
    /// Validated `cuStreamDestroy_v2`.
    pub stream_destroy: StreamDestroyFn,
    /// Validated `cuStreamSynchronize`.
    pub stream_synchronize: StreamSynchronizeFn,
    /// Validated `cuEventCreate`.
    pub event_create: EventCreateFn,
    /// Validated `cuEventDestroy_v2`.
    pub event_destroy: EventDestroyFn,
    /// Validated `cuEventRecord`.
    pub event_record: EventRecordFn,
    /// Validated `cuEventSynchronize`.
    pub event_synchronize: EventSynchronizeFn,
    /// Validated `cuModuleLoadData`.
    pub module_load_data: ModuleLoadDataFn,
    /// Validated `cuModuleUnload`.
    pub module_unload: ModuleUnloadFn,
    /// Validated `cuModuleGetFunction`.
    pub module_get_function: ModuleGetFunctionFn,
    /// Validated `cuLaunchKernel`.
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

    /// Produces the immutable backend-bound unified ABI table.
    #[must_use]
    pub fn common_table(&self) -> ocgpuApi_v1 {
        let mut table = self.raw.common_layout_table();
        table.ocgpuDeviceGetAttribute = Some(self.device_get_attribute);
        table
    }

    /// Produces the raw CUDA ABI table.
    #[must_use]
    pub fn raw_table(&self) -> ocgpuCuApi_v1 {
        self.raw.raw_table()
    }

    /// Borrows the process-lifetime raw CUDA table without copying it.
    #[must_use]
    pub const fn raw_table_ref(&self) -> &ocgpuCuApi_v1 {
        self.raw.raw_table_ref()
    }
}

static UNVALIDATED: OnceLock<Result<UnvalidatedApi, Error>> = OnceLock::new();
static VALIDATED: OnceLock<Result<ValidatedCoreApi, Error>> = OnceLock::new();

/// Loads CUDA and resolves every platform-applicable manifest entry point.
pub fn load_unvalidated() -> Result<&'static UnvalidatedApi, Error> {
    let result = UNVALIDATED.get_or_init(|| {
        let library = ocgpu_loader::load(Backend::Cuda)?;
        build_from_source(
            &LibrarySource::new(library),
            library.loaded_path().to_path_buf(),
        )
    });
    clone_result_ref(result)
}

/// Loads CUDA and validates the complete ABI v1 common core exactly once.
pub fn load() -> Result<&'static ValidatedCoreApi, Error> {
    let result = VALIDATED.get_or_init(|| validate(load_unvalidated()?));
    clone_result_ref(result)
}

/// Loads CUDA from an explicit canonicalized absolute path and validates it.
///
/// # Safety
///
/// The selected library and all of its dependencies must be trusted native code
/// that is safe to initialize and implements the exact CUDA ABIs represented by
/// this crate's generated table. The library must not be replaced with
/// incompatible code between path validation and operating-system loading.
#[cfg(feature = "explicit-library-path")]
pub unsafe fn load_from_absolute(path: &Path) -> Result<&'static ValidatedCoreApi, Error> {
    // SAFETY: this function has the same trusted-library and exact-ABI contract.
    let unvalidated = unsafe { load_unvalidated_from_absolute(path) }?;
    let result = VALIDATED.get_or_init(|| validate(unvalidated));
    clone_result_ref(result)
}

/// Loads CUDA's optional raw inventory from an explicit absolute path.
///
/// # Safety
///
/// The selected library and all of its dependencies must be trusted native code
/// that is safe to initialize and implements the exact CUDA ABIs represented by
/// this crate's generated table. The library must not be replaced with
/// incompatible code between path validation and operating-system loading.
#[cfg(feature = "explicit-library-path")]
pub unsafe fn load_unvalidated_from_absolute(
    path: &Path,
) -> Result<&'static UnvalidatedApi, Error> {
    // SAFETY: the caller guarantees the library trust and ABI requirements.
    let library = unsafe { ocgpu_loader::load_from_absolute(Backend::Cuda, path) }?;
    let unvalidated = UNVALIDATED.get_or_init(|| {
        build_from_source(
            &LibrarySource::new(library),
            library.loaded_path().to_path_buf(),
        )
    });
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
    // First collect every missing field so one diagnostic reports the complete
    // incompatible core profile rather than failing on the first symbol.
    macro_rules! check {
        ($field:ident, $name:literal) => {
            if raw.$field.is_none() {
                missing.push($name);
            }
        };
    }
    check!(init, "cuInit");
    check!(driver_get_version, "cuDriverGetVersion");
    check!(device_get_count, "cuDeviceGetCount");
    check!(device_get, "cuDeviceGet");
    check!(device_get_name, "cuDeviceGetName");
    check!(device_get_attribute, "cuDeviceGetAttribute");
    check!(ctx_create, "cuCtxCreate_v2");
    check!(ctx_destroy, "cuCtxDestroy_v2");
    check!(ctx_set_current, "cuCtxSetCurrent");
    check!(ctx_get_current, "cuCtxGetCurrent");
    check!(ctx_synchronize, "cuCtxSynchronize");
    check!(mem_alloc, "cuMemAlloc_v2");
    check!(mem_free, "cuMemFree_v2");
    check!(memcpy_htod, "cuMemcpyHtoD_v2");
    check!(memcpy_dtoh, "cuMemcpyDtoH_v2");
    check!(stream_create, "cuStreamCreate");
    check!(stream_destroy, "cuStreamDestroy_v2");
    check!(stream_synchronize, "cuStreamSynchronize");
    check!(event_create, "cuEventCreate");
    check!(event_destroy, "cuEventDestroy_v2");
    check!(event_record, "cuEventRecord");
    check!(event_synchronize, "cuEventSynchronize");
    check!(module_load_data, "cuModuleLoadData");
    check!(module_unload, "cuModuleUnload");
    check!(module_get_function, "cuModuleGetFunction");
    check!(launch_kernel, "cuLaunchKernel");
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

    // Every branch above proved these fields present. Pattern matching avoids
    // `unwrap`/`expect`, keeping this path mechanically panic-free.
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
    api_version: i32,
    proc_flags: u64,
    required: bool,
}

struct ProcLookup {
    address: Option<NonNull<c_void>>,
    attempt: ProcAttempt,
}

trait SymbolSource {
    fn proc_supported(&self) -> bool;
    fn proc_address_variant(&self) -> Option<ProcAddressVariant> {
        self.proc_supported().then_some(ProcAddressVariant::V2)
    }
    fn proc_lookup(&self, name: &str, api_version: i32, flags: u64) -> ProcLookup;
    fn direct_lookup(&self, name: &str) -> Option<NonNull<c_void>>;
}

struct LibrarySource {
    library: &'static Library,
    get_proc_address: Option<ProcAddressEntry>,
}

impl LibrarySource {
    fn new(library: &'static Library) -> Self {
        let versioned = inventory_platform_applicable("cuGetProcAddress_v2")
            .then(|| library.find(b"cuGetProcAddress_v2").ok().flatten())
            .flatten()
            .map(|address| {
                // SAFETY: the `_v2` export has the five-argument query-status ABI.
                ProcAddressEntry::V2(unsafe {
                    std::mem::transmute::<*mut c_void, GetProcAddressV2Fn>(address.as_ptr())
                })
            });
        let get_proc_address = versioned.or_else(|| {
            inventory_platform_applicable("cuGetProcAddress")
                .then(|| library.find(b"cuGetProcAddress").ok().flatten())
                .flatten()
                .map(|address| {
                    // SAFETY: the unsuffixed legacy export has four arguments.
                    ProcAddressEntry::Legacy(unsafe {
                        std::mem::transmute::<*mut c_void, GetProcAddressFn>(address.as_ptr())
                    })
                })
        });
        Self {
            library,
            get_proc_address,
        }
    }
}

impl SymbolSource for LibrarySource {
    fn proc_supported(&self) -> bool {
        self.get_proc_address.is_some()
    }

    fn proc_address_variant(&self) -> Option<ProcAddressVariant> {
        self.get_proc_address.map(ProcAddressEntry::variant)
    }

    fn proc_lookup(&self, name: &str, api_version: i32, flags: u64) -> ProcLookup {
        let Some(get_proc_address) = self.get_proc_address else {
            return ProcLookup {
                address: None,
                attempt: ProcAttempt {
                    requested_version: api_version,
                    call_result: -1,
                    query_status: QueryStatus::SymbolNotFound,
                    returned_pointer: false,
                },
            };
        };
        let Ok(name_c) = CString::new(name) else {
            return ProcLookup {
                address: None,
                attempt: ProcAttempt {
                    requested_version: api_version,
                    call_result: -1,
                    query_status: QueryStatus::SymbolNotFound,
                    returned_pointer: false,
                },
            };
        };
        invoke_proc_address(get_proc_address, &name_c, api_version, flags)
    }

    fn direct_lookup(&self, name: &str) -> Option<NonNull<c_void>> {
        self.library.find(name.as_bytes()).ok().flatten()
    }
}

fn invoke_proc_address(
    get_proc_address: ProcAddressEntry,
    name: &CStr,
    api_version: i32,
    flags: u64,
) -> ProcLookup {
    let mut address = std::ptr::null_mut();
    let mut status = -1;
    // SAFETY: output storage is writable and the requested name remains
    // NUL-terminated. Each branch uses the export's exact declared ABI.
    let call_result = match get_proc_address {
        ProcAddressEntry::V2(query) => unsafe {
            query(
                name.as_ptr(),
                &raw mut address,
                api_version,
                flags,
                &raw mut status,
            )
        },
        ProcAddressEntry::Legacy(query) => unsafe {
            query(name.as_ptr(), &raw mut address, api_version, flags)
        },
    };
    let returned_pointer = NonNull::new(address);
    let query_status = match get_proc_address {
        ProcAddressEntry::V2(_) => status.into(),
        ProcAddressEntry::Legacy(_) if returned_pointer.is_some() => QueryStatus::Success,
        ProcAddressEntry::Legacy(_) => QueryStatus::Unknown(-1),
    };
    let address = if call_result == CUDA_SUCCESS && query_status == QueryStatus::Success {
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

struct Resolved {
    address: Option<NonNull<c_void>>,
    report: SymbolReport,
}

fn resolve<S: SymbolSource>(source: &S, spec: SymbolSpec) -> Resolved {
    let mut proc_attempts = Vec::new();
    if source.proc_supported() && spec.api_version > 0 {
        let lookup = source.proc_lookup(spec.proc_name, spec.api_version, spec.proc_flags);
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

fn inventory_descriptor(canonical: &str) -> Option<&'static RawInventoryDescriptor> {
    CUDA_RAW_INVENTORY
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
    table: &mut ocgpuCuApi_v1,
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

fn resolve_inventory_descriptor<S: SymbolSource>(
    source: &S,
    table: &mut ocgpuCuApi_v1,
    descriptor: &RawInventoryDescriptor,
) -> Result<SymbolReport, Error> {
    if !platform_is_applicable(descriptor.platform_mask) {
        return Ok(platform_unavailable_report(descriptor.canonical, false));
    }
    let resolved = resolve(
        source,
        SymbolSpec {
            canonical: descriptor.canonical,
            proc_name: descriptor.proc_name,
            direct_names: descriptor.direct_names,
            api_version: target_proc_version(descriptor),
            proc_flags: descriptor.proc_flags,
            required: false,
        },
    );
    if let Some(address) = resolved.address {
        write_raw_inventory_descriptor(table, descriptor, address)?;
    }
    Ok(resolved.report)
}

fn write_raw_descriptor(
    table: &mut ocgpuCuApi_v1,
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
    table: &mut ocgpuCuApi_v1,
    canonical: &str,
    address: NonNull<c_void>,
) -> Result<(), Error> {
    if let Some(descriptor) = CUDA_RAW_SYMBOLS
        .iter()
        .find(|descriptor| descriptor.canonical == canonical)
    {
        write_raw_descriptor(table, descriptor, address)?;
    }
    Ok(())
}

#[allow(clippy::similar_names, clippy::too_many_lines)]
fn build_from_source<S: SymbolSource>(
    source: &S,
    library_path: PathBuf,
) -> Result<UnvalidatedApi, Error> {
    let mut reports = Vec::with_capacity(CUDA_RAW_INVENTORY.len());
    let mut raw_table = ocgpuCuApi_v1 {
        struct_size: u32::try_from(size_of::<ocgpuCuApi_v1>()).unwrap_or(u32::MAX),
        abi_version: OCGPU_ABI_VERSION_1,
        backend: OCGPU_BACKEND_CUDA,
        flags: 0,
        driver_version: 0,
        reserved0: 0,
        ..ocgpuCuApi_v1::default()
    };

    macro_rules! resolve_field {
        ($type:ty, $canonical:literal, $proc_name:literal, [$($direct:literal),+ $(,)?], $version:expr) => {{
            let descriptor = inventory_descriptor($canonical);
            let fallback_direct_names: &[&str] = &[$($direct),+];
            let resolved = if descriptor
                .is_none_or(|descriptor| platform_is_applicable(descriptor.platform_mask))
            {
                resolve(
                    source,
                    SymbolSpec {
                        canonical: $canonical,
                        proc_name: descriptor.map_or($proc_name, |descriptor| descriptor.proc_name),
                        direct_names: descriptor
                            .map_or(fallback_direct_names, |descriptor| descriptor.direct_names),
                        api_version: descriptor
                            .map_or($version, target_proc_version),
                        proc_flags: descriptor.map_or(
                            GET_PROC_ADDRESS_LEGACY_STREAM,
                            |descriptor| descriptor.proc_flags,
                        ),
                        required: true,
                    },
                )
            } else {
                Resolved {
                    address: None,
                    report: platform_unavailable_report($canonical, true),
                }
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

    let init = resolve_field!(InitFn, "cuInit", "cuInit", ["cuInit"], 2_000);
    let driver_get_version = resolve_field!(
        DriverGetVersionFn,
        "cuDriverGetVersion",
        "cuDriverGetVersion",
        ["cuDriverGetVersion"],
        2_020
    );
    let device_get_count = resolve_field!(
        DeviceGetCountFn,
        "cuDeviceGetCount",
        "cuDeviceGetCount",
        ["cuDeviceGetCount"],
        2_000
    );
    let device_get = resolve_field!(
        DeviceGetFn,
        "cuDeviceGet",
        "cuDeviceGet",
        ["cuDeviceGet"],
        2_000
    );
    let device_get_name = resolve_field!(
        DeviceGetNameFn,
        "cuDeviceGetName",
        "cuDeviceGetName",
        ["cuDeviceGetName"],
        2_000
    );
    let device_get_attribute = resolve_field!(
        DeviceGetAttributeFn,
        "cuDeviceGetAttribute",
        "cuDeviceGetAttribute",
        ["cuDeviceGetAttribute"],
        2_000
    );
    let ctx_create = resolve_field!(
        CtxCreateFn,
        "cuCtxCreate_v2",
        "cuCtxCreate",
        ["cuCtxCreate_v2"],
        3_020
    );
    let ctx_destroy = resolve_field!(
        CtxDestroyFn,
        "cuCtxDestroy_v2",
        "cuCtxDestroy",
        ["cuCtxDestroy_v2"],
        4_000
    );
    let ctx_set_current = resolve_field!(
        CtxSetCurrentFn,
        "cuCtxSetCurrent",
        "cuCtxSetCurrent",
        ["cuCtxSetCurrent"],
        4_000
    );
    let ctx_get_current = resolve_field!(
        CtxGetCurrentFn,
        "cuCtxGetCurrent",
        "cuCtxGetCurrent",
        ["cuCtxGetCurrent"],
        4_000
    );
    let ctx_synchronize = resolve_field!(
        CtxSynchronizeFn,
        "cuCtxSynchronize",
        "cuCtxSynchronize",
        ["cuCtxSynchronize"],
        2_000
    );
    let mem_alloc = resolve_field!(
        MemAllocFn,
        "cuMemAlloc_v2",
        "cuMemAlloc",
        ["cuMemAlloc_v2"],
        3_020
    );
    let mem_free = resolve_field!(
        MemFreeFn,
        "cuMemFree_v2",
        "cuMemFree",
        ["cuMemFree_v2"],
        3_020
    );
    let memcpy_htod = resolve_field!(
        MemcpyHtoDFn,
        "cuMemcpyHtoD_v2",
        "cuMemcpyHtoD",
        ["cuMemcpyHtoD_v2"],
        3_020
    );
    let memcpy_dtoh = resolve_field!(
        MemcpyDtoHFn,
        "cuMemcpyDtoH_v2",
        "cuMemcpyDtoH",
        ["cuMemcpyDtoH_v2"],
        3_020
    );
    let stream_create = resolve_field!(
        StreamCreateFn,
        "cuStreamCreate",
        "cuStreamCreate",
        ["cuStreamCreate"],
        2_000
    );
    let stream_destroy = resolve_field!(
        StreamDestroyFn,
        "cuStreamDestroy_v2",
        "cuStreamDestroy",
        ["cuStreamDestroy_v2"],
        4_000
    );
    let stream_synchronize = resolve_field!(
        StreamSynchronizeFn,
        "cuStreamSynchronize",
        "cuStreamSynchronize",
        ["cuStreamSynchronize"],
        2_000
    );
    let event_create = resolve_field!(
        EventCreateFn,
        "cuEventCreate",
        "cuEventCreate",
        ["cuEventCreate"],
        2_000
    );
    let event_destroy = resolve_field!(
        EventDestroyFn,
        "cuEventDestroy_v2",
        "cuEventDestroy",
        ["cuEventDestroy_v2"],
        4_000
    );
    let event_record = resolve_field!(
        EventRecordFn,
        "cuEventRecord",
        "cuEventRecord",
        ["cuEventRecord"],
        2_000
    );
    let event_synchronize = resolve_field!(
        EventSynchronizeFn,
        "cuEventSynchronize",
        "cuEventSynchronize",
        ["cuEventSynchronize"],
        2_000
    );
    let module_load_data = resolve_field!(
        ModuleLoadDataFn,
        "cuModuleLoadData",
        "cuModuleLoadData",
        ["cuModuleLoadData"],
        2_000
    );
    let module_unload = resolve_field!(
        ModuleUnloadFn,
        "cuModuleUnload",
        "cuModuleUnload",
        ["cuModuleUnload"],
        2_000
    );
    let module_get_function = resolve_field!(
        ModuleGetFunctionFn,
        "cuModuleGetFunction",
        "cuModuleGetFunction",
        ["cuModuleGetFunction"],
        2_000
    );
    let launch_kernel = resolve_field!(
        LaunchKernelFn,
        "cuLaunchKernel",
        "cuLaunchKernel",
        ["cuLaunchKernel"],
        4_000
    );

    let runtime_version = driver_get_version.and_then(|query| {
        let mut version = 0;
        // SAFETY: output points to initialized writable storage and the function
        // pointer was resolved using the exact vendor ABI.
        let result = unsafe { query(&raw mut version) };
        (result == CUDA_SUCCESS).then_some(version)
    });

    raw_table.driver_version = runtime_version.unwrap_or(0);
    for descriptor in CUDA_RAW_INVENTORY {
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
        )?);
    }

    Ok(UnvalidatedApi {
        diagnostics: Diagnostics {
            library_path,
            runtime_version,
            proc_address_support: source.proc_supported(),
            proc_address_variant: source.proc_address_variant(),
            loaded_architecture: std::env::consts::ARCH,
            symbols: reports,
        },
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
        CUDA_SUCCESS, ProcAttempt, ProcLookup, QueryStatus, ResolutionKind, SymbolSource,
        build_from_source, validate,
    };
    use std::cell::Cell;
    use std::collections::BTreeSet;
    use std::ffi::c_void;
    use std::ptr::NonNull;

    const COMMON_ATTRIBUTES: &[i32] = &[
        ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
        ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
        ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
        ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
        ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
        ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
        ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
        ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY,
        ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_WARP_SIZE,
        ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MAX_PITCH,
        ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
        ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_CLOCK_RATE,
        ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT,
        ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT,
        ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_INTEGRATED,
        ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY,
        ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_COMPUTE_MODE,
        ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS,
        ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
        ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
        ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
        ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
        ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
        ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
        ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
        ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_MANAGED_MEMORY,
        ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS,
        ocgpu_abi::OCGPU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS,
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
        CUDA_SUCCESS
    }

    unsafe extern "C" fn mock_driver_version(output: *mut i32) -> i32 {
        // SAFETY: the mock ABI requires callers to supply writable output storage.
        if let Some(output) = unsafe { output.as_mut() } {
            *output = super::COMPILED_API_VERSION;
            CUDA_SUCCESS
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
            CUDA_SUCCESS
        } else {
            1
        }
    }

    unsafe extern "C" fn mock_get_proc_address(
        _name: *const std::ffi::c_char,
        output: *mut *mut c_void,
        _version: i32,
        _flags: u64,
    ) -> i32 {
        // SAFETY: the mock ABI requires callers to supply writable output storage.
        if let Some(output) = unsafe { output.as_mut() } {
            *output = mock_function as *const () as *mut c_void;
            CUDA_SUCCESS
        } else {
            1
        }
    }

    unsafe extern "C" fn mock_get_proc_address_v2(
        _name: *const std::ffi::c_char,
        output: *mut *mut c_void,
        _version: i32,
        _flags: u64,
        status: *mut i32,
    ) -> i32 {
        // SAFETY: the mock ABI requires both pointers to name writable storage.
        if let (Some(output), Some(status)) =
            (unsafe { output.as_mut() }, unsafe { status.as_mut() })
        {
            *output = mock_function as *const () as *mut c_void;
            *status = 2;
            CUDA_SUCCESS
        } else {
            1
        }
    }

    struct MockLibrary {
        proc_supported: bool,
        missing: BTreeSet<&'static str>,
    }

    impl Default for MockLibrary {
        fn default() -> Self {
            Self {
                proc_supported: true,
                missing: BTreeSet::new(),
            }
        }
    }

    impl MockLibrary {
        fn address_for(name: &str) -> NonNull<c_void> {
            let address = match name {
                "cuDriverGetVersion" => mock_driver_version as *const () as *mut c_void,
                "cuDeviceGetAttribute" => mock_device_get_attribute as *const () as *mut c_void,
                _ => mock_function as *const () as *mut c_void,
            };
            NonNull::new(address).expect("function addresses are non-null")
        }
    }

    impl SymbolSource for MockLibrary {
        fn proc_supported(&self) -> bool {
            self.proc_supported
        }

        fn proc_lookup(&self, name: &str, api_version: i32, _flags: u64) -> ProcLookup {
            let address = (!self.missing.contains(name)).then(|| Self::address_for(name));
            ProcLookup {
                attempt: ProcAttempt {
                    requested_version: api_version,
                    call_result: CUDA_SUCCESS,
                    query_status: if address.is_some() {
                        QueryStatus::Success
                    } else {
                        QueryStatus::SymbolNotFound
                    },
                    returned_pointer: address.is_some(),
                },
                address,
            }
        }

        fn direct_lookup(&self, name: &str) -> Option<NonNull<c_void>> {
            let bootstrap_missing =
                matches!(name, "cuGetProcAddress" | "cuGetProcAddress_v2") && !self.proc_supported;
            (!bootstrap_missing && !self.missing.contains(name)).then(|| Self::address_for(name))
        }
    }

    struct TooOldMockLibrary;

    impl SymbolSource for TooOldMockLibrary {
        fn proc_supported(&self) -> bool {
            true
        }

        fn proc_lookup(&self, name: &str, api_version: i32, _flags: u64) -> ProcLookup {
            let available = name != "cuLaunchKernel";
            ProcLookup {
                attempt: ProcAttempt {
                    requested_version: api_version,
                    call_result: CUDA_SUCCESS,
                    query_status: if available {
                        QueryStatus::Success
                    } else {
                        QueryStatus::VersionInsufficient
                    },
                    returned_pointer: available,
                },
                address: available.then(|| MockLibrary::address_for(name)),
            }
        }

        fn direct_lookup(&self, name: &str) -> Option<NonNull<c_void>> {
            (name != "cuLaunchKernel").then(|| MockLibrary::address_for(name))
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
            build_from_source(&MockLibrary::default(), "mock-cuda".into())
                .expect("generated descriptors must fit the mock raw table"),
        ));
        let validated = validate(raw).expect("complete mock exports satisfy core profile");
        assert_eq!(
            validated.diagnostics().runtime_version,
            Some(super::COMPILED_API_VERSION)
        );
        assert_eq!(
            validated.diagnostics().proc_address_variant,
            Some(super::ProcAddressVariant::V2)
        );
        assert_eq!(
            validated.common_table().driver_version,
            super::COMPILED_API_VERSION
        );
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
        assert_eq!(reported_names.len(), super::CUDA_RAW_INVENTORY.len());
        assert!(super::CUDA_RAW_INVENTORY.iter().all(|descriptor| {
            validated
                .diagnostics()
                .symbol(descriptor.canonical)
                .is_some()
        }));
        for bootstrap in ["cuGetProcAddress", "cuGetProcAddress_v2"] {
            let report = validated
                .diagnostics()
                .symbol(bootstrap)
                .expect("both bootstrap exports have independent inventory rows");
            assert_eq!(report.resolution, ResolutionKind::Direct);
            assert!(report.available);
        }
        let raw_table = raw.raw_table();
        assert_eq!(raw_table.backend, ocgpu_abi::OCGPU_BACKEND_CUDA);
        assert_eq!(
            raw_table.struct_size as usize,
            size_of::<ocgpu_abi::ocgpuCuApi_v1>()
        );
        assert_eq!(
            validated.common_table().struct_size as usize,
            size_of::<ocgpu_abi::ocgpuApi_v1>()
        );
        assert!(raw_table.ocgpuCuGetProcAddress.is_some());
        assert!(raw_table.ocgpuCuGetProcAddress_v2.is_some());
        assert!(raw_table.ocgpuCuLaunchKernel.is_some());
    }

    #[test]
    fn missing_symbols_are_reported_together() {
        let raw = Box::leak(Box::new(
            build_from_source(
                &MockLibrary {
                    proc_supported: false,
                    missing: BTreeSet::from(["cuInit", "cuLaunchKernel"]),
                },
                "mock-cuda".into(),
            )
            .expect("generated descriptors must fit the mock raw table"),
        ));
        let error = validate(raw).expect_err("incomplete core must fail validation");
        assert!(raw.raw_table().ocgpuCuInit.is_none());
        let super::Error::MissingCoreSymbols { symbols, .. } = error else {
            panic!("unexpected error category")
        };
        assert_eq!(symbols, ["cuInit", "cuLaunchKernel"]);
    }

    #[test]
    fn version_insufficient_core_symbol_reports_old_backend() {
        let raw = Box::leak(Box::new(
            build_from_source(&TooOldMockLibrary, "mock-old-cuda".into())
                .expect("generated descriptors must fit the mock raw table"),
        ));
        let error = validate(raw).expect_err("version-insufficient core must fail validation");
        let super::Error::BackendTooOld { symbols, .. } = error else {
            panic!("unexpected error category")
        };
        assert_eq!(symbols, ["cuLaunchKernel"]);
    }

    #[test]
    fn proc_address_precedes_direct_export_fallback() {
        let raw = build_from_source(
            &MockLibrary {
                proc_supported: true,
                missing: BTreeSet::new(),
            },
            "mock-cuda".into(),
        )
        .expect("generated descriptors must fit the mock raw table");
        let init = raw
            .diagnostics
            .symbol("cuInit")
            .expect("inventory always reports init");
        assert_eq!(init.resolution, ResolutionKind::ProcAddress);
        assert_eq!(init.proc_attempts.len(), 1);
    }

    #[test]
    fn legacy_and_v2_proc_address_exports_use_their_exact_abis() {
        let name = std::ffi::CString::new("mock").expect("literal is NUL-free");
        let legacy = super::invoke_proc_address(
            super::ProcAddressEntry::Legacy(mock_get_proc_address),
            &name,
            2_000,
            0,
        );
        assert!(legacy.address.is_some());
        assert_eq!(legacy.attempt.query_status, QueryStatus::Success);

        let versioned = super::invoke_proc_address(
            super::ProcAddressEntry::V2(mock_get_proc_address_v2),
            &name,
            13_030,
            0,
        );
        assert!(versioned.address.is_none());
        assert!(versioned.attempt.returned_pointer);
        assert_eq!(
            versioned.attempt.query_status,
            QueryStatus::VersionInsufficient
        );
    }

    #[test]
    fn every_emitted_inventory_entry_populates_its_generated_slot() {
        let raw = build_from_source(&MockLibrary::default(), "mock-cuda".into())
            .expect("generated descriptors must fit the mock raw table");
        let table = raw.raw_table();
        let mut emitted_offsets = BTreeSet::new();
        for descriptor in super::CUDA_RAW_INVENTORY {
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
        assert_eq!(emitted_offsets.len(), super::CUDA_RAW_SYMBOLS.len());
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn generated_resolution_metadata_is_slot_aligned_and_fail_closed() {
        const CORE_METADATA: &[(&str, &str, i32)] = &[
            ("cuInit", "cuInit", 2_000),
            ("cuDriverGetVersion", "cuDriverGetVersion", 2_020),
            ("cuDeviceGetCount", "cuDeviceGetCount", 2_000),
            ("cuDeviceGet", "cuDeviceGet", 2_000),
            ("cuDeviceGetName", "cuDeviceGetName", 2_000),
            ("cuDeviceGetAttribute", "cuDeviceGetAttribute", 2_000),
            ("cuCtxCreate_v2", "cuCtxCreate", 3_020),
            ("cuCtxDestroy_v2", "cuCtxDestroy", 4_000),
            ("cuCtxSetCurrent", "cuCtxSetCurrent", 4_000),
            ("cuCtxGetCurrent", "cuCtxGetCurrent", 4_000),
            ("cuCtxSynchronize", "cuCtxSynchronize", 2_000),
            ("cuMemAlloc_v2", "cuMemAlloc", 3_020),
            ("cuMemFree_v2", "cuMemFree", 3_020),
            ("cuMemcpyHtoD_v2", "cuMemcpyHtoD", 3_020),
            ("cuMemcpyDtoH_v2", "cuMemcpyDtoH", 3_020),
            ("cuStreamCreate", "cuStreamCreate", 2_000),
            ("cuStreamDestroy_v2", "cuStreamDestroy", 4_000),
            ("cuStreamSynchronize", "cuStreamSynchronize", 2_000),
            ("cuEventCreate", "cuEventCreate", 2_000),
            ("cuEventDestroy_v2", "cuEventDestroy", 4_000),
            ("cuEventRecord", "cuEventRecord", 2_000),
            ("cuEventSynchronize", "cuEventSynchronize", 2_000),
            ("cuModuleLoadData", "cuModuleLoadData", 2_000),
            ("cuModuleUnload", "cuModuleUnload", 2_000),
            ("cuModuleGetFunction", "cuModuleGetFunction", 2_000),
            ("cuLaunchKernel", "cuLaunchKernel", 4_000),
        ];
        const DIRECT_ONLY_SYMBOLS: &[&str] = &[
            "cuProfilerStart",
            "cuProfilerStop",
            "cuGetProcAddress",
            "cuGetProcAddress_v2",
            "cuGraphInstantiateWithFlags",
            "cuGraphInstantiate_v2",
            "cuProfilerInitialize",
            "cuGraphInstantiate",
        ];

        assert_eq!(
            super::CUDA_RAW_INVENTORY.len(),
            super::CUDA_RAW_SYMBOLS.len(),
            "runtime inventory must contain only emitted vendor callables"
        );
        let raw = build_from_source(&MockLibrary::default(), "metadata-cuda".into())
            .expect("generated descriptors must fit the mock raw table");
        let direct_raw = build_from_source(
            &MockLibrary {
                proc_supported: false,
                missing: BTreeSet::new(),
            },
            "metadata-direct-cuda".into(),
        )
        .expect("generated descriptors must fit the mock raw table");
        let expected_alias_reports = super::CUDA_RAW_SYMBOLS
            .iter()
            .filter(|descriptor| {
                descriptor.direct_names.first().copied() != Some(descriptor.canonical)
            })
            .count();
        let mut observed_alias_reports = 0;
        for descriptor in super::CUDA_RAW_SYMBOLS {
            let inventory = super::CUDA_RAW_INVENTORY
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

            let core_metadata = CORE_METADATA
                .iter()
                .find(|(canonical, _, _)| *canonical == descriptor.canonical);
            if let Some((_, expected_proc_name, expected_version)) = core_metadata {
                assert_eq!(descriptor.proc_name, *expected_proc_name);
                assert_eq!(descriptor.direct_names, &[descriptor.canonical]);
                assert_eq!(descriptor.proc_version_linux, *expected_version);
                assert_eq!(descriptor.proc_version_windows, *expected_version);
                assert_eq!(descriptor.proc_flags, super::GET_PROC_ADDRESS_LEGACY_STREAM);
                assert_eq!(descriptor.platform_mask, 7);
            }

            let proc_enabled =
                descriptor.proc_version_linux > 0 || descriptor.proc_version_windows > 0;
            assert_eq!(
                proc_enabled,
                !DIRECT_ONLY_SYMBOLS.contains(&descriptor.canonical),
                "CUDA exact-catalog proc classification drifted: {}",
                descriptor.canonical
            );
            assert_eq!(
                descriptor.proc_version_linux, descriptor.proc_version_windows,
                "CUDA catalog baseline is shared across supported platforms"
            );
            let report = raw
                .diagnostics
                .symbol(descriptor.canonical)
                .expect("every emitted slot must have one diagnostic row");
            assert_eq!(report.required, core_metadata.is_some());
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
                assert_eq!(descriptor.proc_flags, super::GET_PROC_ADDRESS_LEGACY_STREAM);
                assert_eq!(report.resolution, ResolutionKind::ProcAddress);
                assert_eq!(report.resolved_name, Some(descriptor.proc_name));
                let direct_report = direct_raw
                    .diagnostics
                    .symbol(descriptor.canonical)
                    .expect("every emitted slot must support exact direct fallback");
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
                        super::CUDA_RAW_INVENTORY
                            .iter()
                            .find(|candidate| candidate.canonical == descriptor.canonical)
                            .expect("typed descriptor has a matching inventory row")
                    ))
                );
            } else {
                assert_eq!(descriptor.proc_version_linux, 0);
                assert!(report.proc_attempts.is_empty());
                assert_eq!(report.resolution, ResolutionKind::Direct);
                assert_eq!(
                    report.resolved_name,
                    descriptor.direct_names.first().copied()
                );
            }
        }
        assert!(expected_alias_reports > 0);
        assert_eq!(observed_alias_reports, expected_alias_reports);
        for helper in ["culib", "is_culib_present"] {
            assert!(
                super::CUDA_RAW_INVENTORY
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
        let mut table = ocgpu_abi::ocgpuCuApi_v1::default();
        let report = super::resolve_inventory_descriptor(&source, &mut table, &descriptor)
            .expect("an unavailable descriptor does not touch its table offset");
        assert_eq!(report.resolution, ResolutionKind::PlatformUnavailable);
        assert!(!report.applicable);
        assert!(report.proc_attempts.is_empty());
        assert_eq!(source.proc_queries.get(), 0);
        assert_eq!(source.direct_queries.get(), 0);
    }

    #[test]
    fn unified_attribute_adapter_restricts_ids_without_changing_the_raw_table() {
        const VENDOR_ONLY_ATTRIBUTE: i32 = 9_999;
        let raw = Box::leak(Box::new(
            build_from_source(&MockLibrary::default(), "mock-cuda".into())
                .expect("generated descriptors must fit the mock raw table"),
        ));
        let validated = validate(raw).expect("complete mock exports satisfy core profile");

        let common_query = validated
            .common_table()
            .ocgpuDeviceGetAttribute
            .expect("validated common table has an attribute adapter");
        let mut common_value = 0;
        for &attribute in COMMON_ATTRIBUTES {
            // SAFETY: the mock accepts this writable output and does not retain it.
            let common_result = unsafe { common_query(&raw mut common_value, attribute, 0) };
            assert_eq!(common_result, CUDA_SUCCESS);
            assert_eq!(common_value, attribute);
        }

        // SAFETY: the adapter rejects this value before calling the mock target.
        let invalid_result =
            unsafe { common_query(&raw mut common_value, VENDOR_ONLY_ATTRIBUTE, 0) };
        assert_eq!(invalid_result, ocgpu_abi::OCGPU_ERROR_INVALID_ARGUMENT);

        let raw_query = raw
            .raw_table()
            .ocgpuCuDeviceGetAttribute
            .expect("the raw entry remains callable");
        let mut raw_value = 0;
        // SAFETY: the raw mock accepts all vendor integer values.
        let raw_result = unsafe { raw_query(&raw mut raw_value, VENDOR_ONLY_ATTRIBUTE, 0) };
        assert_eq!(raw_result, CUDA_SUCCESS);
        assert_eq!(raw_value, VENDOR_ONLY_ATTRIBUTE);
    }
}
