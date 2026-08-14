// SPDX-License-Identifier: CC0-1.0

//! Typed, backend-safe Rust interface to the runtime-loaded ocgpu API.
//!
//! Loading validates the common core profile once. Calls made through a
//! [`Driver`] then use non-null function pointers directly: there is no global
//! backend selection, heap allocation, data copy, or per-call availability
//! branch in the dispatch path.
//!
//! HIP 5, HIP 6, and HIP 7 are loaded as distinct ABI profiles. The typed
//! common API is stable across them; the raw HIP table uses the current HIP 7
//! platform-union layout and leaves entries without reviewed compatibility for
//! the selected profile/version null.
//!
//! Backend marker types make it a compile-time error to combine resources from
//! different runtimes:
//!
//! ```compile_fail
//! use ocgpu::{Backend, Context, Cuda, Hip, Stream};
//!
//! fn requires_one_backend<B: Backend>(
//!     _context: &Context<'_, B>,
//!     _stream: &Stream<'_, '_, B>,
//! ) {}
//!
//! fn rejected(cuda: &Context<'_, Cuda>, hip: &Stream<'_, '_, Hip>) {
//!     requires_one_backend(cuda, hip);
//! }
//! ```

use core::ffi::{CStr, c_char, c_void};
use core::fmt;
use core::marker::PhantomData;
use core::ptr;
#[cfg(feature = "explicit-library-path")]
use std::path::Path;
use std::path::PathBuf;

/// Exact C-compatible declarations.
pub mod sys {
    pub use ocgpu_abi::*;
}

/// Backend-native APIs.
pub mod raw {
    /// CUDA Driver API dispatch.
    #[cfg(feature = "raw-cuda")]
    pub mod cuda {
        pub use ocgpu_cuda::*;
    }

    /// HIP driver-shaped API dispatch using the current HIP 7 platform-union
    /// layout. Nullable entries may be profile-unavailable when the selected
    /// runtime predates that platform's reviewed raw-inventory baseline.
    #[cfg(feature = "raw-hip")]
    pub mod hip {
        pub use ocgpu_hip::*;
    }
}

/// Stable name of the validated common profile.
pub const CORE_PROFILE_NAME: &str = "ocgpu-core-v1";

/// Backend selected for a driver instance.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BackendKind {
    /// NVIDIA CUDA Driver API.
    Cuda,
    /// AMD HIP runtime's driver-shaped API.
    Hip,
}

impl BackendKind {
    /// C ABI backend value.
    #[must_use]
    pub const fn as_raw(self) -> sys::ocgpuBackend {
        match self {
            Self::Cuda => sys::OCGPU_BACKEND_CUDA,
            Self::Hip => sys::OCGPU_BACKEND_HIP,
        }
    }

    /// Stable lowercase spelling used by the CLI and serialized diagnostics.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Hip => "hip",
        }
    }
}

impl fmt::Display for BackendKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// HIP runtime ABI family selected by the loader.
///
/// HIP major releases are separate ABI profiles. The loader never treats a
/// library from one family as though it implemented another family merely
/// because some exported symbol names overlap.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum HipRuntimeProfile {
    /// HIP 5.x runtime ABI.
    Hip5,
    /// HIP 6.x runtime ABI.
    Hip6,
    /// HIP 7.x runtime ABI.
    Hip7,
}

impl HipRuntimeProfile {
    /// Stable lowercase spelling used by serialized diagnostics.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Hip5 => "hip_5",
            Self::Hip6 => "hip_6",
            Self::Hip7 => "hip_7",
        }
    }

    /// Human-readable profile name used by text diagnostics.
    #[must_use]
    pub const fn display_name(self) -> &'static str {
        match self {
            Self::Hip5 => "HIP 5",
            Self::Hip6 => "HIP 6",
            Self::Hip7 => "HIP 7",
        }
    }

    /// Profile bits carried by HIP common and raw table flags.
    #[must_use]
    pub const fn api_flags(self) -> u32 {
        match self {
            Self::Hip5 => sys::OCGPU_API_FLAG_HIP_RUNTIME_PROFILE_5,
            Self::Hip6 => sys::OCGPU_API_FLAG_HIP_RUNTIME_PROFILE_6,
            Self::Hip7 => sys::OCGPU_API_FLAG_HIP_RUNTIME_PROFILE_7,
        }
    }

    /// Decodes a HIP runtime profile from common or raw table flags.
    ///
    /// Returns `None` for CUDA's zero flags and for profile values unknown to
    /// this build.
    #[must_use]
    pub const fn from_api_flags(flags: u32) -> Option<Self> {
        match flags & sys::OCGPU_API_FLAG_HIP_RUNTIME_PROFILE_MASK {
            sys::OCGPU_API_FLAG_HIP_RUNTIME_PROFILE_5 => Some(Self::Hip5),
            sys::OCGPU_API_FLAG_HIP_RUNTIME_PROFILE_6 => Some(Self::Hip6),
            sys::OCGPU_API_FLAG_HIP_RUNTIME_PROFILE_7 => Some(Self::Hip7),
            _ => None,
        }
    }
}

impl fmt::Display for HipRuntimeProfile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.display_name())
    }
}

/// Error returned by the typed API.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// The requested runtime library could not be loaded or validated.
    BackendUnavailable {
        /// Requested backend.
        backend: BackendKind,
        /// Stable diagnostic from the loader.
        detail: String,
    },
    /// The selected runtime is present but too old for the common profile.
    BackendTooOld {
        /// Requested backend.
        backend: BackendKind,
        /// Required vendor entry points rejected by version-aware lookup.
        symbols: Vec<&'static str>,
    },
    /// A known runtime ABI family was found, but its exact version predates
    /// the reviewed compatibility interval.
    BackendRuntimeTooOld {
        /// Requested backend.
        backend: BackendKind,
        /// Exact runtime version and reviewed interval from the loader.
        detail: String,
    },
    /// A runtime library was found but its ABI family did not match the
    /// securely selected backend profile.
    BackendAbiMismatch {
        /// Requested backend.
        backend: BackendKind,
        /// Stable mismatch diagnostic from the loader.
        detail: String,
    },
    /// A required common-profile symbol was not resolved.
    MissingCoreSymbol {
        /// Requested backend.
        backend: BackendKind,
        /// Common ABI field name.
        symbol: &'static str,
    },
    /// A backend operation returned a non-success result.
    Api {
        /// Common operation name.
        operation: &'static str,
        /// Raw operation result. This can be a stable negative `OCGPU_*`
        /// management code or a backend-native CUDA/HIP status value.
        result: sys::ocgpuResult,
    },
    /// A safe wrapper rejected an invalid argument before calling the backend.
    InvalidArgument(&'static str),
    /// A successful backend call unexpectedly returned a null handle.
    NullHandle(&'static str),
    /// A backend count or length could not be represented safely.
    InvalidSize(&'static str),
}

impl Error {
    /// Returns the raw result carried across the C surface.
    ///
    /// The declared negative `OCGPU_*` management codes are stable across
    /// backends. [`Error::Api`] deliberately preserves a backend operation's
    /// native CUDA or HIP status code, which must be interpreted together with
    /// the originating backend.
    #[must_use]
    pub const fn result(&self) -> sys::ocgpuResult {
        match self {
            Self::Api { result, .. } => *result,
            Self::BackendUnavailable { .. } => sys::OCGPU_ERROR_BACKEND_NOT_FOUND,
            Self::BackendTooOld { .. } | Self::BackendRuntimeTooOld { .. } => {
                sys::OCGPU_ERROR_BACKEND_TOO_OLD
            }
            Self::BackendAbiMismatch { .. } => sys::OCGPU_ERROR_ABI_MISMATCH,
            Self::MissingCoreSymbol { .. } => sys::OCGPU_ERROR_SYMBOL_UNAVAILABLE,
            Self::InvalidArgument(_) | Self::InvalidSize(_) => sys::OCGPU_ERROR_INVALID_ARGUMENT,
            Self::NullHandle(_) => sys::OCGPU_ERROR_INTERNAL,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BackendUnavailable { backend, detail } => {
                write!(formatter, "{backend} backend is unavailable: {detail}")
            }
            Self::BackendTooOld { backend, symbols } => write!(
                formatter,
                "{backend} backend is too old for required symbols: {}",
                symbols.join(", ")
            ),
            Self::BackendRuntimeTooOld { backend, detail } => {
                write!(formatter, "{backend} backend runtime is too old: {detail}")
            }
            Self::BackendAbiMismatch { backend, detail } => {
                write!(formatter, "{backend} backend ABI mismatch: {detail}")
            }
            Self::MissingCoreSymbol { backend, symbol } => {
                write!(
                    formatter,
                    "{backend} is missing required core symbol {symbol}"
                )
            }
            Self::Api { operation, result } => {
                write!(formatter, "{operation} failed with ocgpu result {result}")
            }
            Self::InvalidArgument(detail) => write!(formatter, "invalid argument: {detail}"),
            Self::NullHandle(operation) => {
                write!(
                    formatter,
                    "{operation} succeeded but returned a null handle"
                )
            }
            Self::InvalidSize(detail) => write!(formatter, "invalid size: {detail}"),
        }
    }
}

impl std::error::Error for Error {}

/// Result alias for the typed API.
pub type Result<T> = core::result::Result<T, Error>;

mod sealed {
    pub trait Sealed {}
}

/// A statically selected GPU backend.
///
/// This trait is sealed so downstream crates cannot create a backend whose raw
/// handles violate ocgpu's dispatch invariants.
pub trait Backend: sealed::Sealed + Sized + 'static {
    /// Validated backend-native dispatch table.
    type RawApi: 'static;

    /// Backend identity.
    const KIND: BackendKind;

    #[doc(hidden)]
    fn load_raw() -> Result<&'static Self::RawApi>;

    #[cfg(feature = "explicit-library-path")]
    #[doc(hidden)]
    unsafe fn load_raw_from_absolute(path: &Path) -> Result<&'static Self::RawApi>;

    #[doc(hidden)]
    fn common_table(api: &'static Self::RawApi) -> sys::ocgpuApi_v1;
}

/// CUDA backend marker.
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub enum Cuda {}

/// HIP backend marker.
#[cfg(feature = "hip")]
#[derive(Debug)]
pub enum Hip {}

#[cfg(feature = "cuda")]
impl sealed::Sealed for Cuda {}

#[cfg(feature = "hip")]
impl sealed::Sealed for Hip {}

#[cfg(feature = "cuda")]
impl Backend for Cuda {
    type RawApi = ocgpu_cuda::ValidatedCoreApi;

    const KIND: BackendKind = BackendKind::Cuda;

    fn load_raw() -> Result<&'static Self::RawApi> {
        ocgpu_cuda::load().map_err(map_cuda_error)
    }

    #[cfg(feature = "explicit-library-path")]
    unsafe fn load_raw_from_absolute(path: &Path) -> Result<&'static Self::RawApi> {
        // SAFETY: the hidden backend contract is exactly the public Driver
        // contract and is upheld by this method's caller.
        unsafe { ocgpu_cuda::load_from_absolute(path) }.map_err(map_cuda_error)
    }

    fn common_table(api: &'static Self::RawApi) -> sys::ocgpuApi_v1 {
        api.common_table()
    }
}

#[cfg(feature = "hip")]
impl Backend for Hip {
    type RawApi = ocgpu_hip::ValidatedCoreApi;

    const KIND: BackendKind = BackendKind::Hip;

    fn load_raw() -> Result<&'static Self::RawApi> {
        ocgpu_hip::load().map_err(map_hip_error)
    }

    #[cfg(feature = "explicit-library-path")]
    unsafe fn load_raw_from_absolute(path: &Path) -> Result<&'static Self::RawApi> {
        // SAFETY: the hidden backend contract is exactly the public Driver
        // contract and is upheld by this method's caller.
        unsafe { ocgpu_hip::load_from_absolute(path) }.map_err(map_hip_error)
    }

    fn common_table(api: &'static Self::RawApi) -> sys::ocgpuApi_v1 {
        api.common_table()
    }
}

#[cfg(feature = "hip")]
const fn map_hip_runtime_profile(profile: ocgpu_hip::RuntimeProfile) -> HipRuntimeProfile {
    match profile {
        ocgpu_hip::RuntimeProfile::Hip5 => HipRuntimeProfile::Hip5,
        ocgpu_hip::RuntimeProfile::Hip6 => HipRuntimeProfile::Hip6,
        ocgpu_hip::RuntimeProfile::Hip7 => HipRuntimeProfile::Hip7,
    }
}

#[cfg(feature = "cuda")]
fn map_cuda_error(error: ocgpu_cuda::Error) -> Error {
    match error {
        ocgpu_cuda::Error::BackendTooOld { symbols, .. } => Error::BackendTooOld {
            backend: BackendKind::Cuda,
            symbols,
        },
        ocgpu_cuda::Error::MissingCoreSymbols { symbols, .. } => Error::MissingCoreSymbol {
            backend: BackendKind::Cuda,
            symbol: symbols.first().copied().unwrap_or("unknown core symbol"),
        },
        ocgpu_cuda::Error::InvalidRawTableDescriptor { .. } => Error::Api {
            operation: "CUDA raw-table construction",
            result: sys::OCGPU_ERROR_INTERNAL,
        },
        error => Error::BackendUnavailable {
            backend: BackendKind::Cuda,
            detail: error.to_string(),
        },
    }
}

#[cfg(feature = "hip")]
fn map_hip_error(error: ocgpu_hip::Error) -> Error {
    match error {
        ocgpu_hip::Error::BackendTooOld { symbols, .. } => Error::BackendTooOld {
            backend: BackendKind::Hip,
            symbols,
        },
        ocgpu_hip::Error::MissingCoreSymbols { symbols, .. } => Error::MissingCoreSymbol {
            backend: BackendKind::Hip,
            symbol: symbols.first().copied().unwrap_or("unknown core symbol"),
        },
        ocgpu_hip::Error::InvalidRawTableDescriptor { .. } => Error::Api {
            operation: "HIP raw-table construction",
            result: sys::OCGPU_ERROR_INTERNAL,
        },
        error @ ocgpu_hip::Error::UnsupportedRuntimeVersion { .. }
            if error.below_minimum_runtime_version() =>
        {
            Error::BackendRuntimeTooOld {
                backend: BackendKind::Hip,
                detail: error.to_string(),
            }
        }
        error @ (ocgpu_hip::Error::UnsupportedRuntimeProfile { .. }
        | ocgpu_hip::Error::RuntimeProfileMismatch { .. }
        | ocgpu_hip::Error::UnsupportedRuntimeVersion { .. }) => Error::BackendAbiMismatch {
            backend: BackendKind::Hip,
            detail: error.to_string(),
        },
        error => Error::BackendUnavailable {
            backend: BackendKind::Hip,
            detail: error.to_string(),
        },
    }
}

#[derive(Clone, Copy)]
struct CoreFns {
    init: sys::ocgpuInitFn,
    driver_get_version: sys::ocgpuDriverGetVersionFn,
    device_get_count: sys::ocgpuDeviceGetCountFn,
    device_get: sys::ocgpuDeviceGetFn,
    device_get_name: sys::ocgpuDeviceGetNameFn,
    device_get_attribute: sys::ocgpuDeviceGetAttributeFn,
    ctx_create: sys::ocgpuCtxCreateFn,
    ctx_destroy: sys::ocgpuCtxDestroyFn,
    ctx_set_current: sys::ocgpuCtxSetCurrentFn,
    ctx_get_current: sys::ocgpuCtxGetCurrentFn,
    ctx_synchronize: sys::ocgpuCtxSynchronizeFn,
    mem_alloc: sys::ocgpuMemAllocFn,
    mem_free: sys::ocgpuMemFreeFn,
    memcpy_htod: sys::ocgpuMemcpyHtoDFn,
    memcpy_dtoh: sys::ocgpuMemcpyDtoHFn,
    stream_create: sys::ocgpuStreamCreateFn,
    stream_destroy: sys::ocgpuStreamDestroyFn,
    stream_synchronize: sys::ocgpuStreamSynchronizeFn,
    event_create: sys::ocgpuEventCreateFn,
    event_destroy: sys::ocgpuEventDestroyFn,
    event_record: sys::ocgpuEventRecordFn,
    event_synchronize: sys::ocgpuEventSynchronizeFn,
    module_load_data: sys::ocgpuModuleLoadDataFn,
    module_unload: sys::ocgpuModuleUnloadFn,
    module_get_function: sys::ocgpuModuleGetFunctionFn,
    launch_kernel: sys::ocgpuLaunchKernelFn,
}

impl CoreFns {
    fn validate(backend: BackendKind, table: &sys::ocgpuApi_v1) -> Result<Self> {
        macro_rules! required {
            ($field:ident) => {
                table.$field.ok_or(Error::MissingCoreSymbol {
                    backend,
                    symbol: stringify!($field),
                })?
            };
        }
        Ok(Self {
            init: required!(ocgpuInit),
            driver_get_version: required!(ocgpuDriverGetVersion),
            device_get_count: required!(ocgpuDeviceGetCount),
            device_get: required!(ocgpuDeviceGet),
            device_get_name: required!(ocgpuDeviceGetName),
            device_get_attribute: required!(ocgpuDeviceGetAttribute),
            ctx_create: required!(ocgpuCtxCreate),
            ctx_destroy: required!(ocgpuCtxDestroy),
            ctx_set_current: required!(ocgpuCtxSetCurrent),
            ctx_get_current: required!(ocgpuCtxGetCurrent),
            ctx_synchronize: required!(ocgpuCtxSynchronize),
            mem_alloc: required!(ocgpuMemAlloc),
            mem_free: required!(ocgpuMemFree),
            memcpy_htod: required!(ocgpuMemcpyHtoD),
            memcpy_dtoh: required!(ocgpuMemcpyDtoH),
            stream_create: required!(ocgpuStreamCreate),
            stream_destroy: required!(ocgpuStreamDestroy),
            stream_synchronize: required!(ocgpuStreamSynchronize),
            event_create: required!(ocgpuEventCreate),
            event_destroy: required!(ocgpuEventDestroy),
            event_record: required!(ocgpuEventRecord),
            event_synchronize: required!(ocgpuEventSynchronize),
            module_load_data: required!(ocgpuModuleLoadData),
            module_unload: required!(ocgpuModuleUnload),
            module_get_function: required!(ocgpuModuleGetFunction),
            launch_kernel: required!(ocgpuLaunchKernel),
        })
    }
}

/// Immutable, backend-bound driver instance.
pub struct Driver<B: Backend> {
    raw: &'static B::RawApi,
    core: CoreFns,
    metadata: ApiMetadata,
    backend: PhantomData<B>,
}

impl<B: Backend> fmt::Debug for Driver<B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Driver")
            .field("backend", &B::KIND)
            .field("metadata", &self.metadata)
            .finish_non_exhaustive()
    }
}

/// Immutable metadata copied from a negotiated table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ApiMetadata {
    /// Backend bound to the table.
    pub backend: BackendKind,
    /// Negotiated ABI version.
    pub abi_version: u32,
    /// Backend table flags. HIP tables encode their runtime profile in
    /// `OCGPU_API_FLAG_HIP_RUNTIME_PROFILE_MASK`; CUDA flags remain zero in
    /// ABI v1.
    pub flags: u32,
    /// Runtime-reported driver version captured while constructing the table.
    pub driver_version: i32,
    /// Byte size of the table producer's structure.
    pub struct_size: u32,
}

impl ApiMetadata {
    /// Decodes the HIP runtime profile advertised by this table.
    ///
    /// Non-HIP tables return `None` even if their independent flag bits happen
    /// to overlap this field in malformed or future metadata.
    #[must_use]
    pub const fn hip_runtime_profile(&self) -> Option<HipRuntimeProfile> {
        if matches!(self.backend, BackendKind::Hip) {
            HipRuntimeProfile::from_api_flags(self.flags)
        } else {
            None
        }
    }
}

/// Runtime symbol-resolution route.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SymbolResolution {
    /// Version-aware vendor proc-address API.
    ProcAddress,
    /// Direct operating-system symbol lookup.
    Direct,
    /// A legacy direct symbol feeds a typed common-profile adapter while the
    /// current raw-table slot remains unavailable.
    DirectAdapter,
    /// No applicable entry point was found.
    Missing,
    /// The symbol belongs to the compiled inventory but its ABI is not
    /// available in the selected runtime-major profile.
    ProfileUnavailable,
    /// The canonical inventory marks this symbol unavailable on this platform.
    PlatformUnavailable,
}

/// Vendor proc-address bootstrap ABI selected by a backend loader.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProcAddressVariant {
    /// CUDA five-argument `cuGetProcAddress_v2` ABI with query status.
    CudaV2,
    /// CUDA four-argument legacy `cuGetProcAddress` ABI.
    CudaLegacy,
    /// HIP `hipGetProcAddress` ABI.
    Hip,
}

impl ProcAddressVariant {
    /// Stable spelling used by diagnostics and serialized CLI output.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CudaV2 => "cuda_v2",
            Self::CudaLegacy => "cuda_legacy",
            Self::Hip => "hip",
        }
    }
}

/// Backend-neutral status for one runtime symbol.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeSymbolStatus {
    /// Canonical vendor name.
    pub name: &'static str,
    /// Alias or export name that resolved, when available.
    pub resolved_name: Option<&'static str>,
    /// Successful lookup route or missing state.
    pub resolution: SymbolResolution,
    /// Number of version-aware lookup attempts made.
    pub proc_attempts: usize,
    /// Whether the validated common profile requires this symbol.
    pub required: bool,
    /// Whether this inventory entry applies to the current target platform.
    pub applicable: bool,
}

impl RuntimeSymbolStatus {
    /// Whether this exact raw-inventory symbol can be called on the loaded runtime.
    ///
    /// `DirectAdapter` is deliberately false: the typed common operation is
    /// callable, but the differently typed raw-table slot remains null.
    #[must_use]
    pub const fn available(&self) -> bool {
        matches!(
            self.resolution,
            SymbolResolution::ProcAddress | SymbolResolution::Direct
        )
    }
}

/// Cold-path loader and symbol diagnostics independent of a backend crate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BackendDiagnostics {
    /// Backend being described.
    pub backend: BackendKind,
    /// Loaded shared-library path or loader identity.
    pub library_path: PathBuf,
    /// Runtime version used for proc-address negotiation, when reported.
    pub runtime_version: Option<i32>,
    /// Driver version, when separately reported.
    pub driver_version: Option<i32>,
    /// Compile-time exhaustive vendor inventory baseline for this target.
    ///
    /// This remains the current HIP 7 inventory denominator even when an older
    /// profile/version supplies only the common subset; use
    /// `hip_runtime_profile` and per-symbol resolution to determine capabilities.
    pub compiled_api_version: i32,
    /// Selected HIP runtime ABI family, or `None` for non-HIP backends.
    pub hip_runtime_profile: Option<HipRuntimeProfile>,
    /// Whether proc-address resolution is enabled for the selected runtime
    /// profile.
    pub proc_address_support: bool,
    /// Exact bootstrap ABI selected when proc-address resolution is enabled.
    pub proc_address_variant: Option<ProcAddressVariant>,
    /// Architecture of the successfully loaded in-process library.
    pub loaded_architecture: &'static str,
    /// Resolution results for every inventory symbol, including entries that
    /// were not attempted because they do not apply to the current platform.
    pub symbols: Vec<RuntimeSymbolStatus>,
}

impl BackendDiagnostics {
    /// Unavailable applicable symbols in the runtime-resolved inventory,
    /// including deliberate selected-profile omissions.
    pub fn missing_symbols(&self) -> impl Iterator<Item = &RuntimeSymbolStatus> {
        self.symbols
            .iter()
            .filter(|symbol| symbol.applicable && !symbol.available())
    }

    /// Missing symbols required by the validated common profile.
    pub fn missing_required_symbols(&self) -> impl Iterator<Item = &RuntimeSymbolStatus> {
        self.missing_symbols().filter(|symbol| {
            symbol.required && !matches!(symbol.resolution, SymbolResolution::DirectAdapter)
        })
    }

    /// Genuinely missing symbols that affect optional/raw capabilities only.
    ///
    /// Deliberate selected-profile omissions are reported separately by
    /// [`Self::profile_omissions`].
    pub fn missing_optional_symbols(&self) -> impl Iterator<Item = &RuntimeSymbolStatus> {
        self.missing_symbols().filter(|symbol| {
            !symbol.required
                && !matches!(
                    symbol.resolution,
                    SymbolResolution::ProfileUnavailable | SymbolResolution::DirectAdapter
                )
        })
    }

    /// Inventory entries deliberately unavailable on this target platform.
    pub fn platform_omissions(&self) -> impl Iterator<Item = &RuntimeSymbolStatus> {
        self.symbols.iter().filter(|symbol| !symbol.applicable)
    }

    /// Applicable inventory entries intentionally unavailable for the
    /// selected runtime-major profile.
    pub fn profile_omissions(&self) -> impl Iterator<Item = &RuntimeSymbolStatus> {
        self.symbols.iter().filter(|symbol| {
            symbol.applicable
                && matches!(
                    symbol.resolution,
                    SymbolResolution::ProfileUnavailable | SymbolResolution::DirectAdapter
                )
        })
    }
}

/// Loads a backend's optional/raw inventory and returns cold-path diagnostics.
pub fn backend_diagnostics(backend: BackendKind) -> Result<BackendDiagnostics> {
    match backend {
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => {
            let diagnostics =
                ocgpu_cuda::diagnostics().map_err(|error| Error::BackendUnavailable {
                    backend,
                    detail: error.to_string(),
                })?;
            Ok(BackendDiagnostics {
                backend,
                library_path: diagnostics.library_path.clone(),
                runtime_version: diagnostics.runtime_version,
                driver_version: diagnostics.runtime_version,
                compiled_api_version: ocgpu_cuda::COMPILED_API_VERSION,
                hip_runtime_profile: None,
                proc_address_support: diagnostics.proc_address_support,
                proc_address_variant: diagnostics.proc_address_variant.map(
                    |variant| match variant {
                        ocgpu_cuda::ProcAddressVariant::V2 => ProcAddressVariant::CudaV2,
                        ocgpu_cuda::ProcAddressVariant::Legacy => ProcAddressVariant::CudaLegacy,
                    },
                ),
                loaded_architecture: diagnostics.loaded_architecture,
                symbols: diagnostics
                    .symbols
                    .iter()
                    .map(|symbol| RuntimeSymbolStatus {
                        name: symbol.canonical_name,
                        resolved_name: symbol.resolved_name,
                        resolution: match symbol.resolution {
                            ocgpu_cuda::ResolutionKind::ProcAddress => {
                                SymbolResolution::ProcAddress
                            }
                            ocgpu_cuda::ResolutionKind::Direct => SymbolResolution::Direct,
                            ocgpu_cuda::ResolutionKind::Missing => SymbolResolution::Missing,
                            ocgpu_cuda::ResolutionKind::PlatformUnavailable => {
                                SymbolResolution::PlatformUnavailable
                            }
                        },
                        proc_attempts: symbol.proc_attempts.len(),
                        required: symbol.required,
                        applicable: symbol.applicable,
                    })
                    .collect(),
            })
        }
        #[cfg(not(feature = "cuda"))]
        BackendKind::Cuda => Err(feature_disabled(backend)),
        #[cfg(feature = "hip")]
        BackendKind::Hip => {
            let diagnostics = ocgpu_hip::diagnostics().map_err(map_hip_error)?;
            Ok(BackendDiagnostics {
                backend,
                library_path: diagnostics.library_path.clone(),
                runtime_version: diagnostics.runtime_version,
                driver_version: diagnostics.driver_version,
                compiled_api_version: ocgpu_hip::COMPILED_API_VERSION,
                hip_runtime_profile: Some(map_hip_runtime_profile(diagnostics.runtime_profile)),
                proc_address_support: diagnostics.proc_address_support,
                proc_address_variant: diagnostics
                    .proc_address_support
                    .then_some(ProcAddressVariant::Hip),
                loaded_architecture: diagnostics.loaded_architecture,
                symbols: diagnostics
                    .symbols
                    .iter()
                    .map(|symbol| RuntimeSymbolStatus {
                        name: symbol.canonical_name,
                        resolved_name: symbol.resolved_name,
                        resolution: match symbol.resolution {
                            ocgpu_hip::ResolutionKind::ProcAddress => SymbolResolution::ProcAddress,
                            ocgpu_hip::ResolutionKind::Direct => SymbolResolution::Direct,
                            ocgpu_hip::ResolutionKind::DirectAdapter => {
                                SymbolResolution::DirectAdapter
                            }
                            ocgpu_hip::ResolutionKind::Missing => SymbolResolution::Missing,
                            ocgpu_hip::ResolutionKind::ProfileUnavailable => {
                                SymbolResolution::ProfileUnavailable
                            }
                            ocgpu_hip::ResolutionKind::PlatformUnavailable => {
                                SymbolResolution::PlatformUnavailable
                            }
                        },
                        proc_attempts: symbol.proc_attempts.len(),
                        required: symbol.required,
                        applicable: symbol.applicable,
                    })
                    .collect(),
            })
        }
        #[cfg(not(feature = "hip"))]
        BackendKind::Hip => Err(feature_disabled(backend)),
    }
}

impl<B: Backend> Driver<B> {
    /// Loads and initializes one statically typed backend.
    pub fn load() -> Result<Self> {
        Self::from_raw(B::load_raw()?)
    }

    /// Loads a backend from a caller-selected absolute shared-library path.
    ///
    /// The backend's process-wide initialization cell is shared with [`load`](Self::load):
    /// whichever attempt happens first fixes the backend result for the process
    /// lifetime. Relative paths are rejected by the OS-loader layer.
    ///
    /// # Safety
    ///
    /// The selected library and its dependency closure must be trusted, may
    /// run only sound initialization code, and must implement the exact vendor
    /// ABI represented by `B`. The file must not be replaced incompatibly
    /// between path validation and operating-system loading.
    #[cfg(feature = "explicit-library-path")]
    pub unsafe fn load_from_absolute(path: &Path) -> Result<Self> {
        // SAFETY: the caller establishes the trust, constructor, ABI, and file
        // identity requirements documented above.
        Self::from_raw(unsafe { B::load_raw_from_absolute(path) }?)
    }

    fn from_raw(raw: &'static B::RawApi) -> Result<Self> {
        let table = B::common_table(raw);
        if table.abi_version != sys::OCGPU_ABI_VERSION_1 {
            return Err(Error::Api {
                operation: "table ABI negotiation",
                result: sys::OCGPU_ERROR_ABI_MISMATCH,
            });
        }
        if table.backend != B::KIND.as_raw() {
            return Err(Error::Api {
                operation: "table backend validation",
                result: sys::OCGPU_ERROR_INTERNAL,
            });
        }
        let core = CoreFns::validate(B::KIND, &table)?;
        // SAFETY: validation obtained the correctly typed non-null entry.
        check("ocgpuInit", unsafe { (core.init)(0) })?;
        Ok(Self {
            raw,
            core,
            metadata: ApiMetadata {
                backend: B::KIND,
                abi_version: table.abi_version,
                flags: table.flags,
                driver_version: table.driver_version,
                struct_size: table.struct_size,
            },
            backend: PhantomData,
        })
    }

    /// Returns the validated backend-native table.
    #[must_use]
    pub const fn raw_api(&self) -> &'static B::RawApi {
        self.raw
    }

    /// Returns negotiated immutable table metadata.
    #[must_use]
    pub const fn metadata(&self) -> ApiMetadata {
        self.metadata
    }

    /// Queries the runtime-reported driver version.
    pub fn driver_version(&self) -> Result<i32> {
        let mut version = 0_i32;
        // SAFETY: output points to initialized writable storage.
        check("ocgpuDriverGetVersion", unsafe {
            (self.core.driver_get_version)(&raw mut version)
        })?;
        Ok(version)
    }

    /// Returns the number of devices visible to this backend.
    pub fn device_count(&self) -> Result<usize> {
        let mut count = 0_i32;
        // SAFETY: output points to initialized writable storage.
        check("ocgpuDeviceGetCount", unsafe {
            (self.core.device_get_count)(&raw mut count)
        })?;
        usize::try_from(count).map_err(|_| Error::InvalidSize("negative device count"))
    }

    /// Obtains one backend-typed device by ordinal.
    pub fn device(&self, ordinal: usize) -> Result<Device<'_, B>> {
        let ordinal = i32::try_from(ordinal)
            .map_err(|_| Error::InvalidArgument("device ordinal exceeds i32"))?;
        let mut raw = 0;
        // SAFETY: output is writable and ordinal representation was checked.
        check("ocgpuDeviceGet", unsafe {
            (self.core.device_get)(&raw mut raw, ordinal)
        })?;
        Ok(Device { driver: self, raw })
    }

    /// Iterates over all currently visible devices.
    pub fn devices(&self) -> Result<DeviceIter<'_, B>> {
        Ok(DeviceIter {
            driver: self,
            next: 0,
            count: self.device_count()?,
        })
    }

    /// Returns a non-owning view of the backend's current context on this
    /// calling thread.
    ///
    /// # Safety
    ///
    /// When a non-null context is returned, the caller must ensure that its
    /// external owner does not destroy it while the returned view or any copy
    /// of that view is used. The vendor API exposes no ownership information
    /// from which ocgpu could prove that lifetime.
    pub unsafe fn current_context(&self) -> Result<Option<BorrowedContext<'_, B>>> {
        let mut raw = ptr::null_mut();
        // SAFETY: output points to writable handle storage.
        check("ocgpuCtxGetCurrent", unsafe {
            (self.core.ctx_get_current)(&raw mut raw)
        })?;
        Ok((!raw.is_null()).then_some(BorrowedContext { driver: self, raw }))
    }
}

#[cfg(feature = "hip")]
impl Driver<Hip> {
    /// Returns the HIP runtime ABI family selected and validated while loading
    /// this driver.
    ///
    /// Applications can use this value for coarse capability policy. Raw-table
    /// entries must still be checked individually because not every operation
    /// belongs to every runtime-major profile.
    #[must_use]
    pub const fn runtime_profile(&self) -> HipRuntimeProfile {
        map_hip_runtime_profile(self.raw.runtime_profile())
    }
}

/// Iterator over backend-typed devices.
pub struct DeviceIter<'driver, B: Backend> {
    driver: &'driver Driver<B>,
    next: usize,
    count: usize,
}

impl<'driver, B: Backend> Iterator for DeviceIter<'driver, B> {
    type Item = Result<Device<'driver, B>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next == self.count {
            return None;
        }
        let ordinal = self.next;
        self.next += 1;
        Some(self.driver.device(ordinal))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.count - self.next;
        (remaining, Some(remaining))
    }
}

impl<B: Backend> ExactSizeIterator for DeviceIter<'_, B> {}

/// A device handle tied to its originating backend table.
#[derive(Clone, Copy)]
pub struct Device<'driver, B: Backend> {
    driver: &'driver Driver<B>,
    raw: sys::ocgpuDevice,
}

impl<B: Backend> fmt::Debug for Device<'_, B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Device")
            .field("backend", &B::KIND)
            .field("raw", &self.raw)
            .finish()
    }
}

impl<'driver, B: Backend> Device<'driver, B> {
    /// Backend-native device handle.
    #[must_use]
    pub const fn raw(&self) -> sys::ocgpuDevice {
        self.raw
    }

    /// Human-readable device name, lossily converted if a driver returns
    /// non-UTF-8 bytes.
    pub fn name(&self) -> Result<String> {
        const CAPACITY: usize = 256;
        const CAPACITY_I32: i32 = 256;
        let mut buffer = [0 as c_char; CAPACITY];
        // SAFETY: buffer is writable for `capacity` bytes and handle/table match.
        check("ocgpuDeviceGetName", unsafe {
            (self.driver.core.device_get_name)(buffer.as_mut_ptr(), CAPACITY_I32, self.raw)
        })?;
        buffer[CAPACITY - 1] = 0;
        // SAFETY: the last byte is explicitly NUL and the buffer is live.
        let name = unsafe { CStr::from_ptr(buffer.as_ptr()) };
        Ok(name.to_string_lossy().into_owned())
    }

    /// Queries one of the stable `OCGPU_DEVICE_ATTRIBUTE_*` attributes.
    ///
    /// Backend-native attributes remain available through [`raw`]; the unified
    /// surface rejects integers outside its documented semantic intersection.
    pub fn attribute(&self, attribute: sys::ocgpuDeviceAttribute) -> Result<i32> {
        let mut value = 0_i32;
        // SAFETY: output is writable and handle/table share a backend.
        check("ocgpuDeviceGetAttribute", unsafe {
            (self.driver.core.device_get_attribute)(&raw mut value, attribute, self.raw)
        })?;
        Ok(value)
    }

    /// Creates an owned context for this device.
    pub fn create_context(&self, flags: u32) -> Result<Context<'driver, B>> {
        let mut raw = ptr::null_mut();
        // SAFETY: output is writable and device/table share a backend.
        check("ocgpuCtxCreate", unsafe {
            (self.driver.core.ctx_create)(&raw mut raw, flags, self.raw)
        })?;
        if raw.is_null() {
            return Err(Error::NullHandle("ocgpuCtxCreate"));
        }
        Ok(Context {
            driver: self.driver,
            raw,
        })
    }
}

/// A non-owning current-context view.
#[derive(Clone, Copy)]
pub struct BorrowedContext<'driver, B: Backend> {
    driver: &'driver Driver<B>,
    raw: sys::ocgpuContext,
}

impl<B: Backend> BorrowedContext<'_, B> {
    /// Backend-native context handle.
    #[must_use]
    pub const fn raw(&self) -> sys::ocgpuContext {
        self.raw
    }

    /// Makes this context current on the calling thread.
    pub fn make_current(&self) -> Result<()> {
        // SAFETY: this borrowed handle originated from the same validated table.
        check("ocgpuCtxSetCurrent", unsafe {
            (self.driver.core.ctx_set_current)(self.raw)
        })
    }
}

/// Owned backend-typed context.
pub struct Context<'driver, B: Backend> {
    driver: &'driver Driver<B>,
    raw: sys::ocgpuContext,
}

impl<B: Backend> fmt::Debug for Context<'_, B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Context")
            .field("backend", &B::KIND)
            .field("raw", &self.raw)
            .finish()
    }
}

impl<'driver, B: Backend> Context<'driver, B> {
    /// Backend-native context handle.
    #[must_use]
    pub const fn raw(&self) -> sys::ocgpuContext {
        self.raw
    }

    /// Makes this context current on the calling thread.
    pub fn make_current(&self) -> Result<()> {
        // SAFETY: the owned handle originated from the same validated table.
        check("ocgpuCtxSetCurrent", unsafe {
            (self.driver.core.ctx_set_current)(self.raw)
        })
    }

    /// Waits for all preceding work in this context.
    pub fn synchronize(&self) -> Result<()> {
        self.make_current()?;
        // SAFETY: this context is current and the entry was validated.
        check("ocgpuCtxSynchronize", unsafe {
            (self.driver.core.ctx_synchronize)()
        })
    }

    /// Allocates device memory owned by this context.
    pub fn allocate(&self, bytes: usize) -> Result<DeviceMemory<'_, 'driver, B>> {
        if bytes == 0 {
            return Err(Error::InvalidArgument("allocation size must be nonzero"));
        }
        self.make_current()?;
        let mut raw = 0;
        // SAFETY: output is writable, size is nonzero, and context is current.
        check("ocgpuMemAlloc", unsafe {
            (self.driver.core.mem_alloc)(&raw mut raw, bytes)
        })?;
        if raw == 0 {
            return Err(Error::NullHandle("ocgpuMemAlloc"));
        }
        Ok(DeviceMemory {
            context: self,
            raw,
            bytes,
        })
    }

    /// Creates a stream owned by this context.
    pub fn create_stream(&self, flags: u32) -> Result<Stream<'_, 'driver, B>> {
        self.make_current()?;
        let mut raw = ptr::null_mut();
        // SAFETY: output is writable and context is current.
        check("ocgpuStreamCreate", unsafe {
            (self.driver.core.stream_create)(&raw mut raw, flags)
        })?;
        if raw.is_null() {
            return Err(Error::NullHandle("ocgpuStreamCreate"));
        }
        Ok(Stream { context: self, raw })
    }

    /// Creates an event owned by this context.
    pub fn create_event(&self, flags: u32) -> Result<Event<'_, 'driver, B>> {
        self.make_current()?;
        let mut raw = ptr::null_mut();
        // SAFETY: output is writable and context is current.
        check("ocgpuEventCreate", unsafe {
            (self.driver.core.event_create)(&raw mut raw, flags)
        })?;
        if raw.is_null() {
            return Err(Error::NullHandle("ocgpuEventCreate"));
        }
        Ok(Event { context: self, raw })
    }

    /// Loads a vendor module image.
    ///
    /// # Safety
    ///
    /// `image` must point to a complete image accepted by the selected backend
    /// (for example, a NUL-terminated PTX string or a valid cubin/fatbin/HSACO)
    /// and must remain readable for the duration required by that vendor API.
    pub unsafe fn load_module_data(&self, image: *const c_void) -> Result<Module<'_, 'driver, B>> {
        if image.is_null() {
            return Err(Error::InvalidArgument("module image pointer is null"));
        }
        self.make_current()?;
        let mut raw = ptr::null_mut();
        // SAFETY: caller establishes the image contract; output is writable.
        check("ocgpuModuleLoadData", unsafe {
            (self.driver.core.module_load_data)(&raw mut raw, image)
        })?;
        if raw.is_null() {
            return Err(Error::NullHandle("ocgpuModuleLoadData"));
        }
        Ok(Module { context: self, raw })
    }

    /// Loads a NUL-terminated textual module image, such as PTX, without
    /// copying it.
    ///
    /// Binary containers can contain interior NUL bytes and must instead use
    /// [`Self::load_module_data`] with the vendor-specific image contract.
    ///
    /// # Safety
    ///
    /// `image` must contain a complete textual module image accepted by the
    /// selected backend. A [`CStr`] establishes readable NUL-terminated storage
    /// but does not validate the module language or its completeness.
    pub unsafe fn load_module_cstr(&self, image: &CStr) -> Result<Module<'_, 'driver, B>> {
        // SAFETY: the caller establishes that the readable NUL-terminated
        // bytes form a complete textual image accepted by the backend.
        unsafe { self.load_module_data(image.as_ptr().cast()) }
    }
}

impl<B: Backend> Drop for Context<'_, B> {
    fn drop(&mut self) {
        // SAFETY: this type uniquely owns a non-null context from this table.
        let _ = unsafe { (self.driver.core.ctx_destroy)(self.raw) };
    }
}

/// Owned device allocation tied to its context and backend.
pub struct DeviceMemory<'context, 'driver, B: Backend> {
    context: &'context Context<'driver, B>,
    raw: sys::ocgpuDeviceptr,
    bytes: usize,
}

impl<B: Backend> fmt::Debug for DeviceMemory<'_, '_, B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DeviceMemory")
            .field("backend", &B::KIND)
            .field("raw", &self.raw)
            .field("bytes", &self.bytes)
            .finish()
    }
}

impl<B: Backend> DeviceMemory<'_, '_, B> {
    /// Backend-native device pointer.
    #[must_use]
    pub const fn raw(&self) -> sys::ocgpuDeviceptr {
        self.raw
    }

    /// Allocation size in bytes.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.bytes
    }

    /// Whether the allocation has zero length. Allocations created by the safe
    /// API are always nonempty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    /// Copies an entire byte slice into this allocation at offset zero.
    pub fn copy_from(&self, source: &[u8]) -> Result<()> {
        if source.len() > self.bytes {
            return Err(Error::InvalidArgument(
                "host source exceeds device allocation",
            ));
        }
        if source.is_empty() {
            return Ok(());
        }
        self.context.make_current()?;
        // SAFETY: the source is readable, destination owns at least this many
        // bytes, and the allocation's context is current.
        check("ocgpuMemcpyHtoD", unsafe {
            (self.context.driver.core.memcpy_htod)(self.raw, source.as_ptr().cast(), source.len())
        })
    }

    /// Copies bytes from this allocation at offset zero into a host slice.
    pub fn copy_to(&self, destination: &mut [u8]) -> Result<()> {
        if destination.len() > self.bytes {
            return Err(Error::InvalidArgument(
                "host destination exceeds device allocation",
            ));
        }
        if destination.is_empty() {
            return Ok(());
        }
        self.context.make_current()?;
        // SAFETY: destination is writable, source owns at least this many bytes,
        // and the allocation's context is current.
        check("ocgpuMemcpyDtoH", unsafe {
            (self.context.driver.core.memcpy_dtoh)(
                destination.as_mut_ptr().cast(),
                self.raw,
                destination.len(),
            )
        })
    }
}

impl<B: Backend> Drop for DeviceMemory<'_, '_, B> {
    fn drop(&mut self) {
        if self.context.make_current().is_ok() {
            // SAFETY: this type uniquely owns the allocation from this table,
            // and its context was made current successfully. If restoring the
            // context fails, leaking is safer than freeing through an unrelated
            // current context.
            let _ = unsafe { (self.context.driver.core.mem_free)(self.raw) };
        }
    }
}

/// Owned stream tied to its context and backend.
pub struct Stream<'context, 'driver, B: Backend> {
    context: &'context Context<'driver, B>,
    raw: sys::ocgpuStream,
}

impl<B: Backend> Stream<'_, '_, B> {
    /// Backend-native stream handle.
    #[must_use]
    pub const fn raw(&self) -> sys::ocgpuStream {
        self.raw
    }

    /// Waits for all preceding work in this stream.
    pub fn synchronize(&self) -> Result<()> {
        self.context.make_current()?;
        // SAFETY: stream and table share a backend and live context.
        check("ocgpuStreamSynchronize", unsafe {
            (self.context.driver.core.stream_synchronize)(self.raw)
        })
    }
}

impl<B: Backend> Drop for Stream<'_, '_, B> {
    fn drop(&mut self) {
        if self.context.make_current().is_ok() {
            // SAFETY: this type uniquely owns the stream from this table and
            // its context is current.
            let _ = unsafe { (self.context.driver.core.stream_destroy)(self.raw) };
        }
    }
}

/// Owned event tied to its context and backend.
pub struct Event<'context, 'driver, B: Backend> {
    context: &'context Context<'driver, B>,
    raw: sys::ocgpuEvent,
}

impl<'context, 'driver, B: Backend> Event<'context, 'driver, B> {
    /// Backend-native event handle.
    #[must_use]
    pub const fn raw(&self) -> sys::ocgpuEvent {
        self.raw
    }

    /// Records the event on a stream from the same context.
    pub fn record(&self, stream: &Stream<'context, 'driver, B>) -> Result<()> {
        if !ptr::eq(self.context, stream.context) {
            return Err(Error::InvalidArgument(
                "event and stream belong to different contexts",
            ));
        }
        self.context.make_current()?;
        // SAFETY: handles are live and context identity was checked.
        check("ocgpuEventRecord", unsafe {
            (self.context.driver.core.event_record)(self.raw, stream.raw)
        })
    }

    /// Waits until the event has completed.
    pub fn synchronize(&self) -> Result<()> {
        self.context.make_current()?;
        // SAFETY: event and table share a backend and live context.
        check("ocgpuEventSynchronize", unsafe {
            (self.context.driver.core.event_synchronize)(self.raw)
        })
    }
}

impl<B: Backend> Drop for Event<'_, '_, B> {
    fn drop(&mut self) {
        if self.context.make_current().is_ok() {
            // SAFETY: this type uniquely owns the event from this table and
            // its context is current.
            let _ = unsafe { (self.context.driver.core.event_destroy)(self.raw) };
        }
    }
}

/// Owned loaded module tied to its context and backend.
pub struct Module<'context, 'driver, B: Backend> {
    context: &'context Context<'driver, B>,
    raw: sys::ocgpuModule,
}

impl<'context, 'driver, B: Backend> Module<'context, 'driver, B> {
    /// Backend-native module handle.
    #[must_use]
    pub const fn raw(&self) -> sys::ocgpuModule {
        self.raw
    }

    /// Resolves a NUL-terminated function name in this module.
    pub fn function<'module>(
        &'module self,
        name: &CStr,
    ) -> Result<Function<'module, 'context, 'driver, B>> {
        self.context.make_current()?;
        let mut raw = ptr::null_mut();
        // SAFETY: name is NUL-terminated, output is writable, module is live.
        check("ocgpuModuleGetFunction", unsafe {
            (self.context.driver.core.module_get_function)(&raw mut raw, self.raw, name.as_ptr())
        })?;
        if raw.is_null() {
            return Err(Error::NullHandle("ocgpuModuleGetFunction"));
        }
        Ok(Function { module: self, raw })
    }
}

impl<B: Backend> Drop for Module<'_, '_, B> {
    fn drop(&mut self) {
        if self.context.make_current().is_ok() {
            // SAFETY: this type uniquely owns the module from this table and
            // its context is current.
            let _ = unsafe { (self.context.driver.core.module_unload)(self.raw) };
        }
    }
}

/// Kernel launch dimensions and dynamic shared-memory size.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LaunchConfig {
    /// Grid dimensions in blocks.
    pub grid: [u32; 3],
    /// Block dimensions in threads.
    pub block: [u32; 3],
    /// Dynamic shared-memory bytes per block.
    pub shared_memory_bytes: u32,
}

impl LaunchConfig {
    /// Validates nonzero launch dimensions.
    pub fn new(grid: [u32; 3], block: [u32; 3], shared_memory_bytes: u32) -> Result<Self> {
        if grid.contains(&0) || block.contains(&0) {
            return Err(Error::InvalidArgument(
                "grid and block dimensions must be nonzero",
            ));
        }
        Ok(Self {
            grid,
            block,
            shared_memory_bytes,
        })
    }
}

/// Non-owning function handle tied to a live module.
pub struct Function<'module, 'context, 'driver, B: Backend> {
    module: &'module Module<'context, 'driver, B>,
    raw: sys::ocgpuFunction,
}

impl<B: Backend> Function<'_, '_, '_, B> {
    /// Backend-native function handle.
    #[must_use]
    pub const fn raw(&self) -> sys::ocgpuFunction {
        self.raw
    }

    /// Launches this function using the vendor parameter-pointer convention.
    ///
    /// # Safety
    ///
    /// Every entry in `kernel_parameters` must point to readable storage of the
    /// exact type, size, and alignment expected by the device function. The
    /// pointed-to values must remain live until the backend has copied them.
    pub unsafe fn launch(
        &self,
        config: LaunchConfig,
        stream: Option<&Stream<'_, '_, B>>,
        kernel_parameters: &mut [*mut c_void],
    ) -> Result<()> {
        if config.grid.contains(&0) || config.block.contains(&0) {
            return Err(Error::InvalidArgument(
                "grid and block dimensions must be nonzero",
            ));
        }
        if let Some(stream) = stream {
            if !ptr::eq(self.module.context, stream.context) {
                return Err(Error::InvalidArgument(
                    "function and stream belong to different contexts",
                ));
            }
        }
        self.module.context.make_current()?;
        let stream = stream.map_or(ptr::null_mut(), Stream::raw);
        let parameters = if kernel_parameters.is_empty() {
            ptr::null_mut()
        } else {
            kernel_parameters.as_mut_ptr()
        };
        // SAFETY: caller establishes parameter ABI validity; handles share the
        // checked context and all dimensions were validated above.
        check("ocgpuLaunchKernel", unsafe {
            (self.module.context.driver.core.launch_kernel)(
                self.raw,
                config.grid[0],
                config.grid[1],
                config.grid[2],
                config.block[0],
                config.block[1],
                config.block[2],
                config.shared_memory_bytes,
                stream,
                parameters,
                ptr::null_mut(),
            )
        })
    }
}

/// Dynamically selected driver with explicit enum dispatch.
#[cfg(any(feature = "cuda", feature = "hip"))]
pub enum AnyDriver {
    /// CUDA driver.
    #[cfg(feature = "cuda")]
    Cuda(Driver<Cuda>),
    /// HIP driver.
    #[cfg(feature = "hip")]
    Hip(Driver<Hip>),
}

/// Loads a backend and returns its negotiated common C table.
///
/// This is primarily used by `ocgpu-capi`; Rust applications normally use
/// [`Driver`] so required entries are validated into non-null hot-path fields.
#[doc(hidden)]
pub fn negotiated_common_table(backend: BackendKind) -> Result<sys::ocgpuApi_v1> {
    match backend {
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => {
            let raw = Cuda::load_raw()?;
            Ok(Cuda::common_table(raw))
        }
        #[cfg(not(feature = "cuda"))]
        BackendKind::Cuda => Err(feature_disabled(backend)),
        #[cfg(feature = "hip")]
        BackendKind::Hip => {
            let raw = Hip::load_raw()?;
            Ok(Hip::common_table(raw))
        }
        #[cfg(not(feature = "hip"))]
        BackendKind::Hip => Err(feature_disabled(backend)),
    }
}

/// Loads CUDA and returns the raw versioned C table, retaining null optional
/// entries exactly as negotiated with the installed driver.
#[cfg(feature = "cuda")]
#[doc(hidden)]
pub fn negotiated_cuda_table() -> Result<sys::ocgpuCuApi_v1> {
    ocgpu_cuda::load_unvalidated()
        .map(ocgpu_cuda::UnvalidatedApi::raw_table)
        .map_err(map_cuda_error)
}

/// Loads HIP and returns the raw versioned C table, retaining null optional
/// entries exactly as negotiated with the installed runtime.
#[cfg(feature = "hip")]
#[doc(hidden)]
pub fn negotiated_hip_table() -> Result<sys::ocgpuHipApi_v1> {
    ocgpu_hip::load_unvalidated()
        .map(ocgpu_hip::UnvalidatedApi::raw_table)
        .map_err(map_hip_error)
}

#[cfg(any(feature = "cuda", feature = "hip"))]
impl AnyDriver {
    /// Loads the requested backend.
    pub fn load(backend: BackendKind) -> Result<Self> {
        match backend {
            #[cfg(feature = "cuda")]
            BackendKind::Cuda => Driver::<Cuda>::load().map(Self::Cuda),
            #[cfg(not(feature = "cuda"))]
            BackendKind::Cuda => Err(feature_disabled(backend)),
            #[cfg(feature = "hip")]
            BackendKind::Hip => Driver::<Hip>::load().map(Self::Hip),
            #[cfg(not(feature = "hip"))]
            BackendKind::Hip => Err(feature_disabled(backend)),
        }
    }

    /// Bound backend identity.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => BackendKind::Cuda,
            #[cfg(feature = "hip")]
            Self::Hip(_) => BackendKind::Hip,
        }
    }

    /// Runtime-reported driver version.
    pub fn driver_version(&self) -> Result<i32> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(driver) => driver.driver_version(),
            #[cfg(feature = "hip")]
            Self::Hip(driver) => driver.driver_version(),
        }
    }

    /// Number of visible devices.
    pub fn device_count(&self) -> Result<usize> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(driver) => driver.device_count(),
            #[cfg(feature = "hip")]
            Self::Hip(driver) => driver.device_count(),
        }
    }

    /// Owned summaries for all visible devices.
    pub fn device_summaries(&self) -> Result<Vec<DeviceSummary>> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(driver) => summaries(driver),
            #[cfg(feature = "hip")]
            Self::Hip(driver) => summaries(driver),
        }
    }

    /// Negotiated table metadata.
    #[must_use]
    pub const fn metadata(&self) -> ApiMetadata {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(driver) => driver.metadata(),
            #[cfg(feature = "hip")]
            Self::Hip(driver) => driver.metadata(),
        }
    }

    /// Cold-path loader and symbol-resolution diagnostics.
    pub fn diagnostics(&self) -> Result<BackendDiagnostics> {
        backend_diagnostics(self.backend())
    }
}

/// Owned device data suitable for diagnostics and serialization.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceSummary {
    /// Backend identity.
    pub backend: BackendKind,
    /// Enumeration ordinal.
    pub ordinal: usize,
    /// Driver-provided device name.
    pub name: String,
}

#[cfg(any(feature = "cuda", feature = "hip"))]
fn summaries<B: Backend>(driver: &Driver<B>) -> Result<Vec<DeviceSummary>> {
    driver
        .devices()?
        .enumerate()
        .map(|(ordinal, device)| {
            let device = device?;
            Ok(DeviceSummary {
                backend: B::KIND,
                ordinal,
                name: device.name()?,
            })
        })
        .collect()
}

fn check(operation: &'static str, result: sys::ocgpuResult) -> Result<()> {
    if result == sys::OCGPU_SUCCESS {
        Ok(())
    } else {
        Err(Error::Api { operation, result })
    }
}

#[cfg(any(not(feature = "cuda"), not(feature = "hip")))]
fn feature_disabled(backend: BackendKind) -> Error {
    Error::BackendUnavailable {
        backend,
        detail: "backend feature was disabled at compile time".to_owned(),
    }
}

#[cfg(feature = "global-default-backend")]
mod global {
    use super::{AnyDriver, BackendKind, Error, Result};
    use std::sync::OnceLock;

    static DEFAULT: OnceLock<core::result::Result<AnyDriver, Error>> = OnceLock::new();

    /// Returns a lazily loaded convenience backend.
    ///
    /// CUDA is tried before HIP. Prefer explicit [`super::Driver`] instances in
    /// libraries and applications that may use both backends.
    pub fn default_driver() -> Result<&'static AnyDriver> {
        match DEFAULT.get_or_init(|| {
            AnyDriver::load(BackendKind::Cuda).or_else(|_| AnyDriver::load(BackendKind::Hip))
        }) {
            Ok(driver) => Ok(driver),
            Err(error) => Err(error.clone()),
        }
    }
}

#[cfg(feature = "global-default-backend")]
pub use global::default_driver;

#[cfg(test)]
mod tests {
    use super::{
        ApiMetadata, BackendDiagnostics, BackendKind, Error, HipRuntimeProfile, LaunchConfig,
        RuntimeSymbolStatus, SymbolResolution,
    };

    #[test]
    fn backend_values_match_public_abi() {
        assert_eq!(BackendKind::Cuda.as_raw(), super::sys::OCGPU_BACKEND_CUDA);
        assert_eq!(BackendKind::Hip.as_raw(), super::sys::OCGPU_BACKEND_HIP);
    }

    #[test]
    fn hip_runtime_profile_spellings_are_stable() {
        assert_eq!(HipRuntimeProfile::Hip5.as_str(), "hip_5");
        assert_eq!(HipRuntimeProfile::Hip6.as_str(), "hip_6");
        assert_eq!(HipRuntimeProfile::Hip7.as_str(), "hip_7");
        assert_eq!(HipRuntimeProfile::Hip5.to_string(), "HIP 5");
        assert_eq!(HipRuntimeProfile::Hip6.to_string(), "HIP 6");
        assert_eq!(HipRuntimeProfile::Hip7.to_string(), "HIP 7");
        for profile in [
            HipRuntimeProfile::Hip5,
            HipRuntimeProfile::Hip6,
            HipRuntimeProfile::Hip7,
        ] {
            assert_eq!(
                HipRuntimeProfile::from_api_flags(profile.api_flags()),
                Some(profile)
            );
        }
        assert_eq!(HipRuntimeProfile::from_api_flags(0), None);
        assert_eq!(
            HipRuntimeProfile::from_api_flags(super::sys::OCGPU_API_FLAG_HIP_RUNTIME_PROFILE_MASK),
            None
        );
        let metadata = ApiMetadata {
            backend: BackendKind::Hip,
            abi_version: super::sys::OCGPU_ABI_VERSION_1,
            flags: HipRuntimeProfile::Hip6.api_flags(),
            driver_version: 60_400_000,
            struct_size: 0,
        };
        assert_eq!(
            metadata.hip_runtime_profile(),
            Some(HipRuntimeProfile::Hip6)
        );
    }

    #[cfg(feature = "hip")]
    #[test]
    fn hip_driver_exposes_the_selected_runtime_profile() {
        assert_eq!(
            super::map_hip_runtime_profile(ocgpu_hip::RuntimeProfile::Hip5),
            HipRuntimeProfile::Hip5
        );
        assert_eq!(
            super::map_hip_runtime_profile(ocgpu_hip::RuntimeProfile::Hip6),
            HipRuntimeProfile::Hip6
        );
        assert_eq!(
            super::map_hip_runtime_profile(ocgpu_hip::RuntimeProfile::Hip7),
            HipRuntimeProfile::Hip7
        );
        let _: fn(&super::Driver<super::Hip>) -> HipRuntimeProfile =
            super::Driver::<super::Hip>::runtime_profile;
    }

    #[test]
    fn launch_dimensions_must_be_nonzero() {
        assert!(LaunchConfig::new([1, 1, 1], [32, 1, 1], 0).is_ok());
        assert!(matches!(
            LaunchConfig::new([0, 1, 1], [32, 1, 1], 0),
            Err(Error::InvalidArgument(_))
        ));
    }

    #[test]
    fn error_result_mapping_is_stable() {
        assert_eq!(
            Error::InvalidArgument("test").result(),
            super::sys::OCGPU_ERROR_INVALID_ARGUMENT
        );
        assert_eq!(
            Error::BackendTooOld {
                backend: BackendKind::Cuda,
                symbols: vec!["cuLaunchKernel"],
            }
            .result(),
            super::sys::OCGPU_ERROR_BACKEND_TOO_OLD
        );
        assert_eq!(
            Error::BackendRuntimeTooOld {
                backend: BackendKind::Hip,
                detail: "version 50600000 is below reviewed minimum 50700000".to_owned(),
            }
            .result(),
            super::sys::OCGPU_ERROR_BACKEND_TOO_OLD
        );
        assert_eq!(
            Error::MissingCoreSymbol {
                backend: BackendKind::Cuda,
                symbol: "x",
            }
            .result(),
            super::sys::OCGPU_ERROR_SYMBOL_UNAVAILABLE
        );
        assert_eq!(
            Error::BackendAbiMismatch {
                backend: BackendKind::Hip,
                detail: "filename profile disagrees with runtime major".to_owned(),
            }
            .result(),
            super::sys::OCGPU_ERROR_ABI_MISMATCH
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn backend_missing_core_errors_keep_the_symbol_result_category() {
        let error = super::map_cuda_error(ocgpu_cuda::Error::MissingCoreSymbols {
            library: "mock-cuda".into(),
            symbols: vec!["cuInit"],
        });
        assert_eq!(error.result(), super::sys::OCGPU_ERROR_SYMBOL_UNAVAILABLE);
        assert!(matches!(
            error,
            Error::MissingCoreSymbol {
                backend: BackendKind::Cuda,
                symbol: "cuInit"
            }
        ));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn backend_version_evidence_maps_to_too_old_result() {
        let error = super::map_cuda_error(ocgpu_cuda::Error::BackendTooOld {
            library: "mock-old-cuda".into(),
            symbols: vec!["cuLaunchKernel"],
        });
        assert_eq!(error.result(), super::sys::OCGPU_ERROR_BACKEND_TOO_OLD);
        assert!(matches!(
            error,
            Error::BackendTooOld {
                backend: BackendKind::Cuda,
                ref symbols
            } if symbols == &["cuLaunchKernel"]
        ));
    }

    #[cfg(feature = "hip")]
    #[test]
    fn hip_runtime_major_failures_preserve_too_old_and_abi_mismatch_categories() {
        let too_old = super::map_hip_error(ocgpu_hip::Error::BackendTooOld {
            library: "mock-hip4".into(),
            symbols: vec!["hipRuntimeGetVersion"],
        });
        assert_eq!(too_old.result(), super::sys::OCGPU_ERROR_BACKEND_TOO_OLD);

        let unsupported = super::map_hip_error(ocgpu_hip::Error::UnsupportedRuntimeProfile {
            library: "mock-hip8".into(),
            runtime_version: 80_000_000,
            runtime_major: 8,
        });
        assert!(matches!(
            &unsupported,
            Error::BackendAbiMismatch {
                backend: BackendKind::Hip,
                ..
            }
        ));
        assert_eq!(unsupported.result(), super::sys::OCGPU_ERROR_ABI_MISMATCH);

        let mismatch = super::map_hip_error(ocgpu_hip::Error::RuntimeProfileMismatch {
            library: "mock-amdhip64_6".into(),
            expected: ocgpu_hip::RuntimeProfile::Hip6,
            detected: ocgpu_hip::RuntimeProfile::Hip5,
            runtime_version: 50_700_000,
        });
        assert!(matches!(
            &mismatch,
            Error::BackendAbiMismatch {
                backend: BackendKind::Hip,
                ..
            }
        ));
        assert_eq!(mismatch.result(), super::sys::OCGPU_ERROR_ABI_MISMATCH);

        let below_reviewed = super::map_hip_error(ocgpu_hip::Error::UnsupportedRuntimeVersion {
            library: "mock-hip5-old".into(),
            runtime_profile: ocgpu_hip::RuntimeProfile::Hip5,
            runtime_version: 50_600_000,
            minimum_supported: 50_700_000,
            maximum_supported: 50_799_999,
        });
        assert!(matches!(
            &below_reviewed,
            Error::BackendRuntimeTooOld {
                backend: BackendKind::Hip,
                detail
            } if detail.contains("50600000")
                && detail.contains("50700000..=50799999")
        ));
        assert_eq!(
            below_reviewed.result(),
            super::sys::OCGPU_ERROR_BACKEND_TOO_OLD
        );

        let above_reviewed = super::map_hip_error(ocgpu_hip::Error::UnsupportedRuntimeVersion {
            library: "mock-hip6-new".into(),
            runtime_profile: ocgpu_hip::RuntimeProfile::Hip6,
            runtime_version: 60_500_000,
            minimum_supported: 60_100_000,
            maximum_supported: 60_499_999,
        });
        assert!(matches!(
            &above_reviewed,
            Error::BackendAbiMismatch {
                backend: BackendKind::Hip,
                detail
            } if detail.contains("60500000")
                && detail.contains("60100000..=60499999")
        ));
        assert_eq!(
            above_reviewed.result(),
            super::sys::OCGPU_ERROR_ABI_MISMATCH
        );
    }

    #[test]
    fn diagnostics_separate_required_optional_profile_and_platform_omissions() {
        let symbol = |name, resolution, required, applicable| RuntimeSymbolStatus {
            name,
            resolved_name: None,
            resolution,
            proc_attempts: 0,
            required,
            applicable,
        };
        let diagnostics = BackendDiagnostics {
            backend: BackendKind::Cuda,
            library_path: "mock".into(),
            runtime_version: Some(13_030),
            driver_version: Some(13_030),
            compiled_api_version: 13_030,
            hip_runtime_profile: None,
            proc_address_support: true,
            proc_address_variant: Some(super::ProcAddressVariant::CudaV2),
            loaded_architecture: "x86_64",
            symbols: vec![
                symbol("required", SymbolResolution::Missing, true, true),
                symbol("optional", SymbolResolution::Missing, false, true),
                symbol(
                    "newer_profile_only",
                    SymbolResolution::ProfileUnavailable,
                    false,
                    true,
                ),
                symbol(
                    "legacy_common_adapter",
                    SymbolResolution::DirectAdapter,
                    true,
                    true,
                ),
                symbol(
                    "windows_only",
                    SymbolResolution::PlatformUnavailable,
                    false,
                    false,
                ),
                symbol("available", SymbolResolution::Direct, true, true),
            ],
        };
        assert_eq!(diagnostics.missing_required_symbols().count(), 1);
        assert_eq!(diagnostics.missing_optional_symbols().count(), 1);
        assert_eq!(diagnostics.profile_omissions().count(), 2);
        assert_eq!(diagnostics.platform_omissions().count(), 1);
        let adapter = diagnostics
            .symbols
            .iter()
            .find(|symbol| symbol.name == "legacy_common_adapter")
            .expect("adapter diagnostic is present");
        assert!(!adapter.available(), "its newer-shaped raw slot is null");
        assert!(adapter.required, "the common adapter remains required");
    }
}

#[cfg(test)]
mod mock_backend_tests {
    use super::{Backend, BackendKind, Driver, LaunchConfig, Result, sealed, sys};
    use core::ffi::{c_char, c_void};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

    struct Mock;
    struct MockRaw;

    static RAW: MockRaw = MockRaw;
    static MEMORY: [AtomicU8; 64] = [const { AtomicU8::new(0) }; 64];
    static CONTEXT_DROPS: AtomicUsize = AtomicUsize::new(0);
    static MEMORY_DROPS: AtomicUsize = AtomicUsize::new(0);
    static STREAM_DROPS: AtomicUsize = AtomicUsize::new(0);
    static EVENT_DROPS: AtomicUsize = AtomicUsize::new(0);
    static MODULE_DROPS: AtomicUsize = AtomicUsize::new(0);
    static LAUNCHES: AtomicUsize = AtomicUsize::new(0);
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    impl sealed::Sealed for Mock {}

    impl Backend for Mock {
        type RawApi = MockRaw;

        const KIND: BackendKind = BackendKind::Cuda;

        fn load_raw() -> Result<&'static Self::RawApi> {
            Ok(&RAW)
        }

        #[cfg(feature = "explicit-library-path")]
        unsafe fn load_raw_from_absolute(_path: &std::path::Path) -> Result<&'static Self::RawApi> {
            Ok(&RAW)
        }

        fn common_table(_api: &'static Self::RawApi) -> sys::ocgpuApi_v1 {
            sys::ocgpuApi_v1 {
                struct_size: u32::try_from(size_of::<sys::ocgpuApi_v1>()).unwrap_or(u32::MAX),
                abi_version: sys::OCGPU_ABI_VERSION_1,
                backend: sys::OCGPU_BACKEND_CUDA,
                flags: 0,
                driver_version: 13_030,
                reserved0: 0,
                ocgpuInit: Some(init),
                ocgpuDriverGetVersion: Some(driver_get_version),
                ocgpuDeviceGetCount: Some(device_get_count),
                ocgpuDeviceGet: Some(device_get),
                ocgpuDeviceGetName: Some(device_get_name),
                ocgpuDeviceGetAttribute: Some(device_get_attribute),
                ocgpuCtxCreate: Some(ctx_create),
                ocgpuCtxDestroy: Some(ctx_destroy),
                ocgpuCtxSetCurrent: Some(ctx_set_current),
                ocgpuCtxGetCurrent: Some(ctx_get_current),
                ocgpuCtxSynchronize: Some(success_no_args),
                ocgpuMemAlloc: Some(mem_alloc),
                ocgpuMemFree: Some(mem_free),
                ocgpuMemcpyHtoD: Some(memcpy_htod),
                ocgpuMemcpyDtoH: Some(memcpy_dtoh),
                ocgpuStreamCreate: Some(stream_create),
                ocgpuStreamDestroy: Some(stream_destroy),
                ocgpuStreamSynchronize: Some(stream_synchronize),
                ocgpuEventCreate: Some(event_create),
                ocgpuEventDestroy: Some(event_destroy),
                ocgpuEventRecord: Some(event_record),
                ocgpuEventSynchronize: Some(event_synchronize),
                ocgpuModuleLoadData: Some(module_load_data),
                ocgpuModuleUnload: Some(module_unload),
                ocgpuModuleGetFunction: Some(module_get_function),
                ocgpuLaunchKernel: Some(launch_kernel),
            }
        }
    }

    unsafe extern "C" fn init(flags: u32) -> sys::ocgpuResult {
        if flags == 0 {
            sys::OCGPU_SUCCESS
        } else {
            sys::OCGPU_ERROR_INVALID_ARGUMENT
        }
    }

    unsafe extern "C" fn driver_get_version(output: *mut i32) -> sys::ocgpuResult {
        // SAFETY: the typed API supplies writable version storage to this mock.
        let Some(output) = (unsafe { output.as_mut() }) else {
            return sys::OCGPU_ERROR_INVALID_ARGUMENT;
        };
        *output = 13_030;
        sys::OCGPU_SUCCESS
    }

    unsafe extern "C" fn device_get_count(output: *mut i32) -> sys::ocgpuResult {
        // SAFETY: the typed API supplies writable count storage to this mock.
        let Some(output) = (unsafe { output.as_mut() }) else {
            return sys::OCGPU_ERROR_INVALID_ARGUMENT;
        };
        *output = 1;
        sys::OCGPU_SUCCESS
    }

    unsafe extern "C" fn device_get(
        output: *mut sys::ocgpuDevice,
        ordinal: i32,
    ) -> sys::ocgpuResult {
        // SAFETY: the typed API supplies writable device storage to this mock.
        let Some(output) = (unsafe { output.as_mut() }) else {
            return sys::OCGPU_ERROR_INVALID_ARGUMENT;
        };
        if ordinal != 0 {
            return sys::OCGPU_ERROR_INVALID_ARGUMENT;
        }
        *output = 0;
        sys::OCGPU_SUCCESS
    }

    unsafe extern "C" fn device_get_name(
        output: *mut c_char,
        length: i32,
        _device: sys::ocgpuDevice,
    ) -> sys::ocgpuResult {
        const NAME: &[u8] = b"Mock GPU\0";
        let Ok(length) = usize::try_from(length) else {
            return sys::OCGPU_ERROR_INVALID_ARGUMENT;
        };
        if output.is_null() || length < NAME.len() {
            return sys::OCGPU_ERROR_INVALID_ARGUMENT;
        }
        for (index, byte) in NAME.iter().copied().enumerate() {
            // SAFETY: the caller promised `length` writable bytes and the
            // bounds check proves this fixed name fits.
            unsafe {
                output
                    .add(index)
                    .write(c_char::try_from(byte).unwrap_or_default());
            };
        }
        sys::OCGPU_SUCCESS
    }

    unsafe extern "C" fn device_get_attribute(
        output: *mut i32,
        _attribute: sys::ocgpuDeviceAttribute,
        _device: sys::ocgpuDevice,
    ) -> sys::ocgpuResult {
        // SAFETY: the typed API supplies writable attribute storage to this mock.
        let Some(output) = (unsafe { output.as_mut() }) else {
            return sys::OCGPU_ERROR_INVALID_ARGUMENT;
        };
        *output = 42;
        sys::OCGPU_SUCCESS
    }

    unsafe extern "C" fn ctx_create(
        output: *mut sys::ocgpuContext,
        _flags: u32,
        _device: sys::ocgpuDevice,
    ) -> sys::ocgpuResult {
        // SAFETY: the typed API supplies writable context storage to this mock.
        let Some(output) = (unsafe { output.as_mut() }) else {
            return sys::OCGPU_ERROR_INVALID_ARGUMENT;
        };
        *output = 1_usize as sys::ocgpuContext;
        sys::OCGPU_SUCCESS
    }

    unsafe extern "C" fn ctx_destroy(context: sys::ocgpuContext) -> sys::ocgpuResult {
        if context.is_null() {
            return sys::OCGPU_ERROR_INVALID_ARGUMENT;
        }
        CONTEXT_DROPS.fetch_add(1, Ordering::SeqCst);
        sys::OCGPU_SUCCESS
    }

    unsafe extern "C" fn ctx_set_current(context: sys::ocgpuContext) -> sys::ocgpuResult {
        if context.is_null() {
            sys::OCGPU_ERROR_INVALID_ARGUMENT
        } else {
            sys::OCGPU_SUCCESS
        }
    }

    unsafe extern "C" fn ctx_get_current(output: *mut sys::ocgpuContext) -> sys::ocgpuResult {
        // SAFETY: the typed API supplies writable context storage to this mock.
        let Some(output) = (unsafe { output.as_mut() }) else {
            return sys::OCGPU_ERROR_INVALID_ARGUMENT;
        };
        *output = 1_usize as sys::ocgpuContext;
        sys::OCGPU_SUCCESS
    }

    unsafe extern "C" fn success_no_args() -> sys::ocgpuResult {
        sys::OCGPU_SUCCESS
    }

    unsafe extern "C" fn mem_alloc(
        output: *mut sys::ocgpuDeviceptr,
        bytes: usize,
    ) -> sys::ocgpuResult {
        // SAFETY: the typed API supplies writable allocation storage to this mock.
        let Some(output) = (unsafe { output.as_mut() }) else {
            return sys::OCGPU_ERROR_INVALID_ARGUMENT;
        };
        if bytes == 0 || bytes > MEMORY.len() {
            return sys::OCGPU_ERROR_INVALID_ARGUMENT;
        }
        *output = usize::from(bytes != MEMORY.len());
        sys::OCGPU_SUCCESS
    }

    unsafe extern "C" fn mem_free(pointer: sys::ocgpuDeviceptr) -> sys::ocgpuResult {
        if pointer != 1 {
            return sys::OCGPU_ERROR_INVALID_ARGUMENT;
        }
        MEMORY_DROPS.fetch_add(1, Ordering::SeqCst);
        sys::OCGPU_SUCCESS
    }

    unsafe extern "C" fn memcpy_htod(
        destination: sys::ocgpuDeviceptr,
        source: *const c_void,
        bytes: usize,
    ) -> sys::ocgpuResult {
        if destination != 1 || source.is_null() || bytes > MEMORY.len() {
            return sys::OCGPU_ERROR_INVALID_ARGUMENT;
        }
        for (index, cell) in MEMORY.iter().enumerate().take(bytes) {
            // SAFETY: the caller promises `bytes` readable source bytes and the
            // bounds check constrains every offset.
            cell.store(
                unsafe { source.cast::<u8>().add(index).read() },
                Ordering::SeqCst,
            );
        }
        sys::OCGPU_SUCCESS
    }

    unsafe extern "C" fn memcpy_dtoh(
        destination: *mut c_void,
        source: sys::ocgpuDeviceptr,
        bytes: usize,
    ) -> sys::ocgpuResult {
        if source != 1 || destination.is_null() || bytes > MEMORY.len() {
            return sys::OCGPU_ERROR_INVALID_ARGUMENT;
        }
        for (index, cell) in MEMORY.iter().enumerate().take(bytes) {
            // SAFETY: the caller promises `bytes` writable destination bytes and
            // the bounds check constrains every offset.
            unsafe {
                destination
                    .cast::<u8>()
                    .add(index)
                    .write(cell.load(Ordering::SeqCst));
            }
        }
        sys::OCGPU_SUCCESS
    }

    unsafe extern "C" fn stream_create(
        output: *mut sys::ocgpuStream,
        _flags: u32,
    ) -> sys::ocgpuResult {
        write_handle(output, 2_usize as sys::ocgpuStream)
    }

    unsafe extern "C" fn stream_destroy(stream: sys::ocgpuStream) -> sys::ocgpuResult {
        if stream.is_null() {
            return sys::OCGPU_ERROR_INVALID_ARGUMENT;
        }
        STREAM_DROPS.fetch_add(1, Ordering::SeqCst);
        sys::OCGPU_SUCCESS
    }

    unsafe extern "C" fn stream_synchronize(stream: sys::ocgpuStream) -> sys::ocgpuResult {
        non_null_result(stream.cast())
    }

    unsafe extern "C" fn event_create(
        output: *mut sys::ocgpuEvent,
        _flags: u32,
    ) -> sys::ocgpuResult {
        write_handle(output, 3_usize as sys::ocgpuEvent)
    }

    unsafe extern "C" fn event_destroy(event: sys::ocgpuEvent) -> sys::ocgpuResult {
        if event.is_null() {
            return sys::OCGPU_ERROR_INVALID_ARGUMENT;
        }
        EVENT_DROPS.fetch_add(1, Ordering::SeqCst);
        sys::OCGPU_SUCCESS
    }

    unsafe extern "C" fn event_record(
        event: sys::ocgpuEvent,
        stream: sys::ocgpuStream,
    ) -> sys::ocgpuResult {
        if event.is_null() || stream.is_null() {
            sys::OCGPU_ERROR_INVALID_ARGUMENT
        } else {
            sys::OCGPU_SUCCESS
        }
    }

    unsafe extern "C" fn event_synchronize(event: sys::ocgpuEvent) -> sys::ocgpuResult {
        non_null_result(event.cast())
    }

    unsafe extern "C" fn module_load_data(
        output: *mut sys::ocgpuModule,
        image: *const c_void,
    ) -> sys::ocgpuResult {
        if image.is_null() {
            return sys::OCGPU_ERROR_INVALID_ARGUMENT;
        }
        write_handle(output, 4_usize as sys::ocgpuModule)
    }

    unsafe extern "C" fn module_unload(module: sys::ocgpuModule) -> sys::ocgpuResult {
        if module.is_null() {
            return sys::OCGPU_ERROR_INVALID_ARGUMENT;
        }
        MODULE_DROPS.fetch_add(1, Ordering::SeqCst);
        sys::OCGPU_SUCCESS
    }

    unsafe extern "C" fn module_get_function(
        output: *mut sys::ocgpuFunction,
        module: sys::ocgpuModule,
        name: *const c_char,
    ) -> sys::ocgpuResult {
        if module.is_null() || name.is_null() {
            return sys::OCGPU_ERROR_INVALID_ARGUMENT;
        }
        write_handle(output, 5_usize as sys::ocgpuFunction)
    }

    #[allow(clippy::too_many_arguments)]
    unsafe extern "C" fn launch_kernel(
        function: sys::ocgpuFunction,
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        block_x: u32,
        block_y: u32,
        block_z: u32,
        _shared_memory: u32,
        _stream: sys::ocgpuStream,
        _parameters: *mut *mut c_void,
        _extra: *mut *mut c_void,
    ) -> sys::ocgpuResult {
        if function.is_null() || [grid_x, grid_y, grid_z, block_x, block_y, block_z].contains(&0) {
            return sys::OCGPU_ERROR_INVALID_ARGUMENT;
        }
        LAUNCHES.fetch_add(1, Ordering::SeqCst);
        sys::OCGPU_SUCCESS
    }

    fn write_handle<T>(output: *mut *mut T, handle: *mut T) -> sys::ocgpuResult {
        // SAFETY: only the test's FFI mocks call this helper after receiving the
        // writable output pointer from the typed API.
        let Some(output) = (unsafe { output.as_mut() }) else {
            return sys::OCGPU_ERROR_INVALID_ARGUMENT;
        };
        *output = handle;
        sys::OCGPU_SUCCESS
    }

    const fn non_null_result(pointer: *mut c_void) -> sys::ocgpuResult {
        if pointer.is_null() {
            sys::OCGPU_ERROR_INVALID_ARGUMENT
        } else {
            sys::OCGPU_SUCCESS
        }
    }

    #[test]
    fn typed_resources_dispatch_and_release_without_hardware() {
        let _guard = TEST_LOCK.lock().expect("mock test lock");
        CONTEXT_DROPS.store(0, Ordering::SeqCst);
        MEMORY_DROPS.store(0, Ordering::SeqCst);
        STREAM_DROPS.store(0, Ordering::SeqCst);
        EVENT_DROPS.store(0, Ordering::SeqCst);
        MODULE_DROPS.store(0, Ordering::SeqCst);
        LAUNCHES.store(0, Ordering::SeqCst);

        let driver = Driver::<Mock>::load().expect("complete mock table");
        assert_eq!(driver.driver_version().expect("version"), 13_030);
        assert_eq!(driver.device_count().expect("count"), 1);
        // SAFETY: the mock current context is process-static and remains valid
        // for the complete test.
        assert!(
            unsafe { driver.current_context() }
                .expect("current")
                .is_some()
        );
        let device = driver.device(0).expect("device");
        assert_eq!(device.name().expect("name"), "Mock GPU");
        assert_eq!(device.attribute(0).expect("attribute"), 42);

        {
            let context = device.create_context(0).expect("context");
            let memory = context.allocate(8).expect("allocation");
            memory.copy_from(b"ocgpu-v1").expect("HtoD");
            let mut output = [0_u8; 8];
            memory.copy_to(&mut output).expect("DtoH");
            assert_eq!(&output, b"ocgpu-v1");

            let stream = context.create_stream(0).expect("stream");
            let event = context.create_event(0).expect("event");
            event.record(&stream).expect("record");
            event.synchronize().expect("event sync");

            // SAFETY: the mock backend defines this complete literal as an
            // accepted module image and consumes it during the call.
            let module = unsafe { context.load_module_cstr(c"image") }.expect("module");
            let function = module.function(c"ocgpu_noop").expect("function");
            let config = LaunchConfig::new([1, 1, 1], [1, 1, 1], 0).expect("config");
            // SAFETY: mock function has no arguments and validates only handles.
            unsafe {
                function
                    .launch(config, Some(&stream), &mut [])
                    .expect("launch");
            }
            stream.synchronize().expect("stream sync");
            context.synchronize().expect("context sync");
        }

        assert_eq!(CONTEXT_DROPS.load(Ordering::SeqCst), 1);
        assert_eq!(MEMORY_DROPS.load(Ordering::SeqCst), 1);
        assert_eq!(STREAM_DROPS.load(Ordering::SeqCst), 1);
        assert_eq!(EVENT_DROPS.load(Ordering::SeqCst), 1);
        assert_eq!(MODULE_DROPS.load(Ordering::SeqCst), 1);
        assert_eq!(LAUNCHES.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn safe_copy_bounds_fail_before_ffi() {
        let _guard = TEST_LOCK.lock().expect("mock test lock");
        let driver = Driver::<Mock>::load().expect("complete mock table");
        let device = driver.device(0).expect("device");
        let context = device.create_context(0).expect("context");
        let memory = context.allocate(4).expect("allocation");
        assert!(memory.copy_from(&[0; 5]).is_err());
        assert!(memory.copy_to(&mut [0; 5]).is_err());
    }

    #[test]
    fn successful_allocation_must_return_a_nonzero_device_pointer() {
        let _guard = TEST_LOCK.lock().expect("mock test lock");
        let driver = Driver::<Mock>::load().expect("complete mock table");
        let device = driver.device(0).expect("device");
        let context = device.create_context(0).expect("context");
        assert!(matches!(
            context.allocate(MEMORY.len()),
            Err(super::Error::NullHandle("ocgpuMemAlloc"))
        ));
    }
}
