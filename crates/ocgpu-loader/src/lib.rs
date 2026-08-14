// SPDX-License-Identifier: CC0-1.0

//! Small, SDK-free dynamic-library loader used by the GPU backends.
//!
//! A backend library is initialized at most once and is deliberately never
//! unloaded. This is required because callers may retain vendor function
//! pointers and GPU handles until process exit.

#![cfg_attr(not(any(target_os = "linux", target_os = "windows")), allow(dead_code))]

use std::error::Error as StdError;
use std::ffi::{CStr, CString, c_void};
use std::fmt;
use std::mem::{align_of, size_of};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::OnceLock;

#[cfg(all(
    not(target_pointer_width = "64"),
    any(target_os = "linux", target_os = "windows")
))]
compile_error!("ocgpu ABI version 1 supports only 64-bit targets");

/// A dynamically loaded GPU backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Backend {
    /// NVIDIA CUDA Driver API.
    Cuda,
    /// AMD HIP runtime's driver-shaped API.
    Hip,
}

impl Backend {
    /// Human-readable backend name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Cuda => "CUDA",
            Self::Hip => "HIP",
        }
    }

    /// Secure platform-specific default library candidates, in priority order.
    #[must_use]
    pub const fn candidates(self) -> &'static [&'static str] {
        match self {
            #[cfg(target_os = "windows")]
            Self::Cuda => &["nvcuda.dll"],
            #[cfg(target_os = "windows")]
            Self::Hip => &["amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll"],
            #[cfg(target_os = "linux")]
            Self::Cuda => &["libcuda.so.1"],
            #[cfg(target_os = "linux")]
            Self::Hip => &[
                "libamdhip64.so.7",
                "libamdhip64.so.6",
                "libamdhip64.so.5",
                "libamdhip64.so",
            ],
            #[cfg(not(any(target_os = "linux", target_os = "windows")))]
            _ => &[],
        }
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// One failed candidate from a backend-library search.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpenFailure {
    /// Logical or absolute candidate associated with this loader attempt.
    pub candidate: PathBuf,
    /// Operating-system error code, when one was supplied.
    pub os_code: Option<i32>,
    /// Stable owned description captured before another loader call can replace it.
    pub message: String,
}

/// Why an explicit library override was rejected.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InvalidPathReason {
    /// Overrides must be absolute so their meaning cannot change with the CWD.
    NotAbsolute,
    /// The canonical target does not name a regular file.
    NotAFile,
    /// The platform path representation contains an embedded NUL.
    ContainsNul,
    /// Canonicalization or metadata inspection failed.
    Inaccessible,
}

/// Structured dynamic-loader failure.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum LoadError {
    /// None of the platform candidates could be loaded.
    BackendUnavailable {
        /// Backend being loaded.
        backend: Backend,
        /// Every attempted candidate and its corresponding OS failure.
        attempts: Vec<OpenFailure>,
    },
    /// The requested target platform is unsupported.
    UnsupportedPlatform {
        /// Backend being loaded.
        backend: Backend,
    },
    /// An opt-in explicit path did not pass validation.
    InvalidExplicitPath {
        /// Supplied path.
        path: PathBuf,
        /// Validation failure category.
        reason: InvalidPathReason,
        /// Additional OS context, if available.
        detail: Option<String>,
    },
    /// A different library already owns the process-lifetime backend slot.
    AlreadyInitialized {
        /// Backend whose slot was already initialized.
        backend: Backend,
        /// Existing loaded identity.
        loaded: PathBuf,
        /// Newly requested canonical path.
        requested: PathBuf,
    },
    /// A required symbol was not exported by the loaded library.
    SymbolUnavailable {
        /// Loaded library identity.
        library: PathBuf,
        /// Requested symbol, without its trailing NUL.
        symbol: Vec<u8>,
    },
    /// A caller supplied a symbol name containing an embedded NUL.
    InvalidSymbolName {
        /// Rejected bytes.
        symbol: Vec<u8>,
    },
}

impl fmt::Display for LoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BackendUnavailable { backend, attempts } => {
                write!(formatter, "{backend} backend library was not found")?;
                for attempt in attempts {
                    write!(
                        formatter,
                        "; {}: {}",
                        attempt.candidate.display(),
                        attempt.message
                    )?;
                }
                Ok(())
            }
            Self::UnsupportedPlatform { backend } => {
                write!(
                    formatter,
                    "{backend} loading is unsupported on this platform"
                )
            }
            Self::InvalidExplicitPath {
                path,
                reason,
                detail,
            } => {
                write!(
                    formatter,
                    "invalid explicit library path {} ({reason:?})",
                    path.display()
                )?;
                if let Some(detail) = detail {
                    write!(formatter, ": {detail}")?;
                }
                Ok(())
            }
            Self::AlreadyInitialized {
                backend,
                loaded,
                requested,
            } => write!(
                formatter,
                "{backend} is already loaded from {}; cannot replace it with {}",
                loaded.display(),
                requested.display()
            ),
            Self::SymbolUnavailable { library, symbol } => write!(
                formatter,
                "symbol {} is unavailable in {}",
                String::from_utf8_lossy(symbol),
                library.display()
            ),
            Self::InvalidSymbolName { symbol } => write!(
                formatter,
                "symbol name contains an embedded NUL: {:?}",
                String::from_utf8_lossy(symbol)
            ),
        }
    }
}

impl StdError for LoadError {}

/// Failure to populate a generated nullable function-table field.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum TableSlotError {
    /// The generated offset does not fit entirely inside the table.
    OutOfBounds {
        /// Requested byte offset.
        offset: usize,
        /// Total table size in bytes.
        table_size: usize,
    },
    /// The generated offset is not suitably aligned for a function pointer.
    Misaligned {
        /// Requested byte offset.
        offset: usize,
        /// Required byte alignment.
        required_alignment: usize,
    },
}

impl fmt::Display for TableSlotError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfBounds { offset, table_size } => write!(
                formatter,
                "function-table offset {offset} is outside a {table_size}-byte table"
            ),
            Self::Misaligned {
                offset,
                required_alignment,
            } => write!(
                formatter,
                "function-table offset {offset} is not aligned to {required_alignment} bytes"
            ),
        }
    }
}

impl StdError for TableSlotError {}

/// Writes a resolved address into a generated nullable function-table field.
///
/// This is used by generated exhaustive raw inventories, for which every field
/// has a different function signature but the C-compatible nullable function
/// pointers all have the same one-pointer representation.
///
/// # Safety
///
/// `offset` must be the generated `offset_of!` value of an
/// `Option<unsafe extern "C" fn(...)>` field in `T`, and `address` must identify
/// a function with that field's exact ABI and signature. The runtime bounds and
/// alignment checks do not establish those semantic conditions.
#[allow(clippy::cast_ptr_alignment)]
pub unsafe fn write_function_slot<T>(
    table: &mut T,
    offset: usize,
    address: NonNull<c_void>,
) -> Result<(), TableSlotError> {
    type ErasedFunction = unsafe extern "C" fn();
    let table_size = size_of::<T>();
    let slot_size = size_of::<Option<ErasedFunction>>();
    let Some(end) = offset.checked_add(slot_size) else {
        return Err(TableSlotError::OutOfBounds { offset, table_size });
    };
    if end > table_size {
        return Err(TableSlotError::OutOfBounds { offset, table_size });
    }

    let required_alignment = align_of::<Option<ErasedFunction>>();
    let base = std::ptr::from_mut(table).cast::<u8>();
    // SAFETY: the checked range above proves `offset` remains within the table.
    let target = unsafe { base.add(offset) };
    if target.align_offset(required_alignment) != 0 {
        return Err(TableSlotError::Misaligned {
            offset,
            required_alignment,
        });
    }

    // SAFETY: the caller guarantees this code address has the generated field's
    // function ABI. Erasure changes only the Rust type used to copy its bits.
    let function = unsafe { std::mem::transmute::<*mut c_void, ErasedFunction>(address.as_ptr()) };
    // SAFETY: range/alignment were checked and the caller guarantees that this
    // offset identifies a nullable function-pointer field rather than other data.
    unsafe {
        target
            .cast::<Option<ErasedFunction>>()
            .write(Some(function));
    };
    Ok(())
}

/// A process-lifetime shared library.
///
/// There is intentionally no `Drop` implementation. Once published, this
/// handle and every function pointer obtained from it remain valid until the
/// process exits.
pub struct Library {
    handle: NonNull<c_void>,
    backend: Backend,
    loaded_path: PathBuf,
}

// OS loader handles may be used concurrently for read-only symbol lookup.
// SAFETY: both supported operating systems document concurrent symbol lookup,
// and `Library` never mutates or closes its handle.
unsafe impl Send for Library {}
// SAFETY: see the `Send` justification above.
unsafe impl Sync for Library {}

impl fmt::Debug for Library {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Library")
            .field("backend", &self.backend)
            .field("loaded_path", &self.loaded_path)
            .finish_non_exhaustive()
    }
}

impl Library {
    /// Backend represented by this library.
    #[must_use]
    pub const fn backend(&self) -> Backend {
        self.backend
    }

    /// Resolved library identity used for diagnostics.
    ///
    /// Linux default loading records the canonical absolute target selected
    /// before `dlopen`; Windows records the resolved module path when exposed.
    #[must_use]
    pub fn loaded_path(&self) -> &Path {
        &self.loaded_path
    }

    /// Looks up a NUL-terminated symbol without panicking.
    #[must_use]
    pub fn find_cstr(&self, symbol: &CStr) -> Option<NonNull<c_void>> {
        // SAFETY: the library is process-lifetime and `symbol` is NUL-terminated.
        unsafe { platform::find(self.handle, symbol) }
    }

    /// Looks up a symbol name supplied without its terminating NUL.
    pub fn find(&self, symbol: &[u8]) -> Result<Option<NonNull<c_void>>, LoadError> {
        let symbol_c = CString::new(symbol).map_err(|_| LoadError::InvalidSymbolName {
            symbol: symbol.to_vec(),
        })?;
        Ok(self.find_cstr(&symbol_c))
    }

    /// Looks up a required symbol and preserves a structured missing-symbol error.
    pub fn require(&self, symbol: &[u8]) -> Result<NonNull<c_void>, LoadError> {
        self.find(symbol)?
            .ok_or_else(|| LoadError::SymbolUnavailable {
                library: self.loaded_path.clone(),
                symbol: symbol.to_vec(),
            })
    }
}

static CUDA_LIBRARY: OnceLock<Result<Library, LoadError>> = OnceLock::new();
static HIP_LIBRARY: OnceLock<Result<Library, LoadError>> = OnceLock::new();

fn slot(backend: Backend) -> &'static OnceLock<Result<Library, LoadError>> {
    match backend {
        Backend::Cuda => &CUDA_LIBRARY,
        Backend::Hip => &HIP_LIBRARY,
    }
}

/// Loads a backend from its secure platform candidates, at most once.
///
/// On Linux, each fixed basename is resolved major-first through validated
/// absolute `LD_LIBRARY_PATH` directories, a bounded `/etc/ld.so.cache`, and
/// curated absolute `ROCm`, WSL, multiarch, and system directories. Only a
/// canonical absolute regular-file path is passed to `dlopen`, so caller
/// `DT_RPATH`/`DT_RUNPATH` and CWD do not select the top-level runtime.
/// Dependencies remain subject to the loaded object's and ELF system loader's
/// dependency-search policy. Installations outside these sources require the
/// feature-gated, unsafe `load_from_absolute` override.
pub fn load(backend: Backend) -> Result<&'static Library, LoadError> {
    result_ref(slot(backend).get_or_init(|| open_candidates(backend)))
}

/// Loads a backend from a caller-selected absolute path, at most once.
///
/// This escape hatch is deliberately feature-gated. The path is canonicalized
/// and must identify a regular file. Windows uses secure dependency-search
/// flags. Linux rejects relative, empty, and current-directory entries in
/// `LD_LIBRARY_PATH`; the absolute top-level target bypasses the caller's
/// `DT_RPATH` and `DT_RUNPATH`. Dependencies of that target remain subject to
/// the ELF loader's dependency-search policy and the trust contract below.
///
/// # Safety
///
/// The selected library and its dependency closure must be trusted native code
/// whose constructors are safe to execute. Every resolved symbol must implement
/// the exact vendor ABI associated with `backend`, and the file must not be
/// replaced with incompatible code between validation and loading.
#[cfg(feature = "explicit-library-path")]
pub unsafe fn load_from_absolute(
    backend: Backend,
    requested: &Path,
) -> Result<&'static Library, LoadError> {
    let canonical = validate_explicit_path(requested)?;
    let initialized = slot(backend).get_or_init(|| {
        // SAFETY: the caller guarantees that this canonical file and its
        // dependency closure are trusted and implement the selected backend ABI.
        unsafe { open_exact(backend, &canonical) }
    });
    let library = result_ref(initialized)?;
    if paths_identical(library.loaded_path(), &canonical) {
        Ok(library)
    } else {
        Err(LoadError::AlreadyInitialized {
            backend,
            loaded: library.loaded_path().to_path_buf(),
            requested: canonical,
        })
    }
}

fn result_ref(result: &'static Result<Library, LoadError>) -> Result<&'static Library, LoadError> {
    match result {
        Ok(library) => Ok(library),
        Err(error) => Err(error.clone()),
    }
}

fn open_candidates(backend: Backend) -> Result<Library, LoadError> {
    let candidates = backend.candidates();
    if candidates.is_empty() {
        return Err(LoadError::UnsupportedPlatform { backend });
    }

    let mut attempts = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        // SAFETY: platform loading uses fixed, NUL-free names and secure flags.
        match unsafe { platform::open(Path::new(candidate), false) } {
            Ok((handle, loaded_path)) => {
                return Ok(Library {
                    handle,
                    backend,
                    loaded_path,
                });
            }
            Err(failure) => attempts.push(failure),
        }
    }

    Err(LoadError::BackendUnavailable { backend, attempts })
}

#[cfg(feature = "explicit-library-path")]
unsafe fn open_exact(backend: Backend, path: &Path) -> Result<Library, LoadError> {
    // SAFETY: `validate_explicit_path` produced a canonical regular-file path;
    // platform loading excludes CWD for this top-level target, and the caller
    // guarantees that the native code and dependency closure are trusted and
    // ABI-correct.
    match unsafe { platform::open(path, true) } {
        Ok((handle, loaded_path)) => Ok(Library {
            handle,
            backend,
            loaded_path,
        }),
        Err(failure) => Err(LoadError::BackendUnavailable {
            backend,
            attempts: vec![failure],
        }),
    }
}

#[cfg(feature = "explicit-library-path")]
fn validate_explicit_path(path: &Path) -> Result<PathBuf, LoadError> {
    if !path.is_absolute() {
        return Err(LoadError::InvalidExplicitPath {
            path: path.to_path_buf(),
            reason: InvalidPathReason::NotAbsolute,
            detail: None,
        });
    }
    if platform::path_contains_nul(path) {
        return Err(LoadError::InvalidExplicitPath {
            path: path.to_path_buf(),
            reason: InvalidPathReason::ContainsNul,
            detail: None,
        });
    }

    let canonical = path
        .canonicalize()
        .map_err(|error| LoadError::InvalidExplicitPath {
            path: path.to_path_buf(),
            reason: InvalidPathReason::Inaccessible,
            detail: Some(error.to_string()),
        })?;
    let metadata = canonical
        .metadata()
        .map_err(|error| LoadError::InvalidExplicitPath {
            path: path.to_path_buf(),
            reason: InvalidPathReason::Inaccessible,
            detail: Some(error.to_string()),
        })?;
    if !metadata.is_file() {
        return Err(LoadError::InvalidExplicitPath {
            path: canonical,
            reason: InvalidPathReason::NotAFile,
            detail: None,
        });
    }
    Ok(canonical)
}

#[cfg(feature = "explicit-library-path")]
fn paths_identical(left: &Path, right: &Path) -> bool {
    // Both inputs have already passed `canonicalize`. Compare their native OS
    // strings exactly: a lossy UTF-16 conversion on Windows could otherwise
    // collapse two distinct filenames containing unpaired surrogates.
    left == right
}

#[cfg(target_os = "windows")]
mod platform {
    use super::OpenFailure;
    use std::ffi::{CStr, OsString, c_void};
    use std::os::windows::ffi::{OsStrExt, OsStringExt};
    use std::path::{Path, PathBuf};
    use std::ptr::NonNull;

    type HModule = *mut c_void;

    const LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR: u32 = 0x0000_0100;
    const LOAD_LIBRARY_SEARCH_SYSTEM32: u32 = 0x0000_0800;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryExW(file_name: *const u16, file: *mut c_void, flags: u32) -> HModule;
        fn GetProcAddress(module: HModule, proc_name: *const u8) -> *mut c_void;
        fn GetModuleFileNameW(module: HModule, file_name: *mut u16, size: u32) -> u32;
    }

    pub(super) unsafe fn open(
        path: &Path,
        explicit: bool,
    ) -> Result<(NonNull<c_void>, PathBuf), OpenFailure> {
        let candidate = path.to_path_buf();
        let mut wide: Vec<u16> = path.as_os_str().encode_wide().collect();
        if wide.contains(&0) {
            return Err(OpenFailure {
                candidate,
                os_code: None,
                message: "path contains an embedded NUL".to_owned(),
            });
        }
        wide.push(0);
        let flags = if explicit {
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32
        } else {
            // Driver-supplied CUDA and HIP runtimes reside in System32. Restricting
            // a bare DLL name to that directory categorically excludes the CWD.
            LOAD_LIBRARY_SEARCH_SYSTEM32
        };
        // SAFETY: `wide` is a live, NUL-terminated UTF-16 buffer; the reserved
        // file handle is null and the documented secure search flags are used.
        let raw = unsafe { LoadLibraryExW(wide.as_ptr(), std::ptr::null_mut(), flags) };
        let Some(handle) = NonNull::new(raw) else {
            let error = std::io::Error::last_os_error();
            return Err(OpenFailure {
                candidate,
                os_code: error.raw_os_error(),
                message: error.to_string(),
            });
        };

        let loaded_path = if explicit {
            path.to_path_buf()
        } else {
            module_path(handle).unwrap_or_else(|| path.to_path_buf())
        };
        Ok((handle, loaded_path))
    }

    pub(super) unsafe fn find(handle: NonNull<c_void>, symbol: &CStr) -> Option<NonNull<c_void>> {
        // SAFETY: `handle` came from `LoadLibraryExW`, remains loaded forever,
        // and `symbol` is a live NUL-terminated byte string.
        NonNull::new(unsafe { GetProcAddress(handle.as_ptr(), symbol.as_ptr().cast()) })
    }

    fn module_path(handle: NonNull<c_void>) -> Option<PathBuf> {
        let mut capacity = 512_usize;
        while capacity <= 32_768 {
            let mut buffer = vec![0_u16; capacity];
            let size = u32::try_from(buffer.len()).ok()?;
            // SAFETY: the buffer is writable for `size` UTF-16 code units and
            // the process-lifetime module handle is valid.
            let written = unsafe { GetModuleFileNameW(handle.as_ptr(), buffer.as_mut_ptr(), size) };
            if written == 0 {
                return None;
            }
            let written = usize::try_from(written).ok()?;
            if written < buffer.len() {
                buffer.truncate(written);
                return Some(PathBuf::from(OsString::from_wide(&buffer)));
            }
            capacity *= 2;
        }
        None
    }

    #[cfg(feature = "explicit-library-path")]
    pub(super) fn path_contains_nul(path: &Path) -> bool {
        path.as_os_str().encode_wide().any(|unit| unit == 0)
    }
}

#[cfg(target_os = "linux")]
mod platform {
    use super::OpenFailure;
    use std::ffi::{CStr, CString, OsStr, c_char, c_int, c_void};
    use std::fs;
    use std::os::unix::ffi::OsStrExt;
    use std::path::{Path, PathBuf};
    use std::ptr::NonNull;
    use std::sync::OnceLock;

    const RTLD_LOCAL: c_int = 0;
    const RTLD_NOW: c_int = 2;
    // Independently encoded layout facts from glibc's
    // `sysdeps/generic/dl-cache.h`; every offset is bounds-checked below.
    const NEW_CACHE_MAGIC: &[u8] = b"glibc-ld.so.cache1.1";
    const OLD_CACHE_MAGIC: &[u8] = b"ld.so-1.7.0";
    const NEW_CACHE_HEADER_SIZE: usize = 48;
    const NEW_CACHE_ENTRY_SIZE: usize = 24;
    const OLD_CACHE_HEADER_SIZE: usize = 16;
    const OLD_CACHE_ENTRY_SIZE: usize = 12;
    const MAX_LOADER_CACHE_BYTES: u64 = 64 * 1024 * 1024;
    const MAX_LOADER_CACHE_ENTRIES: usize = 1_000_000;
    const MAX_LOADER_CACHE_STRING_BYTES: usize = 4_096;
    const MAX_MATCHING_CACHE_PATHS: usize = 64;

    #[cfg(target_arch = "x86_64")]
    const FIXED_LIBRARY_DIRECTORIES: &[&str] = &[
        "/opt/rocm/lib",
        "/opt/rocm/lib64",
        "/usr/lib/wsl/lib",
        "/lib/x86_64-linux-gnu",
        "/usr/lib/x86_64-linux-gnu",
        "/lib64",
        "/usr/lib64",
        "/lib",
        "/usr/lib",
    ];
    #[cfg(target_arch = "aarch64")]
    const FIXED_LIBRARY_DIRECTORIES: &[&str] = &[
        "/opt/rocm/lib",
        "/opt/rocm/lib64",
        "/usr/lib/wsl/lib",
        "/lib/aarch64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",
        "/lib64",
        "/usr/lib64",
        "/lib",
        "/usr/lib",
    ];
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    const FIXED_LIBRARY_DIRECTORIES: &[&str] = &[
        "/opt/rocm/lib",
        "/opt/rocm/lib64",
        "/usr/lib/wsl/lib",
        "/lib64",
        "/usr/lib64",
        "/lib",
        "/usr/lib",
    ];

    static LOADER_CACHE: OnceLock<Option<Vec<u8>>> = OnceLock::new();

    #[link(name = "dl")]
    unsafe extern "C" {
        fn dlopen(file_name: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        fn dlerror() -> *const c_char;
    }

    pub(super) unsafe fn open(
        path: &Path,
        explicit: bool,
    ) -> Result<(NonNull<c_void>, PathBuf), OpenFailure> {
        let candidate = path.to_path_buf();
        let library_path = std::env::var_os("LD_LIBRARY_PATH");
        let current = std::env::current_dir().ok();
        if validated_library_path_entries(library_path.as_deref(), current.as_deref()).is_err() {
            return Err(OpenFailure {
                candidate,
                os_code: None,
                message:
                    "LD_LIBRARY_PATH contains a relative, tokenized, or current-directory entry"
                        .to_owned(),
            });
        }

        if explicit {
            // SAFETY: explicit paths were canonicalized and checked as regular
            // files by the public validation layer.
            return unsafe { open_absolute(path) };
        }

        let resolved = resolve_default_candidate_paths(
            path,
            library_path.as_deref(),
            current.as_deref(),
            loader_cache(),
            FIXED_LIBRARY_DIRECTORIES,
        )
        .map_err(|message| OpenFailure {
            candidate: candidate.clone(),
            os_code: None,
            message: message.to_owned(),
        })?;
        if resolved.is_empty() {
            return Err(OpenFailure {
                candidate,
                os_code: None,
                message:
                    "no secure absolute candidate was found; caller RPATH/RUNPATH was not searched"
                        .to_owned(),
            });
        }

        let mut failures = Vec::new();
        for absolute in resolved {
            // SAFETY: the resolver returns only canonical absolute regular-file
            // paths and `open_absolute` never searches a bare library name.
            match unsafe { open_absolute(&absolute) } {
                Ok(loaded) => return Ok(loaded),
                Err(failure) => failures.push(format!(
                    "{}: {}",
                    failure.candidate.display(),
                    failure.message
                )),
            }
        }
        Err(OpenFailure {
            candidate,
            os_code: None,
            message: format!(
                "secure absolute candidates failed to load: {}",
                failures.join("; ")
            ),
        })
    }

    unsafe fn open_absolute(path: &Path) -> Result<(NonNull<c_void>, PathBuf), OpenFailure> {
        let candidate = path.to_path_buf();
        if !path.is_absolute() {
            return Err(OpenFailure {
                candidate,
                os_code: None,
                message: "Linux loader target is not absolute".to_owned(),
            });
        }
        let Ok(path_c) = CString::new(path.as_os_str().as_bytes()) else {
            return Err(OpenFailure {
                candidate,
                os_code: None,
                message: "path contains an embedded NUL".to_owned(),
            });
        };
        // SAFETY: `path_c` is NUL-terminated and the required eager/local flags
        // are passed exactly. No `dlclose` is ever performed.
        let raw = unsafe { dlopen(path_c.as_ptr(), RTLD_NOW | RTLD_LOCAL) };
        NonNull::new(raw)
            .map(|handle| (handle, path.to_path_buf()))
            .ok_or_else(|| OpenFailure {
                candidate,
                os_code: None,
                message: take_dl_error(),
            })
    }

    pub(super) unsafe fn find(handle: NonNull<c_void>, symbol: &CStr) -> Option<NonNull<c_void>> {
        // Clear the thread-local error marker before `dlsym` so a null result can
        // be distinguished from an error according to the POSIX contract.
        // SAFETY: `dlerror` has no preconditions.
        let _ = unsafe { dlerror() };
        // SAFETY: the handle is process-lifetime and `symbol` is NUL-terminated.
        let address = unsafe { dlsym(handle.as_ptr(), symbol.as_ptr()) };
        // SAFETY: reads and clears only this thread's loader error.
        let error = unsafe { dlerror() };
        if error.is_null() {
            NonNull::new(address)
        } else {
            None
        }
    }

    fn take_dl_error() -> String {
        // SAFETY: obtains the current thread's loader error pointer, if any.
        let error = unsafe { dlerror() };
        if error.is_null() {
            "dynamic loader did not provide an error".to_owned()
        } else {
            // SAFETY: a non-null `dlerror` result is a NUL-terminated string that
            // remains valid until the next dynamic-loader operation on this thread.
            unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned()
        }
    }

    fn loader_cache() -> Option<&'static [u8]> {
        LOADER_CACHE.get_or_init(read_loader_cache).as_deref()
    }

    fn read_loader_cache() -> Option<Vec<u8>> {
        let path = Path::new("/etc/ld.so.cache");
        let metadata = fs::metadata(path).ok()?;
        if !metadata.is_file() || metadata.len() > MAX_LOADER_CACHE_BYTES {
            return None;
        }
        let bytes = fs::read(path).ok()?;
        (u64::try_from(bytes.len()).ok()? <= MAX_LOADER_CACHE_BYTES).then_some(bytes)
    }

    fn validated_library_path_entries(
        value: Option<&OsStr>,
        current: Option<&Path>,
    ) -> Result<Vec<PathBuf>, ()> {
        let Some(value) = value else {
            return Ok(Vec::new());
        };
        let Some(current) = current else {
            return Err(());
        };
        if value.as_bytes().contains(&b'$') {
            return Err(());
        }
        let canonical_current = current.canonicalize().ok();
        let mut entries = Vec::new();
        // glibc passes `:;` to `fillin_rpath` for `LD_LIBRARY_PATH`; neither
        // separator has an escaping mechanism in this environment variable.
        for bytes in value
            .as_bytes()
            .split(|byte| *byte == b':' || *byte == b';')
        {
            let entry = PathBuf::from(OsStr::from_bytes(bytes));
            if entry.as_os_str().is_empty() || !entry.is_absolute() {
                return Err(());
            }
            if entry == current
                || entry
                    .canonicalize()
                    .ok()
                    .zip(canonical_current.as_ref())
                    .is_some_and(|(entry, cwd)| &entry == cwd)
            {
                return Err(());
            }
            entries.push(entry);
        }
        Ok(entries)
    }

    fn resolve_default_candidate_paths(
        candidate: &Path,
        library_path: Option<&OsStr>,
        current: Option<&Path>,
        cache: Option<&[u8]>,
        fixed_directories: &[&str],
    ) -> Result<Vec<PathBuf>, &'static str> {
        if candidate.file_name() != Some(candidate.as_os_str())
            || candidate.as_os_str().as_bytes().contains(&0)
        {
            return Err("default library candidate is not a NUL-free basename");
        }
        let search_directories = validated_library_path_entries(library_path, current).map_err(
            |()| "LD_LIBRARY_PATH contains a relative, tokenized, or current-directory entry",
        )?;
        let canonical_current = current.and_then(|path| path.canonicalize().ok());
        let mut resolved = Vec::new();
        for directory in search_directories {
            push_canonical_candidate(
                &mut resolved,
                &directory.join(candidate),
                candidate.as_os_str(),
                canonical_current.as_deref(),
            );
        }
        if let Some(cache) = cache {
            for cached in parse_loader_cache(candidate.as_os_str(), cache) {
                push_canonical_candidate(
                    &mut resolved,
                    &cached,
                    candidate.as_os_str(),
                    canonical_current.as_deref(),
                );
            }
        }
        for directory in fixed_directories {
            push_canonical_candidate(
                &mut resolved,
                &Path::new(directory).join(candidate),
                candidate.as_os_str(),
                canonical_current.as_deref(),
            );
        }
        Ok(resolved)
    }

    fn push_canonical_candidate(
        resolved: &mut Vec<PathBuf>,
        raw: &Path,
        exact_basename: &OsStr,
        canonical_current: Option<&Path>,
    ) {
        if raw.file_name() != Some(exact_basename) || !raw.is_absolute() {
            return;
        }
        let Ok(canonical) = raw.canonicalize() else {
            return;
        };
        if !canonical.is_absolute()
            || !canonical
                .metadata()
                .is_ok_and(|metadata| metadata.is_file())
            || !canonical_target_matches_candidate(exact_basename, &canonical)
            || canonical_current.is_some_and(|cwd| canonical.parent() == Some(cwd))
            || resolved.contains(&canonical)
        {
            return;
        }
        resolved.push(canonical);
    }

    fn canonical_target_matches_candidate(candidate: &OsStr, canonical: &Path) -> bool {
        let candidate = candidate.as_bytes();
        if !matches!(
            candidate,
            b"libamdhip64.so.5" | b"libamdhip64.so.6" | b"libamdhip64.so.7"
        ) {
            // CUDA's `libcuda.so.1` commonly resolves to a driver-versioned
            // basename, and unversioned HIP is deliberately profile-neutral.
            return true;
        }
        let Some(target) = canonical.file_name().map(OsStrExt::as_bytes) else {
            return false;
        };
        target == candidate
            || target
                .strip_prefix(candidate)
                .is_some_and(|suffix| suffix.starts_with(b".") && suffix.len() > 1)
    }

    fn parse_loader_cache(candidate: &OsStr, bytes: &[u8]) -> Vec<PathBuf> {
        parse_loader_cache_checked(candidate, bytes).unwrap_or_default()
    }

    fn parse_loader_cache_checked(candidate: &OsStr, bytes: &[u8]) -> Option<Vec<PathBuf>> {
        if bytes.starts_with(NEW_CACHE_MAGIC) {
            return parse_new_loader_cache(candidate, bytes);
        }
        if !bytes.starts_with(OLD_CACHE_MAGIC) || bytes.len() < OLD_CACHE_HEADER_SIZE {
            return None;
        }
        let entries = usize::try_from(read_u32(bytes, 12)?).ok()?;
        if entries > MAX_LOADER_CACHE_ENTRIES {
            return None;
        }
        let entries_bytes = entries.checked_mul(OLD_CACHE_ENTRY_SIZE)?;
        let strings_offset = OLD_CACHE_HEADER_SIZE.checked_add(entries_bytes)?;
        if strings_offset > bytes.len() {
            return None;
        }
        let new_offset = align_to_eight(strings_offset)?;
        if bytes
            .get(new_offset..)
            .is_some_and(|tail| tail.starts_with(NEW_CACHE_MAGIC))
        {
            return parse_new_loader_cache(candidate, bytes.get(new_offset..)?);
        }
        parse_old_loader_cache(candidate, bytes, entries, strings_offset)
    }

    fn parse_new_loader_cache(candidate: &OsStr, bytes: &[u8]) -> Option<Vec<PathBuf>> {
        if bytes.len() < NEW_CACHE_HEADER_SIZE || !bytes.starts_with(NEW_CACHE_MAGIC) {
            return None;
        }
        let endian = bytes.get(28).copied()? & 3;
        #[cfg(target_endian = "little")]
        let expected_endian = 2;
        #[cfg(target_endian = "big")]
        let expected_endian = 3;
        if endian != 0 && endian != expected_endian {
            return None;
        }
        let entries = usize::try_from(read_u32(bytes, 20)?).ok()?;
        if entries > MAX_LOADER_CACHE_ENTRIES {
            return None;
        }
        let strings_length = usize::try_from(read_u32(bytes, 24)?).ok()?;
        let entries_bytes = entries.checked_mul(NEW_CACHE_ENTRY_SIZE)?;
        let strings_offset = NEW_CACHE_HEADER_SIZE.checked_add(entries_bytes)?;
        let strings_end = strings_offset.checked_add(strings_length)?;
        if strings_end > bytes.len() {
            return None;
        }

        let mut paths = Vec::new();
        for index in 0..entries {
            let offset =
                NEW_CACHE_HEADER_SIZE.checked_add(index.checked_mul(NEW_CACHE_ENTRY_SIZE)?)?;
            let key = usize::try_from(read_u32(bytes, offset.checked_add(4)?)?).ok()?;
            let hwcap = read_u64(bytes, offset.checked_add(16)?)?;
            let key = bounded_cache_string(bytes, key, strings_offset, strings_end)?;
            if hwcap == 0 && key == candidate.as_bytes() {
                let value = usize::try_from(read_u32(bytes, offset.checked_add(8)?)?).ok()?;
                let value = bounded_cache_string(bytes, value, strings_offset, strings_end)?;
                push_cache_match(&mut paths, value)?;
            }
        }
        Some(paths)
    }

    fn parse_old_loader_cache(
        candidate: &OsStr,
        bytes: &[u8],
        entries: usize,
        strings_offset: usize,
    ) -> Option<Vec<PathBuf>> {
        let mut paths = Vec::new();
        for index in 0..entries {
            let offset =
                OLD_CACHE_HEADER_SIZE.checked_add(index.checked_mul(OLD_CACHE_ENTRY_SIZE)?)?;
            let key = usize::try_from(read_u32(bytes, offset.checked_add(4)?)?).ok()?;
            let key = strings_offset.checked_add(key)?;
            let key = bounded_cache_string(bytes, key, strings_offset, bytes.len())?;
            if key == candidate.as_bytes() {
                let value = usize::try_from(read_u32(bytes, offset.checked_add(8)?)?).ok()?;
                let value = strings_offset.checked_add(value)?;
                let value = bounded_cache_string(bytes, value, strings_offset, bytes.len())?;
                push_cache_match(&mut paths, value)?;
            }
        }
        Some(paths)
    }

    fn push_cache_match(paths: &mut Vec<PathBuf>, value: &[u8]) -> Option<()> {
        let path = PathBuf::from(OsStr::from_bytes(value));
        if paths.contains(&path) {
            return Some(());
        }
        if paths.len() >= MAX_MATCHING_CACHE_PATHS {
            return None;
        }
        paths.push(path);
        Some(())
    }

    fn bounded_cache_string(
        bytes: &[u8],
        offset: usize,
        strings_offset: usize,
        strings_end: usize,
    ) -> Option<&[u8]> {
        if offset < strings_offset || offset >= strings_end || strings_end > bytes.len() {
            return None;
        }
        let bounded_end = offset
            .checked_add(MAX_LOADER_CACHE_STRING_BYTES)?
            .min(strings_end);
        let tail = bytes.get(offset..bounded_end)?;
        let length = tail.iter().position(|byte| *byte == 0)?;
        tail.get(..length)
    }

    fn read_u32(bytes: &[u8], offset: usize) -> Option<u32> {
        Some(u32::from_ne_bytes(
            bytes.get(offset..offset.checked_add(4)?)?.try_into().ok()?,
        ))
    }

    fn read_u64(bytes: &[u8], offset: usize) -> Option<u64> {
        Some(u64::from_ne_bytes(
            bytes.get(offset..offset.checked_add(8)?)?.try_into().ok()?,
        ))
    }

    fn align_to_eight(value: usize) -> Option<usize> {
        value.checked_add(7).map(|value| value & !7)
    }

    #[cfg(feature = "explicit-library-path")]
    pub(super) fn path_contains_nul(path: &Path) -> bool {
        path.as_os_str().as_bytes().contains(&0)
    }

    #[cfg(test)]
    mod tests {
        use super::{
            FIXED_LIBRARY_DIRECTORIES, NEW_CACHE_ENTRY_SIZE, NEW_CACHE_HEADER_SIZE,
            NEW_CACHE_MAGIC, OLD_CACHE_ENTRY_SIZE, OLD_CACHE_HEADER_SIZE, OLD_CACHE_MAGIC,
            parse_loader_cache, resolve_default_candidate_paths, validated_library_path_entries,
        };
        use std::ffi::OsStr;
        use std::fs;
        use std::os::unix::ffi::OsStrExt;
        use std::path::{Path, PathBuf};
        use std::sync::atomic::{AtomicUsize, Ordering};

        static NEXT_TEST_DIRECTORY: AtomicUsize = AtomicUsize::new(0);

        struct TestDirectory(PathBuf);

        impl TestDirectory {
            fn new() -> Self {
                loop {
                    let unique = NEXT_TEST_DIRECTORY.fetch_add(1, Ordering::Relaxed);
                    let path = std::env::temp_dir()
                        .join(format!("ocgpu-loader-{}-{unique}", std::process::id()));
                    match fs::create_dir(&path) {
                        Ok(()) => return Self(path),
                        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                        Err(error) => panic!("create isolated loader test directory: {error}"),
                    }
                }
            }
        }

        impl Drop for TestDirectory {
            fn drop(&mut self) {
                let _ = fs::remove_dir_all(&self.0);
            }
        }

        #[test]
        fn default_and_explicit_loads_reject_unsafe_dependency_search_entries() {
            let current = Path::new("/work/ocgpu");
            assert!(
                validated_library_path_entries(
                    Some(OsStr::new("relative:/opt/vendor")),
                    Some(current),
                )
                .is_err()
            );
            assert!(
                validated_library_path_entries(
                    Some(OsStr::new("/work/ocgpu:/opt/vendor")),
                    Some(current),
                )
                .is_err()
            );
            assert!(
                validated_library_path_entries(
                    Some(OsStr::new("/opt/vendor:/usr/lib")),
                    Some(current),
                )
                .is_ok()
            );
            assert!(validated_library_path_entries(Some(OsStr::new("")), Some(current),).is_err());
            assert!(
                validated_library_path_entries(
                    Some(OsStr::new("/opt/vendor;;/usr/lib")),
                    Some(current),
                )
                .is_err()
            );
            assert!(
                validated_library_path_entries(
                    Some(OsStr::new("/$ORIGIN/lib:/usr/lib")),
                    Some(current),
                )
                .is_err()
            );
            assert!(validated_library_path_entries(Some(OsStr::new("/opt/vendor")), None).is_err());
        }

        #[test]
        fn absolute_alias_of_cwd_is_rejected() {
            let current = std::env::current_dir().expect("test process has a current directory");
            let alias = current.join(".");
            let search = std::env::join_paths([alias]).expect("one absolute path is valid");
            assert!(validated_library_path_entries(Some(&search), Some(&current)).is_err());
        }

        #[test]
        fn relative_or_empty_rpath_cannot_contribute_a_cwd_candidate() {
            let root = TestDirectory::new();
            let candidate = Path::new("libocgpu-rpath-probe.so.1");
            let cwd_candidate = root.0.join(candidate);
            fs::write(&cwd_candidate, b"not an ELF object").expect("write isolated probe file");

            // RPATH/RUNPATH is intentionally not an input to the resolver. A
            // matching file in CWD therefore cannot become a dlopen target.
            let resolved =
                resolve_default_candidate_paths(candidate, None, Some(&root.0), None, &[])
                    .expect("fixed basename is valid");
            assert!(resolved.is_empty());
        }

        #[test]
        fn safe_absolute_library_path_is_canonicalized_but_cwd_is_not() {
            let root = TestDirectory::new();
            let current = root.0.join("current");
            let safe = root.0.join("safe");
            fs::create_dir(&current).expect("create isolated current directory");
            fs::create_dir(&safe).expect("create isolated safe directory");
            let candidate = Path::new("libocgpu-safe-probe.so.1");
            fs::write(safe.join(candidate), b"probe").expect("write isolated safe probe");
            fs::write(current.join(candidate), b"probe").expect("write isolated cwd probe");
            let search = safe.as_os_str();

            let resolved =
                resolve_default_candidate_paths(candidate, Some(search), Some(&current), None, &[])
                    .expect("absolute non-CWD search directory is accepted");
            assert_eq!(
                resolved,
                [safe.join(candidate).canonicalize().expect("probe exists")]
            );
        }

        #[test]
        fn versioned_hip_candidate_preserves_major_across_symlinks() {
            let root = TestDirectory::new();
            let current = root.0.join("current");
            let cross_major = root.0.join("cross-major");
            let same_major = root.0.join("same-major");
            fs::create_dir(&current).expect("create isolated current directory");
            fs::create_dir(&cross_major).expect("create cross-major directory");
            fs::create_dir(&same_major).expect("create same-major directory");

            let cross_major_target = cross_major.join("libamdhip64.so.6.4.0");
            let same_major_target = same_major.join("libamdhip64.so.7.2.0");
            fs::write(&cross_major_target, b"probe").expect("write HIP 6 target");
            fs::write(&same_major_target, b"probe").expect("write HIP 7 target");
            std::os::unix::fs::symlink(
                cross_major_target
                    .file_name()
                    .expect("HIP 6 target has a basename"),
                cross_major.join("libamdhip64.so.7"),
            )
            .expect("create cross-major HIP symlink");
            std::os::unix::fs::symlink(
                same_major_target
                    .file_name()
                    .expect("HIP 7 target has a basename"),
                same_major.join("libamdhip64.so.7"),
            )
            .expect("create same-major HIP symlink");

            let rejected = resolve_default_candidate_paths(
                Path::new("libamdhip64.so.7"),
                Some(cross_major.as_os_str()),
                Some(&current),
                None,
                &[],
            )
            .expect("versioned HIP candidate is valid");
            assert!(rejected.is_empty());

            let accepted = resolve_default_candidate_paths(
                Path::new("libamdhip64.so.7"),
                Some(same_major.as_os_str()),
                Some(&current),
                None,
                &[],
            )
            .expect("versioned HIP candidate is valid");
            assert_eq!(
                accepted,
                [same_major_target
                    .canonicalize()
                    .expect("same-major target exists")]
            );
        }

        #[test]
        fn hip_major_priority_precedes_every_absolute_search_source() {
            let root = TestDirectory::new();
            let current = root.0.join("current");
            let early_directory = root.0.join("early");
            let cached_directory = root.0.join("cached");
            fs::create_dir(&current).expect("create isolated current directory");
            fs::create_dir(&early_directory).expect("create early search directory");
            fs::create_dir(&cached_directory).expect("create cache search directory");
            let hip6 = early_directory.join("libamdhip64.so.6");
            let hip7 = cached_directory.join("libamdhip64.so.7");
            fs::write(&hip6, b"probe").expect("write HIP 6 probe");
            fs::write(&hip7, b"probe").expect("write HIP 7 probe");
            let cache = new_cache_bytes(&[(OsStr::new("libamdhip64.so.7"), &hip7)]);
            let mut ordered = Vec::new();

            for candidate in crate::Backend::Hip.candidates() {
                let paths = resolve_default_candidate_paths(
                    Path::new(candidate),
                    Some(early_directory.as_os_str()),
                    Some(&current),
                    Some(&cache),
                    &[],
                )
                .expect("generated HIP candidate is a fixed basename");
                ordered.extend(paths);
            }

            assert_eq!(
                ordered,
                [
                    hip7.canonicalize().expect("HIP 7 probe exists"),
                    hip6.canonicalize().expect("HIP 6 probe exists"),
                ]
            );
        }

        #[test]
        fn loader_cache_parser_is_bounded_and_requires_exact_basename() {
            let root = TestDirectory::new();
            let candidate = OsStr::new("libocgpu-cache-probe.so.1");
            let exact = root.0.join(candidate);
            let renamed = root.0.join("different-name.so");
            fs::write(&exact, b"probe").expect("write exact cache probe");
            fs::write(&renamed, b"probe").expect("write renamed cache probe");
            let new_cache = new_cache_bytes(&[(candidate, &exact), (candidate, &renamed)]);
            let old_cache = old_cache_bytes(&[(candidate, &exact)]);
            assert_eq!(
                parse_loader_cache(candidate, &new_cache),
                [exact.clone(), renamed]
            );
            assert_eq!(
                parse_loader_cache(candidate, &old_cache),
                std::slice::from_ref(&exact)
            );

            for end in 0..new_cache.len() {
                let _ = parse_loader_cache(candidate, &new_cache[..end]);
            }
            let resolved = resolve_default_candidate_paths(
                Path::new(candidate),
                None,
                Some(&root.0.join("unrelated-current")),
                Some(&new_cache),
                &[],
            )
            .expect("cache probe basename is valid");
            assert_eq!(
                resolved,
                [exact.canonicalize().expect("exact probe exists")]
            );

            let mut corrupt = new_cache;
            corrupt[20..24].copy_from_slice(&u32::MAX.to_ne_bytes());
            assert!(parse_loader_cache(candidate, &corrupt).is_empty());

            let excessive_paths: Vec<_> = (0..=super::MAX_MATCHING_CACHE_PATHS)
                .map(|index| PathBuf::from(format!("/cache/{index}/libocgpu-cache-probe.so.1")))
                .collect();
            let excessive_entries: Vec<_> = excessive_paths
                .iter()
                .map(|path| (candidate, path.as_path()))
                .collect();
            let excessive_cache = new_cache_bytes(&excessive_entries);
            assert!(parse_loader_cache(candidate, &excessive_cache).is_empty());
        }

        #[test]
        fn fixed_directories_include_supported_multiarch_locations() {
            #[cfg(target_arch = "x86_64")]
            assert!(FIXED_LIBRARY_DIRECTORIES.contains(&"/usr/lib/x86_64-linux-gnu"));
            #[cfg(target_arch = "aarch64")]
            assert!(FIXED_LIBRARY_DIRECTORIES.contains(&"/usr/lib/aarch64-linux-gnu"));
            assert!(FIXED_LIBRARY_DIRECTORIES.contains(&"/opt/rocm/lib"));
            assert!(FIXED_LIBRARY_DIRECTORIES.contains(&"/opt/rocm/lib64"));
            assert!(FIXED_LIBRARY_DIRECTORIES.contains(&"/usr/lib/wsl/lib"));
            assert!(FIXED_LIBRARY_DIRECTORIES.contains(&"/lib"));
            assert!(FIXED_LIBRARY_DIRECTORIES.contains(&"/usr/lib"));
        }

        fn new_cache_bytes(entries: &[(&OsStr, &Path)]) -> Vec<u8> {
            let mut strings = Vec::new();
            let mut offsets = Vec::new();
            let strings_offset = NEW_CACHE_HEADER_SIZE + entries.len() * NEW_CACHE_ENTRY_SIZE;
            for &(name, path) in entries {
                let key = strings_offset + strings.len();
                strings.extend_from_slice(name.as_bytes());
                strings.push(0);
                let value = strings_offset + strings.len();
                strings.extend_from_slice(path.as_os_str().as_bytes());
                strings.push(0);
                offsets.push((key, value));
            }
            let mut bytes = vec![0_u8; strings_offset];
            bytes[..NEW_CACHE_MAGIC.len()].copy_from_slice(NEW_CACHE_MAGIC);
            bytes[20..24].copy_from_slice(
                &u32::try_from(entries.len())
                    .expect("test entry count fits")
                    .to_ne_bytes(),
            );
            bytes[24..28].copy_from_slice(
                &u32::try_from(strings.len())
                    .expect("test string table fits")
                    .to_ne_bytes(),
            );
            for (index, (key, value)) in offsets.into_iter().enumerate() {
                let offset = NEW_CACHE_HEADER_SIZE + index * NEW_CACHE_ENTRY_SIZE;
                bytes[offset + 4..offset + 8]
                    .copy_from_slice(&u32::try_from(key).expect("test key fits").to_ne_bytes());
                bytes[offset + 8..offset + 12]
                    .copy_from_slice(&u32::try_from(value).expect("test value fits").to_ne_bytes());
            }
            bytes.extend_from_slice(&strings);
            bytes
        }

        fn old_cache_bytes(entries: &[(&OsStr, &Path)]) -> Vec<u8> {
            let mut strings = Vec::new();
            let mut offsets = Vec::new();
            for &(name, path) in entries {
                let key = strings.len();
                strings.extend_from_slice(name.as_bytes());
                strings.push(0);
                let value = strings.len();
                strings.extend_from_slice(path.as_os_str().as_bytes());
                strings.push(0);
                offsets.push((key, value));
            }
            let entries_end = OLD_CACHE_HEADER_SIZE + entries.len() * OLD_CACHE_ENTRY_SIZE;
            let mut bytes = vec![0_u8; entries_end];
            bytes[..OLD_CACHE_MAGIC.len()].copy_from_slice(OLD_CACHE_MAGIC);
            bytes[12..16].copy_from_slice(
                &u32::try_from(entries.len())
                    .expect("test entry count fits")
                    .to_ne_bytes(),
            );
            for (index, (key, value)) in offsets.into_iter().enumerate() {
                let offset = OLD_CACHE_HEADER_SIZE + index * OLD_CACHE_ENTRY_SIZE;
                bytes[offset + 4..offset + 8]
                    .copy_from_slice(&u32::try_from(key).expect("test key fits").to_ne_bytes());
                bytes[offset + 8..offset + 12]
                    .copy_from_slice(&u32::try_from(value).expect("test value fits").to_ne_bytes());
            }
            bytes.extend_from_slice(&strings);
            bytes
        }
    }
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
mod platform {
    use super::OpenFailure;
    use std::ffi::{CStr, c_void};
    use std::path::{Path, PathBuf};
    use std::ptr::NonNull;

    pub(super) unsafe fn open(
        path: &Path,
        _explicit: bool,
    ) -> Result<(NonNull<c_void>, PathBuf), OpenFailure> {
        Err(OpenFailure {
            candidate: path.to_path_buf(),
            os_code: None,
            message: "unsupported platform".to_owned(),
        })
    }

    pub(super) unsafe fn find(_handle: NonNull<c_void>, _symbol: &CStr) -> Option<NonNull<c_void>> {
        None
    }

    #[cfg(feature = "explicit-library-path")]
    pub(super) fn path_contains_nul(_path: &Path) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::{Backend, LoadError, TableSlotError};

    unsafe extern "C" fn mock_entry() {}

    #[repr(C)]
    #[derive(Default)]
    struct MockTable {
        prefix: [u32; 6],
        entry: Option<unsafe extern "C" fn(i32) -> i32>,
    }

    #[cfg(feature = "explicit-library-path")]
    #[test]
    fn explicit_path_loading_requires_an_unsafe_caller() {
        let _: unsafe fn(Backend, &std::path::Path) -> Result<&'static super::Library, LoadError> =
            super::load_from_absolute;
    }

    #[test]
    fn candidates_are_fixed_basenames() {
        for backend in [Backend::Cuda, Backend::Hip] {
            for candidate in backend.candidates() {
                let path = std::path::Path::new(candidate);
                assert_eq!(
                    path.file_name().and_then(|name| name.to_str()),
                    Some(*candidate)
                );
                assert!(!path.is_absolute());
                assert!(!candidate.contains('/'));
                assert!(!candidate.contains('\\'));
                assert!(!candidate.as_bytes().contains(&0));
            }
        }
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn hip_candidates_are_newest_first_and_major_specific() {
        assert_eq!(
            Backend::Hip.candidates(),
            &["amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll"]
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn hip_candidates_are_versioned_newest_first_before_unversioned_fallback() {
        assert_eq!(
            Backend::Hip.candidates(),
            &[
                "libamdhip64.so.7",
                "libamdhip64.so.6",
                "libamdhip64.so.5",
                "libamdhip64.so",
            ]
        );
    }

    #[test]
    fn structured_error_preserves_all_attempts() {
        let error = LoadError::BackendUnavailable {
            backend: Backend::Hip,
            attempts: vec![
                super::OpenFailure {
                    candidate: "first".into(),
                    os_code: Some(2),
                    message: "missing".to_owned(),
                },
                super::OpenFailure {
                    candidate: "second".into(),
                    os_code: Some(5),
                    message: "denied".to_owned(),
                },
            ],
        };
        let rendered = error.to_string();
        assert!(rendered.contains("first: missing"));
        assert!(rendered.contains("second: denied"));
    }

    #[test]
    fn generated_function_slot_is_written_after_validation() {
        let mut table = MockTable::default();
        let offset = std::mem::offset_of!(MockTable, entry);
        let address = std::ptr::NonNull::new(mock_entry as *const () as *mut std::ffi::c_void)
            .expect("function addresses are non-null");
        // SAFETY: `offset_of!` names the nullable function-pointer field, and
        // this test never calls the deliberately erased mock signature.
        unsafe { super::write_function_slot(&mut table, offset, address) }
            .expect("valid generated slot must be writable");
        assert!(table.entry.is_some());
    }

    #[test]
    fn generated_function_slot_rejects_invalid_offsets() {
        let mut table = MockTable::default();
        let address = std::ptr::NonNull::new(mock_entry as *const () as *mut std::ffi::c_void)
            .expect("function addresses are non-null");
        // SAFETY: this intentionally invalid offset is rejected before writing.
        let out_of_bounds = unsafe {
            super::write_function_slot(&mut table, std::mem::size_of::<MockTable>(), address)
        };
        assert!(matches!(
            out_of_bounds,
            Err(TableSlotError::OutOfBounds { .. })
        ));

        // SAFETY: this intentionally misaligned offset is rejected before writing.
        let misaligned = unsafe { super::write_function_slot(&mut table, 1, address) };
        assert!(matches!(misaligned, Err(TableSlotError::Misaligned { .. })));
    }

    #[cfg(feature = "explicit-library-path")]
    #[test]
    fn explicit_relative_path_is_rejected_before_loading() {
        let error = super::validate_explicit_path(std::path::Path::new("driver.dll"))
            .expect_err("relative paths must be rejected");
        assert!(matches!(
            error,
            LoadError::InvalidExplicitPath {
                reason: super::InvalidPathReason::NotAbsolute,
                ..
            }
        ));
    }

    #[cfg(feature = "explicit-library-path")]
    #[test]
    fn explicit_directory_is_not_a_library_file() {
        let root = std::env::current_dir().expect("test process has a current directory");
        let error = super::validate_explicit_path(&root)
            .expect_err("directories must not pass explicit path validation");
        assert!(matches!(
            error,
            LoadError::InvalidExplicitPath {
                reason: super::InvalidPathReason::NotAFile,
                ..
            }
        ));
    }
}
