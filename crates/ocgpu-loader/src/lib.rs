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
    /// Candidate that was passed to the operating-system loader.
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
    // platform loading applies flags that exclude the current working directory,
    // and the caller guarantees that the native code is trusted and ABI-correct.
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
    use std::ffi::{CStr, CString, c_char, c_int, c_void};
    use std::os::unix::ffi::OsStrExt;
    use std::path::{Path, PathBuf};
    use std::ptr::NonNull;

    const RTLD_LOCAL: c_int = 0;
    const RTLD_NOW: c_int = 2;
    const RTLD_DI_LINKMAP: c_int = 2;

    #[repr(C)]
    struct LinkMap {
        address: usize,
        name: *mut c_char,
        dynamic: *mut c_void,
        next: *mut Self,
        previous: *mut Self,
    }

    #[link(name = "dl")]
    unsafe extern "C" {
        fn dlopen(file_name: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        fn dlerror() -> *const c_char;
        fn dlinfo(handle: *mut c_void, request: c_int, info: *mut c_void) -> c_int;
    }

    pub(super) unsafe fn open(
        path: &Path,
        explicit: bool,
    ) -> Result<(NonNull<c_void>, PathBuf), OpenFailure> {
        let candidate = path.to_path_buf();
        if has_cwd_search_entry(explicit) {
            return Err(OpenFailure {
                candidate,
                os_code: None,
                message: "LD_LIBRARY_PATH contains a relative or current-directory entry"
                    .to_owned(),
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
            .map(|handle| {
                let loaded_path = module_path(handle).unwrap_or_else(|| path.to_path_buf());
                (handle, loaded_path)
            })
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

    fn module_path(handle: NonNull<c_void>) -> Option<PathBuf> {
        let mut link_map: *mut LinkMap = std::ptr::null_mut();
        // SAFETY: `link_map` is writable pointer storage, the handle is live, and
        // `RTLD_DI_LINKMAP` requests a loader-owned `link_map` pointer.
        let result =
            unsafe { dlinfo(handle.as_ptr(), RTLD_DI_LINKMAP, (&raw mut link_map).cast()) };
        if result != 0 || link_map.is_null() {
            return None;
        }
        // SAFETY: successful `RTLD_DI_LINKMAP` returned a live loader-owned map.
        let name = unsafe { (*link_map).name };
        if name.is_null() {
            return None;
        }
        // SAFETY: `l_name` is a NUL-terminated string owned by the loaded object.
        let bytes = unsafe { CStr::from_ptr(name) }.to_bytes();
        (!bytes.is_empty()).then(|| PathBuf::from(std::ffi::OsStr::from_bytes(bytes)))
    }

    fn has_cwd_search_entry(explicit: bool) -> bool {
        let value = std::env::var_os("LD_LIBRARY_PATH");
        let current = std::env::current_dir().ok();
        has_cwd_search_entry_in(explicit, value.as_deref(), current.as_deref())
    }

    fn has_cwd_search_entry_in(
        _explicit: bool,
        value: Option<&std::ffi::OsStr>,
        current: Option<&Path>,
    ) -> bool {
        let Some(value) = value else {
            return false;
        };
        std::env::split_paths(value).any(|entry| {
            if entry.as_os_str().is_empty() || !entry.is_absolute() {
                return true;
            }
            current.is_some_and(|cwd| {
                if entry == cwd {
                    return true;
                }
                match (entry.canonicalize(), cwd.canonicalize()) {
                    (Ok(entry), Ok(cwd)) => entry == cwd,
                    _ => false,
                }
            })
        })
    }

    #[cfg(feature = "explicit-library-path")]
    pub(super) fn path_contains_nul(path: &Path) -> bool {
        path.as_os_str().as_bytes().contains(&0)
    }

    #[cfg(test)]
    mod tests {
        use super::has_cwd_search_entry_in;
        use std::ffi::OsStr;
        use std::path::Path;

        #[test]
        fn default_and_explicit_loads_reject_unsafe_dependency_search_entries() {
            let current = Path::new("/work/ocgpu");
            assert!(has_cwd_search_entry_in(
                true,
                Some(OsStr::new("relative:/opt/vendor")),
                Some(current),
            ));
            assert!(has_cwd_search_entry_in(
                true,
                Some(OsStr::new("/work/ocgpu:/opt/vendor")),
                Some(current),
            ));
            assert!(!has_cwd_search_entry_in(
                true,
                Some(OsStr::new("/opt/vendor:/usr/lib")),
                Some(current),
            ));
            assert!(has_cwd_search_entry_in(
                true,
                Some(OsStr::new("")),
                Some(current),
            ));
            assert!(has_cwd_search_entry_in(
                false,
                Some(OsStr::new("relative:/opt/vendor")),
                Some(current),
            ));
        }

        #[test]
        fn absolute_alias_of_cwd_is_rejected() {
            let current = std::env::current_dir().expect("test process has a current directory");
            let alias = current.join(".");
            let search = std::env::join_paths([alias]).expect("one absolute path is valid");
            assert!(has_cwd_search_entry_in(true, Some(&search), Some(&current),));
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
