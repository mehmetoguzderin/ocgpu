// SPDX-License-Identifier: CC0-1.0

use core::mem;
use core::ptr;
#[cfg(any(
    not(feature = "cuda"),
    not(feature = "hip"),
    not(feature = "nvrtc"),
    not(feature = "hiprtc")
))]
use ocgpu_abi::OCGPU_ERROR_BACKEND_NOT_FOUND;
use ocgpu_abi::{
    OCGPU_ABI_VERSION_1, OCGPU_BACKEND_CUDA, OCGPU_BACKEND_HIP, OCGPU_ERROR_ABI_MISMATCH,
    OCGPU_ERROR_INVALID_ARGUMENT, OCGPU_SUCCESS, ocgpuApi_v1, ocgpuBackend, ocgpuCuApi_v1,
    ocgpuHipApi_v1, ocgpuHiprtcApi_v1, ocgpuNvrtcApi_v1, ocgpuResult, ocgpuRtcApi_v1,
};
use std::panic::{AssertUnwindSafe, catch_unwind};

const TABLE_PREFIX_SIZE: usize = mem::size_of::<u32>() * 6;

pub(crate) unsafe fn get_api(
    backend: ocgpuBackend,
    requested_abi: u32,
    output_size: usize,
    output: *mut ocgpuApi_v1,
) -> ocgpuResult {
    boundary(|| {
        validate_request(requested_abi, output_size, output)?;
        let backend = match backend {
            OCGPU_BACKEND_CUDA => ocgpu_core::BackendKind::Cuda,
            OCGPU_BACKEND_HIP => ocgpu_core::BackendKind::Hip,
            _ => return Err(OCGPU_ERROR_INVALID_ARGUMENT),
        };
        let table = ocgpu_core::negotiated_common_table(backend).map_err(|error| error.result())?;
        // SAFETY: request validation checked non-null and the complete v1 size;
        // the C caller owns writable storage for the promised byte count.
        unsafe { write_table(output, output_size, &table) };
        Ok(())
    })
}

pub(crate) unsafe fn get_cuda_api(
    requested_abi: u32,
    output_size: usize,
    output: *mut ocgpuCuApi_v1,
) -> ocgpuResult {
    boundary(|| {
        validate_request(requested_abi, output_size, output)?;
        #[cfg(feature = "cuda")]
        {
            let table = ocgpu_core::negotiated_cuda_table().map_err(|error| error.result())?;
            // SAFETY: request validation established the output contract.
            unsafe { write_table(output, output_size, &table) };
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = output;
            Err(OCGPU_ERROR_BACKEND_NOT_FOUND)
        }
    })
}

pub(crate) unsafe fn get_hip_api(
    requested_abi: u32,
    output_size: usize,
    output: *mut ocgpuHipApi_v1,
) -> ocgpuResult {
    boundary(|| {
        validate_request(requested_abi, output_size, output)?;
        #[cfg(feature = "hip")]
        {
            let table = ocgpu_core::negotiated_hip_table().map_err(|error| error.result())?;
            // SAFETY: request validation established the output contract.
            unsafe { write_table(output, output_size, &table) };
            Ok(())
        }
        #[cfg(not(feature = "hip"))]
        {
            let _ = output;
            Err(OCGPU_ERROR_BACKEND_NOT_FOUND)
        }
    })
}

pub(crate) unsafe fn get_rtc_api(
    backend: ocgpuBackend,
    requested_abi: u32,
    output_size: usize,
    output: *mut ocgpuRtcApi_v1,
) -> ocgpuResult {
    boundary(|| {
        validate_request(requested_abi, output_size, output)?;
        match backend {
            OCGPU_BACKEND_CUDA => {
                #[cfg(feature = "nvrtc")]
                {
                    let table =
                        ocgpu_rtc::load_common(backend).map_err(|error| error.as_ocgpu_result())?;
                    // SAFETY: request validation established the output contract.
                    unsafe { write_table(output, output_size, &table) };
                    Ok(())
                }
                #[cfg(not(feature = "nvrtc"))]
                {
                    Err(OCGPU_ERROR_BACKEND_NOT_FOUND)
                }
            }
            OCGPU_BACKEND_HIP => {
                #[cfg(feature = "hiprtc")]
                {
                    let table =
                        ocgpu_rtc::load_common(backend).map_err(|error| error.as_ocgpu_result())?;
                    // SAFETY: request validation established the output contract.
                    unsafe { write_table(output, output_size, &table) };
                    Ok(())
                }
                #[cfg(not(feature = "hiprtc"))]
                {
                    Err(OCGPU_ERROR_BACKEND_NOT_FOUND)
                }
            }
            _ => Err(OCGPU_ERROR_INVALID_ARGUMENT),
        }
    })
}

pub(crate) unsafe fn get_nvrtc_api(
    requested_abi: u32,
    output_size: usize,
    output: *mut ocgpuNvrtcApi_v1,
) -> ocgpuResult {
    boundary(|| {
        validate_request(requested_abi, output_size, output)?;
        #[cfg(feature = "nvrtc")]
        {
            let table = ocgpu_rtc::load_nvrtc_raw().map_err(|error| error.as_ocgpu_result())?;
            // SAFETY: request validation established the output contract.
            unsafe { write_table(output, output_size, &table) };
            Ok(())
        }
        #[cfg(not(feature = "nvrtc"))]
        {
            let _ = output;
            Err(OCGPU_ERROR_BACKEND_NOT_FOUND)
        }
    })
}

pub(crate) unsafe fn get_hiprtc_api(
    requested_abi: u32,
    output_size: usize,
    output: *mut ocgpuHiprtcApi_v1,
) -> ocgpuResult {
    boundary(|| {
        validate_request(requested_abi, output_size, output)?;
        #[cfg(feature = "hiprtc")]
        {
            let table = ocgpu_rtc::load_hiprtc_raw().map_err(|error| error.as_ocgpu_result())?;
            // SAFETY: request validation established the output contract.
            unsafe { write_table(output, output_size, &table) };
            Ok(())
        }
        #[cfg(not(feature = "hiprtc"))]
        {
            let _ = output;
            Err(OCGPU_ERROR_BACKEND_NOT_FOUND)
        }
    })
}

fn validate_request<T>(
    requested_abi: u32,
    output_size: usize,
    output: *mut T,
) -> core::result::Result<(), ocgpuResult> {
    if output.is_null() {
        return Err(OCGPU_ERROR_INVALID_ARGUMENT);
    }
    if requested_abi != OCGPU_ABI_VERSION_1 || output_size < TABLE_PREFIX_SIZE {
        return Err(OCGPU_ERROR_ABI_MISMATCH);
    }
    Ok(())
}

unsafe fn write_table<T>(output: *mut T, output_size: usize, table: &T) {
    let bytes_to_copy = output_size.min(mem::size_of::<T>());
    // SAFETY: caller validation guarantees writable storage for `output_size`
    // bytes. Copying bytes avoids imposing an alignment precondition beyond the
    // C ABI's caller-provided buffer contract. The minimum accepted prefix
    // includes `struct_size`, so a smaller v1 caller can discover the producer's
    // complete table size while receiving every field it knows about.
    unsafe {
        ptr::copy_nonoverlapping(
            ptr::from_ref(table).cast::<u8>(),
            output.cast::<u8>(),
            bytes_to_copy,
        );
    }
}

fn boundary(operation: impl FnOnce() -> core::result::Result<(), ocgpuResult>) -> ocgpuResult {
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(Ok(())) => OCGPU_SUCCESS,
        Ok(Err(result)) => result,
        Err(_) => crate::OCGPU_MANAGEMENT_PANIC_RESULT,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        TABLE_PREFIX_SIZE, boundary, get_api, get_hiprtc_api, get_nvrtc_api, get_rtc_api,
        validate_request, write_table,
    };
    use ocgpu_abi::{
        OCGPU_ABI_VERSION_1, OCGPU_API_FLAG_HIP_RUNTIME_PROFILE_6, OCGPU_BACKEND_CUDA,
        OCGPU_ERROR_ABI_MISMATCH, OCGPU_ERROR_INVALID_ARGUMENT, ocgpuApi_v1, ocgpuHiprtcApi_v1,
        ocgpuNvrtcApi_v1, ocgpuRtcApi_v1,
    };

    #[test]
    fn null_outputs_are_rejected_without_loading_a_backend() {
        // SAFETY: this intentionally supplies a null output to verify that the
        // C boundary rejects it before dereference or backend loading.
        let result = unsafe {
            get_api(
                OCGPU_BACKEND_CUDA,
                OCGPU_ABI_VERSION_1,
                size_of::<ocgpuApi_v1>(),
                core::ptr::null_mut(),
            )
        };
        assert_eq!(result, OCGPU_ERROR_INVALID_ARGUMENT);

        // SAFETY: each call intentionally supplies a null output. Request
        // validation must reject it before attempting any RTC library load.
        unsafe {
            assert_eq!(
                get_rtc_api(
                    OCGPU_BACKEND_CUDA,
                    OCGPU_ABI_VERSION_1,
                    size_of::<ocgpuRtcApi_v1>(),
                    core::ptr::null_mut(),
                ),
                OCGPU_ERROR_INVALID_ARGUMENT
            );
            assert_eq!(
                get_nvrtc_api(
                    OCGPU_ABI_VERSION_1,
                    size_of::<ocgpuNvrtcApi_v1>(),
                    core::ptr::null_mut(),
                ),
                OCGPU_ERROR_INVALID_ARGUMENT
            );
            assert_eq!(
                get_hiprtc_api(
                    OCGPU_ABI_VERSION_1,
                    size_of::<ocgpuHiprtcApi_v1>(),
                    core::ptr::null_mut(),
                ),
                OCGPU_ERROR_INVALID_ARGUMENT
            );
        }
    }

    #[test]
    fn version_and_size_are_negotiated_before_dispatch() {
        let mut storage = core::mem::MaybeUninit::<ocgpuApi_v1>::uninit();
        let pointer = storage.as_mut_ptr();
        assert_eq!(
            validate_request(0, size_of::<ocgpuApi_v1>(), pointer),
            Err(OCGPU_ERROR_ABI_MISMATCH)
        );
        assert_eq!(
            validate_request(OCGPU_ABI_VERSION_1, TABLE_PREFIX_SIZE - 1, pointer),
            Err(OCGPU_ERROR_ABI_MISMATCH)
        );
        assert_eq!(
            validate_request(OCGPU_ABI_VERSION_1, TABLE_PREFIX_SIZE, pointer),
            Ok(())
        );
    }

    #[test]
    fn append_only_negotiation_copies_only_the_callers_known_prefix() {
        #[repr(C, align(8))]
        struct PrefixStorage([u8; 32]);

        let table = ocgpuApi_v1 {
            struct_size: u32::try_from(size_of::<ocgpuApi_v1>()).expect("table fits u32"),
            abi_version: OCGPU_ABI_VERSION_1,
            flags: OCGPU_API_FLAG_HIP_RUNTIME_PROFILE_6,
            ..ocgpuApi_v1::default()
        };
        let mut storage = PrefixStorage([0xa5_u8; 32]);
        // SAFETY: the first 24 bytes are writable and large enough for the v1
        // metadata prefix. The byte-copy implementation permits unaligned C
        // caller storage.
        unsafe {
            write_table(
                core::ptr::from_mut(&mut storage).cast::<ocgpuApi_v1>(),
                TABLE_PREFIX_SIZE,
                &table,
            );
        }
        assert_eq!(
            u32::from_ne_bytes(storage.0[0..4].try_into().expect("four bytes")),
            u32::try_from(size_of::<ocgpuApi_v1>()).expect("table fits u32")
        );
        assert_eq!(
            u32::from_ne_bytes(storage.0[12..16].try_into().expect("four bytes")),
            OCGPU_API_FLAG_HIP_RUNTIME_PROFILE_6,
            "the negotiated C-table prefix must preserve the HIP profile flags"
        );
        assert!(
            storage.0[TABLE_PREFIX_SIZE..]
                .iter()
                .all(|byte| *byte == 0xa5)
        );
    }

    #[test]
    fn unexpected_panics_are_contained_at_the_c_boundary() {
        let result = boundary(|| -> Result<(), ocgpu_abi::ocgpuResult> {
            panic!("synthetic internal failure")
        });
        assert_eq!(result, ocgpu_abi::OCGPU_ERROR_INTERNAL);
    }
}
