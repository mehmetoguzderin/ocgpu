/* SPDX-License-Identifier: CC0-1.0 */

#include <ocgpu/ocgpu.h>
#include <stddef.h>
#include <stdint.h>

#define OCGPU_STATIC_ASSERT(name, expression) \
    typedef char ocgpu_static_assert_##name[(expression) ? 1 : -1]

OCGPU_STATIC_ASSERT(result_width, sizeof(ocgpuResult) == sizeof(int32_t));
OCGPU_STATIC_ASSERT(backend_width, sizeof(ocgpuBackend) == sizeof(uint32_t));
OCGPU_STATIC_ASSERT(device_width, sizeof(ocgpuDevice) == sizeof(int32_t));
OCGPU_STATIC_ASSERT(device_pointer_width, sizeof(ocgpuDeviceptr) == sizeof(uintptr_t));
OCGPU_STATIC_ASSERT(context_pointer_width, sizeof(ocgpuContext) == sizeof(void *));
OCGPU_STATIC_ASSERT(table_prefix_at_zero, offsetof(ocgpuApi_v1, struct_size) == 0u);
OCGPU_STATIC_ASSERT(table_abi_after_size,
                    offsetof(ocgpuApi_v1, abi_version) == sizeof(uint32_t));
OCGPU_STATIC_ASSERT(hip_profile_mask_value,
                    OCGPU_API_FLAG_HIP_RUNTIME_PROFILE_MASK == UINT32_C(0x00ff0000));
OCGPU_STATIC_ASSERT(hip5_profile_value,
                    OCGPU_API_FLAG_HIP_RUNTIME_PROFILE_5 == UINT32_C(0x00050000));
OCGPU_STATIC_ASSERT(hip6_profile_value,
                    OCGPU_API_FLAG_HIP_RUNTIME_PROFILE_6 == UINT32_C(0x00060000));
OCGPU_STATIC_ASSERT(hip7_profile_value,
                    OCGPU_API_FLAG_HIP_RUNTIME_PROFILE_7 == UINT32_C(0x00070000));
OCGPU_STATIC_ASSERT(hip_profiles_fit_mask,
                    ((OCGPU_API_FLAG_HIP_RUNTIME_PROFILE_5 |
                      OCGPU_API_FLAG_HIP_RUNTIME_PROFILE_6 |
                      OCGPU_API_FLAG_HIP_RUNTIME_PROFILE_7) &
                     ~OCGPU_API_FLAG_HIP_RUNTIME_PROFILE_MASK) == UINT32_C(0));

#if !defined(OCGPU_ENABLE_FLAT_C_EXPORTS)
static ocgpuResult compile_public_surface(void)
{
    ocgpuApi_v1 api = {0};
    return ocgpuGetApi(OCGPU_BACKEND_CUDA, OCGPU_ABI_VERSION_1,
                       sizeof(api), &api);
}
#endif

#if defined(OCGPU_ENABLE_FLAT_C_EXPORTS)
static int exercise_flat_stubs(void)
{
    if (ocgpuInit((ocgpuBackend)0, UINT32_C(0)) !=
        OCGPU_ERROR_INVALID_ARGUMENT) {
        return 1;
    }
    if (ocgpuCuInit(UINT32_C(0)) !=
        (ocgpuCUresult)OCGPU_ERROR_SYMBOL_UNAVAILABLE) {
        return 1;
    }
    if (ocgpuHipInit(UINT32_C(0)) !=
        (ocgpuHipError_t)OCGPU_ERROR_SYMBOL_UNAVAILABLE) {
        return 1;
    }
    return 0;
}
#endif

int main(void)
{
#if defined(OCGPU_ENABLE_FLAT_C_EXPORTS)
    return exercise_flat_stubs();
#else
    return compile_public_surface() == OCGPU_ERROR_INTERNAL ? 1 : 0;
#endif
}
