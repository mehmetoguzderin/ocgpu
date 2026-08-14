/* SPDX-License-Identifier: CC0-1.0 */

#include <ocgpu/ocgpu.h>
#include <stdio.h>
#include <string.h>

static int enumerate_backend(ocgpuBackend backend, const char *name)
{
    ocgpuApi_v1 api;
    ocgpuResult result;
    int32_t count = 0;

    memset(&api, 0, sizeof(api));
    result = ocgpuGetApi(backend, OCGPU_ABI_VERSION_1, sizeof(api), &api);
    if (result == OCGPU_ERROR_BACKEND_NOT_FOUND ||
        result == OCGPU_ERROR_BACKEND_TOO_OLD) {
        printf("%s unavailable\n", name);
        return 0;
    }
    if (result != OCGPU_SUCCESS) {
        fprintf(stderr, "%s table error: %d\n", name, (int)result);
        return 1;
    }
    if (api.ocgpuInit == NULL || api.ocgpuDeviceGetCount == NULL) {
        fprintf(stderr, "%s core table is incomplete\n", name);
        return 1;
    }
    result = api.ocgpuInit(0u);
    if (result != OCGPU_SUCCESS) {
        fprintf(stderr, "%s initialization error: %d\n", name, (int)result);
        return 1;
    }
    result = api.ocgpuDeviceGetCount(&count);
    if (result != OCGPU_SUCCESS) {
        fprintf(stderr, "%s enumeration error: %d\n", name, (int)result);
        return 1;
    }
    printf("%s devices: %d\n", name, (int)count);
    return 0;
}

int main(void)
{
    int failed = enumerate_backend(OCGPU_BACKEND_CUDA, "CUDA");
    failed |= enumerate_backend(OCGPU_BACKEND_HIP, "HIP");
    return failed;
}

