// SPDX-License-Identifier: CC0-1.0

// C99 remains the normative consumer ABI. This translation unit only proves
// that the generated header's advertised C++ compatibility mode remains clean.

#define OCGPU_ENABLE_FLAT_C_EXPORTS
#define OCGPU_ENABLE_CUDA
#define OCGPU_ENABLE_HIP
#include <ocgpu/ocgpu.h>

static_assert(OCGPU_ABI_VERSION_1 == 65536u, "ABI version encoding drifted");

int main() {
  const ocgpuResult result = ocgpuInit(static_cast<ocgpuBackend>(0), 0u);
  return result == OCGPU_ERROR_INVALID_ARGUMENT ? 0 : 1;
}
