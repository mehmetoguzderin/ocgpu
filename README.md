<!-- SPDX-License-Identifier: CC0-1.0 -->

# ocgpu

Oxidized Compute-oriented GPU is an SDK-free, runtime-loaded CUDA Driver and
HIP driver-shaped compatibility API for Rust and ISO C99. It provides
backend-bound immutable function tables, typed Rust drivers, generated raw
CUDA/HIP tables, an optional flat C ABI, and an offline diagnostics CLI.

## Build and install

Rust consumers can build the workspace without a CUDA Toolkit, ROCm/HIP SDK,
vendor headers, import libraries, `nvcc`, `hipcc`, bindgen, or libclang:

```sh
cargo build --release -p ocgpu -p ocgpu-cli
```

Tagged releases attach attested Cargo source archives for `ocgpu-abi`,
`ocgpu-loader`, `ocgpu-cuda`, `ocgpu-hip`, and `ocgpu`. Creating those archives
does not claim that the same version has been published to a registry.

To stage the C package for local development, install the exact maintenance
tool and use the committed generated header:

```sh
cargo install cargo-c --version '0.10.24+cargo-0.98.0' --locked
cargo cinstall -p ocgpu-capi --release --destdir ./stage --prefix /usr --libdir /usr/lib
```

On Linux, plain `cargo cinstall` produces a development shared object without
the required `OCGPU_1.0` symbol version. Production packages must use the
mapped final-link and exact export/SONAME audit in
[`docs/release.md`](docs/release.md); the release workflow replaces that
development `.so` with the versioned artifact.

The C installation contains `ocgpu/ocgpu.h`, `ocgpu.pc`, a shared library, and
a static `ocgpu` archive. On Windows the shared consumer links
`ocgpu.dll.lib`; the static consumer links `ocgpu.lib` plus `ws2_32.lib`,
`ntdll.lib`, `userenv.lib`, and `dbghelp.lib`. Both forms dynamically resolve
the installed GPU runtime rather than embedding a vendor driver.

## Use

List backend availability and devices without selecting a process-global
backend:

```sh
cargo run -p ocgpu-cli -- backends
cargo run -p ocgpu-cli -- devices --backend all
cargo run -p ocgpu-cli -- doctor --json
```

Rust applications load a typed backend instance with
`ocgpu::Driver::<ocgpu::Cuda>::load()` or
`ocgpu::Driver::<ocgpu::Hip>::load()`. C applications call `ocgpuGetApi`,
`ocgpuCuGetApi`, or `ocgpuHipGetApi` to obtain an ABI-versioned table. See
`tests/c99/enumerate.c` for a complete allocation-free enumeration consumer.
One application binary can load a supported HIP 5, HIP 6, or HIP 7 runtime;
`ocgpu::backend_diagnostics` and `ocgpu backends --json` report the selected
`hip_5`, `hip_6`, or `hip_7` runtime profile.

The optional flat C ABI is built with cargo-c feature `flat-c-exports` and is
declared to consumers only when they define `OCGPU_ENABLE_FLAT_C_EXPORTS`
before including the header. Unified flat leaves take an explicit
`ocgpuBackend` first; there is no process-global selection. Raw `ocgpuCu*` and
`ocgpuHip*` leaves retain vendor-shaped arguments. A missing result-returning
leaf returns `OCGPU_ERROR_SYMBOL_UNAVAILABLE`; pointer, integer, POD/aggregate,
and void leaves use null, zero, an all-zero value, and no-op sentinels,
respectively, exactly as documented by the generated manifest and header.
Versioned function tables remain the
recommended fast path because flat leaves add a dispatch trampoline.

## Runtime requirements and limits

The target machine still needs a compatible driver/runtime: `libcuda.so.1` or
`nvcuda.dll` for CUDA, and a supported HIP runtime: versioned
`libamdhip64.so.7`, `.so.6`, or `.so.5` (with unversioned `libamdhip64.so` as a
verified Linux fallback), or `amdhip64_7.dll`, `amdhip64_6.dll`, or legacy
`amdhip64.dll` on Windows.
The HIP loader binds each versioned library name to an explicit major profile
and verifies the runtime-reported version before exposing callable entries.
The Linux unversioned fallback makes no filename-based major claim and is
classified solely by a supported runtime-reported version. The loader never
treats same-named exports from different HIP majors as ABI proof.
Missing libraries and optional symbols are reported as capabilities or
structured errors; they never trigger SDK discovery.

The common profile accepts these closed `hipRuntimeGetVersion` ranges:

| Profile | Reviewed floor | Accepted range |
|---|---:|---:|
| HIP 5 | 5.7.0 (`50_731_541`) | `50_731_541..=59_999_999` |
| HIP 6 | 6.1.2 (`60_140_093`) | `60_140_093..=69_999_999` |
| HIP 7 | 7.2.0 (`70_253_210`) | `70_253_210..=79_999_999` |

Later minor and patch releases in the same major are accepted under the
reviewed compatibility policy; earlier builds and unknown majors fail closed.
These are common-profile floors, not exhaustive raw-inventory claims.

The typed `Driver<Hip>` API and the common `ocgpuApi_v1` C table have one
version-independent core contract. The raw `ocgpuHipApi_v1` table retains the
current 535-slot HIP 7 platform-union layout: general/Linux 7.14.60850 and
Windows 7.2.0 inventories with target masks. The exhaustive raw inventory is
enabled only at the reviewed platform baseline (general/Linux 7.14.60850
(`71_460_850`) or Windows 7.2.0 (`70_253_210`)). Earlier supported HIP 7
versions, HIP 6, and HIP 5 expose
only entries reviewed for that selected profile/version. Other raw slots remain
null and diagnostics report `profile_unavailable`. A `direct_adapter`
diagnostic means an exact legacy export backs the common operation through a
typed shim, while its differently typed current-HIP raw slot remains null.
Consumers must test nullable raw entries rather than
assuming that selecting HIP makes every raw operation available. The common
profile is validated as a whole, so a successfully constructed typed/common
driver does not require such per-call checks.

Rust can branch on the validated major without a second loader pass:

```rust
let driver = ocgpu::Driver::<ocgpu::Hip>::load()?;
match driver.runtime_profile() {
    ocgpu::HipRuntimeProfile::Hip5 => { /* HIP 5 capability policy */ }
    ocgpu::HipRuntimeProfile::Hip6 => { /* HIP 6 capability policy */ }
    ocgpu::HipRuntimeProfile::Hip7 => { /* HIP 7 capability policy */ }
}
# Ok::<(), ocgpu::Error>(())
```

C consumers decode the same profile from either negotiated HIP table:

```c
ocgpuApi_v1 api = {0};
ocgpuResult result = ocgpuGetApi(OCGPU_BACKEND_HIP, OCGPU_ABI_VERSION_1,
                                 sizeof(api), &api);
if (result == OCGPU_SUCCESS) {
    uint32_t profile = api.flags & OCGPU_API_FLAG_HIP_RUNTIME_PROFILE_MASK;
    if (profile == OCGPU_API_FLAG_HIP_RUNTIME_PROFILE_6) {
        /* HIP 6 capability policy; the validated common core is callable. */
    }
}
```

CUDA table flags remain zero. Bits outside
`OCGPU_API_FLAG_HIP_RUNTIME_PROFILE_MASK` are reserved for independent table
capabilities and must be preserved when interpreting the profile.

Host-side SDK freedom does not compile device code. Applications provide
precompiled PTX/cubin/fatbin for CUDA or an architecture-compatible HIP/AMDGPU
code object. Runtime compilation, 32-bit targets, vendor-static linking, and
automatic driver installation are outside ABI v1.

The Rust-only `explicit-library-path` feature exposes unsafe
`Driver::<B>::load_from_absolute`: callers must trust the selected library and
its dependency closure, constructors, exact vendor ABI, and file identity
between validation and load. The C ABI v1 intentionally has no explicit-path
getter and always uses secure default discovery.

Developer commands, coverage maintenance, deployment constraints, and evidence
are documented in `docs/developer-guide.md`, `docs/coverage.md`,
`docs/operations.md`, and `docs/traceability.md`. Independently authored project
material is dedicated under CC0-1.0; third-party terms and the release SBOM are
retained in `LICENSES/`.
