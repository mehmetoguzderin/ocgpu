<!-- SPDX-License-Identifier: CC0-1.0 -->

# Local runtime validation, 2026-09-12

This records local Windows x86_64 results, independently of hosted CI. The
machine exposes an NVIDIA GeForce RTX 3060 Laptop GPU (`compute_86`) and AMD
Radeon Graphics (`gfx90c`, HIP 5 profile).

| Invocation | Result |
|---|---|
| NVRTC-only, compile-only | Passed in 0.21 s; NVRTC 12.4 produced 360 PTX bytes and 2,728 CUBIN bytes |
| NVRTC-only, compile and CUDA execution | Passed in 0.35 s; all six common extensions and one no-op launch |
| HIP driver with reviewed precompiled code | Passed in 8.86 s; all six common extensions and one no-op launch |
| Simultaneous CUDA + HIP drivers with precompiled code | Passed in 0.80 s; both backends validated all six extensions and completed one no-op launch each |
| HIPRTC-only, compile-only, compiler default | Passed in 2.51 s; reported HIPRTC 9.0, generated 3,832-byte V6 code object |
| HIPRTC-only, compile-only, explicit V4 | Passed in 3.25 s; generated 3,576-byte V4 code object and 5,016 bitcode bytes |
| HIPRTC-only, compile and HIP execution, V4 | Passed in 3.82 s; all six common extensions and one generated no-op launch |
| Simultaneous NVRTC + HIPRTC, compile-only | Passed in 3.21 s; both workers reached compile/code rendezvous and validated PTX, CUBIN, HIP code object, and bitcode |
| Simultaneous NVRTC + HIPRTC, compile and execution | Passed in 4.24 s; compile/code/launch rendezvous, all six extensions and one generated no-op launch per GPU |

The precompiled dual-driver baseline run occurred September 8; all three RTC
modes were validated September 12. Default HIPRTC discovery originally failed.
After installing the compiler components
from the official ROCm 10 tarball, explicit-path compilation and execution
succeeded. These local results do not claim hosted runner attestations.

The mixed-version installation exposed a dependency-order issue: compiling
with HIPRTC before loading the HIP 5 driver made `ocgpuMemGetInfo` return HIP
status 1. Module inspection showed the driver had bound its implicit
`amd_comgr.dll` import to HIPRTC's newer DLL. Loading the driver first binds its
System32 COMGR; HIPRTC subsequently loads its own COMGR by absolute path. The
execution harness now uses this order, and both DLLs coexist successfully.
Compile-only runs still load no GPU driver through ocgpu.

This compiler defaults to code-object V6. The execution tests explicitly pass
`-mcode-object-version=4` for the installed HIP 5 runtime and assert ELF ABI byte
8 equals 2. The option and ELF ABI interpretation follow the
[LLVM AMDGPU documentation](https://llvm.org/docs/AMDGPUUsage.html#elf-header).

The compiler tests build with `--no-default-features` and select only
`nvrtc,explicit-library-path`, `hiprtc,explicit-library-path`, or
`nvrtc,hiprtc,explicit-library-path`. The CPU-only dual-worker regressions
exercise both compile/code rendezvous and the additional launch rendezvous.
Vendor-mocked tests validate independent NVRTC/HIPRTC output routing, all
required exports, optional outputs, lifecycle errors, and allocation bounds.

Both hardware harnesses use `xtask`'s 45-second process watchdog. Runtime
compilation uses fixed source, nested injected headers, and a checked
preprocessor option; deliberately invalid source checks compiler diagnostics.
The kernel takes no arguments and performs no memory access. Each execution
uses at most two 64-byte allocations per backend for copy checks and launches
one thread once. The common extension checks cover memory information,
device-to-device copy and readback, stream/event queries, an event dependency,
and elapsed event time. The event wait is enqueued on a second nonblocking
stream before host synchronization, then event completion is checked after
that stream completes. Separate fixed-source programs exercise nonempty CUBIN
and HIP bitcode, including exact-size and undersized allocation limits and
repeated output retrieval; they add no GPU launches.

## Selected files

NVRTC was selected explicitly from
`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin`. Only runtime
DLLs were loaded; SDK headers and compiler executables were not used.

HIPRTC components were installed into
`target/rocm-10.0.0/install/bin` from the Windows tarball linked by the
[ROCm 10 installation page](https://rocm.docs.amd.com/en/latest/install/rocm.html):
[therock-dist-windows-multiarch-10.0.0.tar.gz](https://stable.repo.amd.com/rocm/core/tarball/therock-dist-windows-multiarch-10.0.0.tar.gz).
The downloaded archive is 4,796,456,804 bytes with SHA-256
`ebe454fe9ad663655177462187a4c86c72fd0537638f6cbea34660ddebf40056`.
The extracted runtime DLLs are `hiprtc0715.dll`, `hiprtc-builtins0715.dll`, and
`amd_comgr.dll`. Archive HIP headers and package metadata were inspected for
ABI verification; builds and runtime compilation do not consume them.

The HIP package is 7.15.26333, while its `hiprtcVersion` implementation reports
9.0. The wrapper preserves the vendor-reported value; it does not infer runtime
profile or package identity from that value. All 18 exported HIPRTC calls were
checked against the pinned machine ABI. PE imports contain Windows/runtime
dependencies; HIPRTC has no static dependency on the HIP driver.

| File | SHA-256 |
|---|---|
| `nvrtc64_120_0.dll` | `e55461f252b519478e9bf09f8bd1f0f3b6e3064c9270fac6a29d6eb23216835c` |
| `nvrtc-builtins64_124.dll` | `3ba326d21bb242488c0284ebc32fd6dbc94025ecab34862015a5c95eb178c961` |
| Staged `hiprtc0715.dll` | `19b67aa86ba371c49b2def5e5c863346e81c1b6d67caab125cf10a8c405e0043` |
| Staged `hiprtc-builtins0715.dll` | `97a4707260342e6563423a0d586e18744f7ce0de06b8634dc6dde29eceadd196` |
| Staged `amd_comgr.dll` | `213db0bedaa07027ce136ba1797299a2b0cfa2c6bb33584ce3861be8be067937` |
| `C:\Windows\System32\amdhip64.dll` | `f6a64adfef5336490b530942cd2b22fa5d7c07d20c0d4cfb01e5b8c93ba20c94` |
| `C:\Windows\System32\amd_comgr.dll` | `4a00c06240b158d591415273993fea3f16788099341df47d44948ab0cfd79f66` |
| Local `ocgpu_noop_gfx90c_llvm17_x64_roundtrip.hsaco` | `c80071b0d2c4a3472997d4fd4222861dce40bddf9e4faa0989432931af83c8b1` |

The local HIP module is 2,496 bytes. Static inspection confirmed `gfx90c`, no
kernel arguments, zero shared/private storage, a one-thread workgroup limit,
and a four-byte function body `00 00 81 bf`. That body is the GCN `s_endpgm`
encoding documented by the [LLVM instruction test](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.3/llvm/test/MC/AMDGPU/sopp.s).
The module is an existing local test artifact and is not a committed portable
HIP fixture.

## Reproduce compiler validation

Set `OCGPU_RUN_RTC_HARDWARE_SMOKE=1`, choose `OCGPU_RTC_SMOKE_BACKEND=cuda`,
`hip`, or `both`, and supply the corresponding absolute
`OCGPU_NVRTC_LIBRARY`/`OCGPU_HIPRTC_LIBRARY` paths and
`OCGPU_NVRTC_ARCH=compute_86`/`OCGPU_HIPRTC_ARCH=gfx90c` targets for this host.
For each mode, first run `cargo run -p xtask -- rtc-hardware-smoke` with
`OCGPU_RTC_COMPILE_ONLY=1`. After that succeeds, remove the compile-only
variable and run the same command once. Other devices require their own
validated architecture values.

For this installed host, run from the repository root in PowerShell:

```powershell
$env:OCGPU_RUN_RTC_HARDWARE_SMOKE='1'
$env:OCGPU_NVRTC_LIBRARY='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvrtc64_120_0.dll'
$env:OCGPU_HIPRTC_LIBRARY=(Resolve-Path target/rocm-10.0.0/install/bin/hiprtc0715.dll).Path
$env:OCGPU_NVRTC_ARCH='compute_86'
$env:OCGPU_HIPRTC_ARCH='gfx90c'
$env:OCGPU_HIPRTC_CODE_OBJECT_VERSION='4'
$env:OCGPU_RTC_SMOKE_BACKEND='both' # also run separately with cuda and hip
$env:OCGPU_RTC_COMPILE_ONLY='1'
cargo run -p xtask -- rtc-hardware-smoke
# After the compile-only command succeeds:
Remove-Item Env:OCGPU_RTC_COMPILE_ONLY
cargo run -p xtask -- rtc-hardware-smoke
```

`ocgpu compilers --backend cuda|hip|all --json` inspects secure default compiler
discovery independently of GPU driver availability. It does not apply the
hardware harness's explicit library-path variables.

## Repository validation

The common surface grew from 26 to 32 calls (26 required, six optional), with
the original common prefix and all 573 CUDA/535 HIP raw slots preserved.
Generated coverage reports 23 exact and nine adapted common operations, and
bounded hardware-profile breadth increased from 114 to 143 function entries.
Compiler coverage remains separate from the driver oracle denominators.

The local full `xtask ci` passed formatting, strict Clippy, feature checks,
workspace tests, release/flat builds, and warning-denying documentation.
Strict C99/C++ consumers passed. The all-feature release DLL exports exactly
the 1,146 names in `exports/ocgpu-flat.def`, including all six new common calls;
its PE imports contain no CUDA/HIP/RTC/COMGR or other GPU SDK dependencies.

All seven pinned HIP releases passed a fresh source/archive hash replay and
declaration extraction. The rebuilt optional/core profile snapshot matches
the committed SHA-256
`fb220a52315d7780008d60f5a669791849b0239f1beea39005a194229b7943f5`.
Locked offline Rust 1.85 checks passed for the shipping all-feature/minimal
graphs and the hardware harnesses.

Local command transcripts are retained at `target/rtc-validation-final-matrix.log`,
`target/rtc-final-ci.log`, and `target/hip-optional-profile-replay-current.log`.
PowerShell transcripts omit native child output on this host; the observed
test results are recorded above.
