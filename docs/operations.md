<!-- SPDX-License-Identifier: CC0-1.0 -->

# Operations guide

## Deployment contract

Deploy either the shared or static `ocgpu` library plus `ocgpu/ocgpu.h`. In both
forms, vendor entrypoints are resolved at runtime; a static `ocgpu` archive does
not statically contain or link a GPU driver. The host must already have a
compatible display/compute driver that supplies one of these libraries:

| Backend | Linux | Windows |
|---|---|---|
| CUDA | `libcuda.so.1` | `nvcuda.dll` |
| HIP 7 | `libamdhip64.so.7` | `amdhip64_7.dll` |
| HIP 6 | `libamdhip64.so.6` | `amdhip64_6.dll` |
| HIP 5 | `libamdhip64.so.5` | `amdhip64.dll` |

Linux also tries unversioned `libamdhip64.so` after the versioned candidates.
Whichever library is found must report the same HIP major as the candidate
profile. A filename/runtime mismatch or an unsupported future major is an ABI
mismatch, not backend absence; a runtime older than the supported HIP 5 range
is reported as too old.

Common-profile support starts at the exact runtime-reported floors below. Later
minor and patch releases within the same major are accepted; earlier builds and
unknown majors fail closed.

| Profile | Reviewed floor | Accepted `hipRuntimeGetVersion` range |
|---|---:|---:|
| HIP 5 | 5.7.0 (`50_731_541`) | `50_731_541..=59_999_999` |
| HIP 6 | 6.1.2 (`60_140_093`) | `60_140_093..=69_999_999` |
| HIP 7 | 7.2.0 (`70_253_210`) | `70_253_210..=79_999_999` |

The exhaustive raw-inventory floors are separate: general/Linux requires
`71_460_850` (7.14.0), while Windows requires `70_253_210` (7.2.0). Supported
runtimes below those raw floors expose the reviewed common/profile subset and
leave the other current-layout slots null.

Default lookup never searches the current working directory. The Rust-only
`explicit-library-path` feature provides unsafe
`Driver::<B>::load_from_absolute` for an absolute path. Its caller must trust
the library and dependency closure, constructors, exact vendor ABI, and file
identity between validation and OS loading. C ABI v1 deliberately has no
explicit-path getter and always uses secure default discovery; a future C
override would require an append-only, version-negotiated design. The first
initialization attempt for each backend, including its result and chosen path,
wins for the process lifetime. Loaded library handles then remain live so
issued function and GPU handles cannot outlive their code.

Applications must ship suitable precompiled device code: PTX, cubin, or fatbin
for CUDA and an appropriate HSACO/code object or HIP fat binary for HIP. NVRTC
and HIPRTC are outside the core deployment contract.

On Windows, link `ocgpu.dll.lib` when deploying `ocgpu.dll`. Link `ocgpu.lib`
for a static `ocgpu` consumer and include the Rust/OS support libraries
`ws2_32.lib`, `ntdll.lib`, `userenv.lib`, and `dbghelp.lib`. These are operating-system
dependencies; neither link mode imports a CUDA or HIP library.

## Health checks

Fully offline diagnostics load no GPU runtime:

```sh
ocgpu abi
ocgpu coverage
```

Symbol diagnostics load the selected runtime, query its version, and resolve
exports, but do not call backend initialization or enumerate devices:

```sh
ocgpu symbols --backend cuda --missing
ocgpu symbols --backend hip --all
```

Operational health checks initialize backend-bound drivers and enumerate
devices. They do not select a process-global backend, but they are runtime
operations rather than read-only library inspection:

```sh
ocgpu backends
ocgpu devices --backend all
ocgpu doctor --strict
```

Use `--json` for monitoring. Treat a missing backend as an availability state
when that backend is optional. Treat ABI mismatch, a too-old runtime, missing
core symbols, unexpected architecture, or a strict-doctor failure as a rollout
blocker. Optional missing symbols remain null in the backend-bound table and
must yield `OCGPU_ERROR_SYMBOL_UNAVAILABLE` when requested through a convenience
path; they must never panic. For HIP, record `runtime_profile` as well as the
numeric runtime version. `profile_unavailable` means the target platform
applies but the raw operation has no reviewed ABI in the selected HIP major; it
is distinct from both a missing export and `platform_unavailable`.
`direct_adapter` means an exact legacy export backs the validated common
operation through a typed shim while the differently typed current-HIP raw
slot remains null.

The common table and typed Rust driver validate their complete core before use
and are the portable HIP 5/6/7 surface. The raw HIP table keeps one stable ocgpu
layout, but runtimes below the platform's current raw-inventory baseline
populate only ABI-reviewed profile entries. Consumers of a
nullable raw entry must branch on its presence. Selecting a profile/version
below that raw baseline does not promise exhaustive coverage of the current HIP 7 platform-union layout
(general/Linux 7.14.60850 plus Windows 7.2.0, with target masks).

## Safe incident procedure

1. Capture `ocgpu doctor --json`, `ocgpu abi --json`, and the relevant `ocgpu
   symbols --backend ... --all --json` output.
2. Record OS and process architecture, `ocgpu` release checksum, selected
   backend, reported driver/runtime version, and loaded absolute library path.
3. Reproduce with enumeration only. Do not reset a GPU, unload/reload a display
   driver, change clocks or power limits, install a driver, or start a stress
   workload on a production/display machine.
4. Compare the missing entry against the target inventory and the reported HIP
   runtime profile/version. On HIP 5/6 or an earlier supported HIP 7 release,
   confirm whether the entry is deliberately
   `profile_unavailable`; the exhaustive raw denominator remains the committed
   HIP 7 platform union with its exact general/Linux 7.14.60850 and Windows
   7.2.0 baselines.
5. If the unsafe Rust explicit-path API is in use, remove it and retest default
   secure lookup before escalating. There is no C ABI v1 path override.
6. Preserve logs and process exit status. Function tables and backend libraries
   cannot be hot-unloaded; restart only the affected application process during
   an approved maintenance window.

The declared negative `OCGPU_ERROR_*` management codes and `OCGPU_SUCCESS` are
stable numeric results. Once an operation dispatches, a non-success result may
instead be the originating backend's native CUDA or HIP status code; record and
interpret it together with the backend identity. Do not scrape localized text
for alerting. Never print pointer values, kernel arguments, device memory, or
secrets in routine logs.

## Hardware smoke policy

Hardware tests are discovered and run by ordinary test commands, but explicitly
record a capability skip before any device operation unless
`OCGPU_RUN_HARDWARE_SMOKE=1`. Only the manually dispatched workflow on labelled,
dedicated self-hosted runners sets that value. Each job is single-run, bounded to
15 minutes, and selects one backend explicitly. CUDA uses the tiny committed PTX
fixture. AMD Linux and Windows dispatch inputs separately name absolute,
runner-local `OCGPU_HIP_SMOKE_MODULE` paths so an OS-specific path can never be
reused accidentally on the other platform. Each must be a reviewed,
architecture-matched code object; no generic committed HIP binary is claimed.
An enabled test allocates only the fixture's
minimal buffer, verifies one host/device round trip and one no-op launch,
synchronizes, and releases resources.

The smoke test does not install or update drivers, reset devices or contexts it
did not create, alter display state, change persistence/power/clock settings,
loop a workload, or request a reboot. A runner serving an interactive display
must not carry an `ocgpu` hardware label.

The workflow runs `doctor --json` without `--strict` because each dedicated
runner intentionally selects one backend and strict doctor requires every
compiled backend. The selected backend is enforced by the bounded test and by
an exact target-applicable symbol-resolution assertion.

## Rollback

`ocgpu` has no persistent service state or database migration. Roll back by
restoring the prior signed/checksummed application package and restarting only
that application. Do not downgrade the machine's GPU driver as part of an
`ocgpu` rollback. Tables are ABI-versioned and append-only, so callers may also
request their older supported ABI from a newer library.
