<!-- SPDX-License-Identifier: CC0-1.0 -->

# Coverage and oracle maintenance

## Independent baselines

Coverage is evaluated against five committed snapshots:

| Inventory ID | Baseline | Committed file |
|---|---|---|
| `cuda-vendor-13.3-13030` | CUDA Driver API 13.3 Update 1, API 13030 | `oracle/vendor/cuda/13.3-13030.json` |
| `hip-general-7.14.60850` | General HIP 7.14.60850 | `oracle/vendor/hip/general-7.14.60850.json` |
| `hip-windows-7.2.0` | HIP SDK for Windows 7.2.0 | `oracle/vendor/hip/windows-7.2.0.json` |
| `cudarc-0.19.9` | `cudarc::driver::sys` 0.19.9 | `oracle/rust/cudarc-0.19.9.json` |
| `rocmrc-0.5.0` | `rocmrc::hip::sys` 0.5.0 | `oracle/rust/rocmrc-0.5.0.json` |

The vendor inventories decide whether an API exists. The Rust inventories
measure compatibility with exact ecosystem surfaces; they do not override a
vendor signature or platform statement. The two HIP inventories remain
separate because the Windows SDK and general HIP releases have different
baselines.

## HIP multi-major common compatibility

The generated HIP runtime-profile ledger reports a separate common-compatibility
metric of **3/3 reviewed profiles**: HIP 5, HIP 6, and HIP 7. Each profile must
provide all 26 common operations plus the single `hipRuntimeGetVersion`
bootstrap, and the unified attribute adapter is checked against 32 stable HIP
device-attribute values. The 26 operations comprise 25 declaration-compatible
calls and the reviewed `hipMemcpyHtoD` const-qualification adapter where the
legacy declaration differs. Common context synchronization is one of those
ABI-compatible calls but deliberately dispatches to `hipDeviceSynchronize`:
the exhaustive raw table still preserves independent exact fields for both
deprecated `hipCtxSynchronize` and `hipDeviceSynchronize`.

This bounded metric comes from
`oracle/vendor/hip/runtime-profiles.json`. It is independent of the exhaustive
current HIP 7 platform-union raw denominator: it neither adds legacy entries to
that denominator nor claims exhaustive HIP 5/6 raw-table coverage. It records
source and ABI compatibility evidence, not successful HIP 5/6 hardware
execution.

CUDA 13.4 Developer Preview is tracked only as a non-production future input.
ABI v1, canonical denominators, raw exports, and every published metric use the
production CUDA 13.3 Update 1 / API 13030 snapshot; preview declarations are not
silently treated as stable coverage.

Ordinary builds and reports are offline and read only committed data. The
maintainer workflow may consult public authoritative material and crates.io, but
it installs no GPU SDK or vendor compiler and never rewrites reviewed snapshots
without a human diff.

## Runtime-compiler coverage boundary

NVRTC and HIPRTC are intentionally not added to the five existing oracle
snapshots or their coverage denominators by the initial RTC capability. The
common RTC table, `ocgpuNvrtcApi_v1`, and `ocgpuHiprtcApi_v1` have ABI,
required-field, loader, state-machine, and mocked compilation tests, but no
exhaustive vendor-API percentage is claimed. Actual runtime compilation and
launch evidence is recorded separately per compiler and device backend; a CUDA
driver launch is not NVRTC evidence, and a HIP driver launch is not HIPRTC
evidence. A future exhaustive raw-compiler metric requires separately pinned,
reviewed vendor compiler inventories and classifications rather than silently
expanding a driver/runtime denominator.

Within the eleven-call common RTC profile, nine declarations are exact direct
intersections. The two HIPRTC pointer-array const-qualification adapters are
tracked separately and do not inflate exact-common coverage; raw HIPRTC keeps
the pinned source declarations unchanged.

The exact maintainer path is
`.github/scripts/extract-official-oracles.ps1`. It verifies both downloaded
archive hashes and extracted authoritative-header hashes, including CUDA
`cuda.h`/`cudaTypedefs.h` and the separate general/Windows HIP branches, before
producing review-only candidates. Those candidate files are never consumed by
an ordinary build until a reviewed commit replaces the canonical snapshot.

## Normal form

Every entry records an item kind, exact source name, canonical ABI signature,
SHA-256 signature hash, applicable target triples, aliases, version facts,
deprecation state, and evidence locator. Functions and callbacks additionally
record calling convention, ordered parameters, normalized type graphs, pointer
constness, input/output direction, and nullability. An exact source that does
not state an ordinary-pointer nullability contract is recorded explicitly as
`unspecified_by_source`; callbacks may not use that state and require an exact
nullable/non-null fact. Official-vendor records and unions carry Clang-derived
size, alignment, and field offsets for every applicable target. Layouts derived
from pinned Rust `repr(C)` declarations are attached to the exact
`type.oracle_variants` in the canonical manifest/generated inventory and are
enforced by committed Rust and strict-C99 assertions; pointer-only incomplete
tags are never treated as value layouts.

`oracle/vendor/function-union.json` is the deduplicated vendor-led callable
input to canonical raw generation. It groups identical spellings while
retaining separate version/platform/signature variants and exact fetched-source
hashes; the smaller pinned Rust surfaces cannot define the denominator.

`coverage/classifications.json` is intentionally separate from these upstream
facts. It contains exactly one decision and a substantive human reason for each
inventory item. Allowed decisions are `covered_exact`, `covered_adapter`,
`covered_raw_only`, `platform_unavailable`, `deprecated_covered`,
`intentionally_omitted`, `layout_unverified`, and `unrepresentable`. The schema
retains the two incomplete states so candidate updates can explain failures,
but the strict release check rejects either state; a final catalog must resolve
it to coverage, platform absence, deprecation coverage, or a concrete
unrepresentable ABI reason.

The validator rejects:

- missing, duplicate, orphaned, or non-specific classifications;
- a changed normalized signature or SHA-256 signature hash;
- invalid calling convention, parameter direction, pointer, or nullable facts;
- broken alias targets or asymmetric alias declarations;
- invalid, overlapping, or regressed target masks;
- inconsistent size, alignment, or offsets for the same target;
- classification references to absent canonical manifest IDs;
- canonical IDs absent from the generated implementation inventory;
- claimed implementation identifiers absent from generated/shipping Rust code;
- a difference between Windows `.def` and ELF version-script exports; and
- exported identifiers absent from the manifest, generated inventory, or the
  six version-negotiation management getters.

## Reports and embedding

`coverage/coverage.json` is deterministic minified schema-v1 JSON. It contains
separate metric records, classification counts, and a compact symbol list for
the installed CLI. `coverage/coverage.json.deflate` is raw DEFLATE of the exact
JSON bytes before the trailing newline, generated by `miniz_oxide` at level 10.
The CLI embeds and inflates those bytes; it needs neither source files nor a
registry, network, or GPU SDK.

`coverage/coverage.md` presents the metric table and the complete human reason
ledger. No blended percentage is produced. Declaration coverage, common exact
coverage, adapters, raw-only exposure, runtime resolution, bounded hardware
profile breadth, runner execution attestations, and layout verification measure
different facts and cannot honestly be collapsed into one score. The profile
metric records which generated slots the bounded test is designed to exercise;
the execution metric remains zero unless successful runner evidence is actually
published.

Run:

```sh
cargo run -p ocgpu-oracle -- vendor-union --check
cargo run -p ocgpu-oracle -- semantics --check
cargo run -p ocgpu-oracle -- classify --check
cargo run -p ocgpu-oracle -- check
```

`check` validates structure and semantics, classification completeness,
canonical-manifest/generated-implementation accounting, exact export controls,
and byte-for-byte report/DEFLATE freshness. It aggregates discoverable failures
so maintainers can review the entire drift set in one run.
