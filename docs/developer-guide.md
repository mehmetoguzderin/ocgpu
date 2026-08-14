<!-- SPDX-License-Identifier: CC0-1.0 -->

# Developer guide

## Prerequisites

Install the Rust toolchain named in `rust-toolchain.toml`. A normal checkout does
not need a CUDA Toolkit, ROCm/HIP SDK, vendor header, vendor compiler, import
library, `bindgen`, Clang, or libclang. GCC or Clang is needed only for the strict
C99 checks; the maintenance check also compiles the advertised header-only C++
mode while C99 remains the normative consumer ABI. `cargo-c
0.10.24+cargo-0.98.0` is needed only when staging the C package. On Linux,
plain cargo-c output is a development artifact; production packaging must use
the mapped final-link and export/SONAME audit documented in `docs/release.md`.

The publishable Rust source chain is `ocgpu-abi`, `ocgpu-loader`, `ocgpu-cuda`,
`ocgpu-hip`, then `ocgpu`; all inter-crate dependencies carry an explicit
Cargo-compatible `0.1.0` registry requirement as well as a workspace path.
`ocgpu-capi` and `ocgpu-cli` are shipping C and binary packages but are not
registry packages. `ocgpu-codegen`,
`ocgpu-oracle`, and `xtask` are repository maintenance tools. In particular,
only `ocgpu-oracle` declares the exact-pinned `cudarc 0.19.9` and `rocmrc
0.5.0` coverage references. They sit behind an always-false target dependency:
Cargo still resolves and downloads their registry sources for metadata-based
extraction, but neither crate nor its runtime loader dependencies are compiled.

## Reproducible commands

Run commands at the workspace root:

```sh
cargo run -p xtask -- generate
cargo run -p xtask -- check
cargo run -p xtask -- test
cargo run -p xtask -- ci
```

`generate` first refreshes the official-vendor callable union, renders every
artifact derived from `api/ocgpu-api.toml`, refreshes reviewed semantic and
classification catalogs, then rebuilds the coverage report, deterministic
raw-DEFLATE copy, dependency license inventory, and CycloneDX SBOM. `check`
runs each generator in validation-only mode and fails on any byte difference or
unresolved semantic/classification/implementation fact.
`test` adds all workspace tests. `ci` is the complete local format, lint, test,
release-build, and documentation gate.

Strict header compilation is available as:

```sh
CC=clang cargo run -p xtask -- c99
```

The CI workflow additionally compiles, links, and safely runs enumeration-only
consumers against both shared and static `ocgpu` libraries on ordinary hosted
runners. Device allocation, context creation, module loading, and launches occur
only in the separately acknowledged hardware workflow.

For the optional flat C ABI, build cargo-c with `--features flat-c-exports` and
define `OCGPU_ENABLE_FLAT_C_EXPORTS` in the consumer before including
`ocgpu/ocgpu.h`. Unified leaves take the backend as their first argument; raw
CUDA/HIP leaves remain vendor-shaped. Missing result, pointer, integer,
POD/aggregate, and void leaves use `OCGPU_ERROR_SYMBOL_UNAVAILABLE`, null, zero,
an all-zero value, and a no-op sentinel, respectively. These contracts are generated from the
canonical manifest into the public header; consumer code should not infer a
different sentinel from a vendor return type.
Backend-bound tables are still the recommended fast path. Cargo's internal C
library target is `ocgpu_capi` to avoid an MSVC PDB collision with the CLI;
cargo-c installation and `ocgpu.pc` retain the public name `ocgpu`.

HIP runtime discovery has three explicit ABI profiles (`Hip5`, `Hip6`, and
`Hip7`). The typed/common core is projected from the selected profile and is
validated completely before a driver is returned. The raw HIP table retains
the current 535-slot HIP 7 platform-union layout: general/Linux 7.14.60850 and
Windows 7.2.0 inventories with target masks. On HIP 5 and HIP 6, only slots with
reviewed ABI evidence for that major may be callable; all other applicable
slots are null and reported as `profile_unavailable`. Do not infer ABI
compatibility from a shared export name or from the newest table's Rust/C type.
The same fail-closed rule applies to a supported HIP 7 runtime below its
platform raw-inventory baseline: general/Linux requires 7.14.60850
(`71_460_850`), while Windows requires 7.2.0 (`70_253_210`).
`direct_adapter` records the narrower case where an exact legacy declaration is
called through a typed common shim while the current-HIP raw slot remains null.
`Driver<Hip>::runtime_profile()` supports coarse application policy, while raw
consumers must still check each nullable entry.

Secure default runtime discovery is safe. The optional Rust
`explicit-library-path` feature is intentionally unsafe: the caller must uphold
the trust, constructor, ABI, dependency-closure, and file-identity contract of
`Driver::<B>::load_from_absolute`. C ABI v1 exposes no path override.

## Editing the API

1. Change the canonical manifest and its provenance fields.
2. Run `cargo run -p xtask -- generate`.
3. Update all affected committed oracle snapshots only when upstream facts have
   changed; otherwise update `coverage/classifications.json` with one specific
   reason per new item.
4. Review public type layouts, function calling conventions, pointer constness,
   output direction, callback nullability, symbol aliases, version floors, and
   target masks in the generated diff.
5. Add or update Rust, ABI, loader, C99, and negative tests appropriate to the
   change.
6. Run `cargo run -p xtask -- ci` and the import-table workflow.

For a HIP-major change, also update the exact per-profile compatibility ledger,
library-candidate/profile mapping, runtime-major rejection tests, common-core
projection tests, diagnostic profile strings, and C table flag tests. A symbol
may move out of `profile_unavailable` only with reviewed signature, calling
convention, layout, and semantic evidence for that HIP major.

Versioned tables are append-only. Never reorder their prefix or existing
function-pointer fields. Preserve deprecated and version-suffixed vendor names.
Use integer typedefs and constants at the C boundary, not Rust enums. New
nullable callbacks or table entries use `Option<unsafe extern "C" fn(...)>`.

## Updating Rust oracle candidates

The maintainer-only extractor locates exact registry packages with `cargo
metadata`, parses the binding modules using `syn`, and follows source modules and
statically resolvable `include!` files. The target-disabled source dependencies
exist solely to make those exact packages available to Cargo metadata:

```sh
cargo run -p ocgpu-oracle -- extract-rust cudarc --output target/cudarc.json
cargo run -p ocgpu-oracle -- extract-rust rocmrc --output target/rocmrc.json
```

Candidate output is not accepted automatically. Compare it with the committed
snapshot, verify the upstream package checksum in `Cargo.lock`, add a reviewed
classification and concrete reason for every addition, removal, alias change,
signature change, platform change, or layout change, then regenerate coverage.
An unresolved build-script `OUT_DIR` include is a hard error: run the upstream
package's documented source generation in an isolated maintenance environment
and review that material rather than silently omitting it.

## Updating official oracle candidates

The manual oracle-update workflow also runs
`.github/scripts/extract-official-oracles.ps1`. It downloads only exact public
source archives/pages at the committed revisions, verifies archive and exact
header SHA-256 values, constructs small version/standard-header parsing stubs,
and invokes `extract-vendor` plus `extract-cuda-proc-typedefs`. It installs no
Toolkit, HIP SDK, vendor compiler, library, or import file. The resulting files
under `target/oracle-candidates` are review artifacts only; compare every fact
and provenance change before replacing a committed snapshot. Running the same
script locally requires PowerShell, Python 3, Clang, and network access and does
not alter ordinary offline build inputs.

HIP 5/6/7 compatibility evidence has a separate maintainer-only freshness
command:

```powershell
.github/scripts/verify-hip-runtime-profiles.ps1
```

It downloads the exact HIP and CLR archives pinned by the profile ledger,
verifies archive and reviewed-member hashes, independently extracts the common
declarations, transitive types, and device-attribute values for every reviewed
release and target, then compares the result with the committed compact
snapshot. Pass `-ArchiveDirectory` to use a prepopulated verified archive cache.
Only an intentional, reviewed evidence update should use `-Update`; ordinary
builds and checks consume committed files and remain offline.

## Source hygiene

Every independently authored or generated source-like file carries
`SPDX-License-Identifier: CC0-1.0`. Do not copy vendor comments, SDK headers,
wrapper implementation source, or documentation prose. Normalized names,
signatures, constants, aliases, target gates, and independently measured layout
facts are the permitted coverage inputs. The full dedication and notices are in
`LICENSES/`.
