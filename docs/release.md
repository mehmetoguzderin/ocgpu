<!-- SPDX-License-Identifier: CC0-1.0 -->

# Release procedure

Releases are immutable, SDK-free source and binary packages. The tag version
must equal the workspace package version and use `vMAJOR.MINOR.PATCH`.

## Required evidence

From a clean checkout of the intended tag:

1. With Rust 1.85.0, run a locked `--all-features` check over all seven public
   shipping artifacts, then a `--no-default-features` check over the six
   feature-configurable library crates (excluding the CLI so it cannot re-enable
   library defaults through feature unification).
2. Run `cargo run -p xtask -- ci` with Rust 1.97.1.
3. Run the strict C99 workflow with GCC and Clang and the MSVC ABI job.
4. Run Linux and Windows import-table inspection and confirm no CUDA or HIP
   runtime appears as a linked dependency.
5. Cross-build `aarch64-unknown-linux-gnu` and inspect its dynamic section.
6. Run `cargo audit --deny warnings` against the committed lockfile.
7. Verify `cargo tree -p ocgpu -p ocgpu-capi -p ocgpu-cli` contains neither
   oracle crate, `bindgen`, libclang, nor a vendor SDK build helper.
8. Review the canonical manifest, generated header, export lists, normalized
   snapshots, coverage classifications, and raw-DEFLATE report for a clean
   generation diff.
9. Run each applicable hardware job on a dedicated labelled runner. Record a
   failed or unavailable platform explicitly; never substitute one HIP platform
   inventory for another.
10. Build the five Rust source packages in dependency order and inspect every
   archive path before attestation.

## Package construction

The release workflow installs exactly `cargo-c
0.10.24+cargo-0.98.0` and uses `cargo cinstall` to stage the committed header,
static archive, platform shared/import artifacts, and `ocgpu.pc`. On Linux it
replaces cargo-c's development shared object with a manually linked artifact
from the PIC Rust static archive, the generated `ocgpu.map`, and SONAME
`libocgpu.so.1`. The audit requires every public symbol to carry
`@@OCGPU_1.0`, compares the exact export set, and rejects unexpected or vendor
`DT_NEEDED` entries. A flat-enabled Linux package is linked the same way with
`ocgpu-flat.map`. The final public link applies exactly one generated map;
passing an additional map through rustc or cargo-c is deliberately avoided
because their own anonymous version scripts make multiple-script behavior
linker-dependent. Each binary package contains:

- the CLI;
- shared and static `ocgpu` libraries;
- `include/ocgpu/ocgpu.h`;
- `coverage.json` and its human-readable coverage report;
- the complete CC0 dedication, third-party license texts, deterministic
  dependency/license inventory, and CycloneDX SBOM; and
- the project README.

Release publication is blocked on its own locked Rust 1.85.0 all-feature
seven-artifact and no-default six-library checks, RustSec lockfile audit, strict
C99/C++ and generated-layout checks, x64 Linux mapped-ELF audit, case-exact
Windows PE import/export audit, and a separate aarch64 Rust/C cross-build with
exact exports and vendor-`DT_NEEDED` rejection. The publication job depends on
those release-local gates, both x64 package jobs, and the Rust source-package
job; it does not assume that a sibling workflow succeeded.

The verify, x64 package, and aarch64 release jobs first run the same
repository-owned SDK-free preflight used by CI. It rejects toolkit roots,
compiler/configuration tools, vendor headers, and link-time SDK artifacts while
allowing independently installed driver runtimes needed for runtime-only smoke
testing.

Plain `cargo cinstall` is therefore suitable for local staging, but its
unversioned Linux shared object is not a production release artifact.

A separate release job invokes one multi-package `cargo package --locked
--no-verify` operation for `ocgpu-abi`, `ocgpu-loader`, `ocgpu-cuda`,
`ocgpu-hip`, and `ocgpu`. Packaging the workspace set together lets Cargo
prepare the exact path-and-version dependency chain before any version exists in
the registry. The job inspects each archive for required local paths, generates
SHA-256 checksums, attests the five `.crate` files, and attaches them to the tag.
It first fetches the exact lockfile graph, then extracts all five archives,
patches a tiny consumer to the normalized packaged manifests, and runs `cargo
check --offline --all-features`. Thus `--no-verify` skips Cargo's per-package
registry-resolution verifier while the release job still proves the packaged
dependency graph itself is self-contained and compilable; the earlier workspace
quality gate supplies the broader test evidence.

Windows packages distinguish `ocgpu.dll.lib` (shared DLL import library) from
`ocgpu.lib` (static Rust archive). Static MSVC consumers additionally link
`ws2_32.lib`, `ntdll.lib`, `userenv.lib`, and `dbghelp.lib`. Linux packages
retain the versioned shared object and both SONAME symlinks. Release jobs
compile and run a safe enumeration consumer against both the shared and static
installed forms using the staged `ocgpu.pc` (`pkg-config --libs` and
`--libs --static`) before archiving. This verifies that package metadata carries
required platform support libraries instead of relying on a hard-coded consumer
command.

The workflow emits a SHA-256 checksum and GitHub build-provenance attestation for
each archive. Validate that archive extraction does not change permissions or
place files outside its directory. Signatures/attestations identify the build;
they do not change third-party license terms inside compiled Rust or OS runtime
components.

## ABI review

A release is rejected if an existing table field moved, an existing stable ID or
signature hash changed without an approved ABI-major change, a removed vendor
entry disappeared instead of becoming deprecated, or the Windows and ELF export
sets differ. New table fields are appended and guarded by caller `struct_size`.
Reserved fields remain zero. Thirty-two-bit targets are outside ABI v1.

## Optional registry publication

The repository release workflow does not call `cargo publish`, store a
crates.io credential, or claim registry availability. If maintainers separately
authorize registry publication, authenticate only in a protected environment
and publish `ocgpu-abi`, then `ocgpu-loader`, then `ocgpu-cuda` and `ocgpu-hip`,
and finally `ocgpu`. Wait until crates.io resolves each prerequisite version
before publishing its dependants. The tag, source archive, and registry package
must have the same version and content; a GitHub `.crate` attachment is not
evidence of registry publication.

## Recovery

Do not replace files attached to an existing tag. If packaging or provenance is
wrong, publish a new patch version. If a release contains a security flaw,
publish an advisory, mark the affected version, build the fixed tag through the
same workflow, and retain prior checksums and attestations for auditability.
