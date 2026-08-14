<!-- SPDX-License-Identifier: CC0-1.0 -->

# Continuous integration

Hosted jobs deliberately run without CUDA Toolkit or ROCm/HIP SDK packages.
Before any relevant build, a repository-owned PowerShell preflight rejects SDK
root variables, compiler/configuration tools, and known vendor header or
link-time artifact locations; separately installed driver runtimes remain
allowed. The jobs then build Rust, the CLI, and static/shared C libraries for
Windows and Linux, compile strict C99 consumers with GCC and Clang, compile the
advertised C++ header mode with Clang++/MSVC, validate the MSVC ABI, and
cross-build aarch64 Linux with an ordinary system cross-linker. Both x64
targets run all shipping crate tests; hardware tests are discovered there but
capability-skip before any device operation.

Binary inspection is an independent gate. ELF `DT_NEEDED` and PE import tables
must not name CUDA or HIP libraries. Those names may exist only as strings used
by the runtime loader. Windows `.def` and ELF version-script controls must expose
the same reviewed symbol set with case-sensitive comparison. The aarch64 job
also compares its dynamic symbol set exactly and requires the static archive.
C-artifact jobs use exact-pinned `cargo-c`, so
public package names remain `ocgpu` even though the collision-free internal Rust
library target is `ocgpu_capi`.

Generation checks consume committed files and fail on a diff. The quality job
runs formatting, Clippy with warnings denied, all-target/all-feature workspace
tests, release builds, and documentation. It separately checks CUDA-only,
HIP-only, raw-only, explicit-path-only, combined explicit-path, flat-only, and
single-backend flat feature combinations. Security automation audits the exact
lockfile and rejects an oracle or SDK dependency in any shipping graph.

Rust 1.85.0 is a release-critical MSRV gate. CI checks the complete workspace,
all targets, and all features, which includes all seven public shipping
artifacts. It separately checks the six feature-configurable library crates
with default features disabled; the CLI is omitted from this minimal command
because its normal library dependency would re-enable defaults through feature
unification. The release workflow repeats a focused all-feature check over all
seven artifacts and the six-library minimal check. The maintainer-only source
oracles remain exactly resolved in Cargo metadata but sit behind an
always-false target dependency, so their newer loader internals cannot silently
raise the shipping or workspace build floor.

Tag publication repeats the release-critical checks inside `release.yml`:
both locked MSRV feature-surface checks, RustSec audit, strict C/C++ and layout
compilation, x64 Linux/Windows package consumer and import/export audits, and
the aarch64 header/export/dependency gate. Independent push/PR workflows
provide earlier feedback but are not used as implicit release prerequisites.

The four hardware jobs are manual and require the exact acknowledgement
`RUN_BOUNDED_GPU_SMOKE`. Self-hosted labels distinguish NVIDIA/AMD and
Linux/Windows. They are not eligible on hosted or interactive-display runners.
Every job runs the bounded integration test, non-strict diagnostics for its one
selected backend, and a machine assertion that every emitted manifest symbol
applicable to the actual target triple resolves. The Windows AMD job therefore
evaluates Windows HIP rows only. See `operations.md` for the bounded test and
prohibited device-management actions.

The oracle-update workflow produces review artifacts only. It never commits,
opens a pull request, or changes classifications automatically.
