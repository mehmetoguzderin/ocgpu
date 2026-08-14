<!-- SPDX-License-Identifier: CC0-1.0 -->

# Third-party notices

The `ocgpu`-authored source, generated ABI, normalized factual inventories,
tests, and documentation are dedicated under CC0-1.0. Compiled binaries also
contain Rust crates under their own licenses. Those terms are not replaced by
CC0.

`dependencies.json` is the deterministic name/version/license/source/checksum
table for every package in the exact `Cargo.lock`, with explicit
shipping-reachable and repository-development-only scopes. Its CC0 SPDX field
licenses the inventory document, not the listed packages. `ocgpu.cdx.json` is
the CycloneDX 1.5 graph reachable from the `ocgpu`, `ocgpu-capi`, and
`ocgpu-cli` distribution roots with Cargo dev dependencies excluded. A distinct
aggregate distribution component points to those roots without duplicating a
Cargo package reference. The aggregate and `ocgpu`-authored package components
carry CC0; third-party components retain their recorded upstream expressions.
Release packages include both files and the applicable license texts in this
directory. The files are regenerated from `Cargo.lock` plus `cargo metadata --locked` by
`xtask/dependency-manifest.ps1`; `cargo run -p xtask -- check` verifies byte
freshness.

The principal maintenance-only inputs are:

| Package | Use | License | Locked SHA-256 | Source |
|---|---|---|---|---|
| `cbindgen 0.29.4` | generated C-header verification | MPL-2.0 | `2ecb53484c9c167ba674026b656d8a27d7657a58e6066aa902bfb1a4aa00ae20` | `https://crates.io/api/v1/crates/cbindgen/0.29.4/download` |
| `cudarc 0.19.9` | isolated CUDA Rust declaration oracle | MIT OR Apache-2.0 | `804764d10e844da09765a7b2ca9641a0851523d1702efb0d7299d73e31b86e80` | `https://crates.io/api/v1/crates/cudarc/0.19.9/download` |
| `rocmrc 0.5.0` | isolated HIP Rust declaration oracle | MIT OR Apache-2.0 | `766806566f7d4fffd7f53fe065c86ae935a1296ff148395ca4cdf69d9a41cc18` | `https://crates.io/api/v1/crates/rocmrc/0.5.0/download` |
| `cargo-c 0.10.24+cargo-0.98.0` | exact-pinned release packager | MIT | recorded by the release workflow's locked installation | `https://crates.io/crates/cargo-c/0.10.24+cargo-0.98.0` |

Canonical texts retained here cover CC0-1.0, MIT, Apache-2.0 and its LLVM
exception, MPL-2.0, BSD-2-Clause, ISC, Unicode-3.0, Zlib, 0BSD, and Unlicense.
Where a package offers alternatives (including the `r-efi` MIT OR Apache-2.0 OR
LGPL-2.1-or-later expression), the complete expression and upstream repository
remain recorded in `dependencies.json`; this distribution selects and retains a
permitted MIT/Apache branch rather than purporting to relicense the package.

NVIDIA CUDA and AMD ROCm/HIP libraries are discovered at runtime and are not
part of this project. No vendor SDK header, binary, import library,
documentation prose, or source archive is redistributed. The committed oracle
files contain independently normalized facts: identifiers, ABI type graphs,
numeric constants, aliases, platform masks, and measured layouts. Each
inventory records authoritative declaration locators separately from exact
fetched-artifact hashes. NVIDIA, CUDA, AMD, ROCm, and HIP are trademarks of
their respective owners; this project is independent and is not endorsed by
them.
