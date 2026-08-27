<!-- SPDX-License-Identifier: CC0-1.0 -->

# Licensing and provenance

The `ocgpu`-authored source code, generated public API, generated headers,
tests, normalized coverage data, export controls, and documentation are
dedicated under CC0-1.0. The complete legal text is in
`LICENSES/CC0-1.0.txt`; independently authored source-like files carry the
corresponding SPDX identifier.

That statement does not claim every byte of a compiled binary is CC0. Rust
standard/runtime components, operating-system libraries, and dependency code
retain their own terms. `Cargo.lock` identifies the exact Rust dependency graph;
`LICENSES/dependencies.json` records exact package/license/source/checksum facts
and labels each package as shipping-reachable or repository-development-only.
`LICENSES/ocgpu.cdx.json` is the staged CycloneDX SBOM containing only the
non-dev dependency graph reachable from the `ocgpu`, `ocgpu-capi`, and
`ocgpu-cli` distribution roots. Its distinct aggregate product node points to
those roots without reusing a Cargo package reference, and
`LICENSES/THIRD_PARTY_NOTICES.md` explains the boundary.

Tagged releases also attach checksummed, provenance-attested Cargo source
archives for the six publishable Rust crates. The workflow does not publish
them to a registry or treat an archive attachment as registry evidence.

The project is implemented independently. It does not copy or redistribute
NVIDIA or AMD SDK headers, import/static/shared libraries (including
NVRTC/HIPRTC binaries), vendor documentation
prose, `cudarc` or `rocmrc` implementation source, or generated vendor binding
source. Oracle snapshots contain only normalized interoperability facts: public
names, normalized ABI type graphs, numeric values, aliases, version/platform
masks, deprecation facts, and independently verified layouts. Each item has an
evidence locator and a SHA-256 hash of its normalized signature.

`cudarc 0.19.9` and `rocmrc 0.5.0` are exact-pinned development-only comparison
oracles. Always-false target dependencies make Cargo resolve their exact
registry sources without compiling them; a maintainer workflow parses the
declarations from those verified sources. They are not linked into any
workspace artifact or embedded in coverage output. Vendor libraries remain
separately deployed runtime components and are never described as part of
`ocgpu`. Runtime loading support is not permission to redistribute a vendor
compiler; application packagers must obtain compatible NVRTC/HIPRTC components
under the applicable vendor terms.

NVIDIA and CUDA are trademarks of NVIDIA Corporation. AMD, ROCm, and HIP are
trademarks of Advanced Micro Devices, Inc. Use of those names identifies
interoperable APIs and runtime libraries. `ocgpu` is independent, has no vendor
affiliation, and implies no endorsement.

When updating an inventory, record the exact source release, canonical public
URL or crates.io package identity, target/platform scope, and normalization
method. A reviewer must account for every diff with one specific classification
reason. Online material may prepare a candidate snapshot, but committed facts
are never silently refreshed during a normal build.
