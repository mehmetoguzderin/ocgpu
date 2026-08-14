<!-- SPDX-License-Identifier: CC0-1.0 -->

# ocgpu Rust packages

This archive is one source package in the SDK-free `ocgpu` Rust dependency
chain. Registry publication, when separately authorized, must follow this
dependency order:

1. `ocgpu-abi` and `ocgpu-loader`;
2. `ocgpu-cuda` and `ocgpu-hip`;
3. `ocgpu`.

`ocgpu` owns its FFI declarations, secure dynamic loading, backend dispatch,
and typed API. Building these packages does not discover or require a CUDA
Toolkit, ROCm/HIP SDK, vendor header, import library, vendor compiler, bindgen,
or libclang. An installed GPU driver/runtime is still required to use a
backend.

API documentation builds locally with `cargo doc --all-features`; each
manifest configures its corresponding docs.rs location for use after registry
publication.
The complete workspace release additionally contains the C99 package, CLI,
coverage evidence, operations documentation, dependency inventory, SBOM, and
third-party notices.

The independently authored source in this package is dedicated under
CC0-1.0. The complete legal text is included as `LICENSE`. Dependencies
and operating-system/runtime components retain their own terms.
