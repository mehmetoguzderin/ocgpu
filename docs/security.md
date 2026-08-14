<!-- SPDX-License-Identifier: CC0-1.0 -->

# Security policy and threat model

## Reporting

Report suspected vulnerabilities through the repository host's private security
advisory channel. Include the affected version and platform, backend, minimal
reproducer, observed result, and whether an untrusted library path or input
module is involved. Do not include credentials, proprietary kernels, device
memory contents, or public exploit details in an initial report.

Supported releases are the latest patch of the current minor series. Older
versions receive fixes only when the published advisory says so.

## Trust boundaries

`ocgpu` accepts pointers, lengths, opaque handles, module bytes, symbol names,
and, through an unsafe opt-in Rust feature, an explicit library path from its
caller. It loads code supplied by the installed GPU driver/runtime and module
images supplied by the application.
Neither vendor drivers nor application device code are sandboxed by `ocgpu`.

Defenses include:

- secure platform library loading with no current-working-directory search;
- absolute paths only for the unsafe Rust explicit override; C ABI v1 has no
  path override;
- immutable backend-bound function tables and process-lifetime library handles;
- one-time required-core validation before hot-path calls;
- checked sizes, null pointers, ABI versions, and symbol availability;
- nullable C function pointers represented with the guaranteed nullable-pointer
  Rust representation;
- panic-free leaf dispatch and a catching boundary for complex C exports;
- no allocation crossing the Windows CRT boundary without an `ocgpu`
  deallocator; and
- no logging of pointer values, raw buffers, module contents, or kernel
  arguments by default.

Callers remain responsible for buffer validity, handle/backend pairing, device
code provenance, resource limits, and synchronization. A caller of unsafe
`Driver::<B>::load_from_absolute` additionally establishes trust in the chosen
library and dependency closure, constructors, exact vendor ABI, and unchanged
file identity between validation and load. Loading an untrusted GPU module or
shared library is equivalent to loading untrusted native code.

## Supply chain

Shipping crates do not depend on `cudarc`, `rocmrc`, bindgen, GPU SDK headers,
vendor import libraries, or vendor compilers. CI inspects shipping dependency
graphs and binary import tables. Oracle dependencies are exact-pinned and used
only by an unpublished maintenance crate. Generated artifacts and normalized
snapshots are committed and must reproduce byte-for-byte.

Weekly automation audits the exact Cargo lockfile and reviews new dependency
licenses and vulnerabilities. Release artifacts carry SHA-256 checksums and
build-provenance attestations. Action and dependency updates require normal code
review; oracle pins change only with an explicit baseline update.

## Unsafe-code review

Every unsafe block must state or directly establish its invariants: symbol type,
library lifetime, pointer validity, buffer extent, table size, handle origin, or
FFI unwind behavior. Reviewers should treat loader search changes, new exported
pointer types, callback lifetime changes, and module-loading changes as
security-sensitive even when the Rust type checker accepts them.
