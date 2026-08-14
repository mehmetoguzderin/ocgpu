// SPDX-License-Identifier: CC0-1.0

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A normalized upstream API inventory.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Inventory {
    /// Data format version.
    pub schema_version: u32,
    /// SPDX identifier for this independently authored factual dataset.
    pub spdx_license_identifier: String,
    /// Stable identifier used by the coverage catalog.
    pub inventory_id: String,
    /// Human-readable upstream source name.
    pub source_name: String,
    /// Exact upstream release or documentation baseline.
    pub source_version: String,
    /// Primary evidence URL or crate identity.
    pub provenance: String,
    /// Exact fetched inputs, with content hashes separate from declaration locators.
    pub source_artifacts: Vec<SourceArtifact>,
    /// Platforms described by this inventory.
    pub platforms: Vec<String>,
    /// Normalized entries sorted by kind and name.
    pub entries: Vec<Entry>,
}

/// Deduplicated official-vendor callable union used to drive canonical raw generation.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VendorFunctionUnion {
    /// Data format version.
    pub schema_version: u32,
    /// SPDX identifier for this independently authored factual dataset.
    pub spdx_license_identifier: String,
    /// Exact official snapshots contributing callable declarations.
    pub sources: Vec<VendorUnionSource>,
    /// Callable spellings deduplicated by backend, kind, and exact name.
    pub functions: Vec<VendorUnionFunction>,
}

/// One authoritative vendor snapshot contributing to the function union.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VendorUnionSource {
    /// Stable source inventory identifier.
    pub inventory_id: String,
    /// Exact source version.
    pub source_version: String,
    /// Platforms described by the source.
    pub platforms: Vec<String>,
    /// Primary official evidence locator.
    pub provenance: String,
    /// Immutable fetched source artifacts and hashes.
    pub source_artifacts: Vec<SourceArtifact>,
}

/// One official callable spelling and all version/platform declaration variants.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VendorUnionFunction {
    /// Vendor backend family (`cuda` or `hip`).
    pub backend: String,
    /// Function or function-alias category.
    pub kind: ItemKind,
    /// Exact vendor spelling.
    pub name: String,
    /// Source-specific declaration variants, retained instead of flattening version drift.
    pub variants: Vec<VendorUnionVariant>,
}

/// One source/version/platform-specific declaration of a callable spelling.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VendorUnionVariant {
    /// Inventory supplying the declaration.
    pub inventory_id: String,
    /// Exact official baseline.
    pub source_version: String,
    /// Canonical ABI representation.
    pub normalized_signature: String,
    /// SHA-256 over the normalized ABI graph.
    pub signature_hash: String,
    /// Structured ABI for functions; aliases refer to `alias_of` instead.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub abi: Option<Abi>,
    /// Exact canonical target when this variant is an alias.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alias_of: Option<String>,
    /// Alternate spellings attached to a canonical function.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
    /// Target triples for this declaration variant.
    pub platforms: Vec<String>,
    /// First upstream version, when the source declares it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub introduced: Option<String>,
    /// Deprecation baseline, when the source declares it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deprecated: Option<String>,
    /// Exact `cuGetProcAddress` typedef candidates from `cudaTypedefs.h`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub proc_address_candidates: Vec<CudaProcAddressCandidate>,
    /// Declaration-specific official evidence locator.
    pub provenance: String,
}

/// Committed, SDK-free normalization of NVIDIA's versioned function-pointer typedefs.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CudaProcAddressCatalog {
    /// Data format version.
    pub schema_version: u32,
    /// SPDX identifier for this independently authored factual dataset.
    pub spdx_license_identifier: String,
    /// Exact CUDA release described by the typedef header.
    pub source_version: String,
    /// Official documentation for the `cuGetProcAddress` selection contract.
    pub provenance: String,
    /// Immutable archive and extracted-header hashes.
    pub source_artifacts: Vec<SourceArtifact>,
    /// Versioned function-pointer typedefs sorted by symbol, version, variant, and name.
    pub typedefs: Vec<CudaProcAddressCandidate>,
}

/// Stream-semantics branch encoded by a CUDA function-pointer typedef suffix.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CudaProcAddressVariant {
    /// Unsuffixed legacy-stream ABI selection.
    Legacy,
    /// Per-thread-default-stream function without an explicit stream parameter.
    Ptds,
    /// Per-thread-default-stream function with an explicit stream parameter.
    Ptsz,
}

/// One exact `cudaTypedefs.h` candidate for version-aware symbol lookup.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CudaProcAddressCandidate {
    /// Base name passed to `cuGetProcAddress`, without C ABI suffixes.
    pub symbol: String,
    /// Exact typedef spelling in `cudaTypedefs.h`.
    pub typedef_name: String,
    /// CUDA ABI version encoded in the typedef name.
    pub api_version: u32,
    /// Exact stream-semantics query flag (`1` legacy or `2` per-thread).
    pub proc_address_flags: u64,
    /// Typedef suffix semantics.
    pub variant: CudaProcAddressVariant,
    /// Canonical callable type graph including transitive typedef structure.
    pub normalized_signature: String,
    /// SHA-256 over `normalized_signature`.
    pub signature_hash: String,
    /// Structured spelling-level ABI retained for code generation and review.
    pub abi: Abi,
    /// Exact official declaration locator.
    pub provenance: String,
}

/// One immutable source artifact used to produce an inventory.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SourceArtifact {
    /// Role in extraction, such as `authoritative-header` or `supporting-include`.
    pub role: String,
    /// Stable upstream download or source URL.
    pub url: String,
    /// SHA-256 of the exact fetched bytes, prefixed with `sha256:`.
    pub sha256: String,
    /// Upstream tag, commit, or documentation revision.
    pub revision: String,
    /// Path within an archive, or crate module path.
    pub path: String,
}

/// An inventory item.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    /// Item category.
    pub kind: ItemKind,
    /// Exact upstream spelling.
    pub name: String,
    /// Canonical ABI representation used for hashing and comparison.
    pub normalized_signature: String,
    /// SHA-256 of `normalized_signature`, prefixed with `sha256:`.
    pub signature_hash: String,
    /// Evaluated integer value for an enumeration member, when Clang proves one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub numeric_value: Option<i64>,
    /// Structured function ABI, present exactly for functions and callbacks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub abi: Option<Abi>,
    /// Alias spellings resolved to this item, in fallback order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
    /// Target item when this entry itself is an alias.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alias_of: Option<String>,
    /// Applicable targets, each of which must be declared by the inventory.
    pub platforms: Vec<String>,
    /// API version that introduced the entry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub introduced: Option<String>,
    /// API version that deprecated the entry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deprecated: Option<String>,
    /// Per-target layout facts for records and unions.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub layouts: Vec<Layout>,
    /// Evidence locator specific to this entry.
    pub provenance: String,
}

/// Supported inventory item categories.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ItemKind {
    /// Callable function.
    Function,
    /// Alternate function or type name.
    Alias,
    /// Scalar or aggregate type alias.
    Type,
    /// Pointer-sized opaque handle.
    OpaqueHandle,
    /// C-compatible structure.
    Struct,
    /// C-compatible union.
    Union,
    /// Function-pointer type.
    Callback,
    /// Named constant.
    Constant,
    /// Enumeration member.
    EnumValue,
    /// Bit flag.
    Flag,
}

/// Structured callable ABI.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Abi {
    /// Calling convention, normally `C` or `system`.
    pub calling_convention: String,
    /// Normalized return type.
    pub return_type: String,
    /// Ordered parameters.
    pub parameters: Vec<Parameter>,
}

/// One ordered ABI parameter.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Parameter {
    /// Source parameter name.
    pub name: String,
    /// Normalized type graph.
    pub r#type: String,
    /// Pointer qualification.
    pub pointer: PointerKind,
    /// Input/output ownership direction.
    pub direction: Direction,
    /// Whether a pointer or callback may be null; `None` means the declaration does not prove it.
    pub nullable: Option<bool>,
}

/// Pointer qualification retained by signature comparison.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PointerKind {
    /// Value is not a pointer.
    Value,
    /// Read-only pointer.
    Const,
    /// Writable pointer.
    Mut,
    /// Function pointer.
    Callback,
}

/// Parameter data direction.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    /// Declaration syntax alone cannot distinguish input, output, and bidirectional use.
    Unknown,
    /// The exact upstream declaration and documentation state no data-direction contract.
    UnspecifiedBySource,
    /// Caller-to-callee data.
    In,
    /// Callee-to-caller data.
    Out,
    /// Bidirectional data.
    InOut,
}

/// Verified C layout for one target.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Layout {
    /// Rust target triple.
    pub target: String,
    /// Size in bytes.
    pub size: u64,
    /// Alignment in bytes.
    pub alignment: u64,
    /// Field byte offsets keyed by exact source field name.
    pub field_offsets: BTreeMap<String, u64>,
    /// Primary evidence used for the layout.
    pub provenance: String,
}

/// Human-reviewed classification catalog.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CoverageCatalog {
    /// Data format version.
    pub schema_version: u32,
    /// SPDX identifier for the authored classification data.
    pub spdx_license_identifier: String,
    /// Exactly one decision for each `(inventory_id, item_name)` pair.
    pub decisions: Vec<CoverageDecision>,
}

/// Human-reviewed parameter semantics that cannot be proven from declaration syntax.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticCatalog {
    /// Data format version.
    pub schema_version: u32,
    /// SPDX identifier for the authored semantic review data.
    pub spdx_license_identifier: String,
    /// Overrides sorted by inventory, function, and parameter.
    pub parameters: Vec<SemanticOverride>,
}

/// One reviewed direction and nullability fact.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticOverride {
    /// Inventory containing the declaration.
    pub inventory_id: String,
    /// Exact upstream function spelling.
    pub function: String,
    /// Exact parameter name from that inventory.
    pub parameter: String,
    /// Reviewed input/output direction, including an explicit source-unspecified state.
    pub direction: Direction,
    /// Reviewed nullability contract, including an explicit source-unspecified state.
    pub nullability: ReviewedNullability,
    /// Specific explanation for the reviewed decision.
    pub reason: String,
    /// Stable evidence locator for the review.
    pub provenance: String,
}

/// Final reviewed nullability state for a pointer or callback parameter.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReviewedNullability {
    /// The source contract permits a null pointer value.
    Nullable,
    /// The source contract requires a non-null pointer value.
    NonNull,
    /// The exact upstream source does not state an ordinary-pointer nullability contract.
    UnspecifiedBySource,
}

impl ReviewedNullability {
    /// Converts an explicit boolean declaration fact into its reviewed state.
    #[must_use]
    pub const fn from_bool(nullable: bool) -> Self {
        if nullable {
            Self::Nullable
        } else {
            Self::NonNull
        }
    }

    /// Returns the explicit boolean contract, or `None` when upstream leaves it unspecified.
    #[must_use]
    pub const fn as_bool(self) -> Option<bool> {
        match self {
            Self::Nullable => Some(true),
            Self::NonNull => Some(false),
            Self::UnspecifiedBySource => None,
        }
    }
}

/// Classification of one upstream item.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CoverageDecision {
    /// Inventory containing the upstream item.
    pub inventory_id: String,
    /// Normalized category, required because C typedefs and tags may share a spelling.
    pub item_kind: ItemKind,
    /// Exact upstream item name.
    pub item_name: String,
    /// Coverage class.
    pub classification: Classification,
    /// Human explanation; empty or generic reasons are rejected.
    pub reason: String,
    /// Canonical manifest IDs that account for this item.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub manifest_ids: Vec<String>,
    /// Generated Rust or C identifiers implementing this entry.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub implementation_symbols: Vec<String>,
    /// Public dynamic symbols, if any, expected in the export controls.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub export_symbols: Vec<String>,
    /// Whether runtime resolution has been exercised on applicable hardware.
    pub runtime_resolvable: bool,
    /// Whether the bounded hardware smoke profile has exercised the entry.
    pub hardware_smoke: bool,
}

/// Mutually exclusive coverage classes.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Classification {
    /// Common API with identical semantics.
    CoveredExact,
    /// Common API using a documented adapter.
    CoveredAdapter,
    /// Available through a backend-specific raw API.
    CoveredRawOnly,
    /// Not present on the inventory's platform.
    PlatformUnavailable,
    /// Deprecated upstream entry retained by ocgpu.
    DeprecatedCovered,
    /// Deliberate exclusion with a specific reason.
    IntentionallyOmitted,
    /// Known declaration whose binary layout has not been verified.
    LayoutUnverified,
    /// Cannot be represented by the stable C ABI.
    Unrepresentable,
}
