// SPDX-License-Identifier: CC0-1.0

//! Deterministic generator for every committed ocgpu ABI artifact.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::io;
use std::io::Write as IoWrite;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde::{Deserialize, Serialize};

mod profiles;
mod sync;

use profiles::{
    read_hip_runtime_declarations, read_hip_runtime_profiles, render_hip_runtime_profiles,
    validate_hip_runtime_profiles,
};

pub use sync::{SyncReport, sync_rust_oracles};

const MANIFEST_PATH: &str = "api/ocgpu-api.toml";
const ABI_RUST_PATH: &str = "crates/ocgpu-abi/src/generated.rs";
const ABI_LAYOUT_TEST_PATH: &str = "crates/ocgpu-abi/tests/generated_layout.rs";
const ABI_C_LAYOUT_TEST_PATH: &str = "tests/abi/generated_layout.c";
const HEADER_PATH: &str = "include/ocgpu/ocgpu.h";
const CAPI_HEADER_PATH: &str = "crates/ocgpu-capi/assets/ocgpu/ocgpu.h";
const CAPI_EXPORTS_PATH: &str = "crates/ocgpu-capi/assets/generated_exports.rs";
const DEF_PATH: &str = "exports/ocgpu.def";
const MAP_PATH: &str = "exports/ocgpu.map";
const FLAT_DEF_PATH: &str = "exports/ocgpu-flat.def";
const FLAT_MAP_PATH: &str = "exports/ocgpu-flat.map";
const API_INVENTORY_PATH: &str = "crates/ocgpu-codegen/generated/api-inventory.json";
const LOADER_INVENTORY_PATH: &str = "crates/ocgpu-codegen/generated/loader-inventory.json";
const CUDA_SYMBOLS_PATH: &str = "crates/ocgpu-codegen/generated/cuda-symbols.rs";
const HIP_SYMBOLS_PATH: &str = "crates/ocgpu-codegen/generated/hip-symbols.rs";
const CUDA_PACKAGE_SYMBOLS_PATH: &str = "crates/ocgpu-cuda/src/generated_symbols.rs";
const HIP_PACKAGE_SYMBOLS_PATH: &str = "crates/ocgpu-hip/src/generated_symbols.rs";
const CUDA_ORACLE_PATH: &str = "oracle/rust/cudarc-0.19.9.json";
const HIP_ORACLE_PATH: &str = "oracle/rust/rocmrc-0.5.0.json";
const CUDA_VENDOR_PATH: &str = "oracle/vendor/cuda/13.3-13030.json";
const HIP_VENDOR_GENERAL_PATH: &str = "oracle/vendor/hip/general-7.14.60850.json";
const HIP_VENDOR_WINDOWS_PATH: &str = "oracle/vendor/hip/windows-7.2.0.json";
const HIP_RUNTIME_PROFILES_PATH: &str = "oracle/vendor/hip/runtime-profiles.json";
const HIP_RUNTIME_DECLARATIONS_PATH: &str =
    "oracle/vendor/hip/runtime-profile-declarations.json";
const HIP_GENERATED_PROFILES_PATH: &str = "crates/ocgpu-hip/src/generated_profiles.rs";
const VENDOR_FUNCTION_UNION_PATH: &str = "oracle/vendor/function-union.json";

/// Whether generated files are written or only compared with their canonical bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mode {
    /// Update committed generated files.
    Generate,
    /// Fail if any committed generated file differs.
    Check,
}

/// Summary of one generator run.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Report {
    checked: usize,
    changed: Vec<PathBuf>,
    mode: Mode,
}

impl Report {
    /// Human-readable deterministic summary.
    #[must_use]
    pub fn summary(&self) -> String {
        match self.mode {
            Mode::Generate if self.changed.is_empty() => {
                format!("{} generated artifacts already current", self.checked)
            }
            Mode::Generate => format!(
                "updated {} of {} generated artifacts",
                self.changed.len(),
                self.checked
            ),
            Mode::Check => format!("{} generated artifacts are current", self.checked),
        }
    }

    /// Relative paths changed in generate mode.
    #[must_use]
    pub fn changed(&self) -> &[PathBuf] {
        &self.changed
    }
}

/// Generator failure.
#[derive(Debug)]
pub enum Error {
    /// A file could not be read or written.
    Io {
        /// File associated with the failure.
        path: PathBuf,
        /// Underlying operating-system error.
        source: io::Error,
    },
    /// The canonical TOML was not syntactically valid.
    Toml(toml::de::Error),
    /// The manifest violated a semantic ABI invariant.
    Validation(String),
    /// cbindgen could not render the generated declarations.
    Cbindgen(String),
    /// Check mode found stale or missing outputs.
    Stale(Vec<PathBuf>),
    /// JSON serialization failed.
    Json(serde_json::Error),
    /// A committed oracle JSON snapshot was invalid.
    OracleJson {
        /// Snapshot path.
        path: PathBuf,
        /// JSON parser error.
        source: serde_json::Error,
    },
    /// rustfmt could not normalize a generated Rust artifact.
    Rustfmt(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, source } => write!(formatter, "{}: {source}", path.display()),
            Self::Toml(source) => write!(formatter, "invalid {MANIFEST_PATH}: {source}"),
            Self::Validation(message) => write!(formatter, "manifest validation failed: {message}"),
            Self::Cbindgen(message) => write!(formatter, "cbindgen failed: {message}"),
            Self::Stale(paths) => {
                write!(formatter, "generated artifacts are stale:")?;
                for path in paths {
                    write!(formatter, "\n  {}", path.display())?;
                }
                Ok(())
            }
            Self::Json(source) => write!(formatter, "could not serialize generated JSON: {source}"),
            Self::OracleJson { path, source } => {
                write!(
                    formatter,
                    "invalid oracle snapshot {}: {source}",
                    path.display()
                )
            }
            Self::Rustfmt(message) => write!(formatter, "rustfmt failed: {message}"),
        }
    }
}

impl std::error::Error for Error {}

/// Root of the canonical manifest.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ApiManifest {
    /// Manifest metadata and baselines.
    pub manifest: ManifestMetadata,
    /// ABI-exposed scalar and opaque types.
    #[serde(rename = "type")]
    pub types: Vec<TypeEntry>,
    /// ABI constants represented by integer typedefs.
    #[serde(rename = "constant")]
    pub constants: Vec<ConstantEntry>,
    /// Unified operations and their exact backend counterparts.
    #[serde(rename = "function")]
    pub functions: Vec<FunctionEntry>,
    /// Versioned function tables.
    #[serde(rename = "table")]
    pub tables: Vec<TableEntry>,
    /// Exhaustive classification of every pinned Rust-oracle function.
    #[serde(default, rename = "raw_inventory")]
    pub raw_inventory: Vec<RawInventoryEntry>,
}

/// Version and provenance metadata.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ManifestMetadata {
    /// Schema revision.
    pub schema_version: u32,
    /// Public ABI revision.
    pub abi_version: u32,
    /// SPDX identifier applying to authored/generated content.
    pub spdx_license_identifier: String,
    /// CUDA oracle baseline.
    pub cuda_baseline: String,
    /// General HIP oracle baseline.
    pub hip_general_baseline: String,
    /// Windows HIP oracle baseline.
    pub hip_windows_baseline: String,
    /// Supported pointer widths.
    pub pointer_widths: Vec<u32>,
    /// Missing result-returning raw entrypoint sentinel.
    pub raw_missing_result: String,
    /// Missing pointer-returning raw entrypoint sentinel.
    pub raw_missing_pointer: String,
    /// Missing integer-returning raw entrypoint sentinel.
    pub raw_missing_integer: String,
    /// Missing void-returning raw entrypoint behavior.
    pub raw_missing_void: String,
    /// Missing aggregate/POD-returning raw entrypoint sentinel.
    #[serde(default = "default_raw_missing_aggregate")]
    pub raw_missing_aggregate: String,
    /// Result used by the management getter boundary for unexpected panics.
    /// Generated flat leaves remain mechanically panic-free and do not add a
    /// per-call unwind boundary.
    pub raw_panic_result: String,
}

fn default_raw_missing_aggregate() -> String {
    "zeroed".to_owned()
}

/// One public ABI type.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TypeEntry {
    /// Stable numeric identifier.
    pub stable_id: u32,
    /// Public typedef name.
    pub name: String,
    /// `integer`, `pointer_integer`, `opaque_handle`, `alias`, `callback`,
    /// `record`, `union`, `opaque_record`, or `opaque_union`.
    pub kind: String,
    /// Rust scalar spelling for scalar entries.
    pub rust_type: Option<String>,
    /// Opaque structure tag for handle entries.
    pub tag: Option<String>,
    /// Callback result type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub return_type: Option<String>,
    /// Callback parameters.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub params: Vec<Parameter>,
    /// Complete fields for a declaration-derived record or union.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fields: Vec<TypeField>,
    /// Backend owning an oracle-derived raw type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    /// Exact upstream type spelling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vendor_name: Option<String>,
    /// Pinned oracle inventory ID.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub oracle_source: Option<String>,
    /// Pinned oracle signature hash.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub oracle_signature_hash: Option<String>,
    /// Every pinned source-specific declaration retained without flattening.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub oracle_variants: Vec<TypeOracleVariant>,
    /// Per-target layout evidence for this public declaration.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub platform_layouts: Vec<TypeLayoutEvidence>,
    /// Expected size on supported 64-bit targets.
    pub size_64: u32,
    /// Expected alignment on supported 64-bit targets.
    pub align_64: u32,
    /// Hash of the normalized type layout graph.
    pub layout_hash: String,
    /// Source of the layout facts.
    pub layout_provenance: String,
    /// Source of the API name and semantics.
    pub documentation_provenance: String,
}

/// One field in a complete declaration-derived record or union.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TypeField {
    /// Exact bindgen field spelling.
    pub name: String,
    /// Exact C spelling when Rust requires a raw identifier or bindgen suffix.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub c_name: Option<String>,
    /// Normalized public Rust ABI type.
    #[serde(rename = "type")]
    pub type_name: String,
    /// Expected byte offset on supported 64-bit targets.
    pub offset_64: u32,
}

/// One source-specific type declaration retained in the canonical manifest.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TypeOracleVariant {
    /// Pinned inventory identity.
    pub oracle_source: String,
    /// Source release/version label.
    pub source_version: String,
    /// Declaration kind in this source.
    pub oracle_kind: String,
    /// Exact normalized declaration graph.
    pub oracle_signature: String,
    /// Source-provided SHA-256 signature hash.
    pub oracle_signature_hash: String,
    /// Platforms on which this declaration was observed.
    pub platforms: Vec<String>,
    /// Exact compiler/source layout evidence, when the declaration has storage.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub layouts: Vec<TypeLayoutEvidence>,
    /// Exact source provenance.
    pub provenance: String,
}

/// One target-specific type layout fact.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TypeLayoutEvidence {
    /// Rust target triple covered by the evidence.
    pub target: String,
    /// Size in bytes.
    pub size: u32,
    /// Alignment in bytes.
    pub alignment: u32,
    /// Field offsets keyed by exact source spelling.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub field_offsets: BTreeMap<String, u32>,
    /// Compiler/declaration provenance.
    pub provenance: String,
}

/// One integer constant.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ConstantEntry {
    /// Stable numeric identifier.
    pub stable_id: u32,
    /// Macro/constant name.
    pub name: String,
    /// Public integer typedef.
    #[serde(rename = "type")]
    pub type_name: String,
    /// Signed value, preserving error codes.
    pub value: i64,
    /// Backend-native integer values when the unified value requires translation.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub backend_values: BTreeMap<String, i64>,
    /// Target-specific values for vendor constants whose releases differ.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub platform_values: BTreeMap<String, i64>,
    /// Backend owning a vendor-derived constant, absent for unified constants.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    /// Exact vendor spelling before the collision-safe ocgpu backend prefix.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vendor_name: Option<String>,
    /// `constant`, `enum_value`, or `unified`.
    #[serde(default)]
    pub vendor_kind: String,
    /// Every source/platform declaration variant.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub oracle_variants: Vec<ConstantOracleVariant>,
    /// Platforms carrying this value.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub platforms: Vec<String>,
    /// Whether the declaration is an integer constant emitted in Rust/C.
    #[serde(default = "default_true")]
    pub emitted: bool,
    /// `covered_integer` or an item-specific non-integer classification.
    #[serde(default)]
    pub classification: String,
    /// Item-specific coverage explanation.
    #[serde(default)]
    pub reason: String,
    /// Source of the value.
    pub documentation_provenance: String,
    /// Reviewed provenance for the selected numeric value.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub value_provenance: String,
}

fn default_true() -> bool {
    true
}

/// One source-specific constant or enum-value declaration.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ConstantOracleVariant {
    /// Pinned inventory identity.
    pub oracle_source: String,
    /// Source version.
    pub source_version: String,
    /// `constant` or `enum_value`.
    pub oracle_kind: String,
    /// Exact normalized declaration.
    pub oracle_signature: String,
    /// Source signature hash.
    pub oracle_signature_hash: String,
    /// Resolved integer value for this declaration, when it is integer-shaped.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<i64>,
    /// Platforms carrying this declaration.
    pub platforms: Vec<String>,
    /// Source-declared aliases.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
    /// Canonical declaration for an alias.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alias_of: Option<String>,
    /// Source release that introduced the declaration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub introduced: Option<String>,
    /// Source release that deprecated the declaration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deprecated: Option<String>,
    /// Exact source provenance.
    pub provenance: String,
}

/// Function argument metadata.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Parameter {
    /// Parameter name.
    pub name: String,
    /// Normalized Rust ABI type spelling.
    #[serde(rename = "type")]
    pub type_name: String,
    /// `in`, `out`, `inout`, or `unknown` when the pinned declaration does not encode intent.
    pub direction: String,
    /// Whether a pointer may be null, or unknown when declaration facts are insufficient.
    pub nullable: Option<bool>,
    /// `reviewed_override`, `declaration_fact`, or empty for hand-authored common ABI rows.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub semantic_status: String,
    /// Exact declaration or reviewed-sidecar provenance for direction and nullability.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub semantic_provenance: String,
}

/// Platform-specific symbol resolution facts.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PlatformSpec {
    /// Whether the symbol family exists on the platform baseline.
    pub available: bool,
    /// Documentation/source baseline.
    pub baseline: String,
    /// Version passed to a proc-address query.
    pub proc_version: u32,
    /// Ordered symbol candidates.
    pub symbols: Vec<String>,
    /// Human explanation of this platform classification.
    pub reason: String,
}

/// Raw backend mapping for a unified operation.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BackendFunction {
    /// Prefixed ocgpu raw field name.
    pub raw_name: String,
    /// Canonical vendor spelling, including suffixes.
    pub vendor_symbol: String,
    /// Documented aliases with ABI-identical signatures.
    pub aliases: Vec<String>,
    /// Other versioned symbols classified separately from direct aliases.
    pub versioned_alternatives: Vec<String>,
    /// Vendor-compatible result typedef.
    pub return_type: String,
    /// Vendor-compatible arguments.
    pub params: Vec<Parameter>,
    /// Hash of the normalized ABI signature.
    pub signature_hash: String,
    /// Coverage classification.
    pub classification: String,
    /// Human reason for the classification.
    pub classification_reason: String,
    /// Introduced API version.
    pub introduced: String,
    /// Deprecation version or an empty string.
    pub deprecated: String,
    /// Default, versioned, PTDS/PTSZ, or SPT variant.
    pub variant: String,
    /// Whether this is a per-thread-default-stream entrypoint.
    pub per_thread_default_stream: bool,
    /// Proc-address query flags.
    pub proc_address_flags: u64,
    /// Ordered direct/proc-address lookup names.
    pub fallback_order: Vec<String>,
    /// Linux baseline facts.
    pub linux: PlatformSpec,
    /// Windows baseline facts.
    pub windows: PlatformSpec,
}

/// One unified operation and both raw mappings.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FunctionEntry {
    /// Stable dotted identifier.
    pub id: String,
    /// Stable numeric identifier.
    pub stable_id: u32,
    /// API grouping.
    pub group: String,
    /// Unified field name.
    pub common_name: String,
    /// Exact or adapter classification.
    pub classification: String,
    /// Human reason for the classification.
    pub classification_reason: String,
    /// Unified result typedef.
    pub return_type: String,
    /// Unified arguments.
    pub params: Vec<Parameter>,
    /// Hash of the normalized common ABI signature.
    pub signature_hash: String,
    /// Ownership/lifetime rule.
    pub ownership: String,
    /// Nullability rule for outputs and optional inputs.
    pub nullability: String,
    /// Callback behavior or `none`.
    pub callback_behavior: String,
    /// Thread-safety statement.
    pub thread_safety: String,
    /// Layout source.
    pub layout_provenance: String,
    /// Documentation source.
    pub documentation_provenance: String,
    /// CUDA raw mapping.
    pub cuda: BackendFunction,
    /// HIP raw mapping.
    pub hip: BackendFunction,
}

/// One append-only ABI table.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TableEntry {
    /// Public table type name.
    pub name: String,
    /// `common`, `cuda`, or `hip`.
    pub surface: String,
    /// Expected 64-bit size.
    pub size_64: u32,
    /// Expected 64-bit alignment.
    pub align_64: u32,
    /// Hash over size, alignment, and every field offset.
    pub layout_hash: String,
    /// Source of the prefix and append-only layout policy.
    pub layout_provenance: String,
}

/// Classification record for one pinned raw-oracle function.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RawInventoryEntry {
    /// Stable numeric identifier independent of source ordering.
    pub stable_id: u32,
    /// `cuda` or `hip`.
    pub backend: String,
    /// Exact upstream spelling.
    pub vendor_name: String,
    /// Official inventory kind: `function` or ABI-identical `alias`.
    #[serde(default = "default_function_kind")]
    pub vendor_kind: String,
    /// Deterministically prefixed public spelling, absent for non-vendor helpers.
    pub raw_name: Option<String>,
    /// Oracle snapshot identifier.
    pub oracle_source: String,
    /// Oracle's normalized signature graph.
    pub oracle_signature: String,
    /// Oracle's SHA-256 signature hash.
    pub oracle_signature_hash: String,
    /// Every official/Rust source-specific declaration variant retained without flattening.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub oracle_variants: Vec<RawOracleVariant>,
    /// Official alias declarations shadowed by a real function of the same C spelling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alias_collision: Option<RawAliasCollision>,
    /// Raw coverage classification.
    pub classification: String,
    /// Human-readable explanation for the classification.
    pub reason: String,
    /// Oracle platforms on which the entry was observed.
    pub platforms: Vec<String>,
    /// Whether ABI v1 emits a typed raw-table field.
    pub emitted: bool,
    /// Append-only raw-table slot order for emitted entries not linked to the common table.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub table_order: Option<u32>,
    /// Canonical proc-address query name (which may differ from the direct suffix).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proc_name: Option<String>,
    /// Ordered direct symbol candidates, including reviewed ABI-identical aliases.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub direct_names: Vec<String>,
    /// Linux proc-address ABI version; zero means reviewed direct-only resolution.
    #[serde(default)]
    pub proc_version_linux: u32,
    /// Windows proc-address ABI version; zero means reviewed direct-only resolution.
    #[serde(default)]
    pub proc_version_windows: u32,
    /// Vendor proc-address flags.
    #[serde(default)]
    pub proc_flags: u64,
    /// Exact vendor typedef selected for proc-address resolution.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proc_typedef: Option<String>,
    /// Expanded transitive normalized ABI graph of the selected proc typedef.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proc_signature: Option<String>,
    /// Expanded transitive ABI-graph hash of the selected proc typedef.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proc_signature_hash: Option<String>,
    /// Pinned source location proving the selected proc typedef contract.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proc_provenance: Option<String>,
    /// `default`, `versioned`, `ptds`, `ptsz`, `spt`, or `alias`.
    #[serde(default)]
    pub variant: String,
    /// ABI-identical canonical target for official alias declarations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alias_of: Option<String>,
    /// Independently normalized Rust ABI return type for emitted raw-only fields.
    pub abi_return_type: Option<String>,
    /// Independently normalized Rust ABI parameters for emitted raw-only fields.
    #[serde(default)]
    pub abi_params: Vec<Parameter>,
    /// Linked common operation when one exists.
    pub common_id: Option<String>,
    /// Source of the normalized facts.
    pub documentation_provenance: String,
    /// Ownership/lifetime fact for the vendor-shaped call boundary.
    #[serde(default)]
    pub ownership: String,
    /// Nullability interpretation for the per-parameter facts.
    #[serde(default)]
    pub nullability: String,
    /// Callback invocation behavior, or an explicit declaration-only statement.
    #[serde(default)]
    pub callback_behavior: String,
    /// Thread-safety statement for the raw leaf.
    #[serde(default)]
    pub thread_safety: String,
}

fn default_function_kind() -> String {
    "function".to_owned()
}

/// One source-specific raw declaration retained in the canonical manifest.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RawOracleVariant {
    /// Pinned inventory identity.
    pub oracle_source: String,
    /// Declaration kind in this source: `function` or preprocessor/source `alias`.
    #[serde(default = "default_function_kind")]
    pub oracle_kind: String,
    /// Source release/version label.
    pub source_version: String,
    /// Exact normalized declaration graph.
    pub oracle_signature: String,
    /// Source-provided SHA-256 signature hash.
    pub oracle_signature_hash: String,
    /// Platforms on which this declaration variant was observed.
    pub platforms: Vec<String>,
    /// ABI-identical aliases declared by this source variant.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
    /// Canonical declaration for an alias variant.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alias_of: Option<String>,
    /// Exact source provenance.
    pub provenance: String,
}

/// Explicit accounting for an official macro alias that cannot own a second C field name.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RawAliasCollision {
    /// Always `unrepresentable_name_collision` in ABI v1.
    pub classification: String,
    /// ABI-distinct canonical macro target.
    pub alias_target: String,
    /// Item-specific reason the alias cannot be emitted as an independent C field.
    pub reason: String,
    /// Every pinned official declaration of the shadowed alias.
    pub oracle_variants: Vec<RawOracleVariant>,
}

#[derive(Deserialize)]
struct OracleSnapshot {
    inventory_id: String,
    source_version: String,
    provenance: String,
    entries: Vec<OracleEntry>,
}

#[derive(Deserialize)]
struct OracleEntry {
    kind: String,
    name: String,
    normalized_signature: String,
    signature_hash: String,
    platforms: Vec<String>,
    provenance: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HipRuntimeProfiles {
    schema_version: u32,
    spdx_license_identifier: String,
    inventory_id: String,
    scope: String,
    runtime_version_encoding: HipRuntimeVersionEncoding,
    compatibility_policy: HipCompatibilityPolicy,
    table_flag_encoding: HipTableFlagEncoding,
    bootstrap_symbols: Vec<String>,
    release_sets: BTreeMap<String, Vec<String>>,
    profiles: Vec<HipRuntimeProfile>,
    reviewed_releases: Vec<HipReviewedRelease>,
    library_naming_evidence: Vec<HipLibraryNamingEvidence>,
    common_functions: Vec<HipProfileFunction>,
    common_adapters: Vec<HipCommonAdapter>,
    device_attributes: Vec<HipDeviceAttribute>,
    transitive_abi_facts: Vec<HipAbiFact>,
    semantic_reviews: Vec<HipSemanticReview>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HipRuntimeDeclarations {
    schema_version: u32,
    spdx_license_identifier: String,
    inventory_id: String,
    provenance: String,
    snapshots: Vec<HipRuntimeDeclarationSnapshot>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HipRuntimeDeclarationSnapshot {
    release_id: String,
    source_inventory_id: String,
    source_header_sha256: String,
    target_abi: HipTargetAbi,
    functions: Vec<HipDeclaration>,
    transitive_types: Vec<HipTypeDeclaration>,
    device_attributes: Vec<HipDeviceAttribute>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HipTargetAbi {
    pointer_width_bits: u32,
    size_t_width_bits: u32,
    enum_width_bits: u32,
    success_value: i64,
    null_pointer_sentinel: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HipDeclaration {
    name: String,
    normalized_signature: String,
    signature_hash: String,
    platforms: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HipTypeDeclaration {
    name: String,
    kind: String,
    normalized_signature: String,
    signature_hash: String,
    platforms: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HipRuntimeVersionEncoding {
    expression: String,
    source: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HipCompatibilityPolicy {
    rule: String,
    source: String,
    fail_closed: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HipTableFlagEncoding {
    mask: u32,
    shift: u32,
    hip5: u32,
    hip6: u32,
    hip7: u32,
    zero_meaning: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HipRuntimeProfile {
    runtime_major: i32,
    table_flag: u32,
    common_adapter_symbols: Vec<String>,
    reviewed_release_ids: Vec<String>,
    windows: HipPlatformProfile,
    linux: HipPlatformProfile,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HipPlatformProfile {
    library_candidates: Vec<String>,
    runtime_version_min_inclusive: i32,
    runtime_version_max_inclusive: i32,
    proc_address_min_inclusive: Option<i32>,
    raw_inventory_min_inclusive: Option<i32>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HipReviewedRelease {
    id: String,
    rocm_release: String,
    runtime_version: i32,
    #[serde(default)]
    observed_runtime: Option<HipObservedRuntime>,
    hip_commit: String,
    hip_archive_url: String,
    hip_archive_sha256: String,
    hip_header_path: String,
    hip_header_sha256: String,
    hip_version_sha256: String,
    clr_commit: String,
    clr_archive_url: String,
    clr_archive_sha256: String,
    clr_cmake_path: String,
    clr_cmake_sha256: String,
    proc_address_declared: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HipObservedRuntime {
    platform: String,
    library: String,
    sha256: String,
    hip_runtime_get_version: i32,
    file_version: String,
    product_version: String,
    signature_status: String,
    signer_subject: String,
    signer_thumbprint: String,
    scope_note: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HipLibraryNamingEvidence {
    platform: String,
    major: i32,
    names: Vec<String>,
    source: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HipProfileFunction {
    name: String,
    release_set: String,
    signature_hash: String,
    #[serde(default)]
    additional_signatures: Vec<HipAdditionalSignature>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HipAdditionalSignature {
    release_set: String,
    signature_hash: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HipCommonAdapter {
    name: String,
    adapter: String,
    signature_variants: Vec<HipAdapterSignature>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HipAdapterSignature {
    release_set: String,
    normalized_signature: String,
    signature_hash: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HipDeviceAttribute {
    name: String,
    value: i64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HipAbiFact {
    fact: String,
    expected: serde_json::Value,
    proof: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HipSemanticReview {
    operations: Vec<String>,
    finding: String,
    proof: String,
}

#[derive(Deserialize)]
struct VendorFunctionUnion {
    functions: Vec<VendorUnionEntry>,
}

#[derive(Deserialize)]
struct VendorUnionEntry {
    backend: String,
    kind: String,
    name: String,
    variants: Vec<VendorUnionVariant>,
}

#[derive(Deserialize)]
struct VendorUnionVariant {
    inventory_id: String,
    source_version: String,
    normalized_signature: String,
    signature_hash: String,
    #[serde(default)]
    aliases: Vec<String>,
    alias_of: Option<String>,
    platforms: Vec<String>,
    provenance: String,
}

#[derive(Serialize)]
struct LoaderInventory<'a> {
    spdx_license_identifier: &'static str,
    schema_version: u32,
    abi_version: u32,
    cuda: Vec<LoaderFunction<'a>>,
    hip: Vec<LoaderFunction<'a>>,
}

#[derive(Serialize)]
struct ApiInventory<'a> {
    spdx_license_identifier: &'static str,
    #[serde(flatten)]
    manifest: &'a ApiManifest,
}

#[derive(Serialize)]
struct LoaderFunction<'a> {
    stable_id: u32,
    logical_id: &'a str,
    raw_name: &'a str,
    vendor_symbol: &'a str,
    aliases: &'a [String],
    versioned_alternatives: &'a [String],
    signature_hash: &'a str,
    variant: &'a str,
    per_thread_default_stream: bool,
    proc_address_flags: u64,
    fallback_order: &'a [String],
    linux: &'a PlatformSpec,
    windows: &'a PlatformSpec,
}

/// Parse, validate, and generate or check the complete artifact graph.
pub fn run(workspace_root: &Path, mode: Mode) -> Result<Report, Error> {
    let manifest_path = workspace_root.join(MANIFEST_PATH);
    let source = read(&manifest_path)?;
    let manifest: ApiManifest = toml::from_str(&source).map_err(Error::Toml)?;
    validate(&manifest)?;
    validate_oracle_coverage(workspace_root, &manifest)?;
    let hip_profiles = read_hip_runtime_profiles(workspace_root)?;
    let hip_declarations = read_hip_runtime_declarations(workspace_root)?;
    validate_hip_runtime_profiles(
        &manifest,
        &hip_profiles,
        &hip_declarations,
    )?;

    let rust_source = format_rust(&render_rust(&manifest)?)?;
    let layout_test = format_rust(&render_layout_test(&manifest)?)?;
    let c_layout_test = render_c_layout_test(&manifest)?;
    let api_inventory = format!(
        "{}\n",
        serde_json::to_string_pretty(&ApiInventory {
            spdx_license_identifier: "CC0-1.0",
            manifest: &manifest,
        })
        .map_err(Error::Json)?
    );
    let loader_inventory = render_loader_inventory(&manifest)?;
    let cuda_symbols = format_rust(&render_symbol_descriptors(&manifest, "cuda")?)?;
    let hip_symbols = format_rust(&render_symbol_descriptors(&manifest, "hip")?)?;
    let hip_runtime_profiles =
        format_rust(&render_hip_runtime_profiles(&manifest, &hip_profiles)?)?;
    let export_shims = format_rust(&render_export_shims(&manifest))?;
    let def = render_def();
    let map = render_map();
    let flat_def = render_flat_def(&manifest);
    let flat_map = render_flat_map(&manifest);

    let rust_path = workspace_root.join(ABI_RUST_PATH);
    let mut changed = Vec::new();
    if mode == Mode::Generate {
        update(&rust_path, &rust_source, workspace_root, &mut changed)?;
    }

    let header = if mode == Mode::Check && !matches_file(&rust_path, &rust_source)? {
        String::new()
    } else {
        render_header(workspace_root, &manifest)?
    };

    let mut outputs = vec![
        (PathBuf::from(ABI_RUST_PATH), rust_source),
        (PathBuf::from(ABI_LAYOUT_TEST_PATH), layout_test),
        (PathBuf::from(ABI_C_LAYOUT_TEST_PATH), c_layout_test),
        (PathBuf::from(DEF_PATH), def),
        (PathBuf::from(MAP_PATH), map),
        (PathBuf::from(FLAT_DEF_PATH), flat_def),
        (PathBuf::from(FLAT_MAP_PATH), flat_map),
        (PathBuf::from(API_INVENTORY_PATH), api_inventory),
        (PathBuf::from(LOADER_INVENTORY_PATH), loader_inventory),
        (PathBuf::from(CUDA_SYMBOLS_PATH), cuda_symbols.clone()),
        (PathBuf::from(HIP_SYMBOLS_PATH), hip_symbols.clone()),
        (PathBuf::from(CUDA_PACKAGE_SYMBOLS_PATH), cuda_symbols),
        (PathBuf::from(HIP_PACKAGE_SYMBOLS_PATH), hip_symbols),
        (
            PathBuf::from(HIP_GENERATED_PROFILES_PATH),
            hip_runtime_profiles,
        ),
        (PathBuf::from(CAPI_EXPORTS_PATH), export_shims),
    ];
    if header.is_empty() {
        outputs.push((PathBuf::from(HEADER_PATH), String::new()));
        outputs.push((PathBuf::from(CAPI_HEADER_PATH), String::new()));
    } else {
        outputs.push((PathBuf::from(HEADER_PATH), header.clone()));
        outputs.push((PathBuf::from(CAPI_HEADER_PATH), header));
    }

    match mode {
        Mode::Generate => {
            for (relative, content) in &outputs[1..] {
                update(
                    &workspace_root.join(relative),
                    content,
                    workspace_root,
                    &mut changed,
                )?;
            }
            Ok(Report {
                checked: outputs.len(),
                changed,
                mode,
            })
        }
        Mode::Check => {
            let mut stale = Vec::new();
            for (relative, content) in &outputs {
                if content.is_empty() || !matches_file(&workspace_root.join(relative), content)? {
                    stale.push(relative.clone());
                }
            }
            if stale.is_empty() {
                Ok(Report {
                    checked: outputs.len(),
                    changed,
                    mode,
                })
            } else {
                Err(Error::Stale(stale))
            }
        }
    }
}

fn format_rust(source: &str) -> Result<String, Error> {
    let mut child = Command::new("rustfmt")
        .args(["--edition", "2024", "--emit", "stdout"])
        .env("RUSTUP_TOOLCHAIN", "stable")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| Error::Rustfmt(format!("could not start rustfmt: {error}")))?;
    child
        .stdin
        .as_mut()
        .ok_or_else(|| Error::Rustfmt("rustfmt stdin was unavailable".to_owned()))?
        .write_all(source.as_bytes())
        .map_err(|error| Error::Rustfmt(format!("could not write rustfmt input: {error}")))?;
    let output = child
        .wait_with_output()
        .map_err(|error| Error::Rustfmt(format!("could not wait for rustfmt: {error}")))?;
    if !output.status.success() {
        return Err(Error::Rustfmt(
            String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        ));
    }
    String::from_utf8(output.stdout)
        .map_err(|error| Error::Rustfmt(format!("rustfmt output was not UTF-8: {error}")))
}

fn read(path: &Path) -> Result<String, Error> {
    fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn matches_file(path: &Path, expected: &str) -> Result<bool, Error> {
    match fs::read_to_string(path) {
        Ok(actual) => Ok(actual == expected),
        Err(source) if source.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(source) => Err(Error::Io {
            path: path.to_path_buf(),
            source,
        }),
    }
}

fn update(
    path: &Path,
    expected: &str,
    workspace_root: &Path,
    changed: &mut Vec<PathBuf>,
) -> Result<(), Error> {
    if matches_file(path, expected)? {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| Error::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    fs::write(path, expected).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    changed.push(
        path.strip_prefix(workspace_root)
            .unwrap_or(path)
            .to_path_buf(),
    );
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn validate(manifest: &ApiManifest) -> Result<(), Error> {
    if manifest.manifest.schema_version != 1 {
        return Err(Error::Validation("schema_version must be 1".to_owned()));
    }
    if manifest.manifest.abi_version != 0x0001_0000 {
        return Err(Error::Validation(
            "ABI v1 must be encoded as 0x00010000".to_owned(),
        ));
    }
    if manifest.manifest.spdx_license_identifier != "CC0-1.0" {
        return Err(Error::Validation(
            "manifest license must be CC0-1.0".to_owned(),
        ));
    }
    if manifest.manifest.pointer_widths != [64] {
        return Err(Error::Validation(
            "ABI v1 must declare only the 64-bit pointer width".to_owned(),
        ));
    }
    if manifest.manifest.raw_missing_result != "OCGPU_ERROR_SYMBOL_UNAVAILABLE"
        || manifest.manifest.raw_missing_pointer != "null"
        || manifest.manifest.raw_missing_integer != "zero"
        || manifest.manifest.raw_missing_void != "no-op"
        || manifest.manifest.raw_missing_aggregate != "zeroed"
        || manifest.manifest.raw_panic_result != "OCGPU_ERROR_INTERNAL"
    {
        return Err(Error::Validation(
            "flat raw missing-symbol and panic policies must use the documented ABI v1 sentinels"
                .to_owned(),
        ));
    }

    let mut stable_ids = BTreeSet::new();
    let mut type_names = BTreeSet::new();
    for entry in &manifest.types {
        unique_id(&mut stable_ids, entry.stable_id, &entry.name)?;
        if !type_names.insert(entry.name.as_str()) {
            return Err(Error::Validation(format!(
                "duplicate type name {}",
                entry.name
            )));
        }
        if entry.name.contains("__")
            || entry.tag.as_deref().is_some_and(|tag| tag.contains("__"))
            || entry.fields.iter().any(|field| {
                field.name.contains("__")
                    || field
                        .c_name
                        .as_deref()
                        .is_some_and(|name| name.contains("__"))
            })
        {
            return Err(Error::Validation(format!(
                "{} exposes a C++-reserved double-underscore identifier",
                entry.name
            )));
        }
        if !matches!(
            entry.kind.as_str(),
            "integer"
                | "pointer_integer"
                | "opaque_handle"
                | "alias"
                | "callback"
                | "record"
                | "union"
                | "opaque_record"
                | "opaque_union"
        ) {
            return Err(Error::Validation(format!(
                "{} has unsupported type kind {}",
                entry.name, entry.kind
            )));
        }
        if entry.kind == "callback" {
            validate_parameters(&entry.name, &entry.params)?;
            if entry.return_type.is_none() {
                return Err(Error::Validation(format!(
                    "{} callback needs a return type",
                    entry.name
                )));
            }
        }
        if matches!(entry.kind.as_str(), "record" | "union") && entry.fields.is_empty() {
            return Err(Error::Validation(format!(
                "{} complete {} needs fields",
                entry.name, entry.kind
            )));
        }
        if entry.backend.is_some()
            && (entry.vendor_name.is_none()
                || entry.oracle_source.is_none()
                || entry.oracle_signature_hash.is_none()
                || entry.oracle_variants.is_empty()
                || entry.platform_layouts.is_empty())
        {
            return Err(Error::Validation(format!(
                "{} oracle-derived type needs complete source identity",
                entry.name
            )));
        }
        if entry.align_64 == 0
            || !entry.align_64.is_power_of_two()
            || (entry.size_64 != 0 && entry.size_64 % entry.align_64 != 0)
        {
            return Err(Error::Validation(format!(
                "{} has invalid size/alignment {}/{}",
                entry.name, entry.size_64, entry.align_64
            )));
        }
        let mut layout_targets = BTreeSet::new();
        for layout in &entry.platform_layouts {
            validate_target(&layout.target, &entry.name)?;
            if !layout_targets.insert(layout.target.as_str()) {
                return Err(Error::Validation(format!(
                    "{} has duplicate layout evidence for {}",
                    entry.name, layout.target
                )));
            }
            if layout.size != entry.size_64 || layout.alignment != entry.align_64 {
                return Err(Error::Validation(format!(
                    "{} layout evidence for {} is {}/{}, expected {}/{}",
                    entry.name,
                    layout.target,
                    layout.size,
                    layout.alignment,
                    entry.size_64,
                    entry.align_64
                )));
            }
            for field in &entry.fields {
                let c_name = field.c_name.as_deref().unwrap_or(&field.name);
                match layout.field_offsets.get(c_name) {
                    Some(offset) if *offset != field.offset_64 => {
                        return Err(Error::Validation(format!(
                            "{} field {c_name} offset on {} is {offset}, expected {}",
                            entry.name, layout.target, field.offset_64
                        )));
                    }
                    _ => {}
                }
            }
        }
        let expected = format_hash(type_layout_hash(entry));
        if entry.layout_hash != expected {
            return Err(Error::Validation(format!(
                "{} layout_hash is {}, expected {expected}",
                entry.name, entry.layout_hash
            )));
        }
    }

    for entry in &manifest.types {
        if entry
            .oracle_variants
            .iter()
            .any(|variant| variant.oracle_kind == "opaque_handle")
            && matches!(entry.kind.as_str(), "integer" | "pointer_integer")
        {
            return Err(Error::Validation(format!(
                "{} is pointer-shaped in pinned vendor declarations and cannot be emitted as an integer typedef",
                entry.name
            )));
        }
    }
    let hip_deviceptr = manifest
        .types
        .iter()
        .find(|entry| entry.name == "ocgpuHipDeviceptr_t")
        .ok_or_else(|| Error::Validation("missing ocgpuHipDeviceptr_t".to_owned()))?;
    if hip_deviceptr.kind != "alias" || hip_deviceptr.rust_type.as_deref() != Some("*mut c_void") {
        return Err(Error::Validation(
            "ocgpuHipDeviceptr_t must preserve the vendor void* pointer category".to_owned(),
        ));
    }

    let mut constant_names = BTreeSet::new();
    for entry in &manifest.constants {
        unique_id(&mut stable_ids, entry.stable_id, &entry.name)?;
        if !constant_names.insert(entry.name.as_str()) {
            return Err(Error::Validation(format!(
                "duplicate constant {}",
                entry.name
            )));
        }
        if !type_names.contains(entry.type_name.as_str())
            && !matches!(
                entry.type_name.as_str(),
                "u8" | "u16" | "u32" | "u64" | "usize" | "i8" | "i16" | "i32" | "i64"
            )
        {
            return Err(Error::Validation(format!(
                "{} uses unknown type {}",
                entry.name, entry.type_name
            )));
        }
        if entry
            .backend_values
            .keys()
            .any(|backend| !matches!(backend.as_str(), "cuda" | "hip"))
        {
            return Err(Error::Validation(format!(
                "{} has an unknown backend mapping key",
                entry.name
            )));
        }
        if entry.name.starts_with("OCGPU_DEVICE_ATTRIBUTE_")
            && (entry.type_name != "ocgpuDeviceAttribute"
                || entry.backend_values.len() != 2
                || !entry.backend_values.contains_key("cuda")
                || !entry.backend_values.contains_key("hip"))
        {
            return Err(Error::Validation(format!(
                "{} must carry complete CUDA and HIP device-attribute mappings",
                entry.name
            )));
        }
        if let Some(backend) = &entry.backend {
            if !matches!(backend.as_str(), "cuda" | "hip")
                || entry.vendor_name.is_none()
                || !matches!(entry.vendor_kind.as_str(), "constant" | "enum_value")
                || entry.oracle_variants.is_empty()
                || entry.classification.trim().is_empty()
                || entry.reason.trim().is_empty()
            {
                return Err(Error::Validation(format!(
                    "{} has incomplete vendor constant coverage metadata",
                    entry.name
                )));
            }
            if entry.emitted != (entry.classification == "covered_integer") {
                return Err(Error::Validation(format!(
                    "{} emitted state disagrees with classification {}",
                    entry.name, entry.classification
                )));
            }
        }
    }

    for (stable_id, name, value) in [
        (
            0x0002_1010,
            "OCGPU_API_FLAG_HIP_RUNTIME_PROFILE_MASK",
            0x00ff_0000_i64,
        ),
        (
            0x0002_1011,
            "OCGPU_API_FLAG_HIP_RUNTIME_PROFILE_5",
            0x0005_0000_i64,
        ),
        (
            0x0002_1012,
            "OCGPU_API_FLAG_HIP_RUNTIME_PROFILE_6",
            0x0006_0000_i64,
        ),
        (
            0x0002_1013,
            "OCGPU_API_FLAG_HIP_RUNTIME_PROFILE_7",
            0x0007_0000_i64,
        ),
    ] {
        let Some(entry) = manifest.constants.iter().find(|entry| entry.name == name) else {
            return Err(Error::Validation(format!(
                "canonical manifest is missing required table flag {name}"
            )));
        };
        if entry.stable_id != stable_id
            || entry.type_name != "u32"
            || entry.value != value
            || !entry.emitted
            || entry.documentation_provenance.trim().is_empty()
        {
            return Err(Error::Validation(format!(
                "{name} must retain its stable ID, u32 encoding, value, emitted state, and documentation"
            )));
        }
    }

    let mut logical_ids = BTreeSet::new();
    let mut common_names = BTreeSet::new();
    let mut raw_names = BTreeSet::new();
    for entry in &manifest.functions {
        unique_id(&mut stable_ids, entry.stable_id, &entry.id)?;
        if !logical_ids.insert(entry.id.as_str()) {
            return Err(Error::Validation(format!(
                "duplicate function id {}",
                entry.id
            )));
        }
        if !common_names.insert(entry.common_name.as_str()) {
            return Err(Error::Validation(format!(
                "duplicate common function {}",
                entry.common_name
            )));
        }
        if !matches!(entry.classification.as_str(), "exact" | "adapter") {
            return Err(Error::Validation(format!(
                "{} has unsupported common classification {}",
                entry.id, entry.classification
            )));
        }
        validate_parameters(&entry.id, &entry.params)?;
        validate_signature(
            &entry.id,
            &entry.return_type,
            &entry.params,
            &entry.signature_hash,
        )?;
        validate_backend("cuda", entry, &entry.cuda, &mut raw_names)?;
        validate_backend("hip", entry, &entry.hip, &mut raw_names)?;
    }
    for (symbol, parameter, expected_index) in [
        ("hipMemcpyHtoD", "destination", 0_usize),
        ("hipMemcpyDtoH", "source", 1_usize),
    ] {
        let raw = manifest
            .functions
            .iter()
            .map(|function| &function.hip)
            .find(|raw| raw.vendor_symbol == symbol)
            .ok_or_else(|| Error::Validation(format!("missing raw {symbol}")))?;
        let actual = raw.params.get(expected_index);
        if actual.map(|param| (param.name.as_str(), param.type_name.as_str()))
            != Some((parameter, "ocgpuHipDeviceptr_t"))
        {
            return Err(Error::Validation(format!(
                "{symbol} must preserve the pointer-typed ocgpuHipDeviceptr_t raw ABI"
            )));
        }
    }

    let emitted = manifest
        .functions
        .iter()
        .flat_map(|entry| {
            [
                (
                    ("cuda", entry.cuda.vendor_symbol.as_str()),
                    (&entry.cuda.raw_name, &entry.id),
                ),
                (
                    ("hip", entry.hip.vendor_symbol.as_str()),
                    (&entry.hip.raw_name, &entry.id),
                ),
            ]
        })
        .collect::<BTreeMap<_, _>>();
    let mut inventory_keys = BTreeSet::new();
    let mut inventory_slots = BTreeSet::new();
    for entry in &manifest.raw_inventory {
        unique_id(&mut stable_ids, entry.stable_id, &entry.vendor_name)?;
        if !matches!(entry.backend.as_str(), "cuda" | "hip") {
            return Err(Error::Validation(format!(
                "raw inventory {} has invalid backend {}",
                entry.vendor_name, entry.backend
            )));
        }
        if !inventory_keys.insert((entry.backend.as_str(), entry.vendor_name.as_str())) {
            return Err(Error::Validation(format!(
                "duplicate raw inventory entry {}:{}",
                entry.backend, entry.vendor_name
            )));
        }
        if !matches!(entry.vendor_kind.as_str(), "function" | "alias") {
            return Err(Error::Validation(format!(
                "{}:{} has invalid vendor kind {}",
                entry.backend, entry.vendor_name, entry.vendor_kind
            )));
        }
        if (entry.vendor_kind == "alias") != entry.alias_of.is_some() {
            return Err(Error::Validation(format!(
                "{}:{} alias kind/target facts disagree",
                entry.backend, entry.vendor_name
            )));
        }
        if let Some(collision) = &entry.alias_collision {
            if entry.vendor_kind != "function"
                || collision.classification != "unrepresentable_name_collision"
                || collision.alias_target == entry.vendor_name
                || collision.reason.trim().is_empty()
                || collision.oracle_variants.is_empty()
                || collision.oracle_variants.iter().any(|variant| {
                    variant.oracle_kind != "alias"
                        || variant.alias_of.as_deref() != Some(collision.alias_target.as_str())
                })
            {
                return Err(Error::Validation(format!(
                    "{}:{} has incomplete shadowed-alias collision accounting",
                    entry.backend, entry.vendor_name
                )));
            }
        }
        if !matches!(
            entry.classification.as_str(),
            "covered_raw_only"
                | "deprecated_covered"
                | "platform_unavailable"
                | "intentionally_omitted"
                | "layout_unverified"
                | "unrepresentable"
        ) {
            return Err(Error::Validation(format!(
                "{}:{} has unsupported raw classification {}",
                entry.backend, entry.vendor_name, entry.classification
            )));
        }
        if entry.reason.trim().is_empty() {
            return Err(Error::Validation(format!(
                "{}:{} needs a classification reason",
                entry.backend, entry.vendor_name
            )));
        }
        let reason_lower = entry.reason.to_ascii_lowercase();
        if ["pending", "not yet", "placeholder", "to be verified"]
            .iter()
            .any(|phrase| reason_lower.contains(phrase))
        {
            return Err(Error::Validation(format!(
                "{}:{} uses a deferred classification reason",
                entry.backend, entry.vendor_name
            )));
        }
        if entry.ownership.trim().is_empty()
            || entry.nullability.trim().is_empty()
            || entry.callback_behavior.trim().is_empty()
            || entry.thread_safety.trim().is_empty()
        {
            return Err(Error::Validation(format!(
                "{}:{} needs complete raw semantic metadata",
                entry.backend, entry.vendor_name
            )));
        }
        if entry.emitted
            && !matches!(
                entry.variant.as_str(),
                "default" | "versioned" | "ptds" | "ptsz" | "spt" | "alias"
            )
        {
            return Err(Error::Validation(format!(
                "{}:{} has invalid resolution variant {}",
                entry.backend, entry.vendor_name, entry.variant
            )));
        }
        if !entry.oracle_signature_hash.starts_with("sha256:")
            || entry.oracle_signature_hash.len() != 71
        {
            return Err(Error::Validation(format!(
                "{}:{} has malformed oracle signature hash",
                entry.backend, entry.vendor_name
            )));
        }
        match &entry.raw_name {
            Some(raw_name) => {
                let expected = raw_prefixed_name(&entry.backend, &entry.vendor_name)?;
                if raw_name != &expected {
                    return Err(Error::Validation(format!(
                        "{raw_name} must preserve {} exactly as {expected}",
                        entry.vendor_name
                    )));
                }
            }
            None if entry.classification == "intentionally_omitted" => {}
            None => {
                return Err(Error::Validation(format!(
                    "{}:{} needs a prefixed raw name",
                    entry.backend, entry.vendor_name
                )));
            }
        }
        if entry.emitted {
            let proc_name = entry.proc_name.as_deref().ok_or_else(|| {
                Error::Validation(format!(
                    "{}:{} is emitted without a proc-address name",
                    entry.backend, entry.vendor_name
                ))
            })?;
            if proc_name.trim().is_empty()
                || entry.direct_names.is_empty()
                || (entry.vendor_kind == "function"
                    && !entry
                        .direct_names
                        .iter()
                        .any(|name| name == &entry.vendor_name))
                || (entry.vendor_kind == "alias"
                    && !entry
                        .alias_of
                        .as_ref()
                        .is_some_and(|target| entry.direct_names.iter().any(|name| name == target)))
            {
                return Err(Error::Validation(format!(
                    "{}:{} has incomplete resolution names",
                    entry.backend, entry.vendor_name
                )));
            }
            let mut direct = BTreeSet::new();
            if entry.direct_names.iter().any(|name| !direct.insert(name)) {
                return Err(Error::Validation(format!(
                    "{}:{} has duplicate direct resolution names",
                    entry.backend, entry.vendor_name
                )));
            }
            if matches!(entry.variant.as_str(), "spt" | "ptsz" | "ptds")
                && (entry.proc_version_linux != 0 || entry.proc_version_windows != 0)
                && (entry.proc_flags != 2
                    || entry.proc_name.as_deref()
                        != Some(
                            entry
                                .vendor_name
                                .trim_end_matches("_spt")
                                .trim_end_matches("_ptsz")
                                .trim_end_matches("_ptds"),
                        ))
            {
                return Err(Error::Validation(format!(
                    "{}:{} per-thread proc lookup needs the base name and flag 2",
                    entry.backend, entry.vendor_name
                )));
            }
            if let Some((expected_raw_name, expected_common_id)) =
                emitted.get(&(entry.backend.as_str(), entry.vendor_name.as_str()))
            {
                if entry.raw_name.as_ref() != Some(*expected_raw_name)
                    || entry.common_id.as_ref() != Some(*expected_common_id)
                {
                    return Err(Error::Validation(format!(
                        "{}:{} emitted linkage does not match its common function",
                        entry.backend, entry.vendor_name
                    )));
                }
            } else {
                let return_type = entry.abi_return_type.as_deref().ok_or_else(|| {
                    Error::Validation(format!(
                        "{}:{} is emitted without a normalized return type",
                        entry.backend, entry.vendor_name
                    ))
                })?;
                validate_parameters(&entry.vendor_name, &entry.abi_params)?;
                if return_type.contains("bool")
                    || entry
                        .abi_params
                        .iter()
                        .any(|parameter| parameter.type_name.contains("bool"))
                {
                    return Err(Error::Validation(format!(
                        "{}:{} exposes forbidden Rust bool",
                        entry.backend, entry.vendor_name
                    )));
                }
                if entry.common_id.is_some() {
                    return Err(Error::Validation(format!(
                        "{}:{} has an unknown common linkage",
                        entry.backend, entry.vendor_name
                    )));
                }
                let slot = entry.table_order.ok_or_else(|| {
                    Error::Validation(format!(
                        "{}:{} emitted raw-only entry needs an append-only table slot",
                        entry.backend, entry.vendor_name
                    ))
                })?;
                if !inventory_slots.insert((entry.backend.as_str(), slot)) {
                    return Err(Error::Validation(format!(
                        "{} raw-table slot {slot} is duplicated",
                        entry.backend
                    )));
                }
            }
        } else if entry.abi_return_type.is_some() || !entry.abi_params.is_empty() {
            return Err(Error::Validation(format!(
                "{}:{} carries emitted ABI types but is not emitted",
                entry.backend, entry.vendor_name
            )));
        }
    }

    for entry in manifest.raw_inventory.iter().filter(|entry| entry.emitted) {
        match entry.vendor_kind.as_str() {
            "function" if entry.direct_names != [entry.vendor_name.clone()] => {
                return Err(Error::Validation(format!(
                    "{}:{} function slot may direct-resolve only its exact canonical export",
                    entry.backend, entry.vendor_name
                )));
            }
            "alias" => {
                let target_name = entry.alias_of.as_deref().expect("alias target validated");
                if entry.direct_names != [target_name.to_owned()] {
                    return Err(Error::Validation(format!(
                        "{}:{} alias slot must resolve only its typed canonical target {target_name}",
                        entry.backend, entry.vendor_name
                    )));
                }
                let target = manifest
                    .raw_inventory
                    .iter()
                    .find(|candidate| {
                        candidate.backend == entry.backend
                            && candidate.vendor_name == target_name
                            && candidate.emitted
                    })
                    .ok_or_else(|| {
                        Error::Validation(format!(
                            "{}:{} alias target {target_name} is not emitted",
                            entry.backend, entry.vendor_name
                        ))
                    })?;
                let alias_abi = raw_inventory_abi(manifest, entry)?;
                let target_abi = raw_inventory_abi(manifest, target)?;
                if normalized_signature(alias_abi.0, alias_abi.1)
                    != normalized_signature(target_abi.0, target_abi.1)
                {
                    return Err(Error::Validation(format!(
                        "{}:{} alias target {target_name} has a different typed ABI",
                        entry.backend, entry.vendor_name
                    )));
                }
            }
            _ => {}
        }
    }

    for table in &manifest.tables {
        let expected_name = match table.surface.as_str() {
            "common" => "ocgpuApi_v1",
            "cuda" => "ocgpuCuApi_v1",
            "hip" => "ocgpuHipApi_v1",
            other => {
                return Err(Error::Validation(format!(
                    "{} has unsupported table surface {other}",
                    table.name
                )));
            }
        };
        if table.name != expected_name {
            return Err(Error::Validation(format!(
                "{} surface must use table name {expected_name}",
                table.surface
            )));
        }
        let field_count = table_field_count(manifest, &table.surface);
        let expected_size = 24_u32
            .checked_add(
                u32::try_from(field_count)
                    .map_err(|_| Error::Validation("too many functions".to_owned()))?
                    * 8,
            )
            .ok_or_else(|| Error::Validation("table size overflow".to_owned()))?;
        if table.size_64 != expected_size || table.align_64 != 8 {
            return Err(Error::Validation(format!(
                "{} layout must be size {expected_size}, alignment 8",
                table.name
            )));
        }
        let expected_hash = format_hash(table_layout_hash(field_count));
        if table.layout_hash != expected_hash {
            return Err(Error::Validation(format!(
                "{} layout_hash is {}, expected {expected_hash}",
                table.name, table.layout_hash
            )));
        }
    }
    if manifest.tables.len() != 3 {
        return Err(Error::Validation(
            "exactly three v1 tables are required".to_owned(),
        ));
    }
    Ok(())
}

fn raw_inventory_abi<'a>(
    manifest: &'a ApiManifest,
    entry: &'a RawInventoryEntry,
) -> Result<(&'a str, &'a [Parameter]), Error> {
    if let Some(common_id) = &entry.common_id {
        let function = manifest
            .functions
            .iter()
            .find(|function| &function.id == common_id)
            .ok_or_else(|| {
                Error::Validation(format!(
                    "{}:{} references missing common function {common_id}",
                    entry.backend, entry.vendor_name
                ))
            })?;
        let raw = if entry.backend == "cuda" {
            &function.cuda
        } else {
            &function.hip
        };
        Ok((&raw.return_type, &raw.params))
    } else {
        Ok((
            entry.abi_return_type.as_deref().ok_or_else(|| {
                Error::Validation(format!(
                    "{}:{} has no emitted ABI return type",
                    entry.backend, entry.vendor_name
                ))
            })?,
            &entry.abi_params,
        ))
    }
}

// Keeping both Rust and official-vendor correlations in one validation pass
// makes the observed-set completeness check mechanically obvious.
#[allow(clippy::too_many_lines)]
fn validate_oracle_coverage(workspace_root: &Path, manifest: &ApiManifest) -> Result<(), Error> {
    for (backend, relative) in [
        ("cuda", CUDA_ORACLE_PATH),
        ("hip", HIP_ORACLE_PATH),
        ("cuda", CUDA_VENDOR_PATH),
        ("hip", HIP_VENDOR_GENERAL_PATH),
        ("hip", HIP_VENDOR_WINDOWS_PATH),
    ] {
        validate_snapshot_type_constant_coverage(workspace_root, manifest, backend, relative)?;
    }
    let classified = manifest
        .raw_inventory
        .iter()
        .map(|entry| ((entry.backend.as_str(), entry.vendor_name.as_str()), entry))
        .collect::<BTreeMap<_, _>>();
    let mut observed = BTreeSet::new();
    for (backend, relative) in [("cuda", CUDA_ORACLE_PATH), ("hip", HIP_ORACLE_PATH)] {
        let path = workspace_root.join(relative);
        let source = read(&path)?;
        let oracle: OracleSnapshot =
            serde_json::from_str(&source).map_err(|source| Error::OracleJson {
                path: path.clone(),
                source,
            })?;
        for entry in oracle
            .entries
            .iter()
            .filter(|entry| entry.kind == "function")
        {
            let key = (backend, entry.name.as_str());
            let classified_entry = classified.get(&key).ok_or_else(|| {
                Error::Validation(format!(
                    "{} function {} is not classified in the canonical manifest",
                    oracle.inventory_id, entry.name
                ))
            })?;
            let exact = classified_entry.oracle_variants.iter().any(|variant| {
                variant.oracle_source == oracle.inventory_id
                    && variant.oracle_kind == "function"
                    && variant.source_version == oracle.source_version
                    && variant.oracle_signature == entry.normalized_signature
                    && variant.oracle_signature_hash == entry.signature_hash
                    && variant.platforms == entry.platforms
                    && variant.provenance == entry.provenance
            });
            if !exact {
                return Err(Error::Validation(format!(
                    "{} function {} classification is stale",
                    oracle.inventory_id, entry.name
                )));
            }
            observed.insert((backend.to_owned(), entry.name.clone()));
        }
        if oracle.provenance.trim().is_empty() {
            return Err(Error::Validation(format!(
                "{} has empty provenance",
                oracle.inventory_id
            )));
        }
    }

    let vendor_path = workspace_root.join(VENDOR_FUNCTION_UNION_PATH);
    let vendor_source = read(&vendor_path)?;
    let vendor: VendorFunctionUnion =
        serde_json::from_str(&vendor_source).map_err(|source| Error::OracleJson {
            path: vendor_path,
            source,
        })?;
    for function in &vendor.functions {
        let key = (function.backend.as_str(), function.name.as_str());
        let classified_entry = classified.get(&key).ok_or_else(|| {
            Error::Validation(format!(
                "official vendor {} {} is not classified in the canonical manifest",
                function.kind, function.name
            ))
        })?;
        let rust_function_alias_collision = classified_entry.vendor_kind == "function"
            && function.kind == "alias"
            && classified_entry.oracle_variants.iter().any(|variant| {
                variant.oracle_kind == "function"
                    && matches!(
                        variant.oracle_source.as_str(),
                        "cudarc-0.19.9" | "rocmrc-0.5.0"
                    )
            });
        if classified_entry.vendor_kind != function.kind && !rust_function_alias_collision {
            return Err(Error::Validation(format!(
                "{}:{} kind is {}, expected {}",
                function.backend, function.name, classified_entry.vendor_kind, function.kind
            )));
        }
        for expected in &function.variants {
            let exact = classified_entry
                .oracle_variants
                .iter()
                .chain(
                    classified_entry
                        .alias_collision
                        .iter()
                        .flat_map(|collision| &collision.oracle_variants),
                )
                .any(|variant| {
                    variant.oracle_source == expected.inventory_id
                        && variant.oracle_kind == function.kind
                        && variant.source_version == expected.source_version
                        && variant.oracle_signature == expected.normalized_signature
                        && variant.oracle_signature_hash == expected.signature_hash
                        && variant.aliases == expected.aliases
                        && variant.alias_of == expected.alias_of
                        && variant.platforms == expected.platforms
                        && variant.provenance == expected.provenance
                });
            if !exact {
                return Err(Error::Validation(format!(
                    "{}:{} is missing exact {} variant {}",
                    function.backend, function.name, expected.inventory_id, expected.signature_hash
                )));
            }
        }
        observed.insert((function.backend.clone(), function.name.clone()));
    }
    for entry in &manifest.raw_inventory {
        if !observed.contains(&(entry.backend.clone(), entry.vendor_name.clone())) {
            return Err(Error::Validation(format!(
                "raw inventory contains {}:{} absent from pinned Rust and official vendor function inventories",
                entry.backend, entry.vendor_name
            )));
        }
    }
    Ok(())
}

fn validate_snapshot_type_constant_coverage(
    workspace_root: &Path,
    manifest: &ApiManifest,
    backend: &str,
    relative: &str,
) -> Result<(), Error> {
    let path = workspace_root.join(relative);
    let source = read(&path)?;
    let snapshot: OracleSnapshot =
        serde_json::from_str(&source).map_err(|source| Error::OracleJson {
            path: path.clone(),
            source,
        })?;
    for oracle in &snapshot.entries {
        if matches!(
            oracle.kind.as_str(),
            "type" | "opaque_handle" | "callback" | "struct" | "union"
        ) {
            if oracle.name.starts_with("_bindgen")
                || oracle.name.starts_with("__bindgen")
                || oracle.name.contains("BindgenBitfieldUnit")
            {
                continue;
            }
            let public_name = sync::public_type_name(backend, &oracle.name);
            let public = manifest
                .types
                .iter()
                .find(|entry| {
                    (entry.name == public_name
                        || entry.tag.as_deref() == Some(public_name.as_str()))
                        && entry
                            .backend
                            .as_deref()
                            .is_none_or(|owner| owner == backend)
                })
                .ok_or_else(|| {
                    Error::Validation(format!(
                        "{} type {} is absent from the canonical manifest",
                        snapshot.inventory_id, oracle.name
                    ))
                })?;
            if !public.oracle_variants.iter().any(|variant| {
                variant.oracle_source == snapshot.inventory_id
                    && variant.source_version == snapshot.source_version
                    && variant.oracle_kind == oracle.kind
                    && variant.oracle_signature == oracle.normalized_signature
                    && variant.oracle_signature_hash == oracle.signature_hash
                    && variant.platforms == oracle.platforms
                    && variant.provenance == oracle.provenance
            }) {
                return Err(Error::Validation(format!(
                    "{} type {} has stale or missing exact declaration evidence",
                    snapshot.inventory_id, oracle.name
                )));
            }
        } else if matches!(oracle.kind.as_str(), "constant" | "enum_value") {
            let public_name = format!("OCGPU_{}_{}", backend.to_ascii_uppercase(), oracle.name);
            let public = manifest
                .constants
                .iter()
                .find(|entry| entry.name == public_name)
                .ok_or_else(|| {
                    Error::Validation(format!(
                        "{} {} {} is absent from the canonical manifest",
                        snapshot.inventory_id, oracle.kind, oracle.name
                    ))
                })?;
            if !public.oracle_variants.iter().any(|variant| {
                variant.oracle_source == snapshot.inventory_id
                    && variant.source_version == snapshot.source_version
                    && variant.oracle_kind == oracle.kind
                    && variant.oracle_signature == oracle.normalized_signature
                    && variant.oracle_signature_hash == oracle.signature_hash
                    && variant.platforms == oracle.platforms
                    && variant.provenance == oracle.provenance
            }) {
                return Err(Error::Validation(format!(
                    "{} {} {} has stale or missing exact declaration evidence",
                    snapshot.inventory_id, oracle.kind, oracle.name
                )));
            }
        }
    }
    Ok(())
}

fn unique_id(ids: &mut BTreeSet<u32>, id: u32, name: &str) -> Result<(), Error> {
    if ids.insert(id) {
        Ok(())
    } else {
        Err(Error::Validation(format!(
            "stable_id 0x{id:08x} is duplicated at {name}"
        )))
    }
}

fn validate_target(target: &str, owner: &str) -> Result<(), Error> {
    if matches!(
        target,
        "aarch64-unknown-linux-gnu" | "x86_64-unknown-linux-gnu" | "x86_64-pc-windows-msvc"
    ) {
        Ok(())
    } else {
        Err(Error::Validation(format!(
            "{owner} references unsupported target {target}"
        )))
    }
}

fn validate_parameters(owner: &str, params: &[Parameter]) -> Result<(), Error> {
    let mut names = BTreeSet::new();
    for param in params {
        if !names.insert(param.name.as_str()) {
            return Err(Error::Validation(format!(
                "{owner} has duplicate parameter {}",
                param.name
            )));
        }
        if param.name.starts_with("__ocgpu_") {
            return Err(Error::Validation(format!(
                "{owner}.{} uses the generator-reserved flat-shim prefix __ocgpu_",
                param.name
            )));
        }
        if !matches!(param.direction.as_str(), "in" | "out" | "inout" | "unknown") {
            return Err(Error::Validation(format!(
                "{owner}.{} has invalid direction {}",
                param.name, param.direction
            )));
        }
        if !matches!(
            param.semantic_status.as_str(),
            "reviewed_override" | "reviewed_manifest" | "declaration_fact"
        ) {
            return Err(Error::Validation(format!(
                "{owner}.{} has invalid or absent semantic status {}",
                param.name, param.semantic_status
            )));
        }
        if param.semantic_provenance.trim().is_empty() {
            return Err(Error::Validation(format!(
                "{owner}.{} needs semantic provenance",
                param.name
            )));
        }
    }
    Ok(())
}

fn validate_backend<'a>(
    backend: &str,
    function: &FunctionEntry,
    raw: &'a BackendFunction,
    raw_names: &mut BTreeSet<&'a str>,
) -> Result<(), Error> {
    validate_parameters(&raw.raw_name, &raw.params)?;
    validate_signature(
        &raw.raw_name,
        &raw.return_type,
        &raw.params,
        &raw.signature_hash,
    )?;
    if !raw_names.insert(raw.raw_name.as_str()) {
        return Err(Error::Validation(format!(
            "duplicate raw function {}",
            raw.raw_name
        )));
    }
    let expected = raw_prefixed_name(backend, &raw.vendor_symbol)?;
    if raw.raw_name != expected {
        return Err(Error::Validation(format!(
            "{} must preserve {} exactly as {expected}",
            raw.raw_name, raw.vendor_symbol
        )));
    }
    if !matches!(
        raw.classification.as_str(),
        "covered_exact" | "covered_adapter" | "deprecated_covered"
    ) {
        return Err(Error::Validation(format!(
            "{}.{} has unsupported coverage classification {}",
            function.id, backend, raw.classification
        )));
    }
    for (platform, spec) in [("linux", &raw.linux), ("windows", &raw.windows)] {
        if spec.available && spec.symbols.is_empty() {
            return Err(Error::Validation(format!(
                "{}.{}.{platform} is available without symbols",
                function.id, backend
            )));
        }
        if spec.reason.trim().is_empty() {
            return Err(Error::Validation(format!(
                "{}.{}.{platform} needs a classification reason",
                function.id, backend
            )));
        }
    }
    Ok(())
}

fn raw_prefixed_name(backend: &str, vendor_symbol: &str) -> Result<String, Error> {
    let (vendor_prefix, public_prefix) = match backend {
        "cuda" => ("cu", "ocgpuCu"),
        "hip" => ("hip", "ocgpuHip"),
        _ => return Err(Error::Validation(format!("unknown backend {backend}"))),
    };
    let suffix = vendor_symbol.strip_prefix(vendor_prefix).ok_or_else(|| {
        Error::Validation(format!(
            "{vendor_symbol} does not begin with {vendor_prefix}"
        ))
    })?;
    Ok(format!("{public_prefix}{suffix}"))
}

fn validate_signature(
    name: &str,
    return_type: &str,
    params: &[Parameter],
    actual: &str,
) -> Result<(), Error> {
    let expected = format_hash(signature_hash(return_type, params));
    if actual == expected {
        Ok(())
    } else {
        Err(Error::Validation(format!(
            "{name} signature_hash is {actual}, expected {expected}"
        )))
    }
}

fn normalized_signature(return_type: &str, params: &[Parameter]) -> String {
    let mut normalized = String::from("cc=C;ret=");
    normalized.push_str(&return_type.replace(' ', ""));
    normalized.push_str(";args=(");
    for (index, param) in params.iter().enumerate() {
        if index != 0 {
            normalized.push(',');
        }
        normalized.push_str(&param.type_name.replace(' ', ""));
    }
    normalized.push(')');
    normalized
}

fn signature_hash(return_type: &str, params: &[Parameter]) -> u64 {
    fnv1a(normalized_signature(return_type, params).as_bytes())
}

fn type_layout_hash(entry: &TypeEntry) -> u64 {
    let mut normalized = format!(
        "name={};kind={};rust={};tag={};size64={};align64={}",
        entry.name,
        entry.kind,
        entry.rust_type.as_deref().unwrap_or(""),
        entry.tag.as_deref().unwrap_or(""),
        entry.size_64,
        entry.align_64
    );
    if entry.kind == "callback" {
        normalized.push_str(";ret=");
        normalized.push_str(entry.return_type.as_deref().unwrap_or(""));
        normalized.push_str(";args=(");
        for (index, param) in entry.params.iter().enumerate() {
            if index != 0 {
                normalized.push(',');
            }
            normalized.push_str(&param.type_name.replace(' ', ""));
        }
        normalized.push(')');
    }
    if matches!(entry.kind.as_str(), "record" | "union") {
        normalized.push_str(";fields=(");
        for (index, field) in entry.fields.iter().enumerate() {
            if index != 0 {
                normalized.push(',');
            }
            write!(
                normalized,
                "{}:{}:{}@{}",
                field.name,
                field.c_name.as_deref().unwrap_or(&field.name),
                field.type_name.replace(' ', ""),
                field.offset_64
            )
            .expect("String write");
        }
        normalized.push(')');
    }
    fnv1a(normalized.as_bytes())
}

fn table_layout_hash(function_count: usize) -> u64 {
    let mut hash = FNV_OFFSET;
    hash = fnv_u64(hash, 24 + function_count as u64 * 8);
    hash = fnv_u64(hash, 8);
    for offset in [0_u64, 4, 8, 12, 16, 20] {
        hash = fnv_u64(hash, offset);
    }
    for index in 0..function_count {
        hash = fnv_u64(hash, 24 + index as u64 * 8);
    }
    hash
}

fn raw_only_entries<'a>(
    manifest: &'a ApiManifest,
    backend: &'a str,
) -> impl Iterator<Item = &'a RawInventoryEntry> {
    let mut entries = manifest
        .raw_inventory
        .iter()
        .filter(move |entry| entry.backend == backend && entry.emitted && entry.common_id.is_none())
        .collect::<Vec<_>>();
    entries.sort_by_key(|entry| entry.table_order.unwrap_or(u32::MAX));
    entries.into_iter()
}

fn table_field_count(manifest: &ApiManifest, surface: &str) -> usize {
    match surface {
        "common" => manifest.functions.len(),
        "cuda" | "hip" => manifest.functions.len() + raw_only_entries(manifest, surface).count(),
        _ => 0,
    }
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn fnv_u64(mut hash: u64, value: u64) -> u64 {
    for byte in value.to_le_bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn parse_hash(value: &str) -> Result<u64, Error> {
    let digits = value
        .strip_prefix("fnv1a64:")
        .ok_or_else(|| Error::Validation(format!("hash {value:?} must use the fnv1a64: prefix")))?;
    u64::from_str_radix(digits, 16)
        .map_err(|_| Error::Validation(format!("hash {value:?} is not 16 hexadecimal digits")))
}

fn format_hash(hash: u64) -> String {
    format!("fnv1a64:{hash:016x}")
}

#[allow(clippy::too_many_lines)]
fn render_rust(manifest: &ApiManifest) -> Result<String, Error> {
    let mut output = String::from(
        "// SPDX-License-Identifier: CC0-1.0\n\
         // Generated by ocgpu-codegen from api/ocgpu-api.toml. Do not edit.\n\n\
         use core::ffi::{c_char, c_void};\n\n",
    );

    for entry in &manifest.types {
        match entry.kind.as_str() {
            "integer" | "pointer_integer" => {
                let documentation = if entry.name == "ocgpuResult" {
                    "/// Result carrier. The declared `OCGPU_*` management codes are stable;\n\
                     /// a dispatched operation may otherwise return its originating backend's\n\
                     /// native CUDA or HIP status code."
                } else {
                    "/// Stable ABI integer typedef."
                };
                writeln!(
                    output,
                    "{documentation}\n/// Layout {}.\npub type {} = {};\n",
                    entry.layout_hash,
                    entry.name,
                    entry.rust_type.as_deref().expect("validated scalar type")
                )
                .expect("writing to String cannot fail");
            }
            "alias" => {
                writeln!(
                    output,
                    "/// Stable ABI typedef (layout {}).\npub type {} = {};\n",
                    entry.layout_hash,
                    entry.name,
                    entry.rust_type.as_deref().expect("validated alias type")
                )
                .expect("writing to String cannot fail");
            }
            "opaque_handle" => {
                let tag = entry.tag.as_deref().expect("validated opaque tag");
                writeln!(
                    output,
                    "/// Opaque handle tag.\n#[repr(C)]\npub struct {tag} {{\n    _private: [u8; 0],\n}}\n\n/// Nullable opaque handle (layout {}).\npub type {} = *mut {tag};\n",
                    entry.layout_hash, entry.name
                )
                .expect("writing to String cannot fail");
            }
            "callback" => {
                writeln!(
                    output,
                    "/// Nullable vendor callback (layout {}).\npub type {} = Option<unsafe extern \"C\" fn(",
                    entry.layout_hash, entry.name
                )
                .expect("writing to String cannot fail");
                for (index, param) in entry.params.iter().enumerate() {
                    if index != 0 {
                        output.push_str(", ");
                    }
                    write!(output, "{}: {}", param.name, param.type_name).expect("String write");
                }
                writeln!(
                    output,
                    ") -> {}>;\n",
                    entry.return_type.as_deref().expect("validated return type")
                )
                .expect("String write");
            }
            "record" => {
                let repr = rust_record_repr(entry);
                writeln!(
                    output,
                    "/// Complete vendor record (layout {}).\n{repr}\n#[derive(Clone, Copy)]\n#[allow(clippy::pub_underscore_fields)]\npub struct {} {{",
                    entry.layout_hash,
                    entry.name
                )
                .expect("String write");
                for field in &entry.fields {
                    writeln!(
                        output,
                        "    /// Exact declaration-derived vendor field.\n    pub {}: {},",
                        field.name, field.type_name
                    )
                    .expect("String write");
                }
                output.push_str("}\n\n");
            }
            "union" => {
                let repr = rust_record_repr(entry);
                writeln!(
                    output,
                    "/// Complete vendor union (layout {}).\n{repr}\n#[derive(Clone, Copy)]\n#[allow(clippy::pub_underscore_fields)]\npub union {} {{",
                    entry.layout_hash,
                    entry.name
                )
                .expect("String write");
                for field in &entry.fields {
                    writeln!(
                        output,
                        "    /// Exact declaration-derived vendor field.\n    pub {}: {},",
                        field.name, field.type_name
                    )
                    .expect("String write");
                }
                output.push_str("}\n\n");
            }
            "opaque_record" => {
                writeln!(
                    output,
                    "/// Incomplete vendor record; ABI v1 exposes it only behind pointers.\n#[repr(C)]\npub struct {} {{\n    _private: [u8; 0],\n}}\n",
                    entry.name
                )
                .expect("String write");
            }
            "opaque_union" => {
                writeln!(
                    output,
                    "/// Incomplete vendor union; ABI v1 exposes it only behind pointers.\n#[repr(C)]\npub union {} {{\n    _private: [u8; 0],\n}}\n",
                    entry.name
                )
                .expect("String write");
            }
            other => {
                return Err(Error::Validation(format!(
                    "cannot render unsupported type kind {other} for {}",
                    entry.name
                )));
            }
        }
    }

    for entry in manifest.constants.iter().filter(|entry| entry.emitted) {
        render_rust_constant(&mut output, entry)?;
        if entry.name.starts_with("OCGPU_DEVICE_ATTRIBUTE_") {
            let suffix = entry
                .name
                .strip_prefix("OCGPU_DEVICE_ATTRIBUTE_")
                .expect("prefix checked");
            for (backend, type_name, prefix) in [
                (
                    "cuda",
                    "ocgpuCUdevice_attribute",
                    "OCGPU_CUDA_DEVICE_ATTRIBUTE_",
                ),
                (
                    "hip",
                    "ocgpuHipDeviceAttribute_t",
                    "OCGPU_HIP_DEVICE_ATTRIBUTE_",
                ),
            ] {
                let value = entry.backend_values[backend];
                writeln!(
                    output,
                    "/// Backend-native mapping for `{}`.\npub const {prefix}{suffix}: {type_name} = {value};\n",
                    entry.name
                )
                .expect("writing to String cannot fail");
            }
        }
    }

    for entry in &manifest.functions {
        render_fn_alias(
            &mut output,
            &format!("{}Fn", entry.common_name),
            &entry.return_type,
            &entry.params,
            &entry.signature_hash,
        );
    }
    for entry in &manifest.functions {
        render_fn_alias(
            &mut output,
            &format!("{}Fn", entry.cuda.raw_name),
            &entry.cuda.return_type,
            &entry.cuda.params,
            &entry.cuda.signature_hash,
        );
    }
    for entry in &manifest.functions {
        render_fn_alias(
            &mut output,
            &format!("{}Fn", entry.hip.raw_name),
            &entry.hip.return_type,
            &entry.hip.params,
            &entry.hip.signature_hash,
        );
    }
    for backend in ["cuda", "hip"] {
        for entry in raw_only_entries(manifest, backend) {
            render_fn_alias(
                &mut output,
                &format!(
                    "{}Fn",
                    entry
                        .raw_name
                        .as_deref()
                        .expect("validated emitted raw name")
                ),
                entry
                    .abi_return_type
                    .as_deref()
                    .expect("validated emitted return type"),
                &entry.abi_params,
                &entry.oracle_signature_hash,
            );
        }
    }

    for table in &manifest.tables {
        render_table(&mut output, manifest, table)?;
    }

    output.push_str(
        "unsafe extern \"C\" {\n\
         \t/// Negotiate a backend-bound unified ABI table.\n\
         \tpub fn ocgpuGetApi(backend: ocgpuBackend, requested_abi: u32, output_size: usize, output: *mut ocgpuApi_v1) -> ocgpuResult;\n\
         \t/// Negotiate a CUDA raw ABI table.\n\
         \tpub fn ocgpuCuGetApi(requested_abi: u32, output_size: usize, output: *mut ocgpuCuApi_v1) -> ocgpuResult;\n\
         \t/// Negotiate a HIP raw ABI table.\n\
         \tpub fn ocgpuHipGetApi(requested_abi: u32, output_size: usize, output: *mut ocgpuHipApi_v1) -> ocgpuResult;\n\
         }\n",
    );
    Ok(output)
}

fn rust_record_repr(entry: &TypeEntry) -> String {
    if entry.align_64 > 8 {
        format!("#[repr(C, align({}))]", entry.align_64)
    } else {
        "#[repr(C)]".to_owned()
    }
}

fn render_rust_constant(output: &mut String, entry: &ConstantEntry) -> Result<(), Error> {
    let mut groups = BTreeMap::<i64, Vec<String>>::new();
    if entry.platform_values.is_empty() {
        groups.insert(entry.value, Vec::new());
    } else {
        for (platform, value) in &entry.platform_values {
            groups.entry(*value).or_default().push(platform.clone());
        }
    }
    for (value, platforms) in groups {
        let configuration = if platforms.is_empty() {
            None
        } else {
            rust_platform_cfg(&platforms)?
        };
        if let Some(configuration) = configuration {
            writeln!(output, "#[cfg({configuration})]").expect("String write");
        }
        if entry.name.bytes().any(|byte| byte.is_ascii_lowercase()) {
            output.push_str("#[allow(non_upper_case_globals)]\n");
        }
        writeln!(
            output,
            "/// Stable integer constant.\npub const {}: {} = {};\n",
            entry.name,
            entry.type_name,
            separated_decimal(value)
        )
        .expect("writing to String cannot fail");
    }
    Ok(())
}

fn rust_platform_cfg(platforms: &[String]) -> Result<Option<String>, Error> {
    let targets = platforms
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let all = BTreeSet::from([
        "aarch64-unknown-linux-gnu",
        "x86_64-pc-windows-msvc",
        "x86_64-unknown-linux-gnu",
    ]);
    if targets == all {
        return Ok(None);
    }
    let linux = BTreeSet::from(["aarch64-unknown-linux-gnu", "x86_64-unknown-linux-gnu"]);
    if targets == linux {
        return Ok(Some("target_os = \"linux\"".to_owned()));
    }
    if targets == BTreeSet::from(["x86_64-pc-windows-msvc"]) {
        return Ok(Some("target_os = \"windows\"".to_owned()));
    }
    let expressions = targets
        .iter()
        .map(|target| match *target {
            "aarch64-unknown-linux-gnu" => {
                Ok("all(target_os = \"linux\", target_arch = \"aarch64\")")
            }
            "x86_64-unknown-linux-gnu" => {
                Ok("all(target_os = \"linux\", target_arch = \"x86_64\")")
            }
            "x86_64-pc-windows-msvc" => {
                Ok("all(target_os = \"windows\", target_arch = \"x86_64\")")
            }
            unsupported => Err(Error::Validation(format!(
                "constant platform {unsupported} is unsupported"
            ))),
        })
        .collect::<Result<Vec<_>, Error>>()?;
    Ok(Some(if expressions.len() == 1 {
        expressions[0].to_owned()
    } else {
        format!("any({})", expressions.join(", "))
    }))
}

fn separated_decimal(value: i64) -> String {
    let source = value.to_string();
    let (sign, digits) = source
        .strip_prefix('-')
        .map_or(("", source.as_str()), |digits| ("-", digits));
    let first = digits.len() % 3;
    let mut output = String::from(sign);
    if first != 0 {
        output.push_str(&digits[..first]);
    }
    for chunk in digits.as_bytes()[first..].chunks(3) {
        if output.len() > sign.len() {
            output.push('_');
        }
        output.push_str(core::str::from_utf8(chunk).expect("decimal digits are UTF-8"));
    }
    output
}

fn render_fn_alias(
    output: &mut String,
    alias: &str,
    return_type: &str,
    params: &[Parameter],
    hash: &str,
) {
    writeln!(
        output,
        "/// Nullable function-table entry; signature {hash}."
    )
    .expect("writing to String cannot fail");
    write!(output, "pub type {alias} = unsafe extern \"C\" fn(")
        .expect("writing to String cannot fail");
    for (index, param) in params.iter().enumerate() {
        if index != 0 {
            output.push_str(", ");
        }
        write!(output, "{}: {}", param.name, param.type_name)
            .expect("writing to String cannot fail");
    }
    writeln!(output, ") -> {return_type};\n").expect("writing to String cannot fail");
}

fn render_table(
    output: &mut String,
    manifest: &ApiManifest,
    table: &TableEntry,
) -> Result<(), Error> {
    writeln!(
        output,
        "/// Append-only ABI table (layout {}).\n#[repr(C)]\n#[derive(Clone, Copy, Default)]\npub struct {} {{",
        table.layout_hash, table.name
    )
    .expect("writing to String cannot fail");
    output.push_str(
        "    /// Bytes understood by the producer.\n    pub struct_size: u32,\n\
         \t/// Negotiated ABI version.\n    pub abi_version: u32,\n\
         \t/// Backend bound to every entry.\n    pub backend: ocgpuBackend,\n\
         \t/// Negotiated capability bits. HIP tables encode the validated runtime ABI major; CUDA leaves the HIP profile mask zero.\n    pub flags: u32,\n\
         \t/// Runtime-reported driver version.\n    pub driver_version: i32,\n\
         \t/// Reserved and always zero.\n    pub reserved0: u32,\n",
    );
    for entry in &manifest.functions {
        let (field, return_type, params) = match table.surface.as_str() {
            "common" => (&entry.common_name, &entry.return_type, &entry.params),
            "cuda" => (
                &entry.cuda.raw_name,
                &entry.cuda.return_type,
                &entry.cuda.params,
            ),
            "hip" => (
                &entry.hip.raw_name,
                &entry.hip.return_type,
                &entry.hip.params,
            ),
            other => {
                return Err(Error::Validation(format!(
                    "cannot render table {} with unsupported surface {other}",
                    table.name
                )));
            }
        };
        render_table_field(output, field, return_type, params);
    }
    if matches!(table.surface.as_str(), "cuda" | "hip") {
        for entry in raw_only_entries(manifest, &table.surface) {
            render_table_field(
                output,
                entry
                    .raw_name
                    .as_deref()
                    .expect("validated emitted raw name"),
                entry
                    .abi_return_type
                    .as_deref()
                    .expect("validated emitted return type"),
                &entry.abi_params,
            );
        }
    }
    output.push_str("}\n\n");
    let constant_name = upper_snake(&table.name);
    writeln!(
        output,
        "/// Expected 64-bit layout fingerprint.\npub const {constant_name}_LAYOUT_HASH: u64 = 0x{};\n",
        grouped_hex(parse_hash(&table.layout_hash).expect("validated table hash"))
    )
    .expect("String write");
    Ok(())
}

fn render_table_field(output: &mut String, field: &str, return_type: &str, params: &[Parameter]) {
    output.push_str(
        "    /// Nullable when unavailable.\n    #[allow(clippy::type_complexity)]\n    pub ",
    );
    write!(output, "{field}: Option<unsafe extern \"C\" fn(")
        .expect("writing to String cannot fail");
    for (index, param) in params.iter().enumerate() {
        if index != 0 {
            output.push_str(", ");
        }
        write!(output, "{}: {}", param.name, param.type_name)
            .expect("writing to String cannot fail");
    }
    writeln!(output, ") -> {return_type}>,").expect("writing to String cannot fail");
}

#[allow(clippy::too_many_lines)]
fn render_layout_test(manifest: &ApiManifest) -> Result<String, Error> {
    let mut output = String::from(
        "// SPDX-License-Identifier: CC0-1.0\n\
         // Generated by ocgpu-codegen from api/ocgpu-api.toml. Do not edit.\n\n\
         //! Generated ABI layout conformance tests.\n\n\
         #![allow(non_snake_case)]\n\n\
         use core::mem::{align_of, offset_of, size_of};\n\
         use ocgpu_abi::*;\n\n\
         const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;\n\
         const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;\n\n\
         fn feed(mut hash: u64, value: usize) -> u64 {\n\
         \tfor byte in (value as u64).to_le_bytes() {\n\
         \t\thash ^= u64::from(byte);\n\
         \t\thash = hash.wrapping_mul(FNV_PRIME);\n\
         \t}\n\
         \thash\n\
         }\n\n",
    );

    for entry in &manifest.types {
        if matches!(entry.kind.as_str(), "opaque_record" | "opaque_union") || entry.size_64 == 0 {
            continue;
        }
        writeln!(
            output,
            "#[test]\n#[allow(clippy::too_many_lines)]\nfn layout_{}_{:08x}() {{\n    assert_eq!(size_of::<{}>(), {});\n    assert_eq!(align_of::<{}>(), {});",
            upper_snake(&entry.name).to_ascii_lowercase(),
            entry.stable_id,
            entry.name,
            entry.size_64,
            entry.name,
            entry.align_64
        )
        .expect("String write");
        for field in &entry.fields {
            writeln!(
                output,
                "    assert_eq!(offset_of!({}, {}), {});",
                entry.name, field.name, field.offset_64
            )
            .expect("String write");
        }
        output.push_str("}\n\n");
    }

    for table in &manifest.tables {
        let expected = parse_hash(&table.layout_hash)?;
        writeln!(
            output,
            "#[test]\n#[allow(clippy::too_many_lines)]\nfn layout_{}() {{\n    assert_eq!(size_of::<{}>(), {});\n    assert_eq!(align_of::<{}>(), {});\n    let mut hash = feed(FNV_OFFSET, size_of::<{}>());\n    hash = feed(hash, align_of::<{}>());",
            upper_snake(&table.name).to_ascii_lowercase(),
            table.name,
            table.size_64,
            table.name,
            table.align_64,
            table.name,
            table.name
        )
        .expect("String write");
        for field in [
            "struct_size",
            "abi_version",
            "backend",
            "flags",
            "driver_version",
            "reserved0",
        ] {
            writeln!(
                output,
                "    hash = feed(hash, offset_of!({}, {field}));",
                table.name
            )
            .expect("String write");
        }
        for entry in &manifest.functions {
            let field = match table.surface.as_str() {
                "common" => &entry.common_name,
                "cuda" => &entry.cuda.raw_name,
                "hip" => &entry.hip.raw_name,
                other => {
                    return Err(Error::Validation(format!(
                        "cannot render layout for unsupported table surface {other}"
                    )));
                }
            };
            writeln!(
                output,
                "    hash = feed(hash, offset_of!({}, {field}));",
                table.name
            )
            .expect("String write");
        }
        if matches!(table.surface.as_str(), "cuda" | "hip") {
            for entry in raw_only_entries(manifest, &table.surface) {
                let field = entry
                    .raw_name
                    .as_deref()
                    .expect("validated emitted raw name");
                writeln!(
                    output,
                    "    hash = feed(hash, offset_of!({}, {field}));",
                    table.name
                )
                .expect("String write");
            }
        }
        writeln!(
            output,
            "    assert_eq!(hash, 0x{});\n}}\n",
            grouped_hex(expected)
        )
        .expect("String write");
    }
    output.push_str(
        "#[test]\nfn nullable_function_pointer_is_one_pointer() {\n\
         \tassert_eq!(size_of::<Option<ocgpuInitFn>>(), size_of::<usize>());\n\
         \tassert_eq!(size_of::<Option<ocgpuCuInitFn>>(), size_of::<usize>());\n\
         \tassert_eq!(size_of::<Option<ocgpuHipInitFn>>(), size_of::<usize>());\n\
         }\n\n\
         #[test]\nfn abi_v1_pointer_width_is_64() {\n\
         \tassert_eq!(usize::BITS, 64);\n\
         }\n\n\
         #[test]\nfn target_c_long_layout_is_native() {\n\
         \t#[cfg(target_os = \"windows\")]\n\
         \t{\n\
         \t\tassert_eq!(size_of::<core::ffi::c_long>(), 4);\n\
         \t\tassert_eq!(size_of::<core::ffi::c_ulong>(), 4);\n\
         \t}\n\
         \t#[cfg(target_os = \"linux\")]\n\
         \t{\n\
         \t\tassert_eq!(size_of::<core::ffi::c_long>(), 8);\n\
         \t\tassert_eq!(size_of::<core::ffi::c_ulong>(), 8);\n\
         \t}\n\
         }\n",
    );
    Ok(output)
}

#[allow(clippy::too_many_lines)]
fn render_c_layout_test(manifest: &ApiManifest) -> Result<String, Error> {
    let mut output = String::from(
        "/* SPDX-License-Identifier: CC0-1.0 */\n\
         /* Generated by ocgpu-codegen from api/ocgpu-api.toml. Do not edit. */\n\n\
         #include <ocgpu/ocgpu.h>\n\
         #include <stddef.h>\n\
         #include <stdint.h>\n\n\
         #define OCGPU_STATIC_ASSERT(name, expression) \\\n             typedef char ocgpu_static_assert_##name[(expression) ? 1 : -1]\n\n\
         OCGPU_STATIC_ASSERT(abi_v1_pointer_width, UINTPTR_MAX == UINT64_MAX);\n\
         #if defined(_WIN32)\n\
         OCGPU_STATIC_ASSERT(c_long_width, sizeof(long) == 4u);\n\
         OCGPU_STATIC_ASSERT(c_ulong_width, sizeof(unsigned long) == 4u);\n\
         #else\n\
         OCGPU_STATIC_ASSERT(c_long_width, sizeof(long) == 8u);\n\
         OCGPU_STATIC_ASSERT(c_ulong_width, sizeof(unsigned long) == 8u);\n\
         #endif\n\n",
    );
    for entry in &manifest.types {
        if matches!(entry.kind.as_str(), "opaque_record" | "opaque_union") || entry.size_64 == 0 {
            continue;
        }
        let id = format!(
            "{}_{:08x}",
            upper_snake(&entry.name).to_ascii_lowercase(),
            entry.stable_id
        );
        writeln!(
            output,
            "struct ocgpu_align_probe_{id} {{ char byte; {} value; }};\nOCGPU_STATIC_ASSERT({id}_size, sizeof({}) == {}u);\nOCGPU_STATIC_ASSERT({id}_align, offsetof(struct ocgpu_align_probe_{id}, value) == {}u);",
            entry.name, entry.name, entry.size_64, entry.align_64
        )
        .expect("String write");
        for field in &entry.fields {
            let c_name = field.c_name.as_deref().unwrap_or(&field.name);
            writeln!(
                output,
                "OCGPU_STATIC_ASSERT({id}_{}_offset, offsetof({}, {}) == {}u);",
                upper_snake(c_name).to_ascii_lowercase(),
                entry.name,
                c_name,
                field.offset_64
            )
            .expect("String write");
        }
    }
    output.push('\n');
    for entry in manifest.constants.iter().filter(|entry| entry.emitted) {
        render_c_constant_assertion(&mut output, entry)?;
        if let Some(suffix) = entry.name.strip_prefix("OCGPU_DEVICE_ATTRIBUTE_") {
            for (backend, prefix) in [
                ("cuda", "OCGPU_CUDA_DEVICE_ATTRIBUTE_"),
                ("hip", "OCGPU_HIP_DEVICE_ATTRIBUTE_"),
            ] {
                let mapped_name = format!("{prefix}{suffix}");
                let mapped_id = format!(
                    "{}_{:08x}",
                    mapped_name.to_ascii_lowercase(),
                    entry.stable_id
                );
                let mapped_value = entry.backend_values[backend];
                writeln!(
                    output,
                    "OCGPU_STATIC_ASSERT({mapped_id}_value, {mapped_name} == ({mapped_value}));"
                )
                .expect("String write");
            }
        }
    }
    output.push('\n');
    for table in &manifest.tables {
        let id = upper_snake(&table.name).to_ascii_lowercase();
        writeln!(
            output,
            "struct ocgpu_align_probe_{id} {{ char byte; {} value; }};\nOCGPU_STATIC_ASSERT({id}_size, sizeof({}) == {}u);\nOCGPU_STATIC_ASSERT({id}_align, offsetof(struct ocgpu_align_probe_{id}, value) == {}u);",
            table.name, table.name, table.size_64, table.align_64
        )
        .expect("String write");
        for (index, field) in table_field_names(manifest, &table.surface)?
            .iter()
            .enumerate()
        {
            let expected = if index < 6 {
                index * 4
            } else {
                24 + (index - 6) * 8
            };
            writeln!(
                output,
                "OCGPU_STATIC_ASSERT({id}_{}_offset, offsetof({}, {field}) == {expected}u);",
                upper_snake(field).to_ascii_lowercase(),
                table.name
            )
            .expect("String write");
        }
    }
    output.push('\n');
    for table in &manifest.tables {
        let table_id = upper_snake(&table.name).to_ascii_lowercase();
        for field in table_field_names(manifest, &table.surface)?.iter().skip(6) {
            let field_id = upper_snake(field).to_ascii_lowercase();
            writeln!(
                output,
                "OCGPU_STATIC_ASSERT({table_id}_{field_id}_pointer_width, sizeof((({} *)0)->{field}) == sizeof(void (OCGPU_CALL *)(void)));",
                table.name
            )
            .expect("String write");
        }
    }
    output.push_str(
        "\nvoid ocgpu_generated_calling_convention_check(void)\n\
         {\n\
             ocgpuResult (OCGPU_CALL *get_api)(ocgpuBackend, uint32_t, size_t, ocgpuApi_v1 *) = &ocgpuGetApi;\n\
             ocgpuResult (OCGPU_CALL *get_cuda_api)(uint32_t, size_t, ocgpuCuApi_v1 *) = &ocgpuCuGetApi;\n\
             ocgpuResult (OCGPU_CALL *get_hip_api)(uint32_t, size_t, ocgpuHipApi_v1 *) = &ocgpuHipGetApi;\n\
             (void)get_api;\n\
             (void)get_cuda_api;\n\
             (void)get_hip_api;\n\
         }\n",
    );
    Ok(output)
}

fn render_c_constant_assertion(output: &mut String, entry: &ConstantEntry) -> Result<(), Error> {
    let mut groups = BTreeMap::<i64, Vec<String>>::new();
    if entry.platform_values.is_empty() {
        groups.insert(entry.value, Vec::new());
    } else {
        for (platform, value) in &entry.platform_values {
            groups.entry(*value).or_default().push(platform.clone());
        }
    }
    let id = format!(
        "{}_{:08x}",
        entry.name.to_ascii_lowercase(),
        entry.stable_id
    );
    for (index, (value, platforms)) in groups.iter().enumerate() {
        let condition = if platforms.is_empty() {
            None
        } else {
            Some(c_platform_condition(platforms)?)
        };
        if let Some(condition) = &condition {
            writeln!(output, "#if {condition}").expect("String write");
        }
        writeln!(
            output,
            "OCGPU_STATIC_ASSERT({id}_value_{index}, {} == ({}));",
            entry.name, value
        )
        .expect("String write");
        if condition.is_some() {
            output.push_str("#endif\n");
        }
    }
    Ok(())
}

fn c_platform_condition(platforms: &[String]) -> Result<String, Error> {
    let targets = platforms
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let all = BTreeSet::from([
        "aarch64-unknown-linux-gnu",
        "x86_64-pc-windows-msvc",
        "x86_64-unknown-linux-gnu",
    ]);
    if targets == all {
        return Ok("1".to_owned());
    }
    let linux = BTreeSet::from(["aarch64-unknown-linux-gnu", "x86_64-unknown-linux-gnu"]);
    if targets == linux {
        return Ok("!defined(_WIN32)".to_owned());
    }
    if targets == BTreeSet::from(["x86_64-pc-windows-msvc"]) {
        return Ok("defined(_WIN32)".to_owned());
    }
    Ok(targets
        .iter()
        .map(|target| match *target {
            "aarch64-unknown-linux-gnu" => Ok("defined(__aarch64__)"),
            "x86_64-unknown-linux-gnu" => Ok("defined(__x86_64__) && !defined(_WIN32)"),
            "x86_64-pc-windows-msvc" => Ok("defined(_WIN32) && defined(_M_X64)"),
            unsupported => Err(Error::Validation(format!(
                "constant platform {unsupported} is unsupported"
            ))),
        })
        .collect::<Result<Vec<_>, Error>>()?
        .into_iter()
        .map(|condition| format!("({condition})"))
        .collect::<Vec<_>>()
        .join(" || "))
}

fn table_field_names(manifest: &ApiManifest, surface: &str) -> Result<Vec<String>, Error> {
    let mut fields = [
        "struct_size",
        "abi_version",
        "backend",
        "flags",
        "driver_version",
        "reserved0",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<Vec<_>>();
    for entry in &manifest.functions {
        fields.push(match surface {
            "common" => entry.common_name.clone(),
            "cuda" => entry.cuda.raw_name.clone(),
            "hip" => entry.hip.raw_name.clone(),
            other => {
                return Err(Error::Validation(format!(
                    "cannot enumerate unsupported table surface {other}"
                )));
            }
        });
    }
    if matches!(surface, "cuda" | "hip") {
        fields.extend(
            raw_only_entries(manifest, surface)
                .map(|entry| entry.raw_name.clone().expect("validated emitted raw name")),
        );
    }
    Ok(fields)
}

#[allow(clippy::too_many_lines)]
fn render_symbol_descriptors(manifest: &ApiManifest, backend: &str) -> Result<String, Error> {
    let (table, array_name, inventory_name) = match backend {
        "cuda" => ("ocgpuCuApi_v1", "CUDA_RAW_SYMBOLS", "CUDA_RAW_INVENTORY"),
        "hip" => ("ocgpuHipApi_v1", "HIP_RAW_SYMBOLS", "HIP_RAW_INVENTORY"),
        other => {
            return Err(Error::Validation(format!(
                "cannot render symbol descriptors for backend {other}"
            )));
        }
    };
    let mut output = String::from(
        "// SPDX-License-Identifier: CC0-1.0\n\
         // Generated by ocgpu-codegen from api/ocgpu-api.toml. Do not edit.\n\n\
         /// Compile-time raw symbol lookup metadata.\n\
         #[derive(Clone, Copy, Debug, Eq, PartialEq)]\n\
         pub struct RawSymbolDescriptor {\n\
         \t/// Canonical vendor symbol.\n\
         \tpub canonical: &'static str,\n\
         \t/// Name used with the vendor proc-address API.\n\
         \tpub proc_name: &'static str,\n\
         \t/// Direct lookup order, including ABI-identical aliases.\n\
         \tpub direct_names: &'static [&'static str],\n\
         \t/// Linux proc-address version, or zero for direct-only resolution.\n\
         \tpub proc_version_linux: i32,\n\
         \t/// Windows proc-address version, or zero for direct-only resolution.\n\
         \tpub proc_version_windows: i32,\n\
         \t/// Vendor proc-address flags.\n\
         \tpub proc_flags: u64,\n\
         \t/// Byte offset of the nullable field in the generated table.\n\
         \tpub table_offset: usize,\n\
         \t/// Bit 0 Linux `x86_64`, bit 1 Windows `x86_64`, bit 2 Linux `aarch64`.\n\
         \tpub platform_mask: u8,\n\
         }\n\n\
         /// Diagnostic metadata for every raw-oracle function, including un-emitted entries.\n\
         #[derive(Clone, Copy, Debug, Eq, PartialEq)]\n\
         pub struct RawInventoryDescriptor {\n\
         \t/// Canonical upstream function name.\n\
         \tpub canonical: &'static str,\n\
         \t/// Name used with the vendor proc-address API.\n\
         \tpub proc_name: &'static str,\n\
         \t/// Direct lookup order, including ABI-identical aliases.\n\
         \tpub direct_names: &'static [&'static str],\n\
         \t/// Linux proc-address version, or zero for direct-only resolution.\n\
         \tpub proc_version_linux: i32,\n\
         \t/// Windows proc-address version, or zero for direct-only resolution.\n\
         \tpub proc_version_windows: i32,\n\
         \t/// Vendor proc-address flags.\n\
         \tpub proc_flags: u64,\n\
         \t/// Byte offset of an emitted table field; `None` means deliberately un-emitted.\n\
         \tpub table_offset: Option<usize>,\n\
         \t/// Bit 0 Linux `x86_64`, bit 1 Windows `x86_64`, bit 2 Linux `aarch64`.\n\
         \tpub platform_mask: u8,\n\
         \t/// Machine-readable manifest coverage classification.\n\
         \tpub classification: &'static str,\n\
         }\n\n",
    );
    writeln!(
        output,
        "/// Field-aligned descriptors for every callable {backend} ABI v1 entry.\npub const {array_name}: &[RawSymbolDescriptor] = &["
    )
    .expect("String write");
    for function in &manifest.functions {
        let raw = if backend == "cuda" {
            &function.cuda
        } else {
            &function.hip
        };
        let inventory = manifest
            .raw_inventory
            .iter()
            .find(|entry| entry.backend == backend && entry.vendor_name == raw.vendor_symbol);
        let proc_name = inventory
            .and_then(|entry| entry.proc_name.as_deref())
            .unwrap_or(&raw.vendor_symbol);
        let direct_names = merge_resolution_names(
            &raw.fallback_order,
            inventory.map_or(&[], |entry| entry.direct_names.as_slice()),
        );
        let proc_version_linux = if raw.linux.proc_version == 0 {
            inventory.map_or(0, |entry| entry.proc_version_linux)
        } else {
            raw.linux.proc_version
        };
        let proc_version_windows = if raw.windows.proc_version == 0 {
            inventory.map_or(0, |entry| entry.proc_version_windows)
        } else {
            raw.windows.proc_version
        };
        let proc_flags = if raw.proc_address_flags == 0 {
            inventory.map_or(0, |entry| entry.proc_flags)
        } else {
            raw.proc_address_flags
        };
        let mask = platform_mask(&raw.linux.symbols, &raw.windows.symbols, true);
        write_descriptor(
            &mut output,
            table,
            &raw.raw_name,
            &raw.vendor_symbol,
            proc_name,
            &direct_names,
            proc_version_linux,
            proc_version_windows,
            proc_flags,
            mask,
        );
    }
    for entry in raw_only_entries(manifest, backend) {
        let raw_name = entry.raw_name.as_deref().expect("validated raw name");
        let linux = entry
            .platforms
            .iter()
            .any(|platform| platform == "x86_64-unknown-linux-gnu");
        let windows = entry
            .platforms
            .iter()
            .any(|platform| platform == "x86_64-pc-windows-msvc");
        let aarch64 = entry
            .platforms
            .iter()
            .any(|platform| platform == "aarch64-unknown-linux-gnu");
        let mask = u8::from(linux) | (u8::from(windows) << 1) | (u8::from(aarch64) << 2);
        write_descriptor(
            &mut output,
            table,
            raw_name,
            &entry.vendor_name,
            entry.proc_name.as_deref().expect("validated proc name"),
            &entry.direct_names,
            entry.proc_version_linux,
            entry.proc_version_windows,
            entry.proc_flags,
            mask,
        );
    }
    output.push_str("];\n\n");
    writeln!(
        output,
        "/// Exhaustive descriptors for all pinned {backend} raw-oracle functions.\npub const {inventory_name}: &[RawInventoryDescriptor] = &["
    )
    .expect("String write");
    for entry in manifest
        .raw_inventory
        .iter()
        .filter(|entry| entry.backend == backend && entry.raw_name.is_some())
    {
        let common = manifest.functions.iter().find_map(|function| {
            let raw = if backend == "cuda" {
                &function.cuda
            } else {
                &function.hip
            };
            (raw.vendor_symbol == entry.vendor_name).then_some(raw)
        });
        let direct_names = common.map_or_else(
            || entry.direct_names.clone(),
            |raw| merge_resolution_names(&raw.fallback_order, &entry.direct_names),
        );
        let proc_version_linux = common.map_or(entry.proc_version_linux, |raw| {
            if raw.linux.proc_version == 0 {
                entry.proc_version_linux
            } else {
                raw.linux.proc_version
            }
        });
        let proc_version_windows = common.map_or(entry.proc_version_windows, |raw| {
            if raw.windows.proc_version == 0 {
                entry.proc_version_windows
            } else {
                raw.windows.proc_version
            }
        });
        let proc_flags = common.map_or(entry.proc_flags, |raw| {
            if raw.proc_address_flags == 0 {
                entry.proc_flags
            } else {
                raw.proc_address_flags
            }
        });
        let mask = raw_inventory_platform_mask(entry);
        write_inventory_descriptor(
            &mut output,
            table,
            entry,
            &direct_names,
            proc_version_linux,
            proc_version_windows,
            proc_flags,
            mask,
        );
    }
    output.push_str("];\n");
    Ok(output)
}

fn merge_resolution_names(primary: &[String], secondary: &[String]) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut output = Vec::new();
    for name in primary.iter().chain(secondary) {
        if seen.insert(name.as_str()) {
            output.push(name.clone());
        }
    }
    output
}

fn raw_inventory_platform_mask(entry: &RawInventoryEntry) -> u8 {
    let linux = entry
        .platforms
        .iter()
        .any(|platform| platform == "x86_64-unknown-linux-gnu");
    let windows = entry
        .platforms
        .iter()
        .any(|platform| platform == "x86_64-pc-windows-msvc");
    let aarch64 = entry
        .platforms
        .iter()
        .any(|platform| platform == "aarch64-unknown-linux-gnu");
    u8::from(linux) | (u8::from(windows) << 1) | (u8::from(aarch64) << 2)
}

fn platform_mask(linux_symbols: &[String], windows_symbols: &[String], aarch64: bool) -> u8 {
    u8::from(!linux_symbols.is_empty())
        | (u8::from(!windows_symbols.is_empty()) << 1)
        | (u8::from(aarch64 && !linux_symbols.is_empty()) << 2)
}

#[allow(clippy::too_many_arguments)]
fn write_descriptor(
    output: &mut String,
    table: &str,
    raw_name: &str,
    canonical: &str,
    proc_name: &str,
    direct_names: &[String],
    proc_version_linux: u32,
    proc_version_windows: u32,
    proc_flags: u64,
    platform_mask: u8,
) {
    writeln!(
        output,
        "    RawSymbolDescriptor {{\n        canonical: {canonical:?},\n        proc_name: {proc_name:?},\n        direct_names: &{direct_names:?},\n        proc_version_linux: {proc_version_linux},\n        proc_version_windows: {proc_version_windows},\n        proc_flags: {proc_flags},\n        table_offset: core::mem::offset_of!(ocgpu_abi::{table}, {raw_name}),\n        platform_mask: {platform_mask},\n    }},"
    )
    .expect("String write");
}

#[allow(clippy::too_many_arguments)]
fn write_inventory_descriptor(
    output: &mut String,
    table: &str,
    entry: &RawInventoryEntry,
    direct_names: &[String],
    proc_version_linux: u32,
    proc_version_windows: u32,
    proc_flags: u64,
    platform_mask: u8,
) {
    let offset = match (entry.emitted, entry.raw_name.as_deref()) {
        (true, Some(raw_name)) => {
            format!("Some(core::mem::offset_of!(ocgpu_abi::{table}, {raw_name}))")
        }
        _ => "None".to_owned(),
    };
    let canonical = &entry.vendor_name;
    let proc_name = entry.proc_name.as_deref().unwrap_or(canonical);
    let classification = &entry.classification;
    writeln!(
        output,
        "    RawInventoryDescriptor {{\n        canonical: {canonical:?},\n        proc_name: {proc_name:?},\n        direct_names: &{direct_names:?},\n        proc_version_linux: {proc_version_linux},\n        proc_version_windows: {proc_version_windows},\n        proc_flags: {proc_flags},\n        table_offset: {offset},\n        platform_mask: {platform_mask},\n        classification: {classification:?},\n    }},"
    )
    .expect("String write");
}

fn render_header(workspace_root: &Path, manifest: &ApiManifest) -> Result<String, Error> {
    let mut config = cbindgen::Config::from_file(workspace_root.join("cbindgen.toml"))
        .map_err(|error| Error::Cbindgen(error.clone()))?;
    config.header = Some("/* SPDX-License-Identifier: CC0-1.0 */".to_owned());
    config.documentation = true;
    config.sort_by = cbindgen::SortKey::Name;
    config
        .defines
        .insert("target_os = windows".to_owned(), "_WIN32".to_owned());
    config
        .defines
        .insert("target_os = linux".to_owned(), "__linux__".to_owned());
    config.export.include = manifest
        .types
        .iter()
        .map(|entry| entry.name.clone())
        .chain(manifest.tables.iter().map(|entry| entry.name.clone()))
        .collect();
    let bindings = cbindgen::Builder::new()
        .with_crate(workspace_root.join("crates/ocgpu-abi"))
        .with_config(config)
        .generate()
        .map_err(|error| Error::Cbindgen(error.to_string()))?;
    let mut bytes = Vec::new();
    bindings.write(&mut bytes);
    let raw = String::from_utf8(bytes)
        .map_err(|error| Error::Cbindgen(format!("header was not UTF-8: {error}")))?;
    decorate_header(&raw, manifest)
}

// Header decoration owns the C99 normalization, ABI decoration, and generated
// constant casts as one deterministic post-cbindgen transformation.
#[allow(clippy::too_many_lines)]
fn decorate_header(raw: &str, manifest: &ApiManifest) -> Result<String, Error> {
    let macro_block = "\n#ifndef UINTPTR_MAX\n#  error \"ocgpu ABI version 1 requires uintptr_t\"\n#elif UINTPTR_MAX != UINT64_MAX\n#  error \"ocgpu ABI version 1 supports only 64-bit targets\"\n#endif\n\n#ifndef OCGPU_API\n#  if defined(_WIN32) && defined(OCGPU_SHARED)\n#    if defined(OCGPU_BUILDING_LIBRARY)\n#      define OCGPU_API __declspec(dllexport)\n#    else\n#      define OCGPU_API __declspec(dllimport)\n#    endif\n#  elif defined(__GNUC__) && defined(OCGPU_BUILDING_LIBRARY)\n#    define OCGPU_API __attribute__((visibility(\"default\")))\n#  else\n#    define OCGPU_API\n#  endif\n#endif\n\n#ifndef OCGPU_CALL\n#  define OCGPU_CALL\n#endif\n\n#ifndef OCGPU_ALIGN\n#  if defined(_MSC_VER)\n#    define OCGPU_ALIGN(bytes) __declspec(align(bytes))\n#  elif defined(__clang__) || defined(__GNUC__)\n#    define OCGPU_ALIGN(bytes) __attribute__((aligned(bytes)))\n#  else\n#    error \"ocgpu over-aligned ABI records require MSVC, Clang, or GCC\"\n#  endif\n#endif\n";
    let mut output = raw.replacen(
        "#include <stdint.h>\n",
        &format!("#include <stdint.h>\n{macro_block}"),
        1,
    );
    // cbindgen conservatively includes stdlib.h for size_t; stddef.h already
    // supplies the ABI declarations we use and keeps cross-target C99 checks
    // independent of a target libc sysroot.
    output = output.replace("#include <stdlib.h>\n", "");
    let lines = output.lines().collect::<Vec<_>>();
    let mut c99 = String::with_capacity(output.len());
    let mut index = 0;
    while index < lines.len() {
        let line = lines[index];
        let struct_tag = line
            .strip_prefix("typedef struct ")
            .and_then(|rest| rest.strip_suffix(" {"));
        match struct_tag {
            Some(tag)
                if lines
                    .get(index + 1)
                    .is_some_and(|field| field.trim() == "uint8_t _private[0];")
                    && lines
                        .get(index + 2)
                        .is_some_and(|end| end.trim() == format!("}} {tag};")) =>
            {
                writeln!(c99, "typedef struct {tag} {tag};").expect("String write");
                index += 3;
                continue;
            }
            _ => {}
        }
        let union_tag = line
            .strip_prefix("typedef union ")
            .and_then(|rest| rest.strip_suffix(" {"));
        match union_tag {
            Some(tag)
                if lines
                    .get(index + 1)
                    .is_some_and(|field| field.trim() == "uint8_t _private[0];")
                    && lines
                        .get(index + 2)
                        .is_some_and(|end| end.trim() == format!("}} {tag};")) =>
            {
                writeln!(c99, "typedef union {tag} {tag};").expect("String write");
                index += 3;
                continue;
            }
            _ => {}
        }
        let line = if line.contains(" (*ocgpu") {
            line.replacen(" (*ocgpu", " (OCGPU_CALL *ocgpu", 1)
        } else {
            line.to_owned()
        };
        writeln!(c99, "{line}").expect("String write");
        index += 1;
    }
    output = expand_overaligned_records(&c99, manifest)?;
    for getter in ["ocgpuGetApi", "ocgpuCuGetApi", "ocgpuHipGetApi"] {
        output = output.replace(
            &format!("ocgpuResult {getter}("),
            &format!("OCGPU_API ocgpuResult OCGPU_CALL {getter}("),
        );
    }
    output = output.replace(
        "#define OCGPU_ABI_VERSION_1 65536",
        "#define OCGPU_ABI_VERSION_1 UINT32_C(0x00010000)",
    );
    for entry in manifest.constants.iter().filter(|entry| entry.emitted) {
        if entry.name == "OCGPU_ABI_VERSION_1" {
            continue;
        }
        let name = &entry.name;
        let type_name = c_constant_type(&entry.type_name);
        let values = entry
            .platform_values
            .values()
            .copied()
            .chain(core::iter::once(entry.value))
            .collect::<BTreeSet<_>>();
        for value in values {
            output = output.replace(
                &format!("#define {name} {value}"),
                &format!("#define {name} (({type_name}){value})"),
            );
        }
        if let Some(suffix) = entry.name.strip_prefix("OCGPU_DEVICE_ATTRIBUTE_") {
            for (backend, mapped_type, prefix) in [
                (
                    "cuda",
                    "ocgpuCUdevice_attribute",
                    "OCGPU_CUDA_DEVICE_ATTRIBUTE_",
                ),
                (
                    "hip",
                    "ocgpuHipDeviceAttribute_t",
                    "OCGPU_HIP_DEVICE_ATTRIBUTE_",
                ),
            ] {
                let mapped_value = entry.backend_values[backend];
                let mapped_name = format!("{prefix}{suffix}");
                output = output.replace(
                    &format!("#define {mapped_name} {mapped_value}"),
                    &format!("#define {mapped_name} (({mapped_type}){mapped_value})"),
                );
            }
        }
    }
    let flat = render_flat_header(manifest);
    if let Some(index) = output.rfind("#ifdef __cplusplus") {
        output.insert_str(index, &flat);
    } else {
        output.push_str(&flat);
    }
    Ok(output)
}

fn expand_overaligned_records(raw: &str, manifest: &ApiManifest) -> Result<String, Error> {
    let mut output = raw.to_owned();
    for entry in manifest
        .types
        .iter()
        .filter(|entry| entry.align_64 > 8 && matches!(entry.kind.as_str(), "record" | "union"))
    {
        let keyword = if entry.kind == "union" {
            "union"
        } else {
            "struct"
        };
        let forward = format!("typedef {keyword} {} {};", entry.name, entry.name);
        if !output.contains(&forward) {
            return Err(Error::Validation(format!(
                "cbindgen omitted the expected forward declaration for over-aligned {}",
                entry.name
            )));
        }
        let mut definition = format!(
            "typedef {keyword} OCGPU_ALIGN({}) {} {{\n",
            entry.align_64, entry.name
        );
        for field in &entry.fields {
            let name = field.c_name.as_deref().unwrap_or(&field.name);
            writeln!(
                definition,
                "  {};",
                c_field_declaration(manifest, &field.type_name, name, &mut BTreeSet::new())?
            )
            .expect("String write");
        }
        writeln!(definition, "}} {};", entry.name).expect("String write");
        output = output.replacen(&forward, &definition, 1);
    }
    Ok(output)
}

fn c_field_declaration(
    manifest: &ApiManifest,
    rust_type: &str,
    name: &str,
    visiting: &mut BTreeSet<String>,
) -> Result<String, Error> {
    let value = rust_type.trim();
    if let Some(body) = value
        .strip_prefix('[')
        .and_then(|body| body.strip_suffix(']'))
    {
        let (element, count) = body.rsplit_once(';').ok_or_else(|| {
            Error::Validation(format!("malformed generated array field type {rust_type}"))
        })?;
        let count = count.trim().strip_suffix("usize").unwrap_or(count.trim());
        if count.is_empty() || !count.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(Error::Validation(format!(
                "malformed generated array length in {rust_type}"
            )));
        }
        return c_field_declaration(
            manifest,
            element.trim(),
            &format!("{name}[{count}]"),
            visiting,
        );
    }
    let alias_target = manifest
        .types
        .iter()
        .find(|entry| entry.name == value)
        .and_then(|entry| entry.rust_type.as_deref());
    if let Some(target) = alias_target {
        if !visiting.insert(value.to_owned()) {
            return Err(Error::Validation(format!(
                "cyclic generated C field alias at {value}"
            )));
        }
        let declaration = c_field_declaration(manifest, target, name, visiting);
        visiting.remove(value);
        return declaration;
    }
    Ok(c_declaration(value, name))
}

fn c_constant_type(type_name: &str) -> &str {
    match type_name {
        "u8" => "uint8_t",
        "u16" => "uint16_t",
        "u32" => "uint32_t",
        "u64" => "uint64_t",
        "usize" => "uintptr_t",
        "i8" => "int8_t",
        "i16" => "int16_t",
        "i32" => "int32_t",
        "i64" => "int64_t",
        other => other,
    }
}

fn render_flat_header(manifest: &ApiManifest) -> String {
    let mut output = format!(
        "#if defined(OCGPU_ENABLE_FLAT_C_EXPORTS)\n\
         /* Convenience ABI: unified leaf calls take `ocgpuBackend` first; raw CUDA/HIP\n\
          * leaf calls retain their vendor-shaped arguments and return types. There is no\n\
          * process-global backend selection. Missing result symbols return {};\n\
          * pointer returns use {}; integer returns use {}; void calls are {}; aggregate/POD\n\
          * returns use a {} object representation. */\n",
        manifest.manifest.raw_missing_result,
        manifest.manifest.raw_missing_pointer,
        manifest.manifest.raw_missing_integer,
        manifest.manifest.raw_missing_void,
        manifest.manifest.raw_missing_aggregate,
    );
    for function in &manifest.functions {
        let mut params = vec![("backend", "ocgpuBackend")];
        params.extend(
            function
                .params
                .iter()
                .map(|param| (param.name.as_str(), param.type_name.as_str())),
        );
        render_c_function_declaration(
            &mut output,
            &function.common_name,
            &function.return_type,
            &params,
        );
    }
    for backend in ["cuda", "hip"] {
        for function in &manifest.functions {
            let raw = if backend == "cuda" {
                &function.cuda
            } else {
                &function.hip
            };
            let params = raw
                .params
                .iter()
                .map(|param| (param.name.as_str(), param.type_name.as_str()))
                .collect::<Vec<_>>();
            render_c_function_declaration(&mut output, &raw.raw_name, &raw.return_type, &params);
        }
        for entry in raw_only_entries(manifest, backend) {
            let params = entry
                .abi_params
                .iter()
                .map(|param| (param.name.as_str(), param.type_name.as_str()))
                .collect::<Vec<_>>();
            render_c_function_declaration(
                &mut output,
                entry.raw_name.as_deref().expect("validated raw name"),
                entry
                    .abi_return_type
                    .as_deref()
                    .expect("validated return type"),
                &params,
            );
        }
    }
    output.push_str("#endif\n\n");
    output
}

fn render_c_function_declaration(
    output: &mut String,
    name: &str,
    return_type: &str,
    params: &[(&str, &str)],
) {
    write!(
        output,
        "OCGPU_API {}(",
        c_declaration(return_type, &format!("OCGPU_CALL {name}"))
    )
    .expect("String write");
    if params.is_empty() {
        output.push_str("void");
    } else {
        for (index, (param_name, type_name)) in params.iter().enumerate() {
            if index != 0 {
                output.push_str(", ");
            }
            output.push_str(&c_declaration(type_name, param_name));
        }
    }
    output.push_str(");\n");
}

fn c_declaration(rust_type: &str, name: &str) -> String {
    if rust_type == "()" {
        return format!("void {name}");
    }
    let mut remainder = rust_type.trim();
    let mut pointers = Vec::new();
    loop {
        if let Some(rest) = remainder.strip_prefix("*mut ") {
            pointers.push(false);
            remainder = rest.trim();
        } else if let Some(rest) = remainder.strip_prefix("*const ") {
            pointers.push(true);
            remainder = rest.trim();
        } else {
            break;
        }
    }
    let base = match remainder {
        "u8" => "uint8_t",
        "i8" => "int8_t",
        "u16" => "uint16_t",
        "i16" => "int16_t",
        "u32" => "uint32_t",
        "i32" => "int32_t",
        "u64" => "uint64_t",
        "i64" => "int64_t",
        "f32" => "float",
        "f64" => "double",
        "usize" => "size_t",
        "isize" => "ptrdiff_t",
        "c_char" => "char",
        "c_schar" => "signed char",
        "c_uchar" => "unsigned char",
        "c_short" => "short",
        "c_ushort" => "unsigned short",
        "c_int" => "int",
        "c_uint" => "unsigned int",
        "c_long" => "long",
        "c_ulong" => "unsigned long",
        "c_longlong" => "long long",
        "c_ulonglong" => "unsigned long long",
        "c_void" => "void",
        other => other,
    };
    if pointers.is_empty() {
        return format!("{base} {name}");
    }
    let mut output = String::new();
    if pointers.last().copied().unwrap_or(false) {
        output.push_str("const ");
    }
    output.push_str(base);
    output.push(' ');
    for index in (0..pointers.len()).rev() {
        output.push('*');
        if index > 0 && pointers[index - 1] {
            output.push_str(" const");
        }
        output.push(' ');
    }
    output.push_str(name);
    output
}

fn render_def() -> String {
    "; SPDX-License-Identifier: CC0-1.0\n; Generated by ocgpu-codegen. Do not edit.\nEXPORTS\n    ocgpuGetApi\n    ocgpuCuGetApi\n    ocgpuHipGetApi\n".to_owned()
}

fn render_map() -> String {
    "/* SPDX-License-Identifier: CC0-1.0 */\n/* Generated by ocgpu-codegen. Do not edit. */\nOCGPU_1.0 {\n    global:\n        ocgpuGetApi;\n        ocgpuCuGetApi;\n        ocgpuHipGetApi;\n    local:\n        *;\n};\n".to_owned()
}

fn flat_export_names(manifest: &ApiManifest) -> Vec<String> {
    let mut names = ["ocgpuGetApi", "ocgpuCuGetApi", "ocgpuHipGetApi"]
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    names.extend(
        manifest
            .functions
            .iter()
            .map(|entry| entry.common_name.clone()),
    );
    for backend in ["cuda", "hip"] {
        names.extend(manifest.functions.iter().map(|entry| {
            if backend == "cuda" {
                entry.cuda.raw_name.clone()
            } else {
                entry.hip.raw_name.clone()
            }
        }));
        names.extend(
            raw_only_entries(manifest, backend)
                .map(|entry| entry.raw_name.clone().expect("validated emitted raw name")),
        );
    }
    names.sort();
    names.dedup();
    names
}

fn render_flat_def(manifest: &ApiManifest) -> String {
    let mut output = String::from(
        "; SPDX-License-Identifier: CC0-1.0\n; Generated by ocgpu-codegen. Do not edit.\nEXPORTS\n",
    );
    for name in flat_export_names(manifest) {
        writeln!(output, "    {name}").expect("String write");
    }
    output
}

fn render_flat_map(manifest: &ApiManifest) -> String {
    let mut output = String::from(
        "/* SPDX-License-Identifier: CC0-1.0 */\n/* Generated by ocgpu-codegen. Do not edit. */\nOCGPU_1.0 {\n    global:\n",
    );
    for name in flat_export_names(manifest) {
        writeln!(output, "        {name};").expect("String write");
    }
    output.push_str("    local:\n        *;\n};\n");
    output
}

fn flat_rust_imports(manifest: &ApiManifest) -> Vec<String> {
    let mut imports = BTreeSet::from([
        "OCGPU_BACKEND_CUDA".to_owned(),
        "OCGPU_BACKEND_HIP".to_owned(),
        "OCGPU_ERROR_INVALID_ARGUMENT".to_owned(),
        "OCGPU_ERROR_SYMBOL_UNAVAILABLE".to_owned(),
        "ocgpuBackend".to_owned(),
        "ocgpuResult".to_owned(),
    ]);
    let mut collect = |type_name: &str| {
        for token in type_name
            .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
        {
            if token.starts_with("ocgpu") {
                imports.insert(token.to_owned());
            }
        }
    };
    for function in &manifest.functions {
        collect(&function.return_type);
        for param in &function.params {
            collect(&param.type_name);
        }
        for raw in [&function.cuda, &function.hip] {
            collect(&raw.return_type);
            for param in &raw.params {
                collect(&param.type_name);
            }
        }
    }
    for entry in raw_only_entries(manifest, "cuda").chain(raw_only_entries(manifest, "hip")) {
        collect(
            entry
                .abi_return_type
                .as_deref()
                .expect("validated return type"),
        );
        for param in &entry.abi_params {
            collect(&param.type_name);
        }
    }
    imports.into_iter().collect()
}

#[allow(clippy::too_many_lines)]
fn render_export_shims(manifest: &ApiManifest) -> String {
    let mut output = "// SPDX-License-Identifier: CC0-1.0\n\
     // Generated by ocgpu-codegen from api/ocgpu-api.toml. Do not edit.\n\n\
     #[cfg(feature = \"flat-c-exports\")]\n\
     use core::ffi::{c_char, c_void};\n\
     #[cfg(feature = \"flat-c-exports\")]\n\
     use ocgpu_abi::*;\n\n\
     /// Negotiate a backend-bound unified ABI table.\n\
     ///\n\
     /// # Safety\n\
     /// `output` must designate `output_size` writable bytes when non-null.\n\
     #[unsafe(no_mangle)]\n\
     pub unsafe extern \"C\" fn ocgpuGetApi(\n\
     \tbackend: ocgpu_abi::ocgpuBackend,\n\
     \trequested_abi: u32,\n\
     \toutput_size: usize,\n\
     \toutput: *mut ocgpu_abi::ocgpuApi_v1,\n\
     ) -> ocgpu_abi::ocgpuResult {\n\
     \t// SAFETY: This export forwards its documented output-buffer contract unchanged.\n\
     \tunsafe { crate::implementation::get_api(backend, requested_abi, output_size, output) }\n\
     }\n\n\
     /// Negotiate a CUDA raw ABI table.\n\
     ///\n\
     /// # Safety\n\
     /// `output` must designate `output_size` writable bytes when non-null.\n\
     #[unsafe(no_mangle)]\n\
     pub unsafe extern \"C\" fn ocgpuCuGetApi(\n\
     \trequested_abi: u32,\n\
     \toutput_size: usize,\n\
     \toutput: *mut ocgpu_abi::ocgpuCuApi_v1,\n\
     ) -> ocgpu_abi::ocgpuResult {\n\
     \t// SAFETY: This export forwards its documented output-buffer contract unchanged.\n\
     \tunsafe { crate::implementation::get_cuda_api(requested_abi, output_size, output) }\n\
     }\n\n\
     /// Negotiate a HIP raw ABI table.\n\
     ///\n\
     /// # Safety\n\
     /// `output` must designate `output_size` writable bytes when non-null.\n\
     #[unsafe(no_mangle)]\n\
     pub unsafe extern \"C\" fn ocgpuHipGetApi(\n\
     \trequested_abi: u32,\n\
     \toutput_size: usize,\n\
     \toutput: *mut ocgpu_abi::ocgpuHipApi_v1,\n\
     ) -> ocgpu_abi::ocgpuResult {\n\
     \t// SAFETY: This export forwards its documented output-buffer contract unchanged.\n\
     \tunsafe { crate::implementation::get_hip_api(requested_abi, output_size, output) }\n\
     }\n"
    .to_owned();
    output = output.replace(
        "use ocgpu_abi::*;",
        &format!(
            "use ocgpu_abi::{{{}}};",
            flat_rust_imports(manifest).join(", ")
        ),
    );
    writeln!(
        output,
        "\n/// Manifest-selected result for a panic caught at a management getter boundary.\n\
         pub(crate) const OCGPU_MANAGEMENT_PANIC_RESULT: ocgpu_abi::ocgpuResult = ocgpu_abi::{};",
        manifest.manifest.raw_panic_result
    )
    .expect("writing to String cannot fail");
    for function in &manifest.functions {
        render_common_flat_shim(&mut output, function);
    }
    for backend in ["cuda", "hip"] {
        for function in &manifest.functions {
            let raw = if backend == "cuda" {
                &function.cuda
            } else {
                &function.hip
            };
            render_raw_flat_shim(
                &mut output,
                manifest,
                backend,
                &raw.raw_name,
                &raw.return_type,
                &raw.params,
            );
        }
        for entry in raw_only_entries(manifest, backend) {
            render_raw_flat_shim(
                &mut output,
                manifest,
                backend,
                entry.raw_name.as_deref().expect("validated raw name"),
                entry
                    .abi_return_type
                    .as_deref()
                    .expect("validated return type"),
                &entry.abi_params,
            );
        }
    }
    output
}

fn render_common_flat_shim(output: &mut String, function: &FunctionEntry) {
    let name = &function.common_name;
    output.push_str(
        "\n/// Convenience unified leaf export. The first argument selects the backend for this call.\n\
         ///\n\
         /// # Safety\n\
         /// All pointer arguments must satisfy the operation's canonical ABI contract.\n\
         #[cfg(feature = \"flat-c-exports\")]\n\
         #[unsafe(no_mangle)]\n",
    );
    write!(
        output,
        "pub unsafe extern \"C\" fn {name}(backend: ocgpuBackend"
    )
    .expect("String write");
    for param in &function.params {
        write!(output, ", {}: {}", param.name, param.type_name).expect("String write");
    }
    writeln!(output, ") -> ocgpuResult {{").expect("String write");
    output.push_str(
        "    let __ocgpu_backend = match backend {\n\
         \tOCGPU_BACKEND_CUDA => ocgpu_core::BackendKind::Cuda,\n\
         \tOCGPU_BACKEND_HIP => ocgpu_core::BackendKind::Hip,\n\
         \t_ => return OCGPU_ERROR_INVALID_ARGUMENT,\n\
         };\n\
         let __ocgpu_table = match ocgpu_core::negotiated_common_table(__ocgpu_backend) {\n\
         \tOk(table) => table,\n\
         \tErr(error) => return error.result(),\n\
         };\n",
    );
    writeln!(
        output,
        "    let Some(__ocgpu_dispatch) = __ocgpu_table.{name} else {{ return OCGPU_ERROR_SYMBOL_UNAVAILABLE; }};"
    )
    .expect("String write");
    output.push_str(
        "    // SAFETY: The caller upholds this leaf export's documented ABI contract.\n",
    );
    write!(output, "    unsafe {{ __ocgpu_dispatch(").expect("String write");
    write_call_arguments(output, &function.params);
    output.push_str(") }\n}\n");
}

fn render_raw_flat_shim(
    output: &mut String,
    manifest: &ApiManifest,
    backend: &str,
    name: &str,
    return_type: &str,
    params: &[Parameter],
) {
    let feature = backend;
    let module = backend;
    let missing = raw_flat_sentinel(manifest, return_type);
    output.push_str(
        "\n/// Backend-native flat leaf export with the exact generated raw-table signature.\n\
         /// Missing symbols use the manifest's deterministic sentinel policy.\n\
         ///\n\
         /// # Safety\n\
         /// All pointer arguments must satisfy the corresponding vendor ABI contract.\n",
    );
    writeln!(
        output,
        "#[cfg(feature = \"flat-c-exports\")]\n#[unsafe(no_mangle)]"
    )
    .expect("String write");
    if params
        .iter()
        .filter(|parameter| parameter.name.len() == 1)
        .count()
        > 3
    {
        output.push_str("#[allow(clippy::many_single_char_names)]\n");
    }
    write!(output, "pub unsafe extern \"C\" fn {name}(").expect("String write");
    for (index, param) in params.iter().enumerate() {
        if index != 0 {
            output.push_str(", ");
        }
        write!(output, "{}: {}", param.name, param.type_name).expect("String write");
    }
    if return_type == "()" {
        output.push_str(") {\n");
    } else {
        writeln!(output, ") -> {return_type} {{").expect("String write");
    }
    writeln!(output, "    #[cfg(feature = \"{feature}\")]\n    {{").expect("String write");
    writeln!(
        output,
        "        let Ok(__ocgpu_api) = ocgpu_core::raw::{module}::load_unvalidated() else {{ return {missing}; }};\n        let Some(__ocgpu_dispatch) = __ocgpu_api.raw_table_ref().{name} else {{ return {missing}; }};"
    )
    .expect("String write");
    output.push_str(
        "        // SAFETY: The caller upholds this exact vendor leaf's ABI contract.\n\
                 unsafe { __ocgpu_dispatch(",
    );
    write_call_arguments(output, params);
    output.push_str(") }\n    }\n");
    writeln!(output, "    #[cfg(not(feature = \"{feature}\"))]\n    {{").expect("String write");
    if !params.is_empty() {
        output.push_str("        let _ = (");
        write_call_arguments(output, params);
        output.push_str(",);\n");
    }
    writeln!(output, "        {missing}\n    }}\n}}").expect("String write");
}

fn raw_flat_sentinel(manifest: &ApiManifest, return_type: &str) -> String {
    match return_type {
        "ocgpuResult" | "ocgpuCUresult" | "ocgpuHipError_t" => {
            "OCGPU_ERROR_SYMBOL_UNAVAILABLE".to_owned()
        }
        "()" => "()".to_owned(),
        value if value.starts_with("*const ") => "core::ptr::null()".to_owned(),
        value if value.starts_with("*mut ") => "core::ptr::null_mut()".to_owned(),
        "f32" | "f64" => "0.0".to_owned(),
        "u8" | "u16" | "u32" | "u64" | "usize" | "i8" | "i16" | "i32" | "i64" | "isize"
        | "c_char" | "core::ffi::c_long" | "core::ffi::c_ulong" => "0".to_owned(),
        value => manifest
            .types
            .iter()
            .find(|entry| entry.name == value)
            .map_or_else(
                || zeroed_aggregate_sentinel(value),
                |entry| match entry.kind.as_str() {
                    "opaque_handle" => "core::ptr::null_mut()".to_owned(),
                    "callback" => "None".to_owned(),
                    "integer" | "pointer_integer" => "0".to_owned(),
                    "alias" => entry.rust_type.as_deref().map_or_else(
                        || zeroed_aggregate_sentinel(value),
                        |underlying| raw_flat_sentinel(manifest, underlying),
                    ),
                    _ => zeroed_aggregate_sentinel(value),
                },
            ),
    }
}

fn zeroed_aggregate_sentinel(type_name: &str) -> String {
    format!(
        "{{\n            // SAFETY: ABI v1 defines the missing aggregate/POD sentinel as a zeroed object representation.\n            unsafe {{ core::mem::zeroed::<{type_name}>() }}\n        }}"
    )
}

fn write_call_arguments(output: &mut String, params: &[Parameter]) {
    for (index, param) in params.iter().enumerate() {
        if index != 0 {
            output.push_str(", ");
        }
        output.push_str(&param.name);
    }
}

fn render_loader_inventory(manifest: &ApiManifest) -> Result<String, Error> {
    let cuda = manifest
        .functions
        .iter()
        .map(|entry| LoaderFunction {
            stable_id: entry.stable_id,
            logical_id: &entry.id,
            raw_name: &entry.cuda.raw_name,
            vendor_symbol: &entry.cuda.vendor_symbol,
            aliases: &entry.cuda.aliases,
            versioned_alternatives: &entry.cuda.versioned_alternatives,
            signature_hash: &entry.cuda.signature_hash,
            variant: &entry.cuda.variant,
            per_thread_default_stream: entry.cuda.per_thread_default_stream,
            proc_address_flags: entry.cuda.proc_address_flags,
            fallback_order: &entry.cuda.fallback_order,
            linux: &entry.cuda.linux,
            windows: &entry.cuda.windows,
        })
        .collect();
    let hip = manifest
        .functions
        .iter()
        .map(|entry| LoaderFunction {
            stable_id: entry.stable_id,
            logical_id: &entry.id,
            raw_name: &entry.hip.raw_name,
            vendor_symbol: &entry.hip.vendor_symbol,
            aliases: &entry.hip.aliases,
            versioned_alternatives: &entry.hip.versioned_alternatives,
            signature_hash: &entry.hip.signature_hash,
            variant: &entry.hip.variant,
            per_thread_default_stream: entry.hip.per_thread_default_stream,
            proc_address_flags: entry.hip.proc_address_flags,
            fallback_order: &entry.hip.fallback_order,
            linux: &entry.hip.linux,
            windows: &entry.hip.windows,
        })
        .collect();
    let inventory = LoaderInventory {
        spdx_license_identifier: "CC0-1.0",
        schema_version: manifest.manifest.schema_version,
        abi_version: manifest.manifest.abi_version,
        cuda,
        hip,
    };
    Ok(format!(
        "{}\n",
        serde_json::to_string_pretty(&inventory).map_err(Error::Json)?
    ))
}

fn upper_snake(input: &str) -> String {
    let mut output = String::new();
    let mut previous_lower_or_digit = false;
    for character in input.chars() {
        if character.is_ascii_uppercase() && previous_lower_or_digit {
            output.push('_');
        }
        output.push(character.to_ascii_uppercase());
        previous_lower_or_digit = character.is_ascii_lowercase() || character.is_ascii_digit();
    }
    output
}

fn grouped_hex(value: u64) -> String {
    let digits = format!("{value:016x}");
    format!(
        "{}_{}_{}_{}",
        &digits[0..4],
        &digits[4..8],
        &digits[8..12],
        &digits[12..16]
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn canonical_manifest() -> ApiManifest {
        toml::from_str(include_str!("../../../api/ocgpu-api.toml"))
            .expect("canonical manifest parses in generator tests")
    }

    #[test]
    fn exact_prefix_transforms_preserve_suffixes() {
        assert_eq!(
            raw_prefixed_name("cuda", "cuMemAlloc_v2").unwrap(),
            "ocgpuCuMemAlloc_v2"
        );
        assert_eq!(
            raw_prefixed_name("cuda", "cuLaunchKernel_ptsz").unwrap(),
            "ocgpuCuLaunchKernel_ptsz"
        );
        assert_eq!(
            raw_prefixed_name("hip", "hipMemcpy_spt").unwrap(),
            "ocgpuHipMemcpy_spt"
        );
    }

    #[test]
    fn signature_hash_ignores_parameter_names_but_not_pointer_constness() {
        let input = [Parameter {
            name: "input".to_owned(),
            type_name: "*const c_void".to_owned(),
            direction: "in".to_owned(),
            nullable: Some(false),
            semantic_status: "reviewed_manifest".to_owned(),
            semantic_provenance: "unit test".to_owned(),
        }];
        let renamed = [Parameter {
            name: "other".to_owned(),
            ..input[0].clone()
        }];
        let mutable = [Parameter {
            type_name: "*mut c_void".to_owned(),
            ..input[0].clone()
        }];
        assert_eq!(
            signature_hash("ocgpuResult", &input),
            signature_hash("ocgpuResult", &renamed)
        );
        assert_ne!(
            signature_hash("ocgpuResult", &input),
            signature_hash("ocgpuResult", &mutable)
        );
    }

    #[test]
    fn table_layout_hash_changes_when_a_field_is_appended() {
        assert_ne!(table_layout_hash(26), table_layout_hash(27));
    }

    #[test]
    fn raw_flat_sentinels_cover_every_return_category() {
        let manifest = canonical_manifest();
        assert_eq!(
            raw_flat_sentinel(&manifest, "ocgpuResult"),
            "OCGPU_ERROR_SYMBOL_UNAVAILABLE"
        );
        assert_eq!(raw_flat_sentinel(&manifest, "u32"), "0");
        assert_eq!(raw_flat_sentinel(&manifest, "f64"), "0.0");
        assert_eq!(
            raw_flat_sentinel(&manifest, "*const c_void"),
            "core::ptr::null()"
        );

        let opaque = manifest
            .types
            .iter()
            .find(|entry| entry.kind == "opaque_handle")
            .expect("manifest has an opaque handle");
        assert_eq!(
            raw_flat_sentinel(&manifest, &opaque.name),
            "core::ptr::null_mut()"
        );
        let callback = manifest
            .types
            .iter()
            .find(|entry| entry.kind == "callback")
            .expect("manifest has a callback");
        assert_eq!(raw_flat_sentinel(&manifest, &callback.name), "None");
        let pod = manifest
            .types
            .iter()
            .find(|entry| entry.kind == "record")
            .expect("manifest has a POD record");
        assert_eq!(
            raw_flat_sentinel(&manifest, &pod.name),
            format!(
                "{{\n            // SAFETY: ABI v1 defines the missing aggregate/POD sentinel as a zeroed object representation.\n            unsafe {{ core::mem::zeroed::<{}>() }}\n        }}",
                pod.name
            )
        );
    }

    #[test]
    fn zero_argument_raw_stub_has_no_ignored_unit_binding() {
        let manifest = canonical_manifest();
        let raw = manifest
            .functions
            .iter()
            .map(|function| &function.cuda)
            .find(|raw| raw.params.is_empty())
            .expect("manifest has a zero-argument CUDA leaf");
        let mut output = String::new();
        render_raw_flat_shim(
            &mut output,
            &manifest,
            "cuda",
            &raw.raw_name,
            &raw.return_type,
            &raw.params,
        );
        assert!(!output.contains("let _ = ();"));
        assert!(output.contains("#[cfg(not(feature = \"cuda\"))]"));
    }

    #[test]
    fn flat_renderer_always_defines_both_raw_families() {
        let manifest = canonical_manifest();
        let output = render_export_shims(&manifest);
        let cuda = &manifest.functions[0].cuda.raw_name;
        let hip = &manifest.functions[0].hip.raw_name;
        assert!(output.contains(&format!("fn {cuda}(")));
        assert!(output.contains(&format!("fn {hip}(")));
        assert!(output.contains("#[cfg(feature = \"cuda\")]"));
        assert!(output.contains("#[cfg(feature = \"hip\")]"));
    }

    #[test]
    fn flat_missing_aggregate_policy_is_canonical_and_documented() {
        let manifest = canonical_manifest();
        assert_eq!(manifest.manifest.raw_missing_aggregate, "zeroed");
        let header = render_flat_header(&manifest);
        assert!(header.contains("aggregate/POD"));
        assert!(header.contains("zeroed object representation"));
    }

    #[test]
    fn every_generated_unsafe_block_has_a_safety_rationale() {
        let output = render_export_shims(&canonical_manifest());
        assert_eq!(
            output.matches("unsafe {").count(),
            output.matches("// SAFETY:").count()
        );
    }

    #[test]
    fn flat_leaf_renderer_rejects_direct_panic_primitives() {
        let output = render_export_shims(&canonical_manifest());
        assert!(output.contains(
            "OCGPU_MANAGEMENT_PANIC_RESULT: ocgpu_abi::ocgpuResult = \
             ocgpu_abi::OCGPU_ERROR_INTERNAL"
        ));
        for forbidden in [
            "panic!",
            "unreachable!",
            "unimplemented!",
            "todo!",
            "assert!",
            "assert_eq!",
            "assert_ne!",
            "debug_assert!",
            "debug_assert_eq!",
            "debug_assert_ne!",
            "unwrap(",
            "expect(",
            "catch_unwind",
        ] {
            assert!(
                !output.contains(forbidden),
                "generated flat leaf contains direct panic primitive {forbidden}"
            );
        }
        for line in output.lines() {
            if !line.trim_start().starts_with("#[") {
                assert!(
                    !line.contains('['),
                    "generated flat leaf contains a directly panicking indexing expression: {line}"
                );
            }
        }
    }

    #[test]
    fn official_cuda_alias_collisions_are_explicitly_unrepresentable() {
        let manifest = canonical_manifest();
        let collisions = manifest
            .raw_inventory
            .iter()
            .filter_map(|entry| {
                entry
                    .alias_collision
                    .as_ref()
                    .map(|collision| (entry, collision))
            })
            .collect::<Vec<_>>();
        assert_eq!(collisions.len(), 25);
        let (legacy, collision) = collisions
            .iter()
            .find(|(entry, _)| entry.vendor_name == "cuGetProcAddress")
            .expect("cuGetProcAddress collision is classified");
        assert_eq!(legacy.vendor_kind, "function");
        assert_eq!(collision.alias_target, "cuGetProcAddress_v2");
        assert_eq!(collision.classification, "unrepresentable_name_collision");
    }

    #[test]
    fn exact_cuda_proc_typedefs_and_bootstrap_policy_are_retained() {
        let manifest = canonical_manifest();
        let raw = |name: &str| {
            manifest
                .raw_inventory
                .iter()
                .find(|entry| entry.backend == "cuda" && entry.vendor_name == name)
                .expect("named CUDA entry exists")
        };
        assert_eq!(raw("cuDriverGetVersion").proc_version_linux, 2020);
        assert_eq!(raw("cuCtxCreate_v2").proc_version_linux, 3020);
        assert_eq!(raw("cuCtxSetCurrent").proc_version_linux, 4000);
        assert_eq!(raw("cuCtxGetCurrent").proc_version_linux, 4000);
        assert_eq!(raw("cuMemAlloc_v2").proc_version_linux, 3020);
        assert!(raw("cuMemAlloc_v2").proc_typedef.is_some());
        for bootstrap in ["cuGetProcAddress", "cuGetProcAddress_v2"] {
            let entry = raw(bootstrap);
            assert_eq!(entry.proc_version_linux, 0);
            assert_eq!(entry.proc_version_windows, 0);
            assert_eq!(entry.direct_names, [bootstrap]);
        }
    }

    #[test]
    fn public_type_identifiers_never_use_reserved_double_underscores() {
        let manifest = canonical_manifest();
        assert!(manifest.types.iter().all(|entry| {
            !entry.name.contains("__")
                && !entry.tag.as_deref().is_some_and(|tag| tag.contains("__"))
                && entry.fields.iter().all(|field| {
                    !field.name.contains("__")
                        && !field
                            .c_name
                            .as_deref()
                            .is_some_and(|name| name.contains("__"))
                })
        }));
    }

    #[test]
    fn malformed_constant_platforms_return_validation_errors() {
        assert!(rust_platform_cfg(&["unsupported-target".to_owned()]).is_err());
        assert!(c_platform_condition(&["unsupported-target".to_owned()]).is_err());
    }

    #[test]
    fn c_constant_scalar_spellings_are_strict_c99() {
        assert_eq!(c_constant_type("i32"), "int32_t");
        assert_eq!(c_constant_type("u32"), "uint32_t");
        assert_eq!(c_constant_type("u64"), "uint64_t");
        assert_eq!(c_constant_type("usize"), "uintptr_t");
    }
}
