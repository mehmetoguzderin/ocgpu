// SPDX-License-Identifier: CC0-1.0

//! Deterministic maintenance pass that imports declaration facts into the canonical manifest.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use serde::Deserialize;

use super::{
    ApiManifest, ConstantEntry, ConstantOracleVariant, Error, Parameter, RawAliasCollision,
    RawInventoryEntry, RawOracleVariant, TypeEntry, TypeField, TypeLayoutEvidence,
    TypeOracleVariant, format_hash, raw_prefixed_name, table_layout_hash, type_layout_hash,
};

const CUDA_ORACLE: &str = "oracle/rust/cudarc-0.19.9.json";
const HIP_ORACLE: &str = "oracle/rust/rocmrc-0.5.0.json";
const CUDA_VENDOR_TYPES: &str = "oracle/vendor/cuda/13.3-13030.json";
const HIP_VENDOR_TYPES: &str = "oracle/vendor/hip/general-7.14.60850.json";
const HIP_VENDOR_WINDOWS: &str = "oracle/vendor/hip/windows-7.2.0.json";
const VENDOR_UNION: &str = "oracle/vendor/function-union.json";
const SEMANTIC_OVERRIDES: &str = "oracle/semantic-overrides.json";
const CONSTANT_OVERRIDES: &str = "api/vendor-constant-overrides.toml";

/// Results from refreshing canonical Rust-oracle declaration facts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SyncReport {
    /// Number of generated CUDA public type declarations.
    pub cuda_types: usize,
    /// Number of generated HIP public type declarations.
    pub hip_types: usize,
    /// Number of typed CUDA raw-table fields.
    pub cuda_functions: usize,
    /// Number of typed HIP raw-table fields.
    pub hip_functions: usize,
    /// Number of CUDA functions blocked by a by-value unresolved record layout.
    pub cuda_layout_blocked: usize,
    /// Number of HIP functions blocked by a by-value unresolved record layout.
    pub hip_layout_blocked: usize,
}

impl SyncReport {
    /// Concise deterministic CLI summary.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "synced Rust oracles: CUDA {} types / {} callable / {} layout-blocked; HIP {} types / {} callable / {} layout-blocked",
            self.cuda_types,
            self.cuda_functions,
            self.cuda_layout_blocked,
            self.hip_types,
            self.hip_functions,
            self.hip_layout_blocked
        )
    }
}

#[derive(Clone, Debug, Deserialize)]
struct OracleSnapshot {
    inventory_id: String,
    source_version: String,
    provenance: String,
    entries: Vec<OracleEntry>,
}

#[derive(Clone, Debug, Deserialize)]
struct OracleEntry {
    kind: String,
    name: String,
    normalized_signature: String,
    signature_hash: String,
    abi: Option<OracleAbi>,
    platforms: Vec<String>,
    provenance: String,
    #[serde(default)]
    aliases: Vec<String>,
    alias_of: Option<String>,
    introduced: Option<String>,
    deprecated: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct OracleAbi {
    return_type: String,
    parameters: Vec<OracleParameter>,
}

#[derive(Clone, Debug, Deserialize)]
struct OracleParameter {
    name: String,
    #[serde(rename = "type")]
    type_name: String,
    pointer: String,
    direction: String,
    nullable: Option<bool>,
}

#[derive(Clone, Debug, Deserialize)]
struct VendorUnion {
    functions: Vec<VendorFunction>,
}

#[derive(Clone, Debug, Deserialize)]
struct VendorFunction {
    backend: String,
    kind: String,
    name: String,
    variants: Vec<VendorVariant>,
}

#[derive(Clone, Debug, Deserialize)]
struct VendorVariant {
    inventory_id: String,
    source_version: String,
    normalized_signature: String,
    signature_hash: String,
    abi: Option<OracleAbi>,
    #[serde(default)]
    aliases: Vec<String>,
    alias_of: Option<String>,
    platforms: Vec<String>,
    provenance: String,
    #[serde(default)]
    proc_address_candidates: Vec<ProcAddressCandidate>,
}

#[derive(Clone, Debug, Deserialize)]
struct ProcAddressCandidate {
    symbol: String,
    typedef_name: String,
    api_version: u32,
    proc_address_flags: u64,
    variant: String,
    normalized_signature: String,
    signature_hash: String,
    abi: OracleAbi,
    provenance: String,
}

#[derive(Clone, Debug, Deserialize)]
struct VendorTypeSnapshot {
    inventory_id: String,
    source_version: String,
    provenance: String,
    entries: Vec<VendorTypeEntry>,
}

#[derive(Clone, Debug, Deserialize)]
struct VendorTypeEntry {
    kind: String,
    name: String,
    normalized_signature: String,
    signature_hash: String,
    abi: Option<OracleAbi>,
    layouts: Option<Vec<VendorLayout>>,
    #[serde(default)]
    platforms: Vec<String>,
    provenance: String,
    #[serde(default)]
    aliases: Vec<String>,
    alias_of: Option<String>,
    introduced: Option<String>,
    deprecated: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct VendorLayout {
    target: String,
    size: u32,
    alignment: u32,
    field_offsets: BTreeMap<String, u32>,
    provenance: String,
}

fn vendor_layout_evidence(layout: &VendorLayout) -> TypeLayoutEvidence {
    TypeLayoutEvidence {
        target: layout.target.clone(),
        size: layout.size,
        alignment: layout.alignment,
        field_offsets: layout.field_offsets.clone(),
        provenance: layout.provenance.clone(),
    }
}

fn derived_layout_evidence(
    platforms: &[String],
    fields: &[TypeField],
    size: u32,
    alignment: u32,
    provenance: &str,
) -> Vec<TypeLayoutEvidence> {
    let field_offsets = fields
        .iter()
        .map(|field| {
            (
                field.c_name.clone().unwrap_or_else(|| field.name.clone()),
                field.offset_64,
            )
        })
        .collect::<BTreeMap<_, _>>();
    platforms
        .iter()
        .map(|target| TypeLayoutEvidence {
            target: target.clone(),
            size,
            alignment,
            field_offsets: field_offsets.clone(),
            provenance: provenance.to_owned(),
        })
        .collect()
}

fn collect_platform_layouts(
    variants: &[TypeOracleVariant],
) -> Result<Vec<TypeLayoutEvidence>, Error> {
    let mut merged = BTreeMap::<String, TypeLayoutEvidence>::new();
    for layout in variants.iter().flat_map(|variant| &variant.layouts) {
        let Some(existing) = merged.get_mut(&layout.target) else {
            merged.insert(layout.target.clone(), layout.clone());
            continue;
        };
        if existing.size != layout.size || existing.alignment != layout.alignment {
            return Err(Error::Validation(format!(
                "type layout evidence disagrees on {}: {}/{} versus {}/{}",
                layout.target, existing.size, existing.alignment, layout.size, layout.alignment
            )));
        }
        for (field, offset) in &layout.field_offsets {
            match existing.field_offsets.insert(field.clone(), *offset) {
                Some(previous) if previous != *offset => {
                    return Err(Error::Validation(format!(
                        "type field {field} offset disagrees on {}: {previous} versus {offset}",
                        layout.target
                    )));
                }
                _ => {}
            }
        }
        if !existing.provenance.contains(&layout.provenance) {
            existing.provenance.push_str("; ");
            existing.provenance.push_str(&layout.provenance);
        }
    }
    Ok(merged.into_values().collect())
}

#[derive(Clone, Debug, Deserialize)]
struct SemanticOverrides {
    parameters: Vec<SemanticOverride>,
}

#[derive(Clone, Debug, Deserialize)]
struct SemanticOverride {
    inventory_id: String,
    function: String,
    parameter: String,
    direction: String,
    #[serde(default)]
    nullable: Option<bool>,
    #[serde(default)]
    nullability: Option<String>,
    reason: String,
    provenance: String,
}

#[derive(Clone, Debug, Deserialize)]
struct ConstantOverrides {
    constant: Vec<ConstantOverride>,
}

#[derive(Clone, Debug, Deserialize)]
struct ConstantOverride {
    backend: String,
    name: String,
    value: i64,
    provenance: String,
}

#[derive(Clone, Debug)]
struct ConstantFact {
    backend: &'static str,
    inventory_id: String,
    source_version: String,
    kind: String,
    name: String,
    normalized_signature: String,
    signature_hash: String,
    platforms: Vec<String>,
    aliases: Vec<String>,
    alias_of: Option<String>,
    introduced: Option<String>,
    deprecated: Option<String>,
    provenance: String,
}

struct Catalog<'a> {
    backend: &'static str,
    inventory_id: &'a str,
    source_version: &'a str,
    provenance: &'a str,
    declarations: BTreeMap<&'a str, &'a OracleEntry>,
    public_names: BTreeMap<&'a str, String>,
    functions: Vec<&'a OracleEntry>,
}

/// Refresh all stable Rust-oracle types and promote every pointer-safe raw declaration.
#[allow(clippy::too_many_lines)]
pub fn sync_rust_oracles(workspace_root: &Path) -> Result<SyncReport, Error> {
    let manifest_path = workspace_root.join(super::MANIFEST_PATH);
    let manifest_source = read(&manifest_path)?;
    let mut manifest: ApiManifest = toml::from_str(&manifest_source).map_err(Error::Toml)?;
    let cuda = read_oracle(workspace_root, CUDA_ORACLE)?;
    let hip = read_oracle(workspace_root, HIP_ORACLE)?;
    let cuda_vendor_types = read_vendor_types(workspace_root, CUDA_VENDOR_TYPES)?;
    let hip_vendor_types = read_vendor_types(workspace_root, HIP_VENDOR_TYPES)?;
    let hip_vendor_windows = read_vendor_types(workspace_root, HIP_VENDOR_WINDOWS)?;
    let vendor = read_vendor_union(workspace_root)?;
    let semantic_overrides = read_semantic_overrides(workspace_root)?;
    let constant_overrides = read_constant_overrides(workspace_root)?;

    manifest.types.retain(|entry| entry.backend.is_none());
    let existing = manifest
        .types
        .iter()
        .flat_map(|entry| [Some(entry.name.clone()), entry.tag.clone()])
        .flatten()
        .collect::<BTreeSet<_>>();
    let cuda_catalog = Catalog::new("cuda", &cuda);
    let hip_catalog = Catalog::new("hip", &hip);

    let cuda_types = derive_types(&cuda_catalog, &existing, 0x1100_0000)?;
    let mut known = existing;
    known.extend(cuda_types.iter().map(|entry| entry.name.clone()));
    let hip_types = derive_types(&hip_catalog, &known, 0x2100_0000)?;
    manifest.types.extend(cuda_types.iter().cloned());
    manifest.types.extend(hip_types.iter().cloned());
    merge_rust_type_evidence(&mut manifest.types, &cuda_catalog)?;
    merge_rust_type_evidence(&mut manifest.types, &hip_catalog)?;
    let mut official_known = manifest
        .types
        .iter()
        .map(|entry| entry.name.clone())
        .collect::<BTreeSet<_>>();
    let cuda_official_types =
        derive_vendor_types("cuda", &cuda_vendor_types, &official_known, 0x1200_0000)?;
    official_known.extend(cuda_official_types.iter().map(|entry| entry.name.clone()));
    let hip_official_types =
        derive_vendor_types("hip", &hip_vendor_types, &official_known, 0x2200_0000)?;
    manifest.types.extend(cuda_official_types);
    manifest.types.extend(hip_official_types);
    merge_vendor_type_evidence(&mut manifest.types, "cuda", &cuda_vendor_types)?;
    merge_vendor_type_evidence(&mut manifest.types, "hip", &hip_vendor_types)?;
    merge_vendor_type_evidence(&mut manifest.types, "hip", &hip_vendor_windows)?;
    refresh_alias_type_layouts(&mut manifest.types)?;
    for entry in &mut manifest.types {
        entry.layout_hash = format_hash(type_layout_hash(entry));
    }

    manifest.constants.retain(|entry| entry.backend.is_none());
    manifest.constants.extend(derive_vendor_constants(
        &cuda,
        &hip,
        &cuda_vendor_types,
        &hip_vendor_types,
        &hip_vendor_windows,
        &constant_overrides,
    )?);

    let old_slots = capture_existing_slots(&manifest.raw_inventory);
    merge_vendor_inventory(&mut manifest.raw_inventory, &vendor)?;
    promote_functions(
        &mut manifest.raw_inventory,
        &cuda_catalog,
        &semantic_overrides,
    )?;
    promote_functions(
        &mut manifest.raw_inventory,
        &hip_catalog,
        &semantic_overrides,
    )?;
    promote_vendor_functions(
        &mut manifest.raw_inventory,
        &vendor,
        &cuda_catalog,
        &semantic_overrides,
    )?;
    promote_vendor_functions(
        &mut manifest.raw_inventory,
        &vendor,
        &hip_catalog,
        &semantic_overrides,
    )?;
    apply_common_semantic_provenance(&mut manifest, &semantic_overrides);
    apply_cuda_proc_candidates(&mut manifest, &vendor)?;
    populate_raw_semantic_contracts(&mut manifest.raw_inventory, &manifest.types);
    assign_append_only_slots(&mut manifest.raw_inventory, &old_slots);
    refresh_table_layouts(&mut manifest);

    super::validate(&manifest)?;
    super::validate_oracle_coverage(workspace_root, &manifest)?;
    let serialized = toml::to_string_pretty(&manifest).map_err(|error| {
        Error::Validation(format!("could not serialize canonical manifest: {error}"))
    })?;
    let source = format!(
        "# SPDX-License-Identifier: CC0-1.0\n# Canonical ABI manifest; oracle-derived declarations are refreshed by `ocgpu-codegen sync-rust-oracles`.\n\n{}",
        format_manifest_hex(&serialized)
    );
    fs::write(&manifest_path, source).map_err(|source| Error::Io {
        path: manifest_path,
        source,
    })?;

    let cuda_functions = manifest
        .raw_inventory
        .iter()
        .filter(|entry| entry.backend == "cuda" && entry.emitted)
        .count();
    let hip_functions = manifest
        .raw_inventory
        .iter()
        .filter(|entry| entry.backend == "hip" && entry.emitted)
        .count();
    let cuda_layout_blocked = manifest
        .raw_inventory
        .iter()
        .filter(|entry| {
            entry.backend == "cuda"
                && matches!(
                    entry.classification.as_str(),
                    "layout_unverified" | "unrepresentable"
                )
        })
        .count();
    let hip_layout_blocked = manifest
        .raw_inventory
        .iter()
        .filter(|entry| {
            entry.backend == "hip"
                && matches!(
                    entry.classification.as_str(),
                    "layout_unverified" | "unrepresentable"
                )
        })
        .count();
    Ok(SyncReport {
        cuda_types: cuda_types.len(),
        hip_types: hip_types.len(),
        cuda_functions,
        hip_functions,
        cuda_layout_blocked,
        hip_layout_blocked,
    })
}

impl<'a> Catalog<'a> {
    fn new(backend: &'static str, snapshot: &'a OracleSnapshot) -> Self {
        let declarations = snapshot
            .entries
            .iter()
            .filter(|entry| {
                matches!(
                    entry.kind.as_str(),
                    "type" | "alias" | "struct" | "union" | "opaque_handle"
                )
            })
            .map(|entry| (entry.name.as_str(), entry))
            .collect::<BTreeMap<_, _>>();
        let public_names = declarations
            .keys()
            .map(|name| (*name, public_type_name(backend, name)))
            .collect();
        let functions = snapshot
            .entries
            .iter()
            .filter(|entry| entry.kind == "function")
            .collect();
        Self {
            backend,
            inventory_id: &snapshot.inventory_id,
            source_version: &snapshot.source_version,
            provenance: &snapshot.provenance,
            declarations,
            public_names,
            functions,
        }
    }

    fn public_name(&self, source_name: &str) -> Option<String> {
        self.public_names.get(source_name).cloned()
    }
}

fn read_oracle(workspace_root: &Path, relative: &str) -> Result<OracleSnapshot, Error> {
    let path = workspace_root.join(relative);
    let source = read(&path)?;
    serde_json::from_str(&source).map_err(|source| Error::OracleJson { path, source })
}

fn read_vendor_union(workspace_root: &Path) -> Result<VendorUnion, Error> {
    let path = workspace_root.join(VENDOR_UNION);
    let source = read(&path)?;
    serde_json::from_str(&source).map_err(|source| Error::OracleJson { path, source })
}

fn read_vendor_types(workspace_root: &Path, relative: &str) -> Result<VendorTypeSnapshot, Error> {
    let path = workspace_root.join(relative);
    let source = read(&path)?;
    serde_json::from_str(&source).map_err(|source| Error::OracleJson { path, source })
}

fn read_semantic_overrides(workspace_root: &Path) -> Result<SemanticOverrides, Error> {
    let path = workspace_root.join(SEMANTIC_OVERRIDES);
    let source = read(&path)?;
    serde_json::from_str(&source).map_err(|source| Error::OracleJson { path, source })
}

fn read_constant_overrides(workspace_root: &Path) -> Result<ConstantOverrides, Error> {
    let path = workspace_root.join(CONSTANT_OVERRIDES);
    let source = read(&path)?;
    toml::from_str(&source).map_err(Error::Toml)
}

fn read(path: &Path) -> Result<String, Error> {
    fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn derive_types(
    catalog: &Catalog<'_>,
    existing: &BTreeSet<String>,
    stable_base: u32,
) -> Result<Vec<TypeEntry>, Error> {
    let mut output = Vec::new();
    for (index, (source_name, source)) in catalog.declarations.iter().enumerate() {
        if source_name.starts_with("_bindgen")
            || source_name.starts_with("__bindgen")
            || source_name.contains("BindgenBitfieldUnit")
        {
            continue;
        }
        let name = catalog
            .public_name(source_name)
            .expect("catalog names are complete");
        if existing.contains(&name) {
            continue;
        }
        let stable_id = stable_base
            .checked_add(u32::try_from(index).map_err(|_| {
                Error::Validation(format!("too many {} oracle types", catalog.backend))
            })?)
            .ok_or_else(|| Error::Validation("oracle type stable ID overflow".to_owned()))?;
        let mut entry = derive_type(catalog, source, stable_id)?;
        entry.layout_hash = format_hash(type_layout_hash(&entry));
        output.push(entry);
    }
    Ok(output)
}

fn derive_vendor_types(
    backend: &str,
    snapshot: &VendorTypeSnapshot,
    existing: &BTreeSet<String>,
    stable_base: u32,
) -> Result<Vec<TypeEntry>, Error> {
    let mut groups = BTreeMap::<&str, Vec<&VendorTypeEntry>>::new();
    for entry in snapshot.entries.iter().filter(|entry| {
        matches!(
            entry.kind.as_str(),
            "type" | "opaque_handle" | "callback" | "struct" | "union"
        )
    }) {
        groups.entry(&entry.name).or_default().push(entry);
    }
    let mut output = Vec::new();
    for (index, (source_name, entries)) in groups.iter().enumerate() {
        let public_name = public_type_name(backend, source_name);
        if existing.contains(&public_name) {
            continue;
        }
        let primary = entries
            .iter()
            .find(|entry| matches!(entry.kind.as_str(), "opaque_handle" | "callback" | "type"))
            .copied()
            .or_else(|| entries.first().copied())
            .expect("vendor type group is nonempty");
        let stable_id = stable_base
            .checked_add(u32::try_from(index).map_err(|_| {
                Error::Validation(format!("too many {backend} official vendor types"))
            })?)
            .ok_or_else(|| Error::Validation("official type stable ID overflow".to_owned()))?;
        let mut entry =
            derive_vendor_type(backend, snapshot, &groups, primary, entries, stable_id)?;
        entry.layout_hash = format_hash(type_layout_hash(&entry));
        output.push(entry);
    }
    Ok(output)
}

fn merge_rust_type_evidence(types: &mut [TypeEntry], catalog: &Catalog<'_>) -> Result<(), Error> {
    for source in catalog.declarations.values() {
        if source.name.starts_with("_bindgen")
            || source.name.starts_with("__bindgen")
            || source.name.contains("BindgenBitfieldUnit")
        {
            continue;
        }
        let public_name = catalog.public_name(&source.name).ok_or_else(|| {
            Error::Validation(format!(
                "{} type {} has no deterministic public name",
                catalog.inventory_id, source.name
            ))
        })?;
        let public = types
            .iter_mut()
            .find(|entry| {
                entry.name == public_name || entry.tag.as_deref() == Some(public_name.as_str())
            })
            .ok_or_else(|| {
                Error::Validation(format!(
                    "{} type {} has no public manifest declaration",
                    catalog.inventory_id, source.name
                ))
            })?;
        let layouts = if public.name == public_name {
            derived_layout_evidence(
                &source.platforms,
                &public.fields,
                public.size_64,
                public.align_64,
                &format!(
                    "Derived from {} {} repr(C) declaration graph; verified by generated Rust/C layout assertions.",
                    catalog.inventory_id, catalog.source_version
                ),
            )
        } else {
            Vec::new()
        };
        let variant = TypeOracleVariant {
            oracle_source: catalog.inventory_id.to_owned(),
            source_version: catalog.source_version.to_owned(),
            oracle_kind: source.kind.clone(),
            oracle_signature: source.normalized_signature.clone(),
            oracle_signature_hash: source.signature_hash.clone(),
            platforms: source.platforms.clone(),
            layouts,
            provenance: source.provenance.clone(),
        };
        if !public.oracle_variants.iter().any(|existing| {
            existing.oracle_source == variant.oracle_source
                && existing.oracle_kind == variant.oracle_kind
                && existing.oracle_signature_hash == variant.oracle_signature_hash
                && existing.platforms == variant.platforms
        }) {
            public.oracle_variants.push(variant);
        }
        public.platform_layouts = collect_platform_layouts(&public.oracle_variants)?;
    }
    Ok(())
}

// Source correlation, cross-target record reconciliation, and evidence merge
// remain adjacent so an official layout cannot be attached without validation.
#[allow(clippy::too_many_lines)]
fn merge_vendor_type_evidence(
    types: &mut [TypeEntry],
    backend: &str,
    snapshot: &VendorTypeSnapshot,
) -> Result<(), Error> {
    for source in snapshot.entries.iter().filter(|entry| {
        matches!(
            entry.kind.as_str(),
            "type" | "opaque_handle" | "callback" | "struct" | "union"
        )
    }) {
        let public_name = public_type_name(backend, &source.name);
        let public = types
            .iter_mut()
            .find(|entry| {
                entry.name == public_name || entry.tag.as_deref() == Some(public_name.as_str())
            })
            .ok_or_else(|| {
                Error::Validation(format!(
                    "{} type {} has no public manifest declaration",
                    snapshot.inventory_id, source.name
                ))
            })?;
        let source_is_handle_tag = public.name != public_name;
        let exact_layouts = source.layouts.as_deref().map_or_else(Vec::new, |layouts| {
            layouts.iter().map(vendor_layout_evidence).collect()
        });
        let layouts = if source_is_handle_tag {
            Vec::new()
        } else if exact_layouts.is_empty() {
            derived_layout_evidence(
                &source.platforms,
                &public.fields,
                public.size_64,
                public.align_64,
                &format!(
                    "Derived from {} {} declaration graph; verified by generated Rust/C layout assertions.",
                    snapshot.inventory_id, snapshot.source_version
                ),
            )
        } else {
            exact_layouts
        };
        if !source_is_handle_tag && matches!(source.kind.as_str(), "struct" | "union") {
            let had_official_record = public.oracle_variants.iter().any(|variant| {
                !is_rust_oracle(&variant.oracle_source)
                    && matches!(variant.oracle_kind.as_str(), "struct" | "union")
            });
            let (official_fields, size, alignment, _) = derive_vendor_record(backend, source)?;
            if had_official_record && (size != public.size_64 || alignment != public.align_64) {
                return Err(Error::Validation(format!(
                    "official {} record {} cannot replace public layout {}/{} with {size}/{alignment}",
                    snapshot.inventory_id, source.name, public.size_64, public.align_64
                )));
            }
            if !had_official_record {
                public.fields = official_fields;
                public.size_64 = size;
                public.align_64 = alignment;
                for layout in public
                    .oracle_variants
                    .iter_mut()
                    .flat_map(|variant| &mut variant.layouts)
                    .filter(|layout| layout.provenance.starts_with("Derived from "))
                {
                    layout.size = size;
                    layout.alignment = alignment;
                }
            } else if source.kind == "union" {
                for field in official_fields {
                    let c_name = field.c_name.as_deref().unwrap_or(&field.name);
                    if let Some(existing) = public.fields.iter().find(|candidate| {
                        candidate.c_name.as_deref().unwrap_or(&candidate.name) == c_name
                    }) {
                        if existing.type_name != field.type_name
                            || existing.offset_64 != field.offset_64
                        {
                            return Err(Error::Validation(format!(
                                "official union {} field {c_name} differs between pinned targets",
                                source.name
                            )));
                        }
                    } else {
                        public.fields.push(field);
                    }
                }
            } else if public.fields != official_fields {
                return Err(Error::Validation(format!(
                    "official record {} has target-dependent field graph between pinned inventories",
                    source.name
                )));
            }
        }

        for layout in &layouts {
            if layout.size != public.size_64 || layout.alignment != public.align_64 {
                return Err(Error::Validation(format!(
                    "{} {} layout on {} is {}/{}, but public {} is {}/{}",
                    snapshot.inventory_id,
                    source.name,
                    layout.target,
                    layout.size,
                    layout.alignment,
                    public.name,
                    public.size_64,
                    public.align_64
                )));
            }
        }

        let variant = TypeOracleVariant {
            oracle_source: snapshot.inventory_id.clone(),
            source_version: snapshot.source_version.clone(),
            oracle_kind: source.kind.clone(),
            oracle_signature: source.normalized_signature.clone(),
            oracle_signature_hash: source.signature_hash.clone(),
            platforms: source.platforms.clone(),
            layouts,
            provenance: source.provenance.clone(),
        };
        if !public.oracle_variants.iter().any(|existing| {
            existing.oracle_source == variant.oracle_source
                && existing.oracle_kind == variant.oracle_kind
                && existing.oracle_signature_hash == variant.oracle_signature_hash
                && existing.platforms == variant.platforms
        }) {
            public.oracle_variants.push(variant);
        }
        public.platform_layouts = collect_platform_layouts(&public.oracle_variants)?;
    }
    Ok(())
}

fn refresh_alias_type_layouts(types: &mut [TypeEntry]) -> Result<(), Error> {
    for _ in 0..types.len() {
        let by_name = types
            .iter()
            .map(|entry| (entry.name.clone(), (entry.size_64, entry.align_64)))
            .collect::<BTreeMap<_, _>>();
        let mut changed = false;
        for entry in types.iter_mut().filter(|entry| entry.kind == "alias") {
            let Some(target) = entry.rust_type.as_deref() else {
                continue;
            };
            let Some(&(size, alignment)) = by_name.get(target) else {
                continue;
            };
            if entry.size_64 != size || entry.align_64 != alignment {
                entry.size_64 = size;
                entry.align_64 = alignment;
                for layout in entry
                    .oracle_variants
                    .iter_mut()
                    .flat_map(|variant| &mut variant.layouts)
                    .filter(|layout| layout.provenance.starts_with("Derived from "))
                {
                    layout.size = size;
                    layout.alignment = alignment;
                }
                changed = true;
            }
            entry.platform_layouts = collect_platform_layouts(&entry.oracle_variants)?;
        }
        if !changed {
            return Ok(());
        }
    }
    Err(Error::Validation(
        "type alias layout refresh did not converge".to_owned(),
    ))
}

// This routine deliberately keeps collection, cross-source resolution, and
// classification adjacent so each emitted value is auditable against every
// pinned oracle variant.
#[allow(clippy::too_many_lines)]
fn derive_vendor_constants(
    cuda_rust: &OracleSnapshot,
    hip_rust: &OracleSnapshot,
    cuda_vendor: &VendorTypeSnapshot,
    hip_general: &VendorTypeSnapshot,
    hip_windows: &VendorTypeSnapshot,
    overrides: &ConstantOverrides,
) -> Result<Vec<ConstantEntry>, Error> {
    let mut facts = Vec::new();
    append_vendor_constant_facts(&mut facts, "cuda", cuda_vendor);
    append_rust_constant_facts(&mut facts, "cuda", cuda_rust);
    append_vendor_constant_facts(&mut facts, "hip", hip_general);
    append_vendor_constant_facts(&mut facts, "hip", hip_windows);
    append_rust_constant_facts(&mut facts, "hip", hip_rust);

    let mut grouped = BTreeMap::<(&str, &str), Vec<&ConstantFact>>::new();
    for fact in &facts {
        grouped
            .entry((fact.backend, fact.name.as_str()))
            .or_default()
            .push(fact);
    }

    let mut known = BTreeMap::<(&str, &str), i64>::new();
    let mut reviewed = BTreeMap::<(&str, &str), &ConstantOverride>::new();
    for entry in &overrides.constant {
        if !matches!(entry.backend.as_str(), "cuda" | "hip") {
            return Err(Error::Validation(format!(
                "constant override {} has unknown backend {}",
                entry.name, entry.backend
            )));
        }
        let key = (entry.backend.as_str(), entry.name.as_str());
        if !grouped.contains_key(&key) {
            return Err(Error::Validation(format!(
                "constant override {}:{} is absent from pinned inventories",
                entry.backend, entry.name
            )));
        }
        if reviewed.insert(key, entry).is_some() {
            return Err(Error::Validation(format!(
                "duplicate constant override {}:{}",
                entry.backend, entry.name
            )));
        }
        known.insert(key, entry.value);
    }
    loop {
        let mut changed = false;
        for (&key, variants) in &grouped {
            if known.contains_key(&key) {
                continue;
            }
            let values = variants
                .iter()
                .filter_map(|fact| {
                    resolve_constant_expression(
                        constant_expression(&fact.normalized_signature),
                        fact.backend,
                        &known,
                    )
                })
                .collect::<BTreeSet<_>>();
            if values.len() == 1 {
                known.insert(key, *values.first().expect("one resolved constant value"));
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let mut backend_index = BTreeMap::from([("cuda", 0_u32), ("hip", 0_u32)]);
    let mut output = Vec::with_capacity(grouped.len());
    for ((backend, vendor_name), variants) in grouped {
        let index = backend_index
            .get_mut(backend)
            .expect("constant backend is validated during collection");
        let stable_base = if backend == "cuda" {
            0x3000_0000_u32
        } else {
            0x4000_0000_u32
        };
        let stable_id = stable_base
            .checked_add(*index)
            .ok_or_else(|| Error::Validation("vendor constant stable ID overflow".to_owned()))?;
        *index += 1;

        let platforms = variants
            .iter()
            .flat_map(|fact| fact.platforms.iter().cloned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let resolved_variants = variants
            .iter()
            .map(|fact| {
                resolve_constant_expression(
                    constant_expression(&fact.normalized_signature),
                    backend,
                    &known,
                )
            })
            .collect::<Vec<_>>();
        let mut platform_values = BTreeMap::new();
        let mut platform_conflicts = Vec::new();
        for platform in &platforms {
            let official = variants
                .iter()
                .zip(&resolved_variants)
                .filter(|(fact, _)| {
                    !is_rust_oracle(&fact.inventory_id)
                        && fact.platforms.iter().any(|target| target == platform)
                })
                .filter_map(|(_, value)| *value)
                .collect::<BTreeSet<_>>();
            let rust = variants
                .iter()
                .zip(&resolved_variants)
                .filter(|(fact, _)| {
                    is_rust_oracle(&fact.inventory_id)
                        && fact.platforms.iter().any(|target| target == platform)
                })
                .filter_map(|(_, value)| *value)
                .collect::<BTreeSet<_>>();
            let selected = if official.len() == 1 {
                official.first().copied()
            } else if official.len() > 1 {
                platform_conflicts.push(platform.clone());
                None
            } else if rust.len() == 1 {
                rust.first().copied()
            } else if rust.len() > 1 {
                platform_conflicts.push(platform.clone());
                None
            } else {
                known.get(&(backend, vendor_name)).copied()
            };
            if let Some(value) = selected {
                platform_values.insert(platform.clone(), value);
            }
        }

        let all_platforms_resolved = platform_conflicts.is_empty()
            && platforms
                .iter()
                .all(|platform| platform_values.contains_key(platform));
        let emitted = all_platforms_resolved && !platform_values.is_empty();
        let default_value = platform_values
            .get("x86_64-unknown-linux-gnu")
            .or_else(|| platform_values.get("aarch64-unknown-linux-gnu"))
            .or_else(|| platform_values.get("x86_64-pc-windows-msvc"))
            .copied()
            .unwrap_or_default();
        let expressions = variants
            .iter()
            .map(|fact| constant_expression(&fact.normalized_signature))
            .collect::<BTreeSet<_>>();
        let (classification, reason) =
            constant_classification(vendor_name, &expressions, emitted, &platform_conflicts);
        let oracle_variants = variants
            .iter()
            .zip(resolved_variants)
            .map(|(fact, value)| ConstantOracleVariant {
                oracle_source: fact.inventory_id.clone(),
                source_version: fact.source_version.clone(),
                oracle_kind: fact.kind.clone(),
                oracle_signature: fact.normalized_signature.clone(),
                oracle_signature_hash: fact.signature_hash.clone(),
                value,
                platforms: fact.platforms.clone(),
                aliases: fact.aliases.clone(),
                alias_of: fact.alias_of.clone(),
                introduced: fact.introduced.clone(),
                deprecated: fact.deprecated.clone(),
                provenance: fact.provenance.clone(),
            })
            .collect::<Vec<_>>();
        let documentation_provenance = variants
            .iter()
            .map(|fact| {
                format!(
                    "{} {}: {}",
                    fact.inventory_id, fact.source_version, fact.provenance
                )
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
            .join("; ");
        let value_provenance = reviewed.get(&(backend, vendor_name)).map_or_else(
            || {
                format!(
                    "Resolved from exact integer expressions in {}.",
                    variants
                        .iter()
                        .map(|fact| fact.inventory_id.as_str())
                        .collect::<BTreeSet<_>>()
                        .into_iter()
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            },
            |reviewed| reviewed.provenance.clone(),
        );
        output.push(ConstantEntry {
            stable_id,
            name: format!("OCGPU_{}_{vendor_name}", backend.to_ascii_uppercase()),
            type_name: vendor_constant_type(&variants, &platform_values),
            value: default_value,
            backend_values: BTreeMap::new(),
            platform_values,
            backend: Some(backend.to_owned()),
            vendor_name: Some(vendor_name.to_owned()),
            vendor_kind: if variants.iter().any(|fact| fact.kind == "enum_value") {
                "enum_value".to_owned()
            } else {
                "constant".to_owned()
            },
            oracle_variants,
            platforms,
            emitted,
            classification,
            reason,
            documentation_provenance,
            value_provenance,
        });
    }
    Ok(output)
}

fn append_rust_constant_facts(
    output: &mut Vec<ConstantFact>,
    backend: &'static str,
    snapshot: &OracleSnapshot,
) {
    output.extend(
        snapshot
            .entries
            .iter()
            .filter(|entry| matches!(entry.kind.as_str(), "constant" | "enum_value"))
            .map(|entry| ConstantFact {
                backend,
                inventory_id: snapshot.inventory_id.clone(),
                source_version: snapshot.source_version.clone(),
                kind: entry.kind.clone(),
                name: entry.name.clone(),
                normalized_signature: entry.normalized_signature.clone(),
                signature_hash: entry.signature_hash.clone(),
                platforms: entry.platforms.clone(),
                aliases: entry.aliases.clone(),
                alias_of: entry.alias_of.clone(),
                introduced: entry.introduced.clone(),
                deprecated: entry.deprecated.clone(),
                provenance: entry.provenance.clone(),
            }),
    );
}

fn append_vendor_constant_facts(
    output: &mut Vec<ConstantFact>,
    backend: &'static str,
    snapshot: &VendorTypeSnapshot,
) {
    output.extend(
        snapshot
            .entries
            .iter()
            .filter(|entry| matches!(entry.kind.as_str(), "constant" | "enum_value"))
            .map(|entry| ConstantFact {
                backend,
                inventory_id: snapshot.inventory_id.clone(),
                source_version: snapshot.source_version.clone(),
                kind: entry.kind.clone(),
                name: entry.name.clone(),
                normalized_signature: entry.normalized_signature.clone(),
                signature_hash: entry.signature_hash.clone(),
                platforms: entry.platforms.clone(),
                aliases: entry.aliases.clone(),
                alias_of: entry.alias_of.clone(),
                introduced: entry.introduced.clone(),
                deprecated: entry.deprecated.clone(),
                provenance: entry.provenance.clone(),
            }),
    );
}

fn constant_expression(signature: &str) -> &str {
    signature
        .split_once('=')
        .map_or("", |(_, expression)| expression.trim())
}

fn resolve_constant_expression(
    expression: &str,
    backend: &str,
    known: &BTreeMap<(&str, &str), i64>,
) -> Option<i64> {
    let mut compact = expression
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect::<String>();
    loop {
        let stripped = strip_balanced_outer_parentheses(&compact);
        if stripped.len() == compact.len() {
            break;
        }
        compact = stripped.to_owned();
    }
    while let Some(after_cast) = strip_leading_c_cast(&compact) {
        compact = after_cast.to_owned();
        loop {
            let stripped = strip_balanced_outer_parentheses(&compact);
            if stripped.len() == compact.len() {
                break;
            }
            compact = stripped.to_owned();
        }
    }
    if let Some(value) = parse_integer_literal(&compact) {
        return Some(value);
    }
    if let Some(rest) = compact.strip_prefix('~') {
        return resolve_constant_expression(rest, backend, known).map(|value| !value);
    }
    for operator in ['|', '+'] {
        if let Some(index) = find_top_level_operator(&compact, operator) {
            let left = resolve_constant_expression(&compact[..index], backend, known)?;
            let right = resolve_constant_expression(&compact[index + 1..], backend, known)?;
            return if operator == '|' {
                Some(left | right)
            } else {
                left.checked_add(right)
            };
        }
    }
    let identifier = compact.rsplit("::").next().unwrap_or(&compact);
    if identifier
        .bytes()
        .all(|byte| byte == b'_' || byte.is_ascii_alphanumeric())
        && identifier
            .bytes()
            .next()
            .is_some_and(|byte| byte == b'_' || byte.is_ascii_alphabetic())
    {
        return known.get(&(backend, identifier)).copied();
    }
    None
}

fn strip_balanced_outer_parentheses(value: &str) -> &str {
    if !value.starts_with('(') || !value.ends_with(')') {
        return value;
    }
    let mut depth = 0_u32;
    for (index, byte) in value.bytes().enumerate() {
        match byte {
            b'(' => depth += 1,
            b')' => {
                depth = depth.saturating_sub(1);
                if depth == 0 && index + 1 != value.len() {
                    return value;
                }
            }
            _ => {}
        }
    }
    if depth == 0 {
        &value[1..value.len() - 1]
    } else {
        value
    }
}

fn strip_leading_c_cast(value: &str) -> Option<&str> {
    if !value.starts_with('(') {
        return None;
    }
    let close = value.find(')')?;
    let cast = &value[1..close];
    let remainder = &value[close + 1..];
    if remainder.is_empty()
        || cast.is_empty()
        || !cast
            .bytes()
            .all(|byte| byte == b'_' || byte == b'*' || byte.is_ascii_alphanumeric())
        || !cast.bytes().any(|byte| byte.is_ascii_alphabetic())
    {
        return None;
    }
    Some(remainder)
}

fn parse_integer_literal(value: &str) -> Option<i64> {
    let (negative, unsigned) = value
        .strip_prefix('-')
        .map_or((false, value), |value| (true, value));
    let literal = unsigned
        .trim_end_matches(|character: char| matches!(character.to_ascii_uppercase(), 'U' | 'L'));
    let magnitude = if let Some(hex) = literal
        .strip_prefix("0x")
        .or_else(|| literal.strip_prefix("0X"))
    {
        u64::from_str_radix(hex, 16).ok()?
    } else {
        literal.parse::<u64>().ok()?
    };
    if negative {
        let signed = i128::from(magnitude).checked_neg()?;
        i64::try_from(signed).ok()
    } else {
        i64::try_from(magnitude).ok()
    }
}

fn find_top_level_operator(value: &str, operator: char) -> Option<usize> {
    let mut depth = 0_u32;
    for (index, character) in value.char_indices() {
        match character {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            _ if character == operator && depth == 0 => return Some(index),
            _ => {}
        }
    }
    None
}

fn vendor_constant_type(variants: &[&ConstantFact], values: &BTreeMap<String, i64>) -> String {
    if variants.iter().any(|fact| {
        let expression = constant_expression(&fact.normalized_signature);
        expression.contains("void*")
            || expression.contains("CUstream)")
            || expression.contains("hipStream_t)")
    }) {
        return "usize".to_owned();
    }
    let minimum = values.values().copied().min().unwrap_or_default();
    let maximum = values.values().copied().max().unwrap_or_default();
    if minimum < 0 {
        if minimum >= i64::from(i32::MIN) && maximum <= i64::from(i32::MAX) {
            "i32"
        } else {
            "i64"
        }
    } else if variants.iter().any(|fact| fact.kind == "enum_value")
        && maximum <= i64::from(i32::MAX)
    {
        "i32"
    } else if maximum <= i64::from(u32::MAX) {
        "u32"
    } else {
        "u64"
    }
    .to_owned()
}

fn constant_classification(
    vendor_name: &str,
    expressions: &BTreeSet<&str>,
    emitted: bool,
    platform_conflicts: &[String],
) -> (String, String) {
    if emitted {
        return (
            "covered_integer".to_owned(),
            format!(
                "{vendor_name} is emitted as a backend-prefixed integer constant with every pinned source/platform declaration retained."
            ),
        );
    }
    if !platform_conflicts.is_empty() {
        return (
            "conflicting_platform_value".to_owned(),
            format!(
                "{vendor_name} has incompatible integer values within the same pinned target facts for {}.",
                platform_conflicts.join(", ")
            ),
        );
    }
    let classification = if expressions
        .iter()
        .any(|expression| expression.contains("b\"") || expression.contains('"'))
    {
        "non_integer_string"
    } else if expressions
        .iter()
        .all(|expression| *expression == "implicit")
    {
        "source_implicit_value"
    } else if expressions.iter().any(|expression| {
        expression
            .bytes()
            .all(|byte| byte == b'_' || byte == b':' || byte.is_ascii_alphanumeric())
    }) {
        "non_integer_type_alias"
    } else {
        "non_integer_expression"
    };
    (
        classification.to_owned(),
        format!(
            "{vendor_name} is retained but not emitted as an integer because its pinned declaration expression is {}.",
            expressions.iter().copied().collect::<Vec<_>>().join(" or ")
        ),
    )
}

#[allow(clippy::too_many_lines)]
fn derive_vendor_type(
    backend: &str,
    snapshot: &VendorTypeSnapshot,
    groups: &BTreeMap<&str, Vec<&VendorTypeEntry>>,
    primary: &VendorTypeEntry,
    same_name: &[&VendorTypeEntry],
    stable_id: u32,
) -> Result<TypeEntry, Error> {
    let public_name = public_type_name(backend, &primary.name);
    let mut source_for_identity = primary;
    let (kind, rust_type, tag, return_type, params, fields, size_64, align_64, layout_source) =
        match primary.kind.as_str() {
            "callback" => {
                let abi = primary.abi.as_ref().ok_or_else(|| {
                    Error::Validation(format!("official callback {} has no ABI", primary.name))
                })?;
                let params = abi
                    .parameters
                    .iter()
                    .enumerate()
                    .map(|(index, parameter)| {
                        Ok(Parameter {
                            name: rust_parameter_name(&parameter.name, index),
                            type_name: map_vendor_public_type(backend, &parameter.type_name)?,
                            direction: normalized_direction(&parameter.direction),
                            nullable: parameter.nullable,
                            semantic_status: "declaration_fact".to_owned(),
                            semantic_provenance: primary.provenance.clone(),
                        })
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                (
                    "callback",
                    None,
                    None,
                    Some(map_vendor_public_type(backend, &abi.return_type)?),
                    params,
                    Vec::new(),
                    8,
                    8,
                    primary.provenance.clone(),
                )
            }
            "opaque_handle" => (
                "opaque_handle",
                None,
                Some(format!("{public_name}_st")),
                None,
                Vec::new(),
                Vec::new(),
                8,
                8,
                primary.provenance.clone(),
            ),
            "type" => {
                let target = declaration_target(&primary.normalized_signature)?;
                if target.starts_with("enum ") {
                    (
                        "integer",
                        Some("i32".to_owned()),
                        None,
                        None,
                        Vec::new(),
                        Vec::new(),
                        4,
                        4,
                        primary.provenance.clone(),
                    )
                } else if let Some(record) = same_name.iter().find(|entry| {
                    matches!(entry.kind.as_str(), "struct" | "union")
                        && c_record_target(target) == entry.name
                }) {
                    source_for_identity = record;
                    let (fields, size, align, layout_source) =
                        derive_vendor_record(backend, record)?;
                    (
                        if record.kind == "union" {
                            "union"
                        } else {
                            "record"
                        },
                        None,
                        None,
                        None,
                        Vec::new(),
                        fields,
                        size,
                        align,
                        layout_source,
                    )
                } else if c_pointer_backed(target) {
                    (
                        "opaque_handle",
                        None,
                        Some(format!("{public_name}_st")),
                        None,
                        Vec::new(),
                        Vec::new(),
                        8,
                        8,
                        primary.provenance.clone(),
                    )
                } else {
                    let mapped = map_vendor_public_type(backend, target)?;
                    let (size, align) =
                        vendor_source_size_align(groups, target, &mut BTreeSet::new())?;
                    (
                        "alias",
                        Some(mapped),
                        None,
                        None,
                        Vec::new(),
                        Vec::new(),
                        size,
                        align,
                        primary.provenance.clone(),
                    )
                }
            }
            "struct" | "union" => {
                let (fields, size, align, layout_source) = derive_vendor_record(backend, primary)?;
                (
                    if primary.kind == "union" {
                        "union"
                    } else {
                        "record"
                    },
                    None,
                    None,
                    None,
                    Vec::new(),
                    fields,
                    size,
                    align,
                    layout_source,
                )
            }
            other => {
                return Err(Error::Validation(format!(
                    "unsupported official {backend} type kind {other} for {}",
                    primary.name
                )));
            }
        };
    let oracle_variants = same_name
        .iter()
        .map(|variant| {
            let layouts = variant.layouts.as_deref().map_or_else(
                || {
                    derived_layout_evidence(
                        &variant.platforms,
                        &fields,
                        size_64,
                        align_64,
                        &format!(
                            "Derived from {} {} declaration graph; verified by generated Rust/C layout assertions.",
                            snapshot.inventory_id, snapshot.source_version
                        ),
                    )
                },
                |layouts| layouts.iter().map(vendor_layout_evidence).collect(),
            );
            TypeOracleVariant {
                oracle_source: snapshot.inventory_id.clone(),
                source_version: snapshot.source_version.clone(),
                oracle_kind: variant.kind.clone(),
                oracle_signature: variant.normalized_signature.clone(),
                oracle_signature_hash: variant.signature_hash.clone(),
                platforms: variant.platforms.clone(),
                layouts,
                provenance: variant.provenance.clone(),
            }
        })
        .collect::<Vec<_>>();
    let platform_layouts = collect_platform_layouts(&oracle_variants)?;
    Ok(TypeEntry {
        stable_id,
        name: public_name,
        kind: kind.to_owned(),
        rust_type,
        tag,
        return_type,
        params,
        fields,
        backend: Some(backend.to_owned()),
        vendor_name: Some(primary.name.clone()),
        oracle_source: Some(snapshot.inventory_id.clone()),
        oracle_signature_hash: Some(source_for_identity.signature_hash.clone()),
        oracle_variants,
        platform_layouts,
        size_64,
        align_64,
        layout_hash: String::new(),
        layout_provenance: layout_source,
        documentation_provenance: format!(
            "{} {} ({}; {}; {})",
            snapshot.inventory_id,
            snapshot.source_version,
            snapshot.provenance,
            source_for_identity.provenance,
            source_for_identity.signature_hash
        ),
    })
}

fn derive_vendor_record(
    backend: &str,
    source: &VendorTypeEntry,
) -> Result<(Vec<TypeField>, u32, u32, String), Error> {
    let layouts = source.layouts.as_deref().unwrap_or(&[]);
    let layout = layouts.first().ok_or_else(|| {
        Error::Validation(format!(
            "official record {} has no verified layout",
            source.name
        ))
    })?;
    if layouts.iter().any(|candidate| {
        candidate.size != layout.size
            || candidate.alignment != layout.alignment
            || candidate.field_offsets != layout.field_offsets
    }) {
        return Err(Error::Validation(format!(
            "official record {} has target-dependent 64-bit layout",
            source.name
        )));
    }
    let raw_fields = record_fields(&source.normalized_signature)?;
    let mut fields = Vec::with_capacity(raw_fields.len());
    for (index, (name, source_type)) in raw_fields.iter().enumerate() {
        let offset = *layout.field_offsets.get(name).ok_or_else(|| {
            Error::Validation(format!(
                "official record {} field {name} has no verified offset",
                source.name
            ))
        })?;
        let type_name = map_vendor_public_type(backend, source_type).unwrap_or_else(|_| {
            let next_offset = raw_fields
                .iter()
                .skip(index + 1)
                .filter_map(|(next, _)| layout.field_offsets.get(next).copied())
                .filter(|next| *next > offset)
                .min()
                .unwrap_or(layout.size);
            let bytes = next_offset.saturating_sub(offset).max(1);
            if offset % 8 == 0 && bytes % 8 == 0 {
                format!("[u64; {}]", bytes / 8)
            } else if offset % 4 == 0 && bytes % 4 == 0 {
                format!("[u32; {}]", bytes / 4)
            } else {
                format!("[u8; {bytes}]")
            }
        });
        let (field_name, c_name) = public_field_identifiers(name);
        fields.push(TypeField {
            name: field_name,
            c_name,
            type_name,
            offset_64: offset,
        });
    }
    Ok((
        fields,
        layout.size,
        layout.alignment,
        layouts
            .iter()
            .map(|layout| format!("{}: {}", layout.target, layout.provenance))
            .collect::<Vec<_>>()
            .join("; "),
    ))
}

fn vendor_source_size_align(
    groups: &BTreeMap<&str, Vec<&VendorTypeEntry>>,
    source_type: &str,
    visiting: &mut BTreeSet<String>,
) -> Result<(u32, u32), Error> {
    if c_pointer_backed(source_type) {
        return Ok((8, 8));
    }
    let base = c_record_target(source_type);
    if let Some(primitive) = c_primitive_type(base) {
        return primitive_size_align(primitive);
    }
    if !visiting.insert(base.to_owned()) {
        return Err(Error::Validation(format!(
            "cyclic official type alias at {base}"
        )));
    }
    let entries = groups.get(base).ok_or_else(|| {
        Error::Validation(format!(
            "official declaration references unknown type {base}"
        ))
    })?;
    let result = if let Some(record) = entries
        .iter()
        .find(|entry| matches!(entry.kind.as_str(), "struct" | "union"))
    {
        let layout = record
            .layouts
            .as_deref()
            .unwrap_or(&[])
            .first()
            .ok_or_else(|| Error::Validation(format!("official record {base} has no layout")))?;
        (layout.size, layout.alignment)
    } else if entries
        .iter()
        .any(|entry| entry.kind == "opaque_handle" || entry.kind == "callback")
    {
        (8, 8)
    } else {
        let entry = entries[0];
        let target = declaration_target(&entry.normalized_signature)?;
        if target.starts_with("enum ") {
            (4, 4)
        } else {
            vendor_source_size_align(groups, target, visiting)?
        }
    };
    visiting.remove(base);
    Ok(result)
}

fn c_record_target(source_type: &str) -> &str {
    source_type
        .trim()
        .trim_start_matches("const ")
        .trim_start_matches("struct ")
        .trim_start_matches("union ")
        .trim_start_matches("enum ")
        .trim()
}

fn c_pointer_backed(source_type: &str) -> bool {
    source_type.contains('*')
}

fn map_vendor_public_type(backend: &str, value: &str) -> Result<String, Error> {
    let value = value.trim();
    if let Some(open) = value.rfind('[').filter(|_| value.ends_with(']')) {
        let count = value[open + 1..value.len() - 1]
            .trim()
            .parse::<u32>()
            .map_err(|_| Error::Validation(format!("invalid C array type {value}")))?;
        return Ok(format!(
            "[{}; {count}]",
            map_vendor_public_type(backend, value[..open].trim())?
        ));
    }
    let spaced = value.replace('*', " * ");
    let mut tokens = spaced.split_whitespace().collect::<Vec<_>>();
    let pointer_count = tokens.iter().filter(|token| **token == "*").count();
    tokens.retain(|token| *token != "*" && *token != "volatile");
    let base_const = tokens.first().is_some_and(|token| *token == "const");
    tokens.retain(|token| !matches!(*token, "const" | "struct" | "union" | "enum"));
    let base = tokens.join(" ");
    if base.contains("(anonymous") {
        return Err(Error::Validation(format!(
            "anonymous C type requires verified-layout storage: {value}"
        )));
    }
    let mut mapped = c_primitive_type(&base).map_or_else(
        || Ok::<_, Error>(public_type_name(backend, &base)),
        |primitive| Ok(primitive.to_owned()),
    )?;
    for index in 0..pointer_count {
        let kind = if index == 0 && base_const {
            "const"
        } else {
            "mut"
        };
        mapped = format!("*{kind} {mapped}");
    }
    Ok(mapped)
}

fn c_primitive_type(value: &str) -> Option<&'static str> {
    match value.trim() {
        "void" => Some("c_void"),
        "char" | "signed char" => Some("c_char"),
        "unsigned char" | "bool" => Some("u8"),
        "short" | "short int" | "signed short" => Some("i16"),
        "unsigned short" | "unsigned short int" => Some("u16"),
        "int" | "signed int" => Some("i32"),
        "unsigned" | "unsigned int" | "uint32_t" => Some("u32"),
        "long" | "signed long" => Some("core::ffi::c_long"),
        "unsigned long" => Some("core::ffi::c_ulong"),
        "long long" | "signed long long" | "int64_t" => Some("i64"),
        "unsigned long long" | "uint64_t" => Some("u64"),
        "size_t" => Some("usize"),
        "float" => Some("f32"),
        "double" => Some("f64"),
        _ => None,
    }
}

#[allow(clippy::too_many_lines)]
fn derive_type(
    catalog: &Catalog<'_>,
    source: &OracleEntry,
    stable_id: u32,
) -> Result<TypeEntry, Error> {
    let name = public_type_name(catalog.backend, &source.name);
    let (kind, rust_type, tag, return_type, params, fields, size_64, align_64) =
        match source.kind.as_str() {
            "struct" if tuple_struct_target(&source.normalized_signature).is_some() => {
                let target = tuple_struct_target(&source.normalized_signature)
                    .expect("tuple target checked");
                let mapped = map_abi_type(catalog, target)?;
                let (size, align) = abi_type_size_align(catalog, target, &mut BTreeSet::new())?;
                (
                    "alias",
                    Some(mapped),
                    None,
                    None,
                    Vec::new(),
                    Vec::new(),
                    size,
                    align,
                )
            }
            "struct" | "union" => {
                let (fields, size, align) = derive_record(catalog, source)?;
                if fields.is_empty() {
                    (
                        if source.kind == "struct" {
                            "opaque_record"
                        } else {
                            "opaque_union"
                        },
                        None,
                        None,
                        None,
                        Vec::new(),
                        Vec::new(),
                        0,
                        1,
                    )
                } else {
                    (
                        if source.kind == "struct" {
                            "record"
                        } else {
                            "union"
                        },
                        None,
                        None,
                        None,
                        Vec::new(),
                        fields,
                        size,
                        align,
                    )
                }
            }
            "type" | "opaque_handle" if is_enum(&source.normalized_signature) => {
                let repr = enum_repr(&source.normalized_signature)?;
                (
                    "integer",
                    Some(repr.to_owned()),
                    None,
                    None,
                    Vec::new(),
                    Vec::new(),
                    4,
                    4,
                )
            }
            "type" if is_callback(&source.normalized_signature) => {
                let (return_type, params) = parse_callback(catalog, &source.normalized_signature)?;
                (
                    "callback",
                    None,
                    None,
                    Some(return_type),
                    params,
                    Vec::new(),
                    8,
                    8,
                )
            }
            "type" | "alias" | "opaque_handle" => {
                let target = declaration_target(&source.normalized_signature)?;
                if pointer_parts(target).is_some()
                    && !pointer_target_is_declared_record(catalog, target)
                    && !pointer_target_is_primitive(target)
                {
                    (
                        "opaque_handle",
                        None,
                        Some(format!("{name}_st")),
                        None,
                        Vec::new(),
                        Vec::new(),
                        8,
                        8,
                    )
                } else {
                    let mapped = map_abi_type(catalog, target)?;
                    let (size, align) =
                        source_size_align(catalog, &source.name, &mut BTreeSet::new())?;
                    (
                        "alias",
                        Some(mapped),
                        None,
                        None,
                        Vec::new(),
                        Vec::new(),
                        size,
                        align,
                    )
                }
            }
            other => {
                return Err(Error::Validation(format!(
                    "unsupported {} oracle type kind {other} for {}",
                    catalog.backend, source.name
                )));
            }
        };
    let platform_layouts = derived_layout_evidence(
        &source.platforms,
        &fields,
        size_64,
        align_64,
        &format!(
            "Derived from {} {} repr(C) declaration graph; verified by generated Rust/C layout assertions.",
            catalog.inventory_id, catalog.source_version
        ),
    );
    let oracle_variants = vec![TypeOracleVariant {
        oracle_source: catalog.inventory_id.to_owned(),
        source_version: catalog.source_version.to_owned(),
        oracle_kind: source.kind.clone(),
        oracle_signature: source.normalized_signature.clone(),
        oracle_signature_hash: source.signature_hash.clone(),
        platforms: source.platforms.clone(),
        layouts: platform_layouts.clone(),
        provenance: source.provenance.clone(),
    }];
    Ok(TypeEntry {
        stable_id,
        name,
        kind: kind.to_owned(),
        rust_type,
        tag,
        return_type,
        params,
        fields,
        backend: Some(catalog.backend.to_owned()),
        vendor_name: Some(source.name.clone()),
        oracle_source: Some(catalog.inventory_id.to_owned()),
        oracle_signature_hash: Some(source.signature_hash.clone()),
        oracle_variants,
        platform_layouts,
        size_64,
        align_64,
        layout_hash: String::new(),
        layout_provenance:
            "Pinned repr(C) Rust declaration graph and ocgpu ABI v1 64-bit scalar/pointer policy"
                .to_owned(),
        documentation_provenance: format!("{} ({})", catalog.provenance, source.signature_hash),
    })
}

pub(super) fn public_type_name(backend: &str, source_name: &str) -> String {
    let prefixed = if backend == "hip" {
        source_name.strip_prefix("hip").map_or_else(
            || format!("ocgpu{source_name}"),
            |suffix| format!("ocgpuHip{suffix}"),
        )
    } else {
        format!("ocgpu{source_name}")
    };
    sanitize_public_identifier(&prefixed)
}

fn sanitize_public_identifier(source: &str) -> String {
    let mut sanitized = source
        .replace("__bindgen_ty_", "_anon_")
        .replace("__bindgen_", "_");
    while sanitized.contains("__") {
        sanitized = sanitized.replace("__", "_");
    }
    sanitized
}

fn is_enum(signature: &str) -> bool {
    signature.starts_with("enum ")
}

fn enum_repr(signature: &str) -> Result<&'static str, Error> {
    if signature.contains("repr (u32)") {
        Ok("u32")
    } else if signature.contains("repr (i32)") {
        Ok("i32")
    } else {
        Err(Error::Validation(format!(
            "oracle enum has no supported integer representation: {signature}"
        )))
    }
}

fn is_callback(signature: &str) -> bool {
    signature.contains("unsafe extern \"C\" fn")
}

fn declaration_target(signature: &str) -> Result<&str, Error> {
    signature
        .split_once('=')
        .map(|(_, target)| {
            target
                .trim()
                .strip_prefix("self::")
                .unwrap_or(target.trim())
        })
        .ok_or_else(|| Error::Validation(format!("declaration has no target: {signature}")))
}

fn pointer_parts(value: &str) -> Option<(&'static str, &str)> {
    let value = value.trim();
    value
        .strip_prefix("* mut")
        .map(|target| ("mut", target.trim()))
        .or_else(|| {
            value
                .strip_prefix("*mut")
                .map(|target| ("mut", target.trim()))
        })
        .or_else(|| {
            value
                .strip_prefix("* const")
                .map(|target| ("const", target.trim()))
        })
        .or_else(|| {
            value
                .strip_prefix("*const")
                .map(|target| ("const", target.trim()))
        })
}

fn pointer_target_is_primitive(value: &str) -> bool {
    pointer_parts(value).is_some_and(|(_, target)| primitive_type(target).is_some())
}

fn pointer_target_is_declared_record(catalog: &Catalog<'_>, value: &str) -> bool {
    pointer_parts(value).is_some_and(|(_, target)| {
        catalog
            .declarations
            .get(target)
            .is_some_and(|entry| matches!(entry.kind.as_str(), "struct" | "union"))
    })
}

fn primitive_type(value: &str) -> Option<&'static str> {
    let compact = value.replace(' ', "");
    match compact.as_str() {
        "()" => Some("()"),
        "bool" | "u8" | "::core::ffi::c_uchar" => Some("u8"),
        "i8" | "::core::ffi::c_schar" => Some("i8"),
        "u16" | "::core::ffi::c_ushort" => Some("u16"),
        "i16" | "::core::ffi::c_short" => Some("i16"),
        "u32" | "::core::ffi::c_uint" => Some("u32"),
        "i32" | "::core::ffi::c_int" => Some("i32"),
        "u64" | "::core::ffi::c_ulonglong" => Some("u64"),
        "i64" | "::core::ffi::c_longlong" => Some("i64"),
        "usize" => Some("usize"),
        "isize" => Some("isize"),
        "f32" => Some("f32"),
        "f64" => Some("f64"),
        "c_void" | "::core::ffi::c_void" | "core::ffi::c_void" => Some("c_void"),
        "c_char" | "::core::ffi::c_char" | "core::ffi::c_char" => Some("c_char"),
        "::core::ffi::c_long" | "core::ffi::c_long" => Some("core::ffi::c_long"),
        "::core::ffi::c_ulong" | "core::ffi::c_ulong" => Some("core::ffi::c_ulong"),
        _ => None,
    }
}

fn map_abi_type(catalog: &Catalog<'_>, source_type: &str) -> Result<String, Error> {
    if let Some((element, length)) = array_parts(source_type)? {
        return Ok(format!("[{}; {length}]", map_abi_type(catalog, element)?));
    }
    if let Some(storage) = bitfield_storage(source_type) {
        return map_abi_type(catalog, storage);
    }
    if let Some((kind, target)) = pointer_parts(source_type) {
        let mapped = map_abi_type(catalog, target)?;
        return Ok(format!("*{kind} {mapped}"));
    }
    if let Some(primitive) = primitive_type(source_type) {
        return Ok(primitive.to_owned());
    }
    let source_type = source_type
        .trim()
        .strip_prefix("self::")
        .unwrap_or(source_type.trim());
    catalog.public_name(source_type).ok_or_else(|| {
        Error::Validation(format!(
            "{} declaration references unknown type {source_type}",
            catalog.backend
        ))
    })
}

fn derive_record(
    catalog: &Catalog<'_>,
    source: &OracleEntry,
) -> Result<(Vec<TypeField>, u32, u32), Error> {
    let raw_fields = record_fields(&source.normalized_signature)?;
    let is_union = source.kind == "union";
    let mut fields = Vec::with_capacity(raw_fields.len());
    let mut size = 0_u32;
    let mut alignment = 1_u32;
    for (field_name, source_type) in raw_fields {
        let (field_size, field_align) = abi_type_size_align(
            catalog,
            &source_type,
            &mut BTreeSet::from([source.name.clone()]),
        )?;
        alignment = alignment.max(field_align);
        if field_size == 0 {
            continue;
        }
        let offset = if is_union {
            size = size.max(field_size);
            0
        } else {
            size = align_up(size, field_align);
            let offset = size;
            size = size.checked_add(field_size).ok_or_else(|| {
                Error::Validation(format!("{} record size overflow", source.name))
            })?;
            offset
        };
        let (field_name, c_name) = public_field_identifiers(&field_name);
        fields.push(TypeField {
            name: field_name,
            c_name,
            type_name: map_abi_type(catalog, &source_type)?,
            offset_64: offset,
        });
    }
    Ok((fields, align_up(size, alignment), alignment))
}

fn record_fields(signature: &str) -> Result<Vec<(String, String)>, Error> {
    let (_, body) = signature
        .split_once(":{")
        .ok_or_else(|| Error::Validation(format!("record has no field body: {signature}")))?;
    let body = body
        .strip_suffix('}')
        .ok_or_else(|| Error::Validation(format!("record body is not closed: {signature}")))?;
    let mut fields = Vec::new();
    for field in split_top_level(body, ',') {
        let field = field.trim();
        if field.is_empty() {
            continue;
        }
        let field = field.strip_prefix("pub ").unwrap_or(field);
        let (name, type_name) = field.split_once(':').ok_or_else(|| {
            Error::Validation(format!("malformed record field {field:?} in {signature}"))
        })?;
        fields.push((name.trim().to_owned(), type_name.trim().to_owned()));
    }
    Ok(fields)
}

fn tuple_struct_target(signature: &str) -> Option<&str> {
    signature
        .split_once(":(")
        .and_then(|(_, body)| body.strip_suffix(')'))
        .map(|body| {
            body.trim()
                .strip_prefix("pub")
                .unwrap_or(body.trim())
                .trim()
        })
}

fn split_top_level(value: &str, separator: char) -> Vec<&str> {
    let mut output = Vec::new();
    let mut start = 0;
    let mut depth = 0_u32;
    for (index, character) in value.char_indices() {
        match character {
            '[' | '<' | '(' | '{' => depth += 1,
            ']' | '>' | ')' | '}' => depth = depth.saturating_sub(1),
            _ => {}
        }
        if character == separator && depth == 0 {
            output.push(&value[start..index]);
            start = index + character.len_utf8();
        }
    }
    output.push(&value[start..]);
    output
}

fn array_parts(value: &str) -> Result<Option<(&str, u32)>, Error> {
    let value = value.trim();
    let Some(body) = value
        .strip_prefix('[')
        .and_then(|body| body.strip_suffix(']'))
    else {
        return Ok(None);
    };
    let parts = split_top_level(body, ';');
    if parts.len() != 2 {
        return Err(Error::Validation(format!("malformed array type {value}")));
    }
    let count = parts[1]
        .trim()
        .trim_end_matches("usize")
        .parse::<u32>()
        .map_err(|_| Error::Validation(format!("malformed array length in {value}")))?;
    Ok(Some((parts[0].trim(), count)))
}

fn bitfield_storage(value: &str) -> Option<&str> {
    let value = value.trim();
    if !value.starts_with("__BindgenBitfieldUnit") {
        return None;
    }
    let start = value.find('<')? + 1;
    let end = value.rfind('>')?;
    Some(value[start..end].trim())
}

fn abi_type_size_align(
    catalog: &Catalog<'_>,
    source_type: &str,
    visiting: &mut BTreeSet<String>,
) -> Result<(u32, u32), Error> {
    if let Some((element, count)) = array_parts(source_type)? {
        let (element_size, element_align) = abi_type_size_align(catalog, element, visiting)?;
        return Ok((
            element_size.checked_mul(count).ok_or_else(|| {
                Error::Validation(format!("array size overflow in {source_type}"))
            })?,
            element_align,
        ));
    }
    if let Some(storage) = bitfield_storage(source_type) {
        return abi_type_size_align(catalog, storage, visiting);
    }
    if pointer_parts(source_type).is_some() {
        return Ok((8, 8));
    }
    if let Some(primitive) = primitive_type(source_type) {
        return primitive_size_align(primitive);
    }
    source_size_align(catalog, source_type.trim(), visiting)
}

fn record_size_align(
    catalog: &Catalog<'_>,
    entry: &OracleEntry,
    visiting: &mut BTreeSet<String>,
) -> Result<(u32, u32), Error> {
    let is_union = entry.kind == "union";
    let mut size = 0_u32;
    let mut alignment = 1_u32;
    for (_, source_type) in record_fields(&entry.normalized_signature)? {
        let (field_size, field_align) = abi_type_size_align(catalog, &source_type, visiting)?;
        alignment = alignment.max(field_align);
        if is_union {
            size = size.max(field_size);
        } else {
            size = align_up(size, field_align)
                .checked_add(field_size)
                .ok_or_else(|| Error::Validation(format!("{} size overflow", entry.name)))?;
        }
    }
    Ok((align_up(size, alignment), alignment))
}

fn align_up(value: u32, alignment: u32) -> u32 {
    value.div_ceil(alignment) * alignment
}

fn source_size_align(
    catalog: &Catalog<'_>,
    source_name: &str,
    visiting: &mut BTreeSet<String>,
) -> Result<(u32, u32), Error> {
    if !visiting.insert(source_name.to_owned()) {
        return Err(Error::Validation(format!(
            "cyclic {} type alias at {source_name}",
            catalog.backend
        )));
    }
    let result = if let Some(primitive) = primitive_type(source_name) {
        primitive_size_align(primitive)?
    } else {
        let entry = catalog.declarations.get(source_name).ok_or_else(|| {
            Error::Validation(format!("missing {} type {source_name}", catalog.backend))
        })?;
        match entry.kind.as_str() {
            "struct" if tuple_struct_target(&entry.normalized_signature).is_some() => {
                abi_type_size_align(
                    catalog,
                    tuple_struct_target(&entry.normalized_signature).expect("tuple target checked"),
                    visiting,
                )?
            }
            "struct" | "union" => record_size_align(catalog, entry, visiting)?,
            "opaque_handle" => (8, 8),
            "type" if is_enum(&entry.normalized_signature) => (4, 4),
            "type" if is_callback(&entry.normalized_signature) => (8, 8),
            "type" | "alias" => {
                let target = declaration_target(&entry.normalized_signature)?;
                if pointer_parts(target).is_some() {
                    (8, 8)
                } else if let Some(primitive) = primitive_type(target) {
                    primitive_size_align(primitive)?
                } else {
                    source_size_align(catalog, target, visiting)?
                }
            }
            other => {
                return Err(Error::Validation(format!(
                    "cannot size {} {source_name} kind {other}",
                    catalog.backend
                )));
            }
        }
    };
    visiting.remove(source_name);
    Ok(result)
}

fn primitive_size_align(value: &str) -> Result<(u32, u32), Error> {
    match value {
        "()" => Ok((0, 1)),
        "u8" | "i8" | "c_char" | "c_void" => Ok((1, 1)),
        "u16" | "i16" => Ok((2, 2)),
        "u32" | "i32" | "f32" => Ok((4, 4)),
        // ABI v1 targets LP64 Linux and LLP64 Windows. A single size_64 value
        // cannot honestly describe C long, so fail closed if it becomes
        // reachable by value in a generated record or alias. Pointer uses do
        // not pass through this sizing path.
        "core::ffi::c_long" | "core::ffi::c_ulong" => Err(Error::Validation(format!(
            "target-dependent primitive {value} requires per-target layout evidence"
        ))),
        _ => Ok((8, 8)),
    }
}

fn parse_callback(
    catalog: &Catalog<'_>,
    signature: &str,
) -> Result<(String, Vec<Parameter>), Error> {
    let marker = "unsafe extern \"C\" fn (";
    let start = signature
        .find(marker)
        .ok_or_else(|| Error::Validation(format!("malformed callback: {signature}")))?
        + marker.len();
    let tail = &signature[start..];
    let close = tail
        .rfind(')')
        .ok_or_else(|| Error::Validation(format!("malformed callback: {signature}")))?;
    let args = &tail[..close];
    let after = tail[close + 1..].trim();
    let return_source = after
        .strip_prefix("->")
        .map_or("()", |value| value.trim().trim_end_matches('>').trim());
    let return_type = map_abi_type(catalog, return_source)?;
    let mut params = Vec::new();
    if !args.trim().is_empty() {
        for (index, argument) in args.split(',').enumerate() {
            let (name, source_type) = argument.trim().split_once(':').ok_or_else(|| {
                Error::Validation(format!("malformed callback parameter: {argument}"))
            })?;
            let type_name = map_abi_type(catalog, source_type)?;
            let pointer = pointer_parts(source_type);
            params.push(Parameter {
                name: rust_parameter_name(name.trim(), index),
                type_name,
                direction: match pointer {
                    Some(("const", _)) | None => "in".to_owned(),
                    Some(_) => "unknown".to_owned(),
                },
                nullable: if pointer.is_some() { None } else { Some(false) },
                semantic_status: "declaration_fact".to_owned(),
                semantic_provenance: catalog.provenance.to_owned(),
            });
        }
    }
    Ok((return_type, params))
}

#[allow(clippy::too_many_lines)]
fn merge_vendor_inventory(
    inventory: &mut Vec<RawInventoryEntry>,
    union: &VendorUnion,
) -> Result<(), Error> {
    for entry in inventory.iter_mut() {
        entry.alias_collision = None;
        let top_is_rust = is_rust_oracle(&entry.oracle_source);
        entry
            .oracle_variants
            .retain(|variant| is_rust_oracle(&variant.oracle_source));
        if entry.oracle_variants.is_empty() && top_is_rust {
            entry.oracle_variants.push(RawOracleVariant {
                oracle_source: entry.oracle_source.clone(),
                oracle_kind: "function".to_owned(),
                source_version: entry.oracle_source.clone(),
                oracle_signature: entry.oracle_signature.clone(),
                oracle_signature_hash: entry.oracle_signature_hash.clone(),
                platforms: entry.platforms.clone(),
                aliases: Vec::new(),
                alias_of: None,
                provenance: entry.documentation_provenance.clone(),
            });
        }
        if let Some(rust) = entry.oracle_variants.first() {
            rust.oracle_source.clone_into(&mut entry.oracle_source);
            rust.oracle_signature
                .clone_into(&mut entry.oracle_signature);
            rust.oracle_signature_hash
                .clone_into(&mut entry.oracle_signature_hash);
            entry.platforms.clone_from(&rust.platforms);
            rust.provenance
                .clone_into(&mut entry.documentation_provenance);
            "function".clone_into(&mut entry.vendor_kind);
            entry.alias_of = None;
        }
    }

    let mut next = BTreeMap::from([("cuda", 0xe000_0000_u32), ("hip", 0xf000_0000_u32)]);
    for function in &union.functions {
        let position = inventory.iter().position(|entry| {
            entry.backend == function.backend && entry.vendor_name == function.name
        });
        let first = function.variants.first().ok_or_else(|| {
            Error::Validation(format!("vendor union {} has no variants", function.name))
        })?;
        let variants = function
            .variants
            .iter()
            .map(|variant| RawOracleVariant {
                oracle_source: variant.inventory_id.clone(),
                oracle_kind: function.kind.clone(),
                source_version: variant.source_version.clone(),
                oracle_signature: variant.normalized_signature.clone(),
                oracle_signature_hash: variant.signature_hash.clone(),
                platforms: variant.platforms.clone(),
                aliases: variant.aliases.clone(),
                alias_of: variant.alias_of.clone(),
                provenance: variant.provenance.clone(),
            })
            .collect::<Vec<_>>();
        let platforms = function
            .variants
            .iter()
            .flat_map(|variant| variant.platforms.iter().cloned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let (proc_name, direct_names, proc_linux, proc_windows, proc_flags, variant_kind, alias_of) =
            vendor_resolution(function);
        if let Some(position) = position {
            let entry = &mut inventory[position];
            let rust_function_collision = function.kind == "alias"
                && entry
                    .oracle_variants
                    .iter()
                    .any(|variant| is_rust_oracle(&variant.oracle_source));
            if rust_function_collision {
                entry.platforms = entry
                    .platforms
                    .iter()
                    .chain(&platforms)
                    .cloned()
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect();
                "function".clone_into(&mut entry.vendor_kind);
                entry.proc_name = Some(entry.vendor_name.clone());
                entry.direct_names = vec![entry.vendor_name.clone()];
                entry.proc_version_linux = 0;
                entry.proc_version_windows = 0;
                entry.proc_flags = u64::from(entry.backend == "cuda");
                "default".clone_into(&mut entry.variant);
                entry.alias_of = None;
                let targets = variants
                    .iter()
                    .filter_map(|variant| variant.alias_of.clone())
                    .collect::<BTreeSet<_>>();
                if targets.len() != 1 {
                    return Err(Error::Validation(format!(
                        "{}:{} shadowed alias has ambiguous targets: {}",
                        entry.backend,
                        entry.vendor_name,
                        targets.into_iter().collect::<Vec<_>>().join(", ")
                    )));
                }
                let alias_target = targets
                    .into_iter()
                    .next()
                    .ok_or_else(|| Error::Validation("alias target disappeared".to_owned()))?;
                entry.alias_collision = Some(RawAliasCollision {
                    classification: "unrepresentable_name_collision".to_owned(),
                    reason: format!(
                        "Official macro alias {} targets {alias_target}, but a real ABI-distinct exported function already owns the exact C spelling {}; C cannot expose two table fields or leaf exports with one identifier.",
                        entry.vendor_name, entry.vendor_name
                    ),
                    alias_target,
                    oracle_variants: variants.clone(),
                });
            } else {
                function.kind.clone_into(&mut entry.vendor_kind);
                first.inventory_id.clone_into(&mut entry.oracle_source);
                first
                    .normalized_signature
                    .clone_into(&mut entry.oracle_signature);
                first
                    .signature_hash
                    .clone_into(&mut entry.oracle_signature_hash);
                entry.platforms = platforms;
                first
                    .provenance
                    .clone_into(&mut entry.documentation_provenance);
                entry.proc_name = Some(proc_name);
                entry.direct_names = direct_names;
                entry.proc_version_linux = proc_linux;
                entry.proc_version_windows = proc_windows;
                entry.proc_flags = proc_flags;
                entry.variant = variant_kind;
                entry.alias_of = alias_of;
            }
            if !rust_function_collision {
                for variant in variants {
                    if !entry.oracle_variants.iter().any(|existing| {
                        existing.oracle_source == variant.oracle_source
                            && existing.oracle_signature_hash == variant.oracle_signature_hash
                    }) {
                        entry.oracle_variants.push(variant);
                    }
                }
            }
        } else {
            let id = next.get_mut(function.backend.as_str()).ok_or_else(|| {
                Error::Validation(format!("unknown vendor backend {}", function.backend))
            })?;
            let stable_id = *id;
            *id = id
                .checked_add(1)
                .ok_or_else(|| Error::Validation("vendor raw stable ID overflow".to_owned()))?;
            inventory.push(RawInventoryEntry {
                stable_id,
                backend: function.backend.clone(),
                vendor_name: function.name.clone(),
                vendor_kind: function.kind.clone(),
                raw_name: Some(raw_prefixed_name(&function.backend, &function.name)?),
                oracle_source: first.inventory_id.clone(),
                oracle_signature: first.normalized_signature.clone(),
                oracle_signature_hash: first.signature_hash.clone(),
                oracle_variants: variants,
                alias_collision: None,
                classification: "layout_unverified".to_owned(),
                reason: format!(
                    "{} is retained from the official vendor union while its structured ABI graph is validated.",
                    function.name
                ),
                platforms,
                emitted: false,
                table_order: None,
                proc_name: Some(proc_name),
                direct_names,
                proc_version_linux: proc_linux,
                proc_version_windows: proc_windows,
                proc_flags,
                proc_typedef: None,
                proc_signature: None,
                proc_signature_hash: None,
                proc_provenance: None,
                variant: variant_kind,
                alias_of,
                abi_return_type: None,
                abi_params: Vec::new(),
                common_id: None,
                documentation_provenance: first.provenance.clone(),
                ownership: String::new(),
                nullability: String::new(),
                callback_behavior: String::new(),
                thread_safety: String::new(),
            });
        }
    }
    for entry in inventory {
        if entry.proc_name.is_none() {
            let without_stream = ["_ptsz", "_ptds", "_spt"]
                .into_iter()
                .find(|suffix| entry.vendor_name.ends_with(suffix))
                .map_or(entry.vendor_name.as_str(), |suffix| {
                    entry.vendor_name.trim_end_matches(suffix)
                });
            entry.proc_name = Some(strip_numeric_version_suffix(without_stream).to_owned());
        }
        if entry.direct_names.is_empty() {
            entry.direct_names.push(entry.vendor_name.clone());
        }
        if entry.variant.is_empty() {
            let variant = if entry.vendor_name.ends_with("_ptsz") {
                "ptsz"
            } else if entry.vendor_name.ends_with("_ptds") {
                "ptds"
            } else if entry.vendor_name.ends_with("_spt") {
                "spt"
            } else if strip_numeric_version_suffix(&entry.vendor_name) != entry.vendor_name {
                "versioned"
            } else {
                "default"
            };
            variant.clone_into(&mut entry.variant);
        }
        if entry.backend == "cuda" && entry.proc_flags == 0 {
            entry.proc_flags = if matches!(entry.variant.as_str(), "ptsz" | "ptds") {
                2
            } else {
                1
            };
        }
    }
    Ok(())
}

fn is_rust_oracle(source: &str) -> bool {
    matches!(source, "cudarc-0.19.9" | "rocmrc-0.5.0")
}

#[allow(clippy::type_complexity)]
fn vendor_resolution(
    function: &VendorFunction,
) -> (String, Vec<String>, u32, u32, u64, String, Option<String>) {
    let alias_of = function
        .variants
        .iter()
        .find_map(|variant| variant.alias_of.clone());
    let aliases = function
        .variants
        .iter()
        .flat_map(|variant| variant.aliases.iter().cloned())
        .collect::<BTreeSet<_>>();
    // Header aliases can be preprocessor redirects while an older unsuffixed
    // binary export remains ABI-incompatible. Alias convenience slots resolve
    // the normalized canonical target; canonical slots resolve only themselves.
    let direct_names = alias_of.as_ref().map_or_else(
        || vec![function.name.clone()],
        |target| vec![target.clone()],
    );
    let stream_suffix = ["_ptsz", "_ptds", "_spt"]
        .into_iter()
        .find(|suffix| function.name.ends_with(suffix));
    let proc_name = if function.kind == "alias" {
        function.name.clone()
    } else {
        aliases.iter().next().cloned().unwrap_or_else(|| {
            let without_stream = stream_suffix.map_or(function.name.as_str(), |suffix| {
                function.name.trim_end_matches(suffix)
            });
            if function.backend == "cuda" {
                strip_numeric_version_suffix(without_stream).to_owned()
            } else {
                without_stream.to_owned()
            }
        })
    };
    let flags = if function.backend == "cuda" || function.name.ends_with("_spt") {
        if stream_suffix.is_some() { 2 } else { 1 }
    } else {
        0
    };
    // Proc-address lookup is enabled only after an exact vendor typedef ABI
    // match supplies the per-symbol query version. Official declarations alone
    // do not prove that a base-name query returns a suffixed slot ABI.
    let linux = 0;
    let windows = 0;
    let variant = if function.kind == "alias" {
        "alias"
    } else if function.name.ends_with("_ptsz") {
        "ptsz"
    } else if function.name.ends_with("_ptds") {
        "ptds"
    } else if function.name.ends_with("_spt") {
        "spt"
    } else if function.name.contains("_v") {
        "versioned"
    } else {
        "default"
    };
    (
        proc_name,
        direct_names,
        linux,
        windows,
        flags,
        variant.to_owned(),
        alias_of,
    )
}

fn strip_numeric_version_suffix(name: &str) -> &str {
    name.rfind("_v").map_or(name, |index| {
        let version = &name[index + 2..];
        if !version.is_empty() && version.bytes().all(|byte| byte.is_ascii_digit()) {
            &name[..index]
        } else {
            name
        }
    })
}

/// Enables CUDA proc-address resolution only when a pinned `cudaTypedefs.h`
/// candidate has the exact return and parameter types of the generated slot.
/// Candidate graphs that collapse to the same public names but disagree after
/// transitive expansion are treated as ambiguous and fail generation.
// Exact transitive ABI-graph matching is intentionally centralized here: it
// is the fail-closed boundary between vendor typedef facts and loader slots.
#[allow(clippy::too_many_lines)]
fn apply_cuda_proc_candidates(
    manifest: &mut ApiManifest,
    union: &VendorUnion,
) -> Result<(), Error> {
    let candidates_by_symbol = union
        .functions
        .iter()
        .filter(|function| function.backend == "cuda")
        .flat_map(|function| &function.variants)
        .flat_map(|variant| &variant.proc_address_candidates)
        .fold(
            BTreeMap::<&str, Vec<&ProcAddressCandidate>>::new(),
            |mut grouped, candidate| {
                grouped
                    .entry(candidate.symbol.as_str())
                    .or_default()
                    .push(candidate);
                grouped
            },
        );
    let common_abis = manifest
        .functions
        .iter()
        .map(|function| {
            (
                function.id.clone(),
                (
                    function.cuda.return_type.clone(),
                    function
                        .cuda
                        .params
                        .iter()
                        .map(|parameter| parameter.type_name.clone())
                        .collect::<Vec<_>>(),
                ),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let types = &manifest.types;

    for entry in manifest
        .raw_inventory
        .iter_mut()
        .filter(|entry| entry.backend == "cuda" && entry.emitted)
    {
        let (slot_return, slot_params) = if let Some(common_id) = &entry.common_id {
            common_abis.get(common_id).cloned().ok_or_else(|| {
                Error::Validation(format!(
                    "CUDA proc candidate slot {} references missing common operation {common_id}",
                    entry.vendor_name
                ))
            })?
        } else {
            (
                entry.abi_return_type.clone().ok_or_else(|| {
                    Error::Validation(format!(
                        "CUDA proc candidate slot {} has no return type",
                        entry.vendor_name
                    ))
                })?,
                entry
                    .abi_params
                    .iter()
                    .map(|parameter| parameter.type_name.clone())
                    .collect(),
            )
        };
        let slot_return_shape = public_abi_shape(types, &slot_return, &mut BTreeSet::new())?;
        let slot_param_shapes = slot_params
            .iter()
            .map(|parameter| public_abi_shape(types, parameter, &mut BTreeSet::new()))
            .collect::<Result<Vec<_>, Error>>()?;

        entry.proc_version_linux = 0;
        entry.proc_version_windows = 0;
        entry.proc_typedef = None;
        entry.proc_signature = None;
        entry.proc_signature_hash = None;
        entry.proc_provenance = None;

        // The proc-address entrypoints bootstrap resolution itself and must be
        // obtained from the dynamic library directly, never recursively.
        if matches!(
            entry.vendor_name.as_str(),
            "cuGetProcAddress" | "cuGetProcAddress_v2"
        ) {
            continue;
        }

        let query_symbol = entry.proc_name.as_deref().unwrap_or(&entry.vendor_name);
        let Some(candidates) = candidates_by_symbol.get(query_symbol) else {
            continue;
        };
        let mut matches = Vec::new();
        for candidate in candidates {
            let expected_flags = match candidate.variant.as_str() {
                "legacy" => 1,
                "ptsz" | "ptds" => 2,
                other => {
                    return Err(Error::Validation(format!(
                        "CUDA proc typedef {} has unknown variant {other}",
                        candidate.typedef_name
                    )));
                }
            };
            if candidate.proc_address_flags != expected_flags {
                return Err(Error::Validation(format!(
                    "CUDA proc typedef {} variant {} has flags {}, expected {expected_flags}",
                    candidate.typedef_name, candidate.variant, candidate.proc_address_flags
                )));
            }
            let candidate_return = map_vendor_public_type("cuda", &candidate.abi.return_type)?;
            let candidate_params = candidate
                .abi
                .parameters
                .iter()
                .map(|parameter| map_vendor_public_type("cuda", &parameter.type_name))
                .collect::<Result<Vec<_>, Error>>()?;
            let candidate_return_shape =
                public_abi_shape(types, &candidate_return, &mut BTreeSet::new())?;
            let candidate_param_shapes = candidate_params
                .iter()
                .map(|parameter| public_abi_shape(types, parameter, &mut BTreeSet::new()))
                .collect::<Result<Vec<_>, Error>>()?;
            if candidate_return_shape == slot_return_shape
                && candidate_param_shapes == slot_param_shapes
            {
                matches.push(candidate);
            }
        }
        if matches.is_empty() {
            continue;
        }
        matches.sort_by_key(|candidate| {
            (
                candidate.api_version,
                candidate.symbol.as_str(),
                candidate.proc_address_flags,
                candidate.typedef_name.as_str(),
            )
        });
        let selected = matches[0];
        entry.proc_name = Some(selected.symbol.clone());
        entry.proc_version_linux = selected.api_version;
        entry.proc_version_windows = selected.api_version;
        entry.proc_flags = selected.proc_address_flags;
        entry.proc_typedef = Some(selected.typedef_name.clone());
        entry.proc_signature = Some(selected.normalized_signature.clone());
        entry.proc_signature_hash = Some(selected.signature_hash.clone());
        entry.proc_provenance = Some(selected.provenance.clone());
    }

    for function in &mut manifest.functions {
        let Some(inventory) = manifest.raw_inventory.iter().find(|entry| {
            entry.backend == "cuda" && entry.common_id.as_deref() == Some(function.id.as_str())
        }) else {
            continue;
        };
        function.cuda.linux.proc_version = inventory.proc_version_linux;
        function.cuda.windows.proc_version = inventory.proc_version_windows;
        function.cuda.proc_address_flags = inventory.proc_flags;
    }
    Ok(())
}

fn public_abi_shape(
    types: &[TypeEntry],
    type_name: &str,
    visiting: &mut BTreeSet<String>,
) -> Result<String, Error> {
    if let Some((element, count)) = array_parts(type_name)? {
        return Ok(format!(
            "array[{count}]({})",
            public_abi_shape(types, element, visiting)?
        ));
    }
    if let Some((kind, target)) = pointer_parts(type_name) {
        return Ok(format!(
            "ptr[{kind}]({})",
            public_abi_shape(types, target, visiting)?
        ));
    }
    if let Some(primitive) = primitive_type(type_name) {
        return Ok(format!("primitive({primitive})"));
    }
    let name = type_name.trim();
    let entry = types
        .iter()
        .find(|entry| entry.name == name)
        .ok_or_else(|| {
            Error::Validation(format!(
                "proc-address ABI references unknown public type {name}"
            ))
        })?;
    if !visiting.insert(name.to_owned()) {
        return Ok(format!("recursive({name})"));
    }
    let shape = match entry.kind.as_str() {
        "pointer_integer" => "primitive(u64)".to_owned(),
        "integer" | "alias" => public_abi_shape(
            types,
            entry.rust_type.as_deref().ok_or_else(|| {
                Error::Validation(format!("{} has no underlying ABI type", entry.name))
            })?,
            visiting,
        )?,
        "opaque_handle" => format!("opaque_handle({})", entry.tag.as_deref().unwrap_or(name)),
        "callback" => {
            let return_shape = public_abi_shape(
                types,
                entry.return_type.as_deref().ok_or_else(|| {
                    Error::Validation(format!("{} callback has no return type", entry.name))
                })?,
                visiting,
            )?;
            let params = entry
                .params
                .iter()
                .map(|parameter| public_abi_shape(types, &parameter.type_name, visiting))
                .collect::<Result<Vec<_>, Error>>()?;
            format!("callback({})->{return_shape}", params.join(","))
        }
        "record" | "union" => {
            let fields = entry
                .fields
                .iter()
                .map(|field| {
                    Ok(format!(
                        "{}@{}",
                        public_abi_shape(types, &field.type_name, visiting)?,
                        field.offset_64
                    ))
                })
                .collect::<Result<Vec<_>, Error>>()?;
            format!(
                "{}[size={},align={}]({})",
                entry.kind,
                entry.size_64,
                entry.align_64,
                fields.join(",")
            )
        }
        "opaque_record" | "opaque_union" => format!("{}({name})", entry.kind),
        other => {
            return Err(Error::Validation(format!(
                "{} has unsupported ABI-shape kind {other}",
                entry.name
            )));
        }
    };
    visiting.remove(name);
    Ok(shape)
}

fn promote_vendor_functions(
    inventory: &mut [RawInventoryEntry],
    union: &VendorUnion,
    catalog: &Catalog<'_>,
    semantic_overrides: &SemanticOverrides,
) -> Result<(), Error> {
    let functions = union
        .functions
        .iter()
        .filter(|function| function.backend == catalog.backend)
        .map(|function| (function.name.as_str(), function))
        .collect::<BTreeMap<_, _>>();
    for entry in inventory
        .iter_mut()
        .filter(|entry| entry.backend == catalog.backend && entry.common_id.is_none())
    {
        let Some(function) = functions.get(entry.vendor_name.as_str()) else {
            continue;
        };
        if catalog
            .functions
            .iter()
            .any(|rust_function| rust_function.name == entry.vendor_name)
        {
            continue;
        }
        let declaration = if function.kind == "alias" {
            let target = function
                .variants
                .iter()
                .find_map(|variant| variant.alias_of.as_deref())
                .ok_or_else(|| {
                    Error::Validation(format!("alias {} has no target", function.name))
                })?;
            functions.get(target).copied().ok_or_else(|| {
                Error::Validation(format!("alias {} target {target} is absent", function.name))
            })?
        } else {
            function
        };
        let declaration_variant = declaration
            .variants
            .iter()
            .find(|variant| variant.abi.is_some())
            .ok_or_else(|| Error::Validation(format!("{} has no vendor ABI", declaration.name)))?;
        let abi = declaration_variant
            .abi
            .as_ref()
            .expect("ABI variant selected");
        let return_type = map_vendor_abi_type(catalog, &abi.return_type)?;
        let params = abi
            .parameters
            .iter()
            .enumerate()
            .map(|(index, parameter)| {
                let semantics = parameter_semantics(
                    catalog,
                    semantic_overrides,
                    &declaration_variant.inventory_id,
                    &declaration.name,
                    parameter,
                    &declaration_variant.provenance,
                );
                Ok(Parameter {
                    name: rust_parameter_name(&parameter.name, index),
                    type_name: map_vendor_abi_type(catalog, &parameter.type_name)?,
                    direction: semantics.direction,
                    nullable: semantics.nullable,
                    semantic_status: semantics.status,
                    semantic_provenance: semantics.provenance,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        entry.emitted = true;
        entry.abi_return_type = Some(return_type);
        entry.abi_params = params;
        "covered_raw_only".clone_into(&mut entry.classification);
        entry.reason = format!(
            "{} is emitted from the official vendor union with every source/platform variant retained and an exact declaration-derived ABI graph.",
            entry.vendor_name
        );
    }
    Ok(())
}

struct ParameterSemantics {
    direction: String,
    nullable: Option<bool>,
    status: String,
    provenance: String,
}

fn parameter_semantics(
    catalog: &Catalog<'_>,
    overrides: &SemanticOverrides,
    inventory_id: &str,
    function: &str,
    parameter: &OracleParameter,
    declaration_provenance: &str,
) -> ParameterSemantics {
    if let Some(reviewed) = overrides.parameters.iter().find(|reviewed| {
        reviewed.inventory_id == inventory_id
            && reviewed.function == function
            && reviewed.parameter == parameter.name
    }) {
        return ParameterSemantics {
            direction: normalized_direction(&reviewed.direction),
            nullable: semantic_override_nullable(reviewed),
            status: "reviewed_override".to_owned(),
            provenance: format!(
                "{}; {} ({})",
                reviewed.provenance, reviewed.reason, inventory_id
            ),
        };
    }
    let pointer_backed_alias = parameter.pointer != "value"
        || (pointer_parts(&parameter.type_name).is_none()
            && effective_pointer_type(catalog, &parameter.type_name, &mut BTreeSet::new()));
    ParameterSemantics {
        direction: normalized_direction(&parameter.direction),
        nullable: if pointer_backed_alias {
            None
        } else {
            parameter.nullable
        },
        status: "declaration_fact".to_owned(),
        provenance: format!("{inventory_id}: {declaration_provenance}"),
    }
}

fn normalized_direction(direction: &str) -> String {
    match direction {
        "in_out" => "inout".to_owned(),
        "unspecified" | "unspecified_by_source" => "unknown".to_owned(),
        other => other.to_owned(),
    }
}

fn semantic_override_nullable(reviewed: &SemanticOverride) -> Option<bool> {
    reviewed.nullable.or_else(|| {
        reviewed
            .nullability
            .as_deref()
            .and_then(|value| match value {
                "nullable" => Some(true),
                "non_null" => Some(false),
                _ => None,
            })
    })
}

fn effective_pointer_type(
    catalog: &Catalog<'_>,
    source_type: &str,
    visiting: &mut BTreeSet<String>,
) -> bool {
    if pointer_parts(source_type).is_some() {
        return true;
    }
    if primitive_type(source_type).is_some() {
        return false;
    }
    let name = source_type
        .trim()
        .trim_start_matches("const ")
        .strip_prefix("self::")
        .unwrap_or(source_type.trim().trim_start_matches("const "));
    if !visiting.insert(name.to_owned()) {
        return false;
    }
    let result = catalog.declarations.get(name).is_some_and(|entry| {
        if entry.kind == "opaque_handle" || is_callback(&entry.normalized_signature) {
            true
        } else if matches!(entry.kind.as_str(), "type" | "alias") {
            declaration_target(&entry.normalized_signature)
                .is_ok_and(|target| effective_pointer_type(catalog, target, visiting))
        } else if entry.kind == "struct" {
            tuple_struct_target(&entry.normalized_signature)
                .is_some_and(|target| effective_pointer_type(catalog, target, visiting))
        } else {
            false
        }
    });
    visiting.remove(name);
    result
}

fn map_vendor_abi_type(catalog: &Catalog<'_>, value: &str) -> Result<String, Error> {
    map_vendor_public_type(catalog.backend, value)
}

fn populate_raw_semantic_contracts(inventory: &mut [RawInventoryEntry], types: &[TypeEntry]) {
    for entry in inventory {
        "Vendor-shaped raw call: ocgpu neither acquires nor extends ownership; ownership transfer, when any, is defined by the pinned vendor contract."
            .clone_into(&mut entry.ownership);
        "Each parameter records reviewed nullability or explicit unknown when the declaration and reviewed overrides do not establish it."
            .clone_into(&mut entry.nullability);
        let has_callback = entry.abi_params.iter().any(|parameter| {
            public_type_is_callback(types, &parameter.type_name, &mut BTreeSet::new())
        }) || entry.abi_return_type.as_deref().is_some_and(|return_type| {
            public_type_is_callback(types, return_type, &mut BTreeSet::new())
        });
        let callback_behavior = if has_callback {
            "The callback ABI is preserved exactly; invocation timing, reentrancy, and lifetime remain vendor-defined unless a reviewed parameter override states otherwise."
        } else {
            "none"
        };
        callback_behavior.clone_into(&mut entry.callback_behavior);
        "Raw calls add no serialization; vendor thread-safety and context rules apply."
            .clone_into(&mut entry.thread_safety);
    }
}

fn apply_common_semantic_provenance(manifest: &mut ApiManifest, overrides: &SemanticOverrides) {
    for function in &mut manifest.functions {
        if let Some(version) = cuda_api_version(&function.cuda.introduced) {
            function.cuda.linux.proc_version = version;
            function.cuda.windows.proc_version = version;
        }
        function.cuda.proc_address_flags = if function.cuda.per_thread_default_stream
            || matches!(function.cuda.variant.as_str(), "ptsz" | "ptds")
        {
            2
        } else {
            1
        };
        let current_context = if function.cuda.vendor_symbol == "cuCtxSetCurrent" {
            function
                .cuda
                .params
                .iter_mut()
                .find(|parameter| parameter.name == "context")
        } else {
            None
        };
        if let Some(context) = current_context {
            context.nullable = Some(true);
            "reviewed_override".clone_into(&mut context.semantic_status);
            "CUDA 13.3 cuda.h cuCtxSetCurrent contract: NULL unbinds the current context (header lines 6605-6613)."
                .clone_into(&mut context.semantic_provenance);
        }
        for raw in [&mut function.cuda, &mut function.hip] {
            for parameter in &mut raw.params {
                let matching = || {
                    overrides.parameters.iter().filter(|reviewed| {
                        reviewed.function == raw.vendor_symbol
                            && reviewed.parameter.eq_ignore_ascii_case(&parameter.name)
                    })
                };
                if let Some(reviewed) = matching()
                    .find(|reviewed| semantic_override_nullable(reviewed).is_some())
                    .or_else(|| matching().next())
                {
                    parameter.direction = normalized_direction(&reviewed.direction);
                    if let Some(nullable) = semantic_override_nullable(reviewed) {
                        parameter.nullable = Some(nullable);
                    }
                    "reviewed_override".clone_into(&mut parameter.semantic_status);
                    parameter.semantic_provenance = format!(
                        "{}; {} ({})",
                        reviewed.provenance, reviewed.reason, reviewed.inventory_id
                    );
                } else {
                    "reviewed_manifest".clone_into(&mut parameter.semantic_status);
                    function
                        .documentation_provenance
                        .clone_into(&mut parameter.semantic_provenance);
                }
            }
        }
        for parameter in &mut function.params {
            "reviewed_manifest".clone_into(&mut parameter.semantic_status);
            function
                .documentation_provenance
                .clone_into(&mut parameter.semantic_provenance);
        }
    }
}

fn cuda_api_version(introduced: &str) -> Option<u32> {
    let (major, minor) = introduced.trim().split_once('.')?;
    Some(major.parse::<u32>().ok()? * 1000 + minor.parse::<u32>().ok()? * 10)
}

fn public_type_is_callback(
    types: &[TypeEntry],
    type_name: &str,
    visiting: &mut BTreeSet<String>,
) -> bool {
    let base = type_name
        .trim()
        .trim_start_matches("*const ")
        .trim_start_matches("*mut ");
    if !visiting.insert(base.to_owned()) {
        return false;
    }
    let result = types
        .iter()
        .find(|entry| entry.name == base)
        .is_some_and(|entry| {
            entry.kind == "callback"
                || (entry.kind == "alias"
                    && entry.rust_type.as_deref().is_some_and(|underlying| {
                        public_type_is_callback(types, underlying, visiting)
                    }))
        });
    visiting.remove(base);
    result
}

#[allow(clippy::too_many_lines)]
fn promote_functions(
    inventory: &mut [RawInventoryEntry],
    catalog: &Catalog<'_>,
    semantic_overrides: &SemanticOverrides,
) -> Result<(), Error> {
    let functions = catalog
        .functions
        .iter()
        .copied()
        .map(|entry| (entry.name.as_str(), entry))
        .collect::<BTreeMap<_, _>>();
    for raw in inventory
        .iter_mut()
        .filter(|entry| entry.backend == catalog.backend)
    {
        let rust_function = functions.get(raw.vendor_name.as_str()).copied();
        if let Some(function) = rust_function {
            raw.oracle_variants
                .retain(|variant| variant.oracle_source != catalog.inventory_id);
            raw.oracle_variants.push(RawOracleVariant {
                oracle_source: catalog.inventory_id.to_owned(),
                oracle_kind: "function".to_owned(),
                source_version: catalog.source_version.to_owned(),
                oracle_signature: function.normalized_signature.clone(),
                oracle_signature_hash: function.signature_hash.clone(),
                platforms: function.platforms.clone(),
                aliases: Vec::new(),
                alias_of: None,
                provenance: function.provenance.clone(),
            });
        }
        if raw.common_id.is_some() {
            continue;
        }
        let Some(function) = rust_function else {
            continue;
        };
        if raw.raw_name.is_none() {
            raw.emitted = false;
            raw.abi_return_type = None;
            raw.abi_params.clear();
            "intentionally_omitted".clone_into(&mut raw.classification);
            raw.reason = format!(
                "{} is a Rust crate loader helper rather than a vendor C entrypoint, so it has no raw C symbol or table slot.",
                raw.vendor_name
            );
            continue;
        }
        let Some(abi) = &function.abi else {
            raw.emitted = false;
            raw.abi_return_type = None;
            raw.abi_params.clear();
            "unrepresentable".clone_into(&mut raw.classification);
            raw.reason = format!(
                "{} has no C ABI declaration object in {}.",
                raw.vendor_name, catalog.inventory_id
            );
            continue;
        };
        if abi.return_type.contains("libloading")
            || abi
                .parameters
                .iter()
                .any(|parameter| parameter.type_name.contains("libloading"))
        {
            raw.raw_name = None;
            raw.emitted = false;
            raw.abi_return_type = None;
            raw.abi_params.clear();
            "intentionally_omitted".clone_into(&mut raw.classification);
            raw.reason = format!(
                "{} is a Rust crate loader helper returning a libloading object, not a vendor C entrypoint.",
                raw.vendor_name
            );
            continue;
        }
        verify_complete_layout(catalog, &abi.return_type, &mut BTreeSet::new())?;
        for param in &abi.parameters {
            verify_complete_layout(catalog, &param.type_name, &mut BTreeSet::new())?;
        }
        let return_type = map_abi_type(catalog, &abi.return_type)?;
        let params = abi
            .parameters
            .iter()
            .enumerate()
            .map(|(index, parameter)| {
                let semantics = parameter_semantics(
                    catalog,
                    semantic_overrides,
                    catalog.inventory_id,
                    &function.name,
                    parameter,
                    &function.provenance,
                );
                Ok(Parameter {
                    name: rust_parameter_name(&parameter.name, index),
                    type_name: map_abi_type(catalog, &parameter.type_name)?,
                    direction: semantics.direction,
                    nullable: semantics.nullable,
                    semantic_status: semantics.status,
                    semantic_provenance: semantics.provenance,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        raw.raw_name = Some(raw_prefixed_name(catalog.backend, &raw.vendor_name)?);
        raw.emitted = true;
        raw.abi_return_type = Some(return_type);
        raw.abi_params = params;
        "covered_raw_only".clone_into(&mut raw.classification);
        raw.reason = format!(
            "{} is emitted from the exact {} declaration; every transitive by-value dependency is a scalar, integer typedef, opaque handle, or callback, and incomplete records occur only behind pointers.",
            raw.vendor_name, catalog.inventory_id
        );
    }
    Ok(())
}

fn verify_complete_layout(
    catalog: &Catalog<'_>,
    source_type: &str,
    visiting: &mut BTreeSet<String>,
) -> Result<(), Error> {
    if pointer_parts(source_type).is_some() || primitive_type(source_type).is_some() {
        return Ok(());
    }
    let name = source_type
        .trim()
        .strip_prefix("self::")
        .unwrap_or(source_type.trim());
    if !visiting.insert(name.to_owned()) {
        return Ok(());
    }
    let Some(entry) = catalog.declarations.get(name) else {
        visiting.remove(name);
        return Err(Error::Validation(format!(
            "{} function references unknown by-value type {name}",
            catalog.backend
        )));
    };
    match entry.kind.as_str() {
        "struct" if tuple_struct_target(&entry.normalized_signature).is_some() => {
            verify_complete_layout(
                catalog,
                tuple_struct_target(&entry.normalized_signature).expect("tuple target checked"),
                visiting,
            )?;
        }
        "struct" | "union" => {
            let _ = record_size_align(catalog, entry, visiting)?;
        }
        "type"
            if is_enum(&entry.normalized_signature) || is_callback(&entry.normalized_signature) => {
        }
        "type" | "alias" => {
            verify_complete_layout(
                catalog,
                declaration_target(&entry.normalized_signature)?,
                visiting,
            )?;
        }
        _ => {}
    }
    visiting.remove(name);
    Ok(())
}

fn rust_parameter_name(source: &str, index: usize) -> String {
    let candidate = if source.is_empty() {
        format!("arg{index}")
    } else {
        source.to_owned()
    };
    if matches!(
        candidate.as_str(),
        "as" | "break"
            | "const"
            | "continue"
            | "crate"
            | "else"
            | "enum"
            | "extern"
            | "false"
            | "fn"
            | "for"
            | "if"
            | "impl"
            | "in"
            | "let"
            | "loop"
            | "match"
            | "mod"
            | "move"
            | "mut"
            | "pub"
            | "ref"
            | "return"
            | "self"
            | "Self"
            | "static"
            | "struct"
            | "super"
            | "trait"
            | "true"
            | "type"
            | "unsafe"
            | "use"
            | "where"
            | "while"
            | "async"
            | "await"
            | "dyn"
    ) {
        format!("{candidate}_")
    } else {
        candidate
    }
}

fn rust_field_identifier(source: &str) -> String {
    if matches!(
        source,
        "as" | "break"
            | "const"
            | "continue"
            | "crate"
            | "else"
            | "enum"
            | "extern"
            | "false"
            | "fn"
            | "for"
            | "if"
            | "impl"
            | "in"
            | "let"
            | "loop"
            | "match"
            | "mod"
            | "move"
            | "mut"
            | "pub"
            | "ref"
            | "return"
            | "self"
            | "Self"
            | "static"
            | "struct"
            | "super"
            | "trait"
            | "true"
            | "type"
            | "unsafe"
            | "use"
            | "where"
            | "while"
            | "async"
            | "await"
            | "dyn"
    ) {
        format!("r#{source}")
    } else {
        source.to_owned()
    }
}

fn public_field_identifiers(source: &str) -> (String, Option<String>) {
    let sanitized = if source.starts_with("__") {
        let suffix = source
            .trim_start_matches('_')
            .strip_prefix("bindgen_")
            .unwrap_or_else(|| source.trim_start_matches('_'));
        format!("ocgpu_{suffix}")
    } else {
        sanitize_public_identifier(source)
    };
    let rust = rust_field_identifier(&sanitized);
    let c_name = (sanitized == source && rust != source).then(|| source.to_owned());
    (rust, c_name)
}

fn capture_existing_slots(inventory: &[RawInventoryEntry]) -> BTreeMap<(String, String), u32> {
    let mut next = BTreeMap::<String, u32>::new();
    let mut slots = BTreeMap::new();
    for entry in inventory
        .iter()
        .filter(|entry| entry.emitted && entry.common_id.is_none())
    {
        let slot = entry.table_order.unwrap_or_else(|| {
            let value = *next.get(&entry.backend).unwrap_or(&0);
            next.insert(entry.backend.clone(), value + 1);
            value
        });
        slots.insert((entry.backend.clone(), entry.vendor_name.clone()), slot);
        next.entry(entry.backend.clone())
            .and_modify(|value| *value = (*value).max(slot + 1))
            .or_insert(slot + 1);
    }
    slots
}

fn assign_append_only_slots(
    inventory: &mut [RawInventoryEntry],
    old_slots: &BTreeMap<(String, String), u32>,
) {
    for backend in ["cuda", "hip"] {
        let mut next = old_slots
            .iter()
            .filter(|((owner, _), _)| owner == backend)
            .map(|(_, slot)| slot + 1)
            .max()
            .unwrap_or(0);
        for entry in inventory
            .iter_mut()
            .filter(|entry| entry.backend == backend && entry.emitted && entry.common_id.is_none())
        {
            let key = (entry.backend.clone(), entry.vendor_name.clone());
            entry.table_order = old_slots.get(&key).copied().or_else(|| {
                let slot = next;
                next += 1;
                Some(slot)
            });
        }
        for entry in inventory.iter_mut().filter(|entry| {
            entry.backend == backend && (!entry.emitted || entry.common_id.is_some())
        }) {
            entry.table_order = None;
        }
    }
}

fn refresh_table_layouts(manifest: &mut ApiManifest) {
    for table in &mut manifest.tables {
        let raw_count = manifest
            .raw_inventory
            .iter()
            .filter(|entry| {
                entry.backend == table.surface && entry.emitted && entry.common_id.is_none()
            })
            .count();
        let count = manifest.functions.len()
            + if table.surface == "common" {
                0
            } else {
                raw_count
            };
        table.size_64 = 24 + u32::try_from(count).expect("table count fits u32") * 8;
        table.layout_hash = format_hash(table_layout_hash(count));
    }
}

fn format_manifest_hex(serialized: &str) -> String {
    serialized
        .lines()
        .map(|line| {
            if let Some(value) = line
                .strip_prefix("stable_id = ")
                .and_then(|value| value.parse::<u32>().ok())
            {
                return format!("stable_id = 0x{value:08x}");
            }
            if let Some(value) = line
                .strip_prefix("abi_version = ")
                .and_then(|value| value.parse::<u32>().ok())
            {
                return format!("abi_version = 0x{value:08x}");
            }
            line.to_owned()
        })
        .collect::<Vec<_>>()
        .join("\n")
        + "\n"
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vendor_union() -> VendorUnion {
        serde_json::from_str(include_str!("../../../oracle/vendor/function-union.json"))
            .expect("committed vendor union parses")
    }

    #[test]
    fn cuda_macro_aliases_never_fallback_to_legacy_unsuffixed_exports() {
        let union = vendor_union();
        let canonical = union
            .functions
            .iter()
            .find(|function| function.name == "cuMemAlloc_v2")
            .expect("CUDA union has cuMemAlloc_v2");
        let (_, direct, linux, windows, _, _, _) = vendor_resolution(canonical);
        assert_eq!(direct, ["cuMemAlloc_v2"]);
        assert_eq!((linux, windows), (0, 0));

        let alias = union
            .functions
            .iter()
            .find(|function| function.name == "cuMemAlloc" && function.kind == "alias")
            .expect("CUDA union has cuMemAlloc macro alias");
        let (_, direct, linux, windows, _, variant, target) = vendor_resolution(alias);
        assert_eq!(direct, ["cuMemAlloc_v2"]);
        assert_eq!((linux, windows), (0, 0));
        assert_eq!(variant, "alias");
        assert_eq!(target.as_deref(), Some("cuMemAlloc_v2"));
    }

    #[test]
    fn hip_spt_proc_lookup_uses_base_name_and_per_thread_flag() {
        let union = vendor_union();
        let function = union
            .functions
            .iter()
            .find(|function| function.name == "hipEventRecord_spt")
            .expect("HIP union has an SPT function");
        let (proc_name, direct, linux, _, flags, variant, _) = vendor_resolution(function);
        assert_eq!(proc_name, "hipEventRecord");
        assert_eq!(direct, ["hipEventRecord_spt"]);
        assert_eq!(linux, 0);
        assert_eq!(flags, 2);
        assert_eq!(variant, "spt");
    }

    #[test]
    fn numeric_cuda_suffix_is_removed_only_for_proc_queries() {
        assert_eq!(strip_numeric_version_suffix("cuMemAlloc_v2"), "cuMemAlloc");
        assert_eq!(strip_numeric_version_suffix("cuMemcpy_spt"), "cuMemcpy_spt");
    }

    #[test]
    fn target_dependent_c_long_never_gets_a_shared_layout() {
        assert!(primitive_size_align("core::ffi::c_long").is_err());
        assert!(primitive_size_align("core::ffi::c_ulong").is_err());
        assert_eq!(primitive_size_align("u32").unwrap(), (4, 4));
        assert_eq!(primitive_size_align("u64").unwrap(), (8, 8));
    }
}
