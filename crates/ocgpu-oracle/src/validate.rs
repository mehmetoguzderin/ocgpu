// SPDX-License-Identifier: CC0-1.0

use crate::hash::sha256;
use crate::model::{
    Classification, CoverageCatalog, CoverageDecision, CudaProcAddressCatalog,
    CudaProcAddressVariant, Direction, Entry, Inventory, ItemKind, PointerKind,
    ReviewedNullability, SemanticCatalog, SemanticOverride, SourceArtifact, VendorFunctionUnion,
};
use crate::vendor_union::build_vendor_function_union;
use serde_json::Value as JsonValue;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;
use toml::Value as TomlValue;

const INVENTORY_PATHS: [&str; 5] = [
    "oracle/vendor/cuda/13.3-13030.json",
    "oracle/vendor/hip/general-7.14.60850.json",
    "oracle/vendor/hip/windows-7.2.0.json",
    "oracle/rust/cudarc-0.19.9.json",
    "oracle/rust/rocmrc-0.5.0.json",
];

const EXPECTED_INVENTORIES: [&str; 5] = [
    "cuda-vendor-13.3-13030",
    "hip-general-7.14.60850",
    "hip-windows-7.2.0",
    "cudarc-0.19.9",
    "rocmrc-0.5.0",
];

type EntryKey<'a> = (&'a str, ItemKind, &'a str);
type SemanticKey<'a> = (&'a str, &'a str, &'a str);

/// Aggregate validation failure. All discoverable violations are reported in one run.
#[derive(Debug, Error)]
#[error("coverage validation failed with {count} error(s):\n{details}")]
pub struct ValidationError {
    count: usize,
    details: String,
}

/// Counts from a successful validation run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ValidationSummary {
    /// Committed independent inventories.
    pub inventories: usize,
    /// Total normalized entries.
    pub entries: usize,
    /// Human-reviewed classifications.
    pub decisions: usize,
    /// Canonical manifest item IDs seen by the validator.
    pub manifest_entries: usize,
    /// Controlled C export symbols.
    pub exports: usize,
    /// Reviewed HIP runtime-major compatibility profiles.
    pub hip_runtime_profiles: usize,
    /// Common HIP operations proven across those profiles.
    pub hip_common_functions: usize,
    /// HIP device-attribute values proven across those profiles.
    pub hip_device_attributes: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct HipProfileCounts {
    profiles: usize,
    common_functions: usize,
    device_attributes: usize,
}

/// Loads the five required snapshots and the separate human classification catalog.
pub fn read_inputs(
    repository_root: &Path,
) -> Result<(Vec<Inventory>, CoverageCatalog), ValidationError> {
    let mut errors = Vec::new();
    let mut inventories = Vec::new();
    for relative in INVENTORY_PATHS {
        let path = repository_root.join(relative);
        match read_json::<Inventory>(&path) {
            Ok(inventory) => inventories.push(inventory),
            Err(error) => errors.push(error),
        }
    }
    let catalog_path = repository_root.join("coverage/classifications.json");
    let catalog = match read_json::<CoverageCatalog>(&catalog_path) {
        Ok(catalog) => Some(catalog),
        Err(error) => {
            errors.push(error);
            None
        }
    };
    match (errors.is_empty(), catalog) {
        (true, Some(catalog)) => Ok((inventories, catalog)),
        (true, None) => Err(validation_error(&[
            "coverage catalog was not loaded despite a successful read".to_owned(),
        ])),
        (false, _) => Err(validation_error(&errors)),
    }
}

/// Validates classification, signatures, layouts, aliases, platforms, canonical-manifest
/// accounting, generated implementation inventory, and export controls.
pub fn validate_repository(repository_root: &Path) -> Result<ValidationSummary, ValidationError> {
    let (inventories, catalog) = read_inputs(repository_root)?;
    let mut errors = Vec::new();
    validate_inventory_set(&inventories, &mut errors);
    let entries = entry_index(&inventories, &mut errors);
    let semantic_path = repository_root.join("oracle/semantic-overrides.json");
    let semantic_catalog = match read_json::<SemanticCatalog>(&semantic_path) {
        Ok(catalog) => Some(catalog),
        Err(error) => {
            errors.push(error);
            None
        }
    };
    let semantics = semantic_catalog
        .as_ref()
        .map_or_else(BTreeMap::new, |catalog| {
            semantic_index(catalog, &entries, &mut errors)
        });
    validate_entries(&inventories, &entries, &semantics, &mut errors);
    let decisions = decision_index(&catalog, &mut errors);
    validate_classifications(&entries, &decisions, &mut errors);
    validate_cross_inventory_facts(&inventories, &mut errors);

    let proc_catalog_path = repository_root.join("oracle/vendor/cuda/13.3-13030-proc-address.json");
    let proc_catalog = match read_json::<CudaProcAddressCatalog>(&proc_catalog_path) {
        Ok(catalog) => {
            validate_cuda_proc_catalog(&catalog, &mut errors);
            Some(catalog)
        }
        Err(error) => {
            errors.push(error);
            None
        }
    };
    let vendor_union_path = repository_root.join("oracle/vendor/function-union.json");
    let vendor_union = match read_json::<VendorFunctionUnion>(&vendor_union_path) {
        Ok(union) => Some(union),
        Err(error) => {
            errors.push(error);
            None
        }
    };
    if let (Some(proc_catalog), Some(vendor_union)) = (&proc_catalog, &vendor_union) {
        validate_vendor_union(&inventories, proc_catalog, vendor_union, &mut errors);
    }

    let manifest_path = repository_root.join("api/ocgpu-api.toml");
    let manifest = read_toml(&manifest_path, &mut errors);
    let manifest_ids = manifest
        .as_ref()
        .map_or_else(BTreeSet::new, collect_manifest_ids);
    let oracle_manifest_ids = manifest
        .as_ref()
        .map_or_else(BTreeSet::new, collect_oracle_accountable_manifest_ids);
    let manifest_symbols = manifest
        .as_ref()
        .map_or_else(BTreeSet::new, collect_ocgpu_symbols_toml);
    validate_manifest_accounting(&decisions, &manifest_ids, &oracle_manifest_ids, &mut errors);

    let hip_profile_counts = validate_hip_runtime_profile_ledger(repository_root, &mut errors);

    let generated_path = repository_root.join("crates/ocgpu-codegen/generated/api-inventory.json");
    let generated = read_json_value(&generated_path, &mut errors);
    let generated_ids = generated
        .as_ref()
        .map_or_else(BTreeSet::new, collect_generated_ids);
    validate_generated_inventory(&manifest_ids, &generated_ids, &mut errors);

    let implementation_text = read_implementation_text(repository_root, &mut errors);
    validate_implementation_symbols(&decisions, &implementation_text, &mut errors);

    let def_exports = read_def_exports(&repository_root.join("exports/ocgpu.def"), &mut errors);
    let map_exports = read_map_exports(&repository_root.join("exports/ocgpu.map"), &mut errors);
    let generated_symbols = generated
        .as_ref()
        .map_or_else(BTreeSet::new, collect_ocgpu_symbols_json);
    validate_exports(
        &decisions,
        &manifest_symbols,
        &generated_symbols,
        &def_exports,
        &map_exports,
        &mut errors,
    );

    if errors.is_empty() {
        Ok(ValidationSummary {
            inventories: inventories.len(),
            entries: entries.len(),
            decisions: decisions.len(),
            manifest_entries: manifest_ids.len(),
            exports: def_exports.len(),
            hip_runtime_profiles: hip_profile_counts.profiles,
            hip_common_functions: hip_profile_counts.common_functions,
            hip_device_attributes: hip_profile_counts.device_attributes,
        })
    } else {
        Err(validation_error(&errors))
    }
}

#[allow(clippy::too_many_lines)]
fn validate_hip_runtime_profile_ledger(
    repository_root: &Path,
    errors: &mut Vec<String>,
) -> HipProfileCounts {
    let path = repository_root.join("oracle/vendor/hip/runtime-profiles.json");
    let Some(ledger) = read_json_value(&path, errors) else {
        return HipProfileCounts::default();
    };
    let Some(root) = ledger.as_object() else {
        errors.push(format!("{} must contain a JSON object", path.display()));
        return HipProfileCounts::default();
    };
    if root.get("schema_version").and_then(JsonValue::as_u64) != Some(1)
        || root.get("inventory_id").and_then(JsonValue::as_str) != Some("hip-runtime-profiles")
    {
        errors.push("HIP runtime-profile ledger metadata is stale".to_owned());
    }
    let profiles = root
        .get("profiles")
        .and_then(JsonValue::as_array)
        .map_or(&[][..], Vec::as_slice);
    let exact = root
        .get("common_functions")
        .and_then(JsonValue::as_array)
        .map_or(&[][..], Vec::as_slice);
    let adapters = root
        .get("common_adapters")
        .and_then(JsonValue::as_array)
        .map_or(&[][..], Vec::as_slice);
    let attributes = root
        .get("device_attributes")
        .and_then(JsonValue::as_array)
        .map_or(&[][..], Vec::as_slice);
    let releases = root
        .get("reviewed_releases")
        .and_then(JsonValue::as_array)
        .map_or(&[][..], Vec::as_slice);

    let expected_profiles = [
        (7, 0x0007_0000_u64, 70_253_210_i64, Some(71_460_850_i64)),
        (6, 0x0006_0000_u64, 60_140_093_i64, None),
        (5, 0x0005_0000_u64, 50_731_541_i64, None),
    ];
    if profiles.len() != expected_profiles.len() {
        errors.push("HIP runtime-profile ledger must contain exactly profiles 7/6/5".to_owned());
    }
    for (profile, (major, flag, minimum, linux_raw_floor)) in profiles.iter().zip(expected_profiles)
    {
        let windows = profile.get("windows");
        let linux = profile.get("linux");
        let maximum = i64::from((major + 1) * 10_000_000 - 1);
        if profile.get("runtime_major").and_then(JsonValue::as_i64) != Some(i64::from(major))
            || profile.get("table_flag").and_then(JsonValue::as_u64) != Some(flag)
            || windows
                .and_then(|value| value.get("runtime_version_min_inclusive"))
                .and_then(JsonValue::as_i64)
                != Some(minimum)
            || linux
                .and_then(|value| value.get("runtime_version_min_inclusive"))
                .and_then(JsonValue::as_i64)
                != Some(minimum)
            || windows
                .and_then(|value| value.get("runtime_version_max_inclusive"))
                .and_then(JsonValue::as_i64)
                != Some(maximum)
            || linux
                .and_then(|value| value.get("runtime_version_max_inclusive"))
                .and_then(JsonValue::as_i64)
                != Some(maximum)
            || linux
                .and_then(|value| value.get("raw_inventory_min_inclusive"))
                .and_then(JsonValue::as_i64)
                != linux_raw_floor
        {
            errors.push(format!(
                "HIP {major} runtime-profile interval is stale or unsafe"
            ));
        }
    }

    let function_names = exact
        .iter()
        .chain(adapters)
        .filter_map(|entry| entry.get("name").and_then(JsonValue::as_str))
        .collect::<BTreeSet<_>>();
    let attribute_names = attributes
        .iter()
        .filter_map(|entry| entry.get("name").and_then(JsonValue::as_str))
        .collect::<BTreeSet<_>>();
    if exact.len() != 25
        || adapters.len() != 1
        || function_names.len() != 26
        || !function_names.contains("hipMemcpyHtoD")
        || attribute_names.len() != 32
    {
        errors.push(
            "HIP profile coverage must remain 25 raw-exact + 1 adapted common operations and 32 attributes"
                .to_owned(),
        );
    }
    let semantic_reviews = root
        .get("semantic_reviews")
        .and_then(JsonValue::as_array)
        .map_or(&[][..], Vec::as_slice);
    let mut semantic_operations = BTreeSet::new();
    let mut duplicate_semantic_operation = false;
    for review in semantic_reviews {
        let operations = review
            .get("operations")
            .and_then(JsonValue::as_array)
            .map_or(&[][..], Vec::as_slice);
        if operations.is_empty()
            || review
                .get("finding")
                .and_then(JsonValue::as_str)
                .is_none_or(str::is_empty)
            || review
                .get("proof")
                .and_then(JsonValue::as_str)
                .is_none_or(str::is_empty)
        {
            errors.push("HIP semantic review evidence is incomplete".to_owned());
        }
        for operation in operations.iter().filter_map(JsonValue::as_str) {
            duplicate_semantic_operation |= !semantic_operations.insert(operation);
        }
    }
    if semantic_reviews.len() != 7
        || duplicate_semantic_operation
        || semantic_operations != function_names
    {
        errors.push(
            "HIP semantic review operation union must equal all 26 common operations exactly"
                .to_owned(),
        );
    }
    if releases.len() != 7 {
        errors.push(
            "HIP profile ledger must retain all seven reviewed release observations".to_owned(),
        );
    }
    for release in releases {
        let id = release
            .get("id")
            .and_then(JsonValue::as_str)
            .unwrap_or("<unknown>");
        let hashes = [
            "hip_archive_sha256",
            "hip_header_sha256",
            "hip_version_sha256",
            "clr_archive_sha256",
            "clr_cmake_sha256",
        ];
        let mut unique = BTreeSet::new();
        for field in hashes {
            let value = release.get(field).and_then(JsonValue::as_str).unwrap_or("");
            if value.len() != 71
                || !value.starts_with("sha256:")
                || !value[7..].bytes().all(|byte| byte.is_ascii_hexdigit())
            {
                errors.push(format!("{id} {field} is not a canonical SHA-256"));
            }
            if !unique.insert(value) {
                errors.push(format!("{id} reuses an archive/member hash without proof"));
            }
        }
    }
    let mut declaration_function_names = function_names
        .iter()
        .map(|name| (*name).to_owned())
        .collect::<BTreeSet<_>>();
    declaration_function_names.insert("hipRuntimeGetVersion".to_owned());
    let attribute_values = attributes
        .iter()
        .filter_map(|entry| {
            Some((
                entry.get("name")?.as_str()?.to_owned(),
                entry.get("value")?.as_i64()?,
            ))
        })
        .collect::<BTreeMap<_, _>>();
    validate_hip_runtime_declarations(
        repository_root,
        releases,
        &declaration_function_names,
        &attribute_values,
        errors,
    );
    HipProfileCounts {
        profiles: profiles.len(),
        common_functions: function_names.len(),
        device_attributes: attribute_names.len(),
    }
}

#[allow(clippy::too_many_lines)]
fn validate_hip_runtime_declarations(
    repository_root: &Path,
    releases: &[JsonValue],
    expected_functions: &BTreeSet<String>,
    expected_attributes: &BTreeMap<String, i64>,
    errors: &mut Vec<String>,
) {
    let path = repository_root.join("oracle/vendor/hip/runtime-profile-declarations.json");
    let Some(document) = read_json_value(&path, errors) else {
        return;
    };
    let Some(root) = document.as_object() else {
        errors.push(format!("{} must contain a JSON object", path.display()));
        return;
    };
    if root.get("schema_version").and_then(JsonValue::as_u64) != Some(1)
        || root
            .get("spdx_license_identifier")
            .and_then(JsonValue::as_str)
            != Some("CC0-1.0")
        || root.get("inventory_id").and_then(JsonValue::as_str)
            != Some("hip-runtime-profile-declarations")
        || root
            .get("provenance")
            .and_then(JsonValue::as_str)
            .is_none_or(str::is_empty)
    {
        errors.push("HIP runtime-profile declaration metadata is stale".to_owned());
    }
    let snapshots = root
        .get("snapshots")
        .and_then(JsonValue::as_array)
        .map_or(&[][..], Vec::as_slice);
    let expected = [
        (
            "hip-5.7.31541",
            "hip-profile-5.7.0-review",
            "authoritative-hip-header",
            &[
                "aarch64-unknown-linux-gnu",
                "x86_64-pc-windows-msvc",
                "x86_64-unknown-linux-gnu",
            ][..],
        ),
        (
            "hip-5.7.31921",
            "hip-profile-5.7.1-review",
            "authoritative-hip-header",
            &[
                "aarch64-unknown-linux-gnu",
                "x86_64-pc-windows-msvc",
                "x86_64-unknown-linux-gnu",
            ][..],
        ),
        (
            "hip-6.1.40093",
            "hip-profile-6.1.2-review",
            "authoritative-hip-header",
            &[
                "aarch64-unknown-linux-gnu",
                "x86_64-pc-windows-msvc",
                "x86_64-unknown-linux-gnu",
            ][..],
        ),
        (
            "hip-6.2.41134",
            "hip-profile-6.2.4-review",
            "authoritative-hip-header",
            &[
                "aarch64-unknown-linux-gnu",
                "x86_64-pc-windows-msvc",
                "x86_64-unknown-linux-gnu",
            ][..],
        ),
        (
            "hip-6.4.43484",
            "hip-profile-6.4.2-review",
            "authoritative-hip-header",
            &[
                "aarch64-unknown-linux-gnu",
                "x86_64-pc-windows-msvc",
                "x86_64-unknown-linux-gnu",
            ][..],
        ),
        (
            "hip-7.2.53210",
            "hip-profile-7.2.53210-review",
            "authoritative-hip-header",
            &[
                "aarch64-unknown-linux-gnu",
                "x86_64-pc-windows-msvc",
                "x86_64-unknown-linux-gnu",
            ][..],
        ),
        (
            "hip-7.14.60850",
            "hip-profile-7.14.60850-review",
            "semantic-hip-header",
            &["aarch64-unknown-linux-gnu", "x86_64-unknown-linux-gnu"][..],
        ),
    ];
    if snapshots.len() != expected.len() || releases.len() != expected.len() {
        errors.push("HIP runtime-profile declaration release set is incomplete".to_owned());
        return;
    }

    for ((snapshot, release), (id, inventory_id, header_role, platforms)) in
        snapshots.iter().zip(releases).zip(expected)
    {
        let release_id = release.get("id").and_then(JsonValue::as_str);
        let header = snapshot.get("source_header_artifact");
        let clr = snapshot.get("source_clr_artifact");
        let snapshot_platforms = json_string_values(snapshot.get("source_inventory_platforms"));
        if release_id != Some(id)
            || snapshot.get("release_id").and_then(JsonValue::as_str) != Some(id)
            || snapshot
                .get("source_inventory_id")
                .and_then(JsonValue::as_str)
                != Some(inventory_id)
            || snapshot_platforms.as_slice() != platforms
            || header
                .and_then(|value| value.get("role"))
                .and_then(JsonValue::as_str)
                != Some(header_role)
            || header.and_then(|value| value.get("url")) != release.get("hip_archive_url")
            || header.and_then(|value| value.get("sha256")) != release.get("hip_header_sha256")
            || header.and_then(|value| value.get("path")) != release.get("hip_header_path")
            || header
                .and_then(|value| value.get("revision"))
                .and_then(JsonValue::as_str)
                .is_none_or(str::is_empty)
            || clr
                .and_then(|value| value.get("role"))
                .and_then(JsonValue::as_str)
                != Some("supporting-clr-source")
            || clr.and_then(|value| value.get("url")) != release.get("clr_archive_url")
            || clr.and_then(|value| value.get("sha256")) != release.get("clr_archive_sha256")
            || clr
                .and_then(|value| value.get("path"))
                .and_then(JsonValue::as_str)
                != Some("hipamd/include")
            || clr
                .and_then(|value| value.get("revision"))
                .and_then(JsonValue::as_str)
                .is_none_or(str::is_empty)
        {
            errors.push(format!(
                "{id} compact declaration source/header/platform binding is stale"
            ));
        }

        let target_abi = snapshot.get("target_abi");
        if target_abi
            .and_then(|value| value.get("pointer_width_bits"))
            .and_then(JsonValue::as_u64)
            != Some(64)
            || target_abi
                .and_then(|value| value.get("size_t_width_bits"))
                .and_then(JsonValue::as_u64)
                != Some(64)
            || target_abi
                .and_then(|value| value.get("enum_width_bits"))
                .and_then(JsonValue::as_u64)
                != Some(32)
            || target_abi
                .and_then(|value| value.get("success_value"))
                .and_then(JsonValue::as_i64)
                != Some(0)
            || target_abi
                .and_then(|value| value.get("null_pointer_sentinel"))
                .and_then(JsonValue::as_str)
                != Some("all-bits-zero")
        {
            errors.push(format!("{id} compact target ABI facts are stale"));
        }

        let functions = snapshot
            .get("functions")
            .and_then(JsonValue::as_array)
            .map_or(&[][..], Vec::as_slice);
        let function_names = functions
            .iter()
            .filter_map(|entry| entry.get("name").and_then(JsonValue::as_str))
            .map(str::to_owned)
            .collect::<BTreeSet<_>>();
        if functions.len() != 27 || &function_names != expected_functions {
            errors.push(format!("{id} compact function declaration set is stale"));
        }
        for entry in functions {
            validate_compact_declaration_entry(id, entry, platforms, errors);
        }

        let types = snapshot
            .get("transitive_types")
            .and_then(JsonValue::as_array)
            .map_or(&[][..], Vec::as_slice);
        if types.len() != 9 {
            errors.push(format!("{id} compact transitive type set is stale"));
        }
        for entry in types {
            validate_compact_declaration_entry(id, entry, platforms, errors);
        }

        let attributes = snapshot
            .get("device_attributes")
            .and_then(JsonValue::as_array)
            .map_or(&[][..], Vec::as_slice);
        let attribute_values = attributes
            .iter()
            .filter_map(|entry| {
                Some((
                    entry.get("name")?.as_str()?.to_owned(),
                    entry.get("value")?.as_i64()?,
                ))
            })
            .collect::<BTreeMap<_, _>>();
        if attributes.len() != 32 || &attribute_values != expected_attributes {
            errors.push(format!("{id} compact device-attribute values are stale"));
        }
        for entry in attributes {
            validate_compact_declaration_entry(id, entry, platforms, errors);
            let name = entry.get("name").and_then(JsonValue::as_str).unwrap_or("");
            if !entry
                .get("normalized_signature")
                .and_then(JsonValue::as_str)
                .is_some_and(|signature| signature.starts_with(&format!("enum-value {name}=")))
            {
                errors.push(format!("{id} attribute {name} declaration is malformed"));
            }
        }
    }
}

fn validate_compact_declaration_entry(
    release: &str,
    entry: &JsonValue,
    expected_platforms: &[&str],
    errors: &mut Vec<String>,
) {
    let name = entry.get("name").and_then(JsonValue::as_str).unwrap_or("");
    let signature = entry
        .get("normalized_signature")
        .and_then(JsonValue::as_str)
        .unwrap_or("");
    let hash = entry
        .get("signature_hash")
        .and_then(JsonValue::as_str)
        .unwrap_or("");
    let platforms = json_string_values(entry.get("platforms"));
    if name.is_empty()
        || signature.is_empty()
        || !is_canonical_sha256(hash)
        || hash != sha256(signature)
        || platforms.as_slice() != expected_platforms
    {
        errors.push(format!(
            "{release} compact declaration {name} has stale signature/platform evidence"
        ));
    }
}

fn json_string_values(value: Option<&JsonValue>) -> Vec<&str> {
    value
        .and_then(JsonValue::as_array)
        .map_or(&[][..], Vec::as_slice)
        .iter()
        .filter_map(JsonValue::as_str)
        .collect()
}

fn is_canonical_sha256(value: &str) -> bool {
    value.len() == 71
        && value.starts_with("sha256:")
        && value[7..].bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn validate_inventory_set(inventories: &[Inventory], errors: &mut Vec<String>) {
    let actual = inventories
        .iter()
        .map(|inventory| inventory.inventory_id.as_str())
        .collect::<BTreeSet<_>>();
    let expected = EXPECTED_INVENTORIES.into_iter().collect::<BTreeSet<_>>();
    if actual != expected {
        errors.push(format!(
            "inventory IDs differ: expected {expected:?}, found {actual:?}"
        ));
    }
    for inventory in inventories {
        if inventory.schema_version != 1 {
            errors.push(format!(
                "{} uses unsupported schema version {}",
                inventory.inventory_id, inventory.schema_version
            ));
        }
        if inventory.spdx_license_identifier != "CC0-1.0" {
            errors.push(format!(
                "{} must declare SPDX license CC0-1.0",
                inventory.inventory_id
            ));
        }
        if inventory.source_name.trim().is_empty()
            || inventory.source_version.trim().is_empty()
            || inventory.provenance.trim().is_empty()
        {
            errors.push(format!(
                "{} has incomplete source provenance",
                inventory.inventory_id
            ));
        }
        if inventory.source_artifacts.is_empty() {
            errors.push(format!(
                "{} records no exact fetched source artifacts",
                inventory.inventory_id
            ));
        }
        let artifact_keys = inventory
            .source_artifacts
            .iter()
            .map(|artifact| (artifact.role.as_str(), artifact.url.as_str()))
            .collect::<Vec<_>>();
        if !strictly_sorted_unique(&artifact_keys) {
            errors.push(format!(
                "{} source artifacts must be sorted by role and URL with no duplicates",
                inventory.inventory_id
            ));
        }
        for artifact in &inventory.source_artifacts {
            let hash = artifact.sha256.strip_prefix("sha256:").unwrap_or("");
            if artifact.role.trim().is_empty()
                || !artifact.url.starts_with("https://")
                || hash.len() != 64
                || !hash.chars().all(|character| character.is_ascii_hexdigit())
                || artifact.revision.trim().is_empty()
                || artifact.path.trim().is_empty()
            {
                errors.push(format!(
                    "{} has an incomplete or invalid source artifact record for {}",
                    inventory.inventory_id, artifact.role
                ));
            }
        }
        if inventory.platforms.is_empty() {
            errors.push(format!(
                "{} has no platform baseline",
                inventory.inventory_id
            ));
        }
        if !strictly_sorted_unique(&inventory.platforms) {
            errors.push(format!(
                "{} platforms must be sorted and unique",
                inventory.inventory_id
            ));
        }
        let keys = inventory
            .entries
            .iter()
            .map(|entry| (entry.kind, entry.name.as_str()))
            .collect::<Vec<_>>();
        if !strictly_sorted_unique(&keys) {
            errors.push(format!(
                "{} entries must be sorted by kind and name with no duplicates",
                inventory.inventory_id
            ));
        }
    }
}

fn validate_cuda_proc_catalog(catalog: &CudaProcAddressCatalog, errors: &mut Vec<String>) {
    if catalog.schema_version != 1 {
        errors.push(format!(
            "CUDA proc-address catalog uses unsupported schema version {}",
            catalog.schema_version
        ));
    }
    if catalog.spdx_license_identifier != "CC0-1.0" {
        errors.push("CUDA proc-address catalog must declare SPDX license CC0-1.0".to_owned());
    }
    if !catalog.source_version.contains("13030")
        || !catalog.provenance.starts_with("https://docs.nvidia.com/")
    {
        errors.push("CUDA proc-address catalog has invalid baseline provenance".to_owned());
    }
    validate_source_artifact_set(
        "CUDA proc-address catalog",
        &catalog.source_artifacts,
        errors,
    );
    if !catalog
        .source_artifacts
        .iter()
        .any(|artifact| artifact.role == "authoritative-proc-address-typedef-header")
    {
        errors.push(
            "CUDA proc-address catalog lacks the exact typedef-header source hash".to_owned(),
        );
    }
    let keys = catalog
        .typedefs
        .iter()
        .map(|candidate| {
            (
                candidate.symbol.as_str(),
                candidate.api_version,
                candidate.variant,
                candidate.typedef_name.as_str(),
            )
        })
        .collect::<Vec<_>>();
    if !strictly_sorted_unique(&keys) {
        errors.push(
            "CUDA proc-address typedefs must be sorted and unique by symbol/version/variant/name"
                .to_owned(),
        );
    }
    for candidate in &catalog.typedefs {
        validate_cuda_proc_candidate(candidate, errors);
    }
}

fn validate_source_artifact_set(
    label: &str,
    artifacts: &[SourceArtifact],
    errors: &mut Vec<String>,
) {
    let keys = artifacts
        .iter()
        .map(|artifact| (artifact.role.as_str(), artifact.url.as_str()))
        .collect::<Vec<_>>();
    if artifacts.is_empty() || !strictly_sorted_unique(&keys) {
        errors.push(format!(
            "{label} source artifacts must be non-empty, sorted, and unique"
        ));
    }
    for artifact in artifacts {
        let hash = artifact.sha256.strip_prefix("sha256:").unwrap_or("");
        if artifact.role.trim().is_empty()
            || !artifact.url.starts_with("https://")
            || hash.len() != 64
            || !hash.chars().all(|character| character.is_ascii_hexdigit())
            || artifact.revision.trim().is_empty()
            || artifact.path.trim().is_empty()
        {
            errors.push(format!("{label} contains an invalid source artifact"));
        }
    }
}

fn validate_cuda_proc_candidate(
    candidate: &crate::model::CudaProcAddressCandidate,
    errors: &mut Vec<String>,
) {
    let suffix = match candidate.variant {
        CudaProcAddressVariant::Legacy => "",
        CudaProcAddressVariant::Ptds => "_ptds",
        CudaProcAddressVariant::Ptsz => "_ptsz",
    };
    let expected_name = format!(
        "PFN_{}_v{}{suffix}",
        candidate.symbol, candidate.api_version
    );
    let expected_flags = match candidate.variant {
        CudaProcAddressVariant::Legacy => 1,
        CudaProcAddressVariant::Ptds | CudaProcAddressVariant::Ptsz => 2,
    };
    let key = format!("CUDA proc typedef {}", candidate.typedef_name);
    if candidate.typedef_name != expected_name
        || !candidate.symbol.strip_prefix("cu").is_some_and(|suffix| {
            suffix
                .chars()
                .next()
                .is_some_and(|character| character.is_ascii_uppercase())
        })
        || candidate.api_version == 0
        || candidate.api_version > 13_030
        || candidate.proc_address_flags != expected_flags
    {
        errors.push(format!("{key} has inconsistent name/version/flag metadata"));
    }
    if candidate.abi.calling_convention != "system"
        || candidate.abi.return_type.trim().is_empty()
        || candidate
            .abi
            .parameters
            .iter()
            .any(|parameter| parameter.r#type.trim().is_empty())
    {
        errors.push(format!("{key} has an incomplete structured ABI"));
    }
    let expected_hash = sha256(&candidate.normalized_signature);
    if candidate.signature_hash != expected_hash
        || !candidate
            .normalized_signature
            .starts_with("abi[calling_convention=system](")
    {
        errors.push(format!("{key} has an invalid normalized ABI graph/hash"));
    }
    if contains_extractor_path(&candidate.normalized_signature)
        || candidate.provenance.len() < 32
        || !candidate.provenance.contains(&candidate.typedef_name)
    {
        errors.push(format!("{key} has non-portable or incomplete provenance"));
    }
}

fn validate_vendor_union(
    inventories: &[Inventory],
    proc_catalog: &CudaProcAddressCatalog,
    actual: &VendorFunctionUnion,
    errors: &mut Vec<String>,
) {
    if actual.schema_version != 2 || actual.spdx_license_identifier != "CC0-1.0" {
        errors.push("vendor function union must use schema 2 and SPDX CC0-1.0".to_owned());
    }
    let expected = build_vendor_function_union(inventories, proc_catalog);
    match (
        serde_json::to_value(&expected),
        serde_json::to_value(actual),
    ) {
        (Ok(expected), Ok(actual)) if expected != actual => errors.push(
            "vendor function union is stale relative to official snapshots/proc typedefs"
                .to_owned(),
        ),
        (Err(error), _) | (_, Err(error)) => errors.push(format!(
            "vendor function union could not be normalized for freshness validation: {error}"
        )),
        _ => {}
    }
    for function in &actual.functions {
        if function.backend != "cuda" {
            continue;
        }
        for variant in &function.variants {
            if variant.proc_address_candidates.is_empty() {
                errors.push(format!(
                    "CUDA union entry {} has no exact proc-address typedef candidates",
                    function.name
                ));
            }
        }
    }
}

fn contains_extractor_path(value: &str) -> bool {
    value.contains("target/oracle-source")
        || value.contains("target\\oracle-source")
        || value.contains(":\\Users\\")
}

fn entry_index<'a>(
    inventories: &'a [Inventory],
    errors: &mut Vec<String>,
) -> BTreeMap<EntryKey<'a>, &'a Entry> {
    let mut index = BTreeMap::new();
    for inventory in inventories {
        for entry in &inventory.entries {
            let key = (
                inventory.inventory_id.as_str(),
                entry.kind,
                entry.name.as_str(),
            );
            if index.insert(key, entry).is_some() {
                errors.push(format!(
                    "duplicate inventory entry {}::{:?}::{}",
                    key.0, key.1, key.2
                ));
            }
        }
    }
    index
}

fn validate_entries<'a>(
    inventories: &'a [Inventory],
    entries: &BTreeMap<EntryKey<'a>, &'a Entry>,
    semantics: &BTreeMap<SemanticKey<'_>, &SemanticOverride>,
    errors: &mut Vec<String>,
) {
    for inventory in inventories {
        let platform_set = inventory.platforms.iter().collect::<BTreeSet<_>>();
        for entry in &inventory.entries {
            let key = format!("{}::{}", inventory.inventory_id, entry.name);
            if entry.name.trim().is_empty() || entry.normalized_signature.trim().is_empty() {
                errors.push(format!("{key} has an empty name or normalized signature"));
            }
            if contains_extractor_path(&entry.normalized_signature) {
                errors.push(format!(
                    "{key} normalized signature embeds a maintainer checkout path"
                ));
            }
            let expected_hash = sha256(&entry.normalized_signature);
            if entry.signature_hash != expected_hash {
                errors.push(format!(
                    "{key} signature hash mismatch: expected {expected_hash}, found {}",
                    entry.signature_hash
                ));
            }
            if entry.platforms.is_empty() || !strictly_sorted_unique(&entry.platforms) {
                errors.push(format!(
                    "{key} platforms must be non-empty, sorted, and unique"
                ));
            }
            for platform in &entry.platforms {
                if !platform_set.contains(platform) {
                    errors.push(format!(
                        "{key} declares platform {platform} outside its inventory mask"
                    ));
                }
            }
            let requires_abi = matches!(entry.kind, ItemKind::Function | ItemKind::Callback);
            if requires_abi != entry.abi.is_some() {
                errors.push(format!(
                    "{key} must have structured ABI exactly when it is a function or callback"
                ));
            }
            if let Some(abi) = &entry.abi {
                validate_abi(
                    &key,
                    &inventory.inventory_id,
                    &entry.name,
                    abi,
                    semantics,
                    errors,
                );
            }
            if let Some(target) = &entry.alias_of {
                let target_name = target.rsplit("::").next().unwrap_or(target);
                if !entries.keys().any(|(inventory_id, _, name)| {
                    *inventory_id == inventory.inventory_id && *name == target_name
                }) {
                    errors.push(format!("{key} alias target {target} is absent"));
                }
            }
            for alias in &entry.aliases {
                let Some(alias_entry) = entries
                    .iter()
                    .find(|((inventory_id, kind, name), _)| {
                        *inventory_id == inventory.inventory_id
                            && *kind == ItemKind::Alias
                            && *name == alias
                    })
                    .map(|(_, entry)| *entry)
                else {
                    errors.push(format!("{key} names absent alias {alias}"));
                    continue;
                };
                let alias_target = alias_entry
                    .alias_of
                    .as_deref()
                    .and_then(|value| value.rsplit("::").next());
                if alias_target != Some(entry.name.as_str()) {
                    errors.push(format!("{key} alias {alias} does not point back to it"));
                }
            }
            validate_layouts(
                &key,
                entry,
                inventory.inventory_id.starts_with("cuda-vendor-")
                    || inventory.inventory_id.starts_with("hip-"),
                errors,
            );
            if entry.provenance.trim().len() < 12 {
                errors.push(format!("{key} has insufficient entry provenance"));
            }
        }
    }
}

fn validate_abi(
    key: &str,
    inventory_id: &str,
    function: &str,
    abi: &crate::model::Abi,
    semantics: &BTreeMap<SemanticKey<'_>, &SemanticOverride>,
    errors: &mut Vec<String>,
) {
    if !matches!(abi.calling_convention.as_str(), "C" | "system") {
        errors.push(format!(
            "{key} has non-C calling convention {}",
            abi.calling_convention
        ));
    }
    if abi.return_type.trim().is_empty() {
        errors.push(format!("{key} has an empty return type"));
    }
    let mut parameter_names = BTreeSet::new();
    for parameter in &abi.parameters {
        if !parameter_names.insert(parameter.name.as_str()) {
            errors.push(format!("{key} repeats parameter {}", parameter.name));
        }
        if parameter.r#type.trim().is_empty() {
            errors.push(format!(
                "{key} parameter {} has an empty type",
                parameter.name
            ));
        }
        let appears_pointer = parameter.r#type.contains('*')
            || parameter.r#type.contains("Option<")
            || parameter.r#type.contains("Option <");
        if appears_pointer && parameter.pointer == PointerKind::Value {
            errors.push(format!(
                "{key} parameter {} pointer classification disagrees with its type graph",
                parameter.name
            ));
        }
        if parameter.direction != Direction::In && parameter.pointer == PointerKind::Value {
            errors.push(format!(
                "{key} parameter {} is output-classified but is not a pointer",
                parameter.name
            ));
        }
        if parameter.nullable == Some(true) && parameter.pointer == PointerKind::Value {
            errors.push(format!(
                "{key} parameter {} is nullable but is not a pointer or callback",
                parameter.name
            ));
        }
        let reviewed = semantics.get(&(inventory_id, function, parameter.name.as_str()));
        if let Some(reviewed) = reviewed {
            if parameter.direction != Direction::Unknown
                && parameter.direction != reviewed.direction
            {
                errors.push(format!(
                    "{key} parameter {} reviewed direction disagrees with the declaration fact",
                    parameter.name
                ));
            }
            if parameter.nullable.is_some_and(|value| {
                reviewed
                    .nullability
                    .as_bool()
                    .is_some_and(|reviewed| value != reviewed)
            }) {
                errors.push(format!(
                    "{key} parameter {} reviewed nullability disagrees with the declaration fact",
                    parameter.name
                ));
            }
        }
        if parameter.direction == Direction::Unknown && reviewed.is_none() {
            errors.push(format!(
                "{key} parameter {} direction is unresolved; add a reviewed semantic override",
                parameter.name
            ));
        }
        if parameter.pointer != PointerKind::Value
            && parameter.nullable.is_none()
            && reviewed.is_none()
        {
            errors.push(format!(
                "{key} parameter {} nullability is unresolved; add a reviewed semantic override",
                parameter.name
            ));
        }
    }
}

#[allow(clippy::too_many_lines)]
fn semantic_index<'a>(
    catalog: &'a SemanticCatalog,
    entries: &BTreeMap<EntryKey<'_>, &Entry>,
    errors: &mut Vec<String>,
) -> BTreeMap<SemanticKey<'a>, &'a SemanticOverride> {
    if catalog.schema_version != 1 {
        errors.push(format!(
            "semantic catalog uses unsupported schema version {}",
            catalog.schema_version
        ));
    }
    if catalog.spdx_license_identifier != "CC0-1.0" {
        errors.push("coverage catalog must declare SPDX license CC0-1.0".to_owned());
    }
    let ordered = catalog
        .parameters
        .iter()
        .map(|fact| {
            (
                fact.inventory_id.as_str(),
                fact.function.as_str(),
                fact.parameter.as_str(),
            )
        })
        .collect::<Vec<_>>();
    if !strictly_sorted_unique(&ordered) {
        errors.push(
            "semantic overrides must be sorted by inventory, function, and parameter with no duplicates"
                .to_owned(),
        );
    }
    let mut output = BTreeMap::new();
    for fact in &catalog.parameters {
        let key = (
            fact.inventory_id.as_str(),
            fact.function.as_str(),
            fact.parameter.as_str(),
        );
        let entry = entries
            .get(&(
                fact.inventory_id.as_str(),
                ItemKind::Function,
                fact.function.as_str(),
            ))
            .or_else(|| {
                entries.get(&(
                    fact.inventory_id.as_str(),
                    ItemKind::Callback,
                    fact.function.as_str(),
                ))
            });
        let Some(entry) = entry else {
            errors.push(format!(
                "semantic override {}::{}::{} has no callable inventory entry",
                key.0, key.1, key.2
            ));
            continue;
        };
        let parameter = entry.abi.as_ref().and_then(|abi| {
            abi.parameters
                .iter()
                .find(|parameter| parameter.name == fact.parameter)
        });
        match parameter {
            None => errors.push(format!(
                "semantic override {}::{}::{} has no parameter declaration",
                key.0, key.1, key.2
            )),
            Some(parameter) if parameter.pointer == PointerKind::Value => errors.push(format!(
                "semantic override {}::{}::{} targets a value parameter",
                key.0, key.1, key.2
            )),
            Some(parameter)
                if parameter.pointer == PointerKind::Callback
                    && fact.nullability == ReviewedNullability::UnspecifiedBySource =>
            {
                errors.push(format!(
                    "semantic override {}::{}::{} leaves callback nullability unspecified",
                    key.0, key.1, key.2
                ));
            }
            Some(parameter)
                if parameter.nullable.is_some()
                    && fact.nullability == ReviewedNullability::UnspecifiedBySource =>
            {
                errors.push(format!(
                    "semantic override {}::{}::{} discards explicit declaration nullability",
                    key.0, key.1, key.2
                ));
            }
            Some(_) => {}
        }
        if fact.direction == Direction::Unknown {
            errors.push(format!(
                "semantic override {}::{}::{} remains unresolved",
                key.0, key.1, key.2
            ));
        }
        if parameter.is_some_and(|parameter| {
            parameter.pointer == PointerKind::Callback
                && fact.direction == Direction::UnspecifiedBySource
        }) {
            errors.push(format!(
                "semantic override {}::{}::{} leaves callback direction unspecified",
                key.0, key.1, key.2
            ));
        }
        if fact.reason.trim().len() < 32 || fact.provenance.trim().len() < 20 {
            errors.push(format!(
                "semantic override {}::{}::{} lacks a specific reason or provenance",
                key.0, key.1, key.2
            ));
        }
        output.insert(key, fact);
    }
    output
}

fn validate_layouts(
    key: &str,
    entry: &Entry,
    require_vendor_layout: bool,
    errors: &mut Vec<String>,
) {
    if !entry.layouts.is_empty() && !matches!(entry.kind, ItemKind::Struct | ItemKind::Union) {
        errors.push(format!(
            "{key} carries layout facts but is not a record or union"
        ));
    }
    let mut targets = BTreeSet::new();
    for layout in &entry.layouts {
        if !targets.insert(layout.target.as_str()) {
            errors.push(format!("{key} repeats layout target {}", layout.target));
        }
        if !entry.platforms.contains(&layout.target) {
            errors.push(format!(
                "{key} has layout for non-applicable target {}",
                layout.target
            ));
        }
        if layout.size == 0
            || layout.alignment == 0
            || !layout.alignment.is_power_of_two()
            || layout.size % layout.alignment != 0
        {
            errors.push(format!(
                "{key} has invalid size/alignment for {}",
                layout.target
            ));
        }
        for (field, offset) in &layout.field_offsets {
            if *offset >= layout.size {
                errors.push(format!(
                    "{key} field {field} offset {offset} is outside size {} on {}",
                    layout.size, layout.target
                ));
            }
        }
        if layout.provenance.trim().len() < 12 {
            errors.push(format!("{key} layout {} lacks provenance", layout.target));
        }
        let expected_fields = normalized_record_fields(&entry.normalized_signature);
        let actual_fields = layout
            .field_offsets
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>();
        if expected_fields != actual_fields {
            errors.push(format!(
                "{key} layout {} fields differ: declaration-only={:?}, layout-only={:?}",
                layout.target,
                expected_fields
                    .difference(&actual_fields)
                    .collect::<Vec<_>>(),
                actual_fields
                    .difference(&expected_fields)
                    .collect::<Vec<_>>()
            ));
        }
    }
    if require_vendor_layout && matches!(entry.kind, ItemKind::Struct | ItemKind::Union) {
        let expected = entry
            .platforms
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        if targets != expected {
            errors.push(format!(
                "{key} layout targets differ: expected {expected:?}, found {targets:?}"
            ));
        }
    }
}

fn normalized_record_fields(signature: &str) -> BTreeSet<String> {
    let Some((_, body)) = signature.split_once(":{") else {
        return BTreeSet::new();
    };
    let body = body.strip_suffix('}').unwrap_or(body);
    let mut depth = 0_i32;
    let mut start = 0_usize;
    let mut fields = Vec::new();
    for (index, character) in body.char_indices() {
        match character {
            '(' | '[' | '<' => depth += 1,
            ')' | ']' | '>' => depth -= 1,
            ',' if depth == 0 => {
                fields.push(&body[start..index]);
                start = index + 1;
            }
            _ => {}
        }
    }
    if start < body.len() {
        fields.push(&body[start..]);
    }
    fields
        .into_iter()
        .filter_map(|field| field.split_once(':').map(|(name, _)| name.to_owned()))
        .collect()
}

fn decision_index<'a>(
    catalog: &'a CoverageCatalog,
    errors: &mut Vec<String>,
) -> BTreeMap<EntryKey<'a>, &'a CoverageDecision> {
    if catalog.schema_version != 1 {
        errors.push(format!(
            "coverage catalog uses unsupported schema version {}",
            catalog.schema_version
        ));
    }
    if catalog.spdx_license_identifier != "CC0-1.0" {
        errors.push("semantic catalog must declare SPDX license CC0-1.0".to_owned());
    }
    let ordered = catalog
        .decisions
        .iter()
        .map(|decision| {
            (
                decision.inventory_id.as_str(),
                decision.item_kind,
                decision.item_name.as_str(),
            )
        })
        .collect::<Vec<_>>();
    if !strictly_sorted_unique(&ordered) {
        errors.push(
            "coverage decisions must be sorted by inventory and item with no duplicates".to_owned(),
        );
    }
    catalog
        .decisions
        .iter()
        .map(|decision| {
            (
                (
                    decision.inventory_id.as_str(),
                    decision.item_kind,
                    decision.item_name.as_str(),
                ),
                decision,
            )
        })
        .collect()
}

fn validate_classifications<'a>(
    entries: &BTreeMap<EntryKey<'a>, &'a Entry>,
    decisions: &BTreeMap<EntryKey<'a>, &'a CoverageDecision>,
    errors: &mut Vec<String>,
) {
    for key in entries.keys() {
        if !decisions.contains_key(key) {
            errors.push(format!(
                "{}::{:?}::{} has no classification",
                key.0, key.1, key.2
            ));
        }
    }
    for (key, decision) in decisions {
        let Some(entry) = entries.get(key) else {
            errors.push(format!(
                "{}::{:?}::{} classification has no inventory item",
                key.0, key.1, key.2
            ));
            continue;
        };
        if decision.reason.trim().len() < 24 {
            errors.push(format!(
                "{}::{:?}::{} needs a specific human-readable reason",
                key.0, key.1, key.2
            ));
        }
        if !strictly_sorted_unique(&decision.manifest_ids)
            || !strictly_sorted_unique(&decision.implementation_symbols)
            || !strictly_sorted_unique(&decision.export_symbols)
        {
            errors.push(format!(
                "{}::{:?}::{} manifest, implementation, and export lists must be sorted and unique",
                key.0, key.1, key.2
            ));
        }
        let represented = matches!(
            decision.classification,
            Classification::CoveredExact
                | Classification::CoveredAdapter
                | Classification::CoveredRawOnly
                | Classification::DeprecatedCovered
        );
        if represented
            && (decision.manifest_ids.is_empty() || decision.implementation_symbols.is_empty())
        {
            errors.push(format!(
                "{}::{:?}::{} is covered but lacks a manifest ID or implementation symbol",
                key.0, key.1, key.2
            ));
        }
        if decision.classification == Classification::DeprecatedCovered
            && entry.deprecated.is_none()
        {
            errors.push(format!(
                "{}::{:?}::{} is deprecated-covered but the inventory has no deprecation fact",
                key.0, key.1, key.2
            ));
        }
        if decision.classification == Classification::LayoutUnverified
            && !matches!(entry.kind, ItemKind::Struct | ItemKind::Union)
        {
            errors.push(format!(
                "{}::{:?}::{} is layout-unverified but is not a record or union",
                key.0, key.1, key.2
            ));
        }
        if matches!(
            decision.classification,
            Classification::IntentionallyOmitted | Classification::LayoutUnverified
        ) {
            errors.push(format!(
                "{}::{:?}::{} remains in a deferred coverage classification",
                key.0, key.1, key.2
            ));
        }
        let lower_reason = decision.reason.to_ascii_lowercase();
        if ["not yet", "pending automated", "deferred verification"]
            .iter()
            .any(|phrase| lower_reason.contains(phrase))
        {
            errors.push(format!(
                "{}::{:?}::{} uses deferred or placeholder language in its coverage reason",
                key.0, key.1, key.2
            ));
        }
        if decision.runtime_resolvable
            && !matches!(entry.kind, ItemKind::Function | ItemKind::Alias)
        {
            errors.push(format!(
                "{}::{:?}::{} marks a non-callable runtime-resolvable",
                key.0, key.1, key.2
            ));
        }
        if decision.hardware_smoke && !decision.runtime_resolvable {
            errors.push(format!(
                "{}::{:?}::{} has hardware-smoke evidence without runtime-resolution evidence",
                key.0, key.1, key.2
            ));
        }
    }
}

fn validate_cross_inventory_facts(inventories: &[Inventory], errors: &mut Vec<String>) {
    let mut facts: BTreeMap<(&str, ItemKind, &str), (&str, &Entry)> = BTreeMap::new();
    for inventory in inventories {
        let family = if inventory.inventory_id.starts_with("hip-") {
            "hip"
        } else if inventory.inventory_id.starts_with("cuda-") {
            "cuda"
        } else {
            continue;
        };
        for entry in &inventory.entries {
            let key = (family, entry.kind, entry.name.as_str());
            if let Some((prior_inventory, prior)) = facts.get(&key) {
                let overlapping_platform = prior
                    .platforms
                    .iter()
                    .any(|platform| entry.platforms.contains(platform));
                if *prior_inventory == inventory.inventory_id || !overlapping_platform {
                    continue;
                }
                if prior.normalized_signature != entry.normalized_signature {
                    errors.push(format!(
                        "signature drift for {} between {} and {}",
                        entry.name, prior_inventory, inventory.inventory_id
                    ));
                }
                for layout in &entry.layouts {
                    if let Some(prior_layout) = prior
                        .layouts
                        .iter()
                        .find(|value| value.target == layout.target)
                    {
                        if prior_layout.size != layout.size
                            || prior_layout.alignment != layout.alignment
                            || prior_layout.field_offsets != layout.field_offsets
                        {
                            errors.push(format!(
                                "layout drift for {} on {} between {} and {}",
                                entry.name, layout.target, prior_inventory, inventory.inventory_id
                            ));
                        }
                    }
                }
            } else {
                facts.insert(key, (inventory.inventory_id.as_str(), entry));
            }
        }
    }
}

fn validate_manifest_accounting(
    decisions: &BTreeMap<EntryKey<'_>, &CoverageDecision>,
    manifest_ids: &BTreeSet<String>,
    oracle_manifest_ids: &BTreeSet<String>,
    errors: &mut Vec<String>,
) {
    if manifest_ids.is_empty() {
        errors.push("canonical manifest exposes no string IDs".to_owned());
        return;
    }
    let accounted = decisions
        .values()
        .flat_map(|decision| decision.manifest_ids.iter())
        .collect::<BTreeSet<_>>();
    for decision in decisions.values() {
        for id in &decision.manifest_ids {
            if !manifest_ids.contains(id) {
                errors.push(format!(
                    "{}::{} references absent manifest ID {id}",
                    decision.inventory_id, decision.item_name
                ));
            }
        }
    }
    for id in oracle_manifest_ids {
        if !accounted.contains(id) {
            errors.push(format!(
                "manifest entry {id} has no oracle accounting decision"
            ));
        }
    }
}

fn validate_generated_inventory(
    manifest_ids: &BTreeSet<String>,
    generated_ids: &BTreeSet<String>,
    errors: &mut Vec<String>,
) {
    if generated_ids.is_empty() {
        errors.push("generated API inventory exposes no string IDs".to_owned());
        return;
    }
    for id in manifest_ids.difference(generated_ids) {
        errors.push(format!(
            "manifest entry {id} has no generated implementation record"
        ));
    }
    for id in generated_ids.difference(manifest_ids) {
        errors.push(format!(
            "generated implementation record {id} is absent from the manifest"
        ));
    }
}

fn validate_implementation_symbols(
    decisions: &BTreeMap<EntryKey<'_>, &CoverageDecision>,
    implementation_text: &str,
    errors: &mut Vec<String>,
) {
    for decision in decisions.values() {
        for symbol in &decision.implementation_symbols {
            if !contains_identifier(implementation_text, symbol) {
                errors.push(format!(
                    "{}::{} claims missing implementation symbol {symbol}",
                    decision.inventory_id, decision.item_name
                ));
            }
        }
    }
}

fn validate_exports(
    decisions: &BTreeMap<EntryKey<'_>, &CoverageDecision>,
    manifest_symbols: &BTreeSet<String>,
    generated_symbols: &BTreeSet<String>,
    def_exports: &BTreeSet<String>,
    map_exports: &BTreeSet<String>,
    errors: &mut Vec<String>,
) {
    if def_exports != map_exports {
        errors.push(format!(
            "Windows and ELF export controls differ: def-only={:?}, map-only={:?}",
            def_exports.difference(map_exports).collect::<Vec<_>>(),
            map_exports.difference(def_exports).collect::<Vec<_>>()
        ));
    }
    let expected = decisions
        .values()
        .flat_map(|decision| decision.export_symbols.iter().cloned())
        .collect::<BTreeSet<_>>();
    for symbol in &expected {
        if !def_exports.contains(symbol) {
            errors.push(format!(
                "classified public export {symbol} is absent from export controls"
            ));
        }
    }
    let bootstrap = ["ocgpuGetApi", "ocgpuCuGetApi", "ocgpuHipGetApi"]
        .into_iter()
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    for symbol in def_exports {
        if !expected.contains(symbol)
            && !manifest_symbols.contains(symbol)
            && !generated_symbols.contains(symbol)
            && !bootstrap.contains(symbol)
        {
            errors.push(format!(
                "public export {symbol} is absent from the canonical manifest"
            ));
        }
    }
}

fn read_implementation_text(repository_root: &Path, errors: &mut Vec<String>) -> String {
    let mut files = Vec::new();
    collect_files(&repository_root.join("crates"), &mut files, errors);
    files.sort();
    let mut output = String::new();
    for path in files {
        if path.extension().and_then(|value| value.to_str()) != Some("rs")
            || path
                .components()
                .any(|component| component.as_os_str() == "ocgpu-oracle")
        {
            continue;
        }
        match fs::read_to_string(&path) {
            Ok(source) => {
                output.push_str(&source);
                output.push('\n');
            }
            Err(error) => errors.push(format!("cannot read {}: {error}", path.display())),
        }
    }
    output
}

fn collect_files(root: &Path, output: &mut Vec<PathBuf>, errors: &mut Vec<String>) {
    let entries = match fs::read_dir(root) {
        Ok(entries) => entries,
        Err(error) => {
            errors.push(format!("cannot list {}: {error}", root.display()));
            return;
        }
    };
    for entry in entries {
        match entry {
            Ok(entry) => {
                let path = entry.path();
                if path.is_dir() {
                    collect_files(&path, output, errors);
                } else {
                    output.push(path);
                }
            }
            Err(error) => errors.push(format!("cannot enumerate {}: {error}", root.display())),
        }
    }
}

fn contains_identifier(text: &str, symbol: &str) -> bool {
    text.match_indices(symbol).any(|(start, _)| {
        let before = text[..start].chars().next_back();
        let end = start + symbol.len();
        let after = text[end..].chars().next();
        before.is_none_or(|character| !is_identifier_character(character))
            && after.is_none_or(|character| !is_identifier_character(character))
    })
}

fn is_identifier_character(character: char) -> bool {
    character == '_' || character.is_ascii_alphanumeric()
}

fn read_def_exports(path: &Path, errors: &mut Vec<String>) -> BTreeSet<String> {
    let Some(source) = read_text(path, errors) else {
        return BTreeSet::new();
    };
    source
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("ocgpu"))
        .filter_map(|line| line.split_whitespace().next())
        .map(|symbol| symbol.split('=').next().unwrap_or(symbol).to_owned())
        .collect()
}

fn read_map_exports(path: &Path, errors: &mut Vec<String>) -> BTreeSet<String> {
    let Some(source) = read_text(path, errors) else {
        return BTreeSet::new();
    };
    source
        .lines()
        .map(str::trim)
        .filter_map(|line| line.strip_suffix(';'))
        .filter(|line| line.starts_with("ocgpu"))
        .map(str::to_owned)
        .collect()
}

fn read_toml(path: &Path, errors: &mut Vec<String>) -> Option<TomlValue> {
    let source = read_text(path, errors)?;
    match toml::from_str::<TomlValue>(&source) {
        Ok(value) => Some(value),
        Err(error) => {
            errors.push(format!("invalid TOML {}: {error}", path.display()));
            None
        }
    }
}

fn read_json_value(path: &Path, errors: &mut Vec<String>) -> Option<JsonValue> {
    let source = read_text(path, errors)?;
    match serde_json::from_str(&source) {
        Ok(value) => Some(value),
        Err(error) => {
            errors.push(format!("invalid JSON {}: {error}", path.display()));
            None
        }
    }
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, String> {
    let source = fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    serde_json::from_str(&source)
        .map_err(|error| format!("invalid JSON {}: {error}", path.display()))
}

fn read_text(path: &Path, errors: &mut Vec<String>) -> Option<String> {
    match fs::read_to_string(path) {
        Ok(source) => Some(source),
        Err(error) => {
            errors.push(format!("cannot read {}: {error}", path.display()));
            None
        }
    }
}

fn collect_manifest_ids(value: &TomlValue) -> BTreeSet<String> {
    let mut output = collect_toml_strings(value, "id");
    let Some(table) = value.as_table() else {
        return output;
    };
    for (section, prefix) in [("type", "type"), ("constant", "constant")] {
        for item in table
            .get(section)
            .and_then(TomlValue::as_array)
            .into_iter()
            .flatten()
        {
            if let Some(name) = item.get("name").and_then(TomlValue::as_str) {
                output.insert(format!("{prefix}.{name}"));
            }
        }
    }
    for item in table
        .get("raw_inventory")
        .and_then(TomlValue::as_array)
        .into_iter()
        .flatten()
    {
        if let (Some(backend), Some(stable_id)) = (
            item.get("backend").and_then(TomlValue::as_str),
            item.get("stable_id").and_then(TomlValue::as_integer),
        ) {
            output.insert(format!("raw.{backend}.{stable_id:08x}"));
        }
    }
    output
}

fn collect_oracle_accountable_manifest_ids(value: &TomlValue) -> BTreeSet<String> {
    collect_manifest_ids(value)
        .into_iter()
        .filter(|id| {
            id.starts_with("raw.")
                || (!id.starts_with("type.") && !id.starts_with("constant."))
                || id.starts_with("type.ocgpuCU")
                || id.starts_with("type.ocgpuHip")
                || id.starts_with("constant.OCGPU_CUDA_")
                || id.starts_with("constant.OCGPU_HIP_")
        })
        .collect()
}

fn collect_generated_ids(value: &JsonValue) -> BTreeSet<String> {
    let mut output = collect_json_strings(value, "id");
    for (section, prefix) in [("type", "type"), ("constant", "constant")] {
        for item in value
            .get(section)
            .and_then(JsonValue::as_array)
            .into_iter()
            .flatten()
        {
            if let Some(name) = item.get("name").and_then(JsonValue::as_str) {
                output.insert(format!("{prefix}.{name}"));
            }
        }
    }
    for item in value
        .get("raw_inventory")
        .and_then(JsonValue::as_array)
        .into_iter()
        .flatten()
    {
        if let (Some(backend), Some(stable_id)) = (
            item.get("backend").and_then(JsonValue::as_str),
            item.get("stable_id").and_then(JsonValue::as_u64),
        ) {
            output.insert(format!("raw.{backend}.{stable_id:08x}"));
        }
    }
    output
}

fn collect_ocgpu_symbols_toml(value: &TomlValue) -> BTreeSet<String> {
    let mut output = BTreeSet::new();
    walk_toml(value, &mut |candidate| {
        if candidate.starts_with("ocgpu") && candidate.chars().all(is_symbol_character) {
            output.insert(candidate.to_owned());
        }
    });
    output
}

fn collect_ocgpu_symbols_json(value: &JsonValue) -> BTreeSet<String> {
    let mut output = BTreeSet::new();
    walk_json(value, &mut |candidate| {
        if candidate.starts_with("ocgpu") && candidate.chars().all(is_symbol_character) {
            output.insert(candidate.to_owned());
        }
    });
    output
}

fn is_symbol_character(character: char) -> bool {
    character == '_' || character.is_ascii_alphanumeric()
}

fn collect_toml_strings(value: &TomlValue, sought_key: &str) -> BTreeSet<String> {
    let mut output = BTreeSet::new();
    match value {
        TomlValue::Table(table) => {
            for (key, nested) in table {
                if key == sought_key {
                    if let Some(value) = nested.as_str() {
                        output.insert(value.to_owned());
                    }
                }
                output.extend(collect_toml_strings(nested, sought_key));
            }
        }
        TomlValue::Array(values) => {
            for nested in values {
                output.extend(collect_toml_strings(nested, sought_key));
            }
        }
        _ => {}
    }
    output
}

fn collect_json_strings(value: &JsonValue, sought_key: &str) -> BTreeSet<String> {
    let mut output = BTreeSet::new();
    match value {
        JsonValue::Object(object) => {
            for (key, nested) in object {
                if key == sought_key {
                    if let Some(value) = nested.as_str() {
                        output.insert(value.to_owned());
                    }
                }
                output.extend(collect_json_strings(nested, sought_key));
            }
        }
        JsonValue::Array(values) => {
            for nested in values {
                output.extend(collect_json_strings(nested, sought_key));
            }
        }
        _ => {}
    }
    output
}

fn walk_toml(value: &TomlValue, visitor: &mut impl FnMut(&str)) {
    match value {
        TomlValue::String(value) => visitor(value),
        TomlValue::Array(values) => {
            for nested in values {
                walk_toml(nested, visitor);
            }
        }
        TomlValue::Table(table) => {
            for nested in table.values() {
                walk_toml(nested, visitor);
            }
        }
        _ => {}
    }
}

fn walk_json(value: &JsonValue, visitor: &mut impl FnMut(&str)) {
    match value {
        JsonValue::String(value) => visitor(value),
        JsonValue::Array(values) => {
            for nested in values {
                walk_json(nested, visitor);
            }
        }
        JsonValue::Object(object) => {
            for nested in object.values() {
                walk_json(nested, visitor);
            }
        }
        _ => {}
    }
}

fn strictly_sorted_unique<T: Ord>(values: &[T]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

fn validation_error(errors: &[String]) -> ValidationError {
    let mut details = String::new();
    for error in errors {
        let _ = writeln!(&mut details, "- {error}");
    }
    ValidationError {
        count: errors.len(),
        details,
    }
}

#[cfg(test)]
mod tests {
    use super::{contains_identifier, strictly_sorted_unique};

    #[test]
    fn identifier_search_respects_boundaries() {
        assert!(contains_identifier("fn ocgpuInit()", "ocgpuInit"));
        assert!(!contains_identifier("fn xocgpuInit()", "ocgpuInit"));
    }

    #[test]
    fn ordering_rejects_duplicates() {
        assert!(strictly_sorted_unique(&["a", "b"]));
        assert!(!strictly_sorted_unique(&["a", "a"]));
    }
}
