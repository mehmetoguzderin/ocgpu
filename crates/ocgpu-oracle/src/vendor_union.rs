// SPDX-License-Identifier: CC0-1.0

//! Deterministic union of the authoritative vendor callable snapshots.

use crate::model::{
    CudaProcAddressCandidate, CudaProcAddressCatalog, CudaProcAddressVariant, Inventory, ItemKind,
    VendorFunctionUnion, VendorUnionFunction, VendorUnionSource, VendorUnionVariant,
};
use std::collections::BTreeMap;

/// Builds a vendor-led callable union without flattening legitimate version or platform drift.
#[must_use]
pub fn build_vendor_function_union(
    inventories: &[Inventory],
    cuda_proc_addresses: &CudaProcAddressCatalog,
) -> VendorFunctionUnion {
    let vendor = inventories
        .iter()
        .filter(|inventory| {
            inventory.inventory_id.starts_with("cuda-vendor-")
                || inventory.inventory_id.starts_with("hip-general-")
                || inventory.inventory_id.starts_with("hip-windows-")
        })
        .collect::<Vec<_>>();
    let mut functions = BTreeMap::<(String, ItemKind, String), Vec<VendorUnionVariant>>::new();
    for inventory in &vendor {
        let backend = if inventory.inventory_id.starts_with("cuda-") {
            "cuda"
        } else {
            "hip"
        };
        for entry in &inventory.entries {
            if !matches!(entry.kind, ItemKind::Function | ItemKind::Alias) {
                continue;
            }
            functions
                .entry((backend.to_owned(), entry.kind, entry.name.clone()))
                .or_default()
                .push(VendorUnionVariant {
                    inventory_id: inventory.inventory_id.clone(),
                    source_version: inventory.source_version.clone(),
                    normalized_signature: entry.normalized_signature.clone(),
                    signature_hash: entry.signature_hash.clone(),
                    abi: entry.abi.clone(),
                    alias_of: entry.alias_of.clone(),
                    aliases: entry.aliases.clone(),
                    platforms: entry.platforms.clone(),
                    introduced: entry.introduced.clone(),
                    deprecated: entry.deprecated.clone(),
                    proc_address_candidates: if backend == "cuda" {
                        cuda_candidates_for_entry(
                            &entry.name,
                            entry.alias_of.as_deref(),
                            cuda_proc_addresses,
                        )
                    } else {
                        Vec::new()
                    },
                    provenance: entry.provenance.clone(),
                });
        }
    }
    let functions = functions
        .into_iter()
        .map(|((backend, kind, name), mut variants)| {
            variants.sort_by(|left, right| {
                (
                    &left.inventory_id,
                    &left.source_version,
                    &left.signature_hash,
                    &left.platforms,
                )
                    .cmp(&(
                        &right.inventory_id,
                        &right.source_version,
                        &right.signature_hash,
                        &right.platforms,
                    ))
            });
            VendorUnionFunction {
                backend,
                kind,
                name,
                variants,
            }
        })
        .collect();
    let sources = vendor
        .into_iter()
        .map(|inventory| VendorUnionSource {
            inventory_id: inventory.inventory_id.clone(),
            source_version: inventory.source_version.clone(),
            platforms: inventory.platforms.clone(),
            provenance: inventory.provenance.clone(),
            source_artifacts: inventory.source_artifacts.clone(),
        })
        .collect();
    VendorFunctionUnion {
        schema_version: 2,
        spdx_license_identifier: "CC0-1.0".to_owned(),
        sources,
        functions,
    }
}

fn cuda_candidates_for(
    function_name: &str,
    catalog: &CudaProcAddressCatalog,
) -> Vec<CudaProcAddressCandidate> {
    let (base, variant) = cuda_proc_key(function_name);
    catalog
        .typedefs
        .iter()
        .filter(|candidate| candidate.symbol == base && candidate.variant == variant)
        .cloned()
        .collect()
}

fn cuda_candidates_for_entry(
    name: &str,
    alias_of: Option<&str>,
    catalog: &CudaProcAddressCatalog,
) -> Vec<CudaProcAddressCandidate> {
    let candidates = cuda_candidates_for(name, catalog);
    if !candidates.is_empty() {
        return candidates;
    }
    alias_of.map_or_else(Vec::new, |target| {
        let target = target.rsplit("::").next().unwrap_or(target);
        cuda_candidates_for(target, catalog)
    })
}

fn cuda_proc_key(function_name: &str) -> (&str, CudaProcAddressVariant) {
    let (without_stream_suffix, variant) = if let Some(name) = function_name.strip_suffix("_ptds") {
        (name, CudaProcAddressVariant::Ptds)
    } else if let Some(name) = function_name.strip_suffix("_ptsz") {
        (name, CudaProcAddressVariant::Ptsz)
    } else {
        (function_name, CudaProcAddressVariant::Legacy)
    };
    let base = without_stream_suffix
        .rsplit_once("_v")
        .filter(|(_, suffix)| {
            !suffix.is_empty() && suffix.chars().all(|character| character.is_ascii_digit())
        })
        .map_or(without_stream_suffix, |(base, _)| base);
    (base, variant)
}

#[cfg(test)]
mod tests {
    use super::build_vendor_function_union;
    use crate::model::{CudaProcAddressCatalog, Entry, Inventory, ItemKind, SourceArtifact};

    fn proc_catalog() -> CudaProcAddressCatalog {
        CudaProcAddressCatalog {
            schema_version: 1,
            spdx_license_identifier: "CC0-1.0".to_owned(),
            source_version: "test".to_owned(),
            provenance: "https://example.invalid/proc".to_owned(),
            source_artifacts: Vec::new(),
            typedefs: Vec::new(),
        }
    }

    fn inventory(id: &str, version: &str, signature: &str) -> Inventory {
        Inventory {
            schema_version: 1,
            spdx_license_identifier: "CC0-1.0".to_owned(),
            inventory_id: id.to_owned(),
            source_name: id.to_owned(),
            source_version: version.to_owned(),
            provenance: format!("https://example.invalid/{id}"),
            source_artifacts: vec![SourceArtifact {
                role: "authoritative-header".to_owned(),
                url: format!("https://example.invalid/{id}.h"),
                sha256: format!("sha256:{}", "0".repeat(64)),
                revision: version.to_owned(),
                path: "include/vendor.h".to_owned(),
            }],
            platforms: vec!["x86_64-unknown-linux-gnu".to_owned()],
            entries: vec![Entry {
                kind: ItemKind::Function,
                name: "hipExample".to_owned(),
                normalized_signature: signature.to_owned(),
                signature_hash: format!("sha256:{}", "1".repeat(64)),
                numeric_value: None,
                abi: None,
                aliases: Vec::new(),
                alias_of: None,
                platforms: vec!["x86_64-unknown-linux-gnu".to_owned()],
                introduced: None,
                deprecated: None,
                layouts: Vec::new(),
                provenance: format!("https://example.invalid/{id}#hipExample"),
            }],
        }
    }

    #[test]
    fn keeps_version_variants_under_one_vendor_spelling() {
        let union = build_vendor_function_union(
            &[
                inventory("hip-general-one", "one", "fn hipExample()->int"),
                inventory("hip-windows-two", "two", "fn hipExample()->long"),
            ],
            &proc_catalog(),
        );
        assert_eq!(union.functions.len(), 1);
        assert_eq!(union.functions[0].variants.len(), 2);
    }
}
