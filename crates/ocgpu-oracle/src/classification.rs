// SPDX-License-Identifier: CC0-1.0

//! Deterministic seed catalog derived from reviewed coverage policy and generated facts.

use crate::model::{Classification, CoverageCatalog, CoverageDecision, Entry, Inventory, ItemKind};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

type OracleKey = (String, ItemKind, String);

/// Builds a complete decision ledger from the committed inventories and code-generator inventory.
///
/// Every covered decision is joined by inventory ID, item kind, and exact normalized-signature
/// hash. Name similarity is never coverage evidence. Missing or ambiguous generated facts remain
/// deliberately red under the strict validator.
#[must_use]
pub fn build_seed_catalog(inventories: &[Inventory], generated: &Value) -> CoverageCatalog {
    let raw = raw_records(generated);
    let raw_spellings = raw_spellings(generated);
    let common = common_classifications(generated);
    let declarations = generated_declarations(generated);
    let mut decisions = Vec::new();
    for inventory in inventories {
        let backend = backend_for_inventory(&inventory.inventory_id);
        for entry in &inventory.entries {
            decisions.push(classify_entry(
                inventory,
                entry,
                backend,
                &raw,
                &raw_spellings,
                &common,
                &declarations,
            ));
        }
    }
    decisions.sort_by(|left, right| {
        (&left.inventory_id, left.item_kind, &left.item_name).cmp(&(
            &right.inventory_id,
            right.item_kind,
            &right.item_name,
        ))
    });
    CoverageCatalog {
        schema_version: 1,
        spdx_license_identifier: "CC0-1.0".to_owned(),
        decisions,
    }
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn classify_entry(
    inventory: &Inventory,
    entry: &Entry,
    backend: &str,
    raw: &BTreeMap<OracleKey, RawRecord>,
    raw_spellings: &BTreeSet<(String, ItemKind, String)>,
    common: &BTreeMap<String, Classification>,
    declarations: &BTreeMap<OracleKey, GeneratedDeclaration>,
) -> CoverageDecision {
    let mut decision = empty_decision(inventory, entry);
    let key = oracle_key(inventory, entry);

    // Rust bindings use `alias` for type aliases as well as callable aliases. Exact generated
    // declaration evidence therefore takes precedence over callable-spelling handling.
    if let Some(generated) = declarations.get(&key) {
        if !generated.emitted {
            decision.classification = Classification::Unrepresentable;
            decision.reason = generated.reason.clone().unwrap_or_else(|| {
                format!(
                    "{} is retained as exact source evidence, but its non-integer declaration cannot be emitted in the backend-prefixed integer-constant namespace.",
                    entry.name
                )
            });
            decision.manifest_ids.push(generated.manifest_id());
            return decision;
        }
        if matches!(entry.kind, ItemKind::Struct | ItemKind::Union)
            && !generated.layout_verified
            && !generated.opaque_record_projection
        {
            decision.classification = Classification::LayoutUnverified;
            decision.reason = format!(
                "{} has an exact generated declaration record, but its generated oracle variant lacks size, alignment, or field-offset evidence for an applicable target.",
                entry.name
            );
            return decision;
        }
        decision.classification = Classification::CoveredRawOnly;
        decision.reason = if generated.opaque_record_projection {
            format!(
                "{} is a zero-sized Rust marker for an incomplete vendor tag used only behind {}; the pointee is never passed by value and the generated opaque-handle pointer representation is enforced by committed Rust and strict-C99 assertions.",
                entry.name, generated.name
            )
        } else if matches!(entry.kind, ItemKind::Struct | ItemKind::Union) {
            format!(
                "{} is emitted as {} from the exact {} normalized-signature hash; every applicable target layout is enforced by committed generated Rust and strict-C99 assertions.",
                entry.name, generated.name, inventory.inventory_id
            )
        } else {
            format!(
                "{} is emitted as {} from the exact {} item-kind and normalized-signature hash, preserving the pinned source fact without name-based inference.",
                entry.name, generated.name, inventory.inventory_id
            )
        };
        decision.manifest_ids.push(generated.manifest_id());
        decision.implementation_symbols.push(generated.name.clone());
        if let Some(common_success) = common_success_constant(backend, &entry.name) {
            decision
                .manifest_ids
                .push(format!("constant.{common_success}"));
            decision
                .implementation_symbols
                .push(common_success.to_owned());
            decision.manifest_ids.sort();
            decision.implementation_symbols.sort();
        }
        return decision;
    }

    if inventory.inventory_id == "rocmrc-0.5.0" && entry.name.starts_with("_bindgen_ty_") {
        decision.classification = Classification::Unrepresentable;
        decision.reason = format!(
            "{} is an anonymous bindgen implementation name rather than a stable vendor C spelling; its named enumeration values are independently retained and emitted as backend-prefixed constants.",
            entry.name
        );
        return decision;
    }
    if inventory.inventory_id == "rocmrc-0.5.0" && entry.name == "__BindgenBitfieldUnit" {
        decision.classification = Classification::Unrepresentable;
        "__BindgenBitfieldUnit is a generic Rust bindgen storage helper rather than a vendor C ABI declaration; each containing vendor record is emitted with its reviewed byte storage and target layout assertions."
            .clone_into(&mut decision.reason);
        return decision;
    }

    if matches!(entry.kind, ItemKind::Function | ItemKind::Alias) {
        if let Some(record) = raw.get(&key) {
            decision.manifest_ids.push(record.manifest_id());
            if record.emitted {
                decision.classification = record
                    .common_id
                    .as_ref()
                    .and_then(|id| common.get(id))
                    .copied()
                    .unwrap_or(record.classification);
                if decision.classification == Classification::DeprecatedCovered
                    && entry.deprecated.is_none()
                {
                    decision.classification = Classification::CoveredRawOnly;
                }
                decision.reason = format!(
                    "{} is emitted in the generated {backend} raw table as {} from the exact {} item-kind and normalized-signature hash; runtime loading resolves this callable slot.",
                    entry.name, record.raw_name, inventory.inventory_id
                );
                if let Some(common_id) = &record.common_id {
                    decision.manifest_ids.push(common_id.clone());
                }
                decision.manifest_ids.sort();
                decision.manifest_ids.dedup();
                decision
                    .implementation_symbols
                    .push(record.raw_name.clone());
                decision.runtime_resolvable = true;
                decision.hardware_smoke = record
                    .common_id
                    .as_deref()
                    .is_some_and(is_hardware_profile_common_id);
                return decision;
            }

            decision.classification = if record.classification
                == Classification::IntentionallyOmitted
                && record.reason.contains("Rust crate loader helper")
            {
                Classification::Unrepresentable
            } else {
                record.classification
            };
            decision.reason.clone_from(&record.reason);
            return decision;
        }

        // NVIDIA exposes 25 macro spellings whose transformed public names collide with real
        // function spellings having different ABI graphs. The real function slot wins; treating
        // the macro as that slot would make a call through the wrong function-pointer type.
        if entry.kind == ItemKind::Alias
            && raw_spellings.contains(&(backend.to_owned(), ItemKind::Function, entry.name.clone()))
        {
            decision.classification = Classification::Unrepresentable;
            decision.reason = format!(
                "{} is an authoritative macro alias to {}, but its deterministic transformed C name collides with a real {backend} function slot having a different exact ABI graph; the real function wins and the alias is not emitted to prevent an unsafe call through the wrong signature.",
                entry.name,
                entry.alias_of.as_deref().unwrap_or("its canonical target")
            );
            return decision;
        }

        decision.reason = format!(
            "{} is an authoritative upstream callable or alias with no exact canonical raw-inventory variant keyed by source, kind, and normalized-signature hash; strict validation rejects this unaccounted surface.",
            entry.name
        );
        return decision;
    }

    decision.reason = format!(
        "{} is retained in the normalized {} inventory, but no exact generated declaration variant keyed by source, kind, and normalized-signature hash accounts for this {} item.",
        entry.name,
        inventory.source_name,
        kind_name(entry.kind)
    );
    decision
}

fn common_classifications(generated: &Value) -> BTreeMap<String, Classification> {
    generated
        .get("function")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|value| {
            let id = value.get("id")?.as_str()?.to_owned();
            let classification = match value.get("classification")?.as_str()? {
                "exact" => Classification::CoveredExact,
                "adapter" => Classification::CoveredAdapter,
                _ => return None,
            };
            Some((id, classification))
        })
        .collect()
}

fn empty_decision(inventory: &Inventory, entry: &Entry) -> CoverageDecision {
    CoverageDecision {
        inventory_id: inventory.inventory_id.clone(),
        item_kind: entry.kind,
        item_name: entry.name.clone(),
        classification: Classification::IntentionallyOmitted,
        reason: String::new(),
        manifest_ids: Vec::new(),
        implementation_symbols: Vec::new(),
        export_symbols: Vec::new(),
        runtime_resolvable: false,
        hardware_smoke: false,
    }
}

#[derive(Clone, Debug)]
struct RawRecord {
    stable_id: u64,
    backend: String,
    raw_name: String,
    classification: Classification,
    reason: String,
    emitted: bool,
    common_id: Option<String>,
}

impl RawRecord {
    fn manifest_id(&self) -> String {
        format!("raw.{}.{:08x}", self.backend, self.stable_id)
    }
}

fn raw_records(generated: &Value) -> BTreeMap<OracleKey, RawRecord> {
    let mut output = BTreeMap::new();
    for value in generated
        .get("raw_inventory")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let Some(record) = raw_record(value) else {
            continue;
        };
        for variant in value
            .get("oracle_variants")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if let Some(key) = variant_key(variant) {
                output.entry(key).or_insert_with(|| record.clone());
            }
        }
    }
    output
}

fn raw_record(value: &Value) -> Option<RawRecord> {
    let backend = value.get("backend")?.as_str()?.to_owned();
    Some(RawRecord {
        stable_id: value.get("stable_id")?.as_u64()?,
        backend,
        raw_name: value
            .get("raw_name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        classification: parse_classification(value.get("classification")?.as_str()?),
        reason: value.get("reason")?.as_str()?.to_owned(),
        emitted: value.get("emitted")?.as_bool()?,
        common_id: value
            .get("common_id")
            .and_then(Value::as_str)
            .map(str::to_owned),
    })
}

fn raw_spellings(generated: &Value) -> BTreeSet<(String, ItemKind, String)> {
    generated
        .get("raw_inventory")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|value| {
            Some((
                value.get("backend")?.as_str()?.to_owned(),
                parse_item_kind(value.get("vendor_kind")?.as_str()?)?,
                value.get("vendor_name")?.as_str()?.to_owned(),
            ))
        })
        .collect()
}

#[derive(Clone, Debug)]
struct GeneratedDeclaration {
    name: String,
    manifest_prefix: &'static str,
    emitted: bool,
    layout_verified: bool,
    opaque_record_projection: bool,
    reason: Option<String>,
}

impl GeneratedDeclaration {
    fn manifest_id(&self) -> String {
        format!("{}.{}", self.manifest_prefix, self.name)
    }
}

fn generated_declarations(generated: &Value) -> BTreeMap<OracleKey, GeneratedDeclaration> {
    let mut output = BTreeMap::new();
    for (section, prefix) in [("type", "type"), ("constant", "constant")] {
        for item in generated
            .get(section)
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let Some(name) = item.get("name").and_then(Value::as_str) else {
                continue;
            };
            let emitted = section != "constant"
                || item
                    .get("emitted")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
            let generated_kind = item.get("kind").and_then(Value::as_str);
            let reason = item
                .get("reason")
                .and_then(Value::as_str)
                .map(str::to_owned);
            for variant in item
                .get("oracle_variants")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let Some(key) = variant_key(variant) else {
                    continue;
                };
                let layout_verified = variant_layout_verified(variant, key.1);
                let opaque_record_projection = key.1 == ItemKind::Struct
                    && generated_kind == Some("opaque_handle")
                    && variant
                        .get("layouts")
                        .and_then(Value::as_array)
                        .is_none_or(Vec::is_empty);
                output.entry(key).or_insert_with(|| GeneratedDeclaration {
                    name: name.to_owned(),
                    manifest_prefix: prefix,
                    emitted,
                    layout_verified,
                    opaque_record_projection,
                    reason: reason.clone(),
                });
            }
        }
    }
    output
}

fn variant_layout_verified(variant: &Value, kind: ItemKind) -> bool {
    if !matches!(kind, ItemKind::Struct | ItemKind::Union) {
        return true;
    }
    let platforms = variant
        .get("platforms")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect::<BTreeSet<_>>();
    let layouts = variant
        .get("layouts")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|layout| layout.get("target").and_then(Value::as_str))
        .collect::<BTreeSet<_>>();
    !platforms.is_empty() && platforms.is_subset(&layouts)
}

fn oracle_key(inventory: &Inventory, entry: &Entry) -> OracleKey {
    (
        inventory.inventory_id.clone(),
        entry.kind,
        entry.signature_hash.clone(),
    )
}

fn variant_key(variant: &Value) -> Option<OracleKey> {
    Some((
        variant.get("oracle_source")?.as_str()?.to_owned(),
        parse_item_kind(variant.get("oracle_kind")?.as_str()?)?,
        variant.get("oracle_signature_hash")?.as_str()?.to_owned(),
    ))
}

fn parse_classification(value: &str) -> Classification {
    match value {
        "covered_exact" => Classification::CoveredExact,
        "covered_adapter" => Classification::CoveredAdapter,
        "covered_raw_only" => Classification::CoveredRawOnly,
        "platform_unavailable" => Classification::PlatformUnavailable,
        "deprecated_covered" => Classification::DeprecatedCovered,
        "unrepresentable" => Classification::Unrepresentable,
        _ => Classification::IntentionallyOmitted,
    }
}

fn parse_item_kind(value: &str) -> Option<ItemKind> {
    match value {
        "function" => Some(ItemKind::Function),
        "alias" => Some(ItemKind::Alias),
        "type" => Some(ItemKind::Type),
        "opaque_handle" => Some(ItemKind::OpaqueHandle),
        "struct" => Some(ItemKind::Struct),
        "union" => Some(ItemKind::Union),
        "callback" => Some(ItemKind::Callback),
        "constant" => Some(ItemKind::Constant),
        "enum_value" => Some(ItemKind::EnumValue),
        "flag" => Some(ItemKind::Flag),
        _ => None,
    }
}

fn backend_for_inventory(inventory_id: &str) -> &str {
    if inventory_id.contains("cuda") || inventory_id.starts_with("cudarc") {
        "cuda"
    } else {
        "hip"
    }
}

fn kind_name(kind: ItemKind) -> &'static str {
    match kind {
        ItemKind::Function => "function",
        ItemKind::Alias => "alias",
        ItemKind::Type => "type",
        ItemKind::OpaqueHandle => "opaque-handle",
        ItemKind::Struct => "struct",
        ItemKind::Union => "union",
        ItemKind::Callback => "callback",
        ItemKind::Constant => "constant",
        ItemKind::EnumValue => "enum-value",
        ItemKind::Flag => "flag",
    }
}

fn is_hardware_profile_common_id(id: &str) -> bool {
    matches!(
        id,
        "init.initialize"
            | "driver.get_version"
            | "device.get_count"
            | "device.get"
            | "device.get_name"
            | "context.create"
            | "context.destroy"
            | "context.synchronize"
            | "memory.allocate"
            | "memory.free"
            | "memory.copy_htod"
            | "memory.copy_dtoh"
            | "stream.create"
            | "stream.destroy"
            | "stream.synchronize"
            | "event.create"
            | "event.destroy"
            | "event.record"
            | "event.synchronize"
            | "module.load_data"
            | "module.unload"
            | "module.get_function"
            | "launch.kernel"
    )
}

fn common_success_constant(backend: &str, source_name: &str) -> Option<&'static str> {
    match (backend, source_name) {
        ("cuda", "CUDA_SUCCESS") => Some("OCGPU_CUDA_SUCCESS"),
        ("hip", "hipSuccess" | "HIP_SUCCESS") => Some("OCGPU_HIP_SUCCESS"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{is_hardware_profile_common_id, parse_item_kind};
    use crate::model::ItemKind;

    #[test]
    fn generated_item_kinds_are_parsed_exactly() {
        assert_eq!(parse_item_kind("function"), Some(ItemKind::Function));
        assert_eq!(parse_item_kind("enum_value"), Some(ItemKind::EnumValue));
        assert_eq!(parse_item_kind("integer"), None);
    }

    #[test]
    fn bounded_hardware_profile_excludes_unexercised_queries() {
        assert!(is_hardware_profile_common_id("launch.kernel"));
        assert!(!is_hardware_profile_common_id("device.get_attribute"));
        assert!(!is_hardware_profile_common_id("context.get_current"));
    }
}
