// SPDX-License-Identifier: CC0-1.0

use crate::model::{
    Abi, Direction, Entry, Inventory, ItemKind, PointerKind, ReviewedNullability, SemanticCatalog,
    SemanticOverride,
};
use std::collections::{BTreeMap, BTreeSet};
use toml::Value;

/// Builds the reviewed semantic sidecar from the canonical common/raw mappings.
#[allow(clippy::too_many_lines)] // The manifest-to-five-inventory join is clearer as one audited pass.
pub fn build_semantic_catalog(
    inventories: &[Inventory],
    manifest: &Value,
) -> Result<SemanticCatalog, String> {
    let functions = manifest
        .get("function")
        .and_then(Value::as_array)
        .ok_or_else(|| "canonical manifest has no function array".to_owned())?;
    let inventory_map = inventories
        .iter()
        .map(|inventory| (inventory.inventory_id.as_str(), inventory))
        .collect::<BTreeMap<_, _>>();
    let mut parameters = BTreeMap::new();

    for function in functions {
        let table = function
            .as_table()
            .ok_or_else(|| "canonical function entry is not a table".to_owned())?;
        let manifest_id = table
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| "canonical function lacks an id".to_owned())?;
        for (backend, inventory_ids) in [
            (
                "cuda",
                ["cuda-vendor-13.3-13030", "cudarc-0.19.9"].as_slice(),
            ),
            (
                "hip",
                [
                    "hip-general-7.14.60850",
                    "hip-windows-7.2.0",
                    "rocmrc-0.5.0",
                ]
                .as_slice(),
            ),
        ] {
            let backend_table = table
                .get(backend)
                .and_then(Value::as_table)
                .ok_or_else(|| format!("function {manifest_id} lacks {backend} mapping"))?;
            let vendor_symbol = backend_table
                .get("vendor_symbol")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("function {manifest_id} lacks {backend}.vendor_symbol"))?;
            let reviewed = backend_table
                .get("params")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("function {manifest_id} lacks {backend}.params"))?;

            for inventory_id in inventory_ids {
                let inventory = inventory_map
                    .get(inventory_id)
                    .ok_or_else(|| format!("required inventory {inventory_id} is absent"))?;
                let Some(entry) = inventory
                    .entries
                    .iter()
                    .find(|entry| entry.kind == ItemKind::Function && entry.name == vendor_symbol)
                else {
                    continue;
                };
                let abi = entry
                    .abi
                    .as_ref()
                    .ok_or_else(|| format!("{inventory_id}::{vendor_symbol} lacks ABI facts"))?;
                if abi.parameters.len() != reviewed.len() {
                    return Err(format!(
                        "{inventory_id}::{vendor_symbol} has {} parameters but manifest {manifest_id}.{backend} has {}",
                        abi.parameters.len(),
                        reviewed.len()
                    ));
                }
                if !manifest_params_match_abi(reviewed, abi) {
                    return Err(format!(
                        "{inventory_id}::{vendor_symbol} normalized parameter type graph differs from manifest {manifest_id}.{backend}; semantic facts cannot be propagated by position"
                    ));
                }
                for (index, (oracle, review)) in
                    abi.parameters.iter().zip(reviewed.iter()).enumerate()
                {
                    if oracle.pointer == PointerKind::Value {
                        continue;
                    }
                    let review = review.as_table().ok_or_else(|| {
                        format!("{manifest_id}.{backend}.params[{index}] is not a table")
                    })?;
                    let direction = match review.get("direction").and_then(Value::as_str) {
                        Some("in") => Direction::In,
                        Some("out") => Direction::Out,
                        Some("inout") => Direction::InOut,
                        value => {
                            return Err(format!(
                                "{manifest_id}.{backend}.params[{index}] has invalid direction {value:?}"
                            ));
                        }
                    };
                    let nullable = review.get("nullable").and_then(Value::as_bool).ok_or_else(
                        || {
                            format!(
                                "{manifest_id}.{backend}.params[{index}] lacks boolean nullability"
                            )
                        },
                    )?;
                    let declaration_direction_drift =
                        oracle.direction != Direction::Unknown && oracle.direction != direction;
                    let declaration_nullability_drift =
                        oracle.nullable.is_some_and(|source| source != nullable);
                    let reviewed_direction = if declaration_direction_drift {
                        oracle.direction
                    } else {
                        direction
                    };
                    let reviewed_nullability = if declaration_nullability_drift {
                        ReviewedNullability::from_bool(oracle.nullable.unwrap_or(nullable))
                    } else if declaration_direction_drift
                        && oracle.pointer != PointerKind::Callback
                        && oracle.nullable.is_none()
                    {
                        ReviewedNullability::UnspecifiedBySource
                    } else {
                        ReviewedNullability::from_bool(nullable)
                    };
                    let (reason, provenance) = if declaration_direction_drift
                        || declaration_nullability_drift
                    {
                        (
                            format!(
                                "The source-specific {inventory_id} declaration fact differs from the canonical {manifest_id} mapping; the exact source fact is retained as reviewed version/platform drift."
                            ),
                            format!(
                                "{}; api/ocgpu-api.toml#function.{manifest_id}.{backend}.params[{index}]",
                                entry.provenance
                            ),
                        )
                    } else {
                        (
                            format!(
                                "Reviewed backend mapping for canonical manifest function {manifest_id}; declaration position and semantic contract were checked together."
                            ),
                            format!(
                                "api/ocgpu-api.toml#function.{manifest_id}.{backend}.params[{index}]"
                            ),
                        )
                    };
                    insert_override(
                        &mut parameters,
                        SemanticOverride {
                            inventory_id: (*inventory_id).to_owned(),
                            function: vendor_symbol.to_owned(),
                            parameter: oracle.name.clone(),
                            direction: reviewed_direction,
                            nullability: reviewed_nullability,
                            reason,
                            provenance,
                        },
                    )?;
                }
            }
        }
    }
    propagate_identical_abi_facts(inventories, &mut parameters)?;
    apply_reviewed_callback_contracts(inventories, &mut parameters)?;
    record_source_unspecified_nullability(inventories, &mut parameters)?;
    Ok(SemanticCatalog {
        schema_version: 1,
        spdx_license_identifier: "CC0-1.0".to_owned(),
        parameters: parameters.into_values().collect(),
    })
}

fn manifest_params_match_abi(reviewed: &[Value], abi: &Abi) -> bool {
    abi.parameters.iter().zip(reviewed).all(|(oracle, review)| {
        let Some(review_type) = review
            .as_table()
            .and_then(|table| table.get("type"))
            .and_then(Value::as_str)
        else {
            return false;
        };
        canonical_type(&oracle.r#type) == canonical_type(review_type)
    })
}

fn insert_override(
    output: &mut BTreeMap<(String, String, String), SemanticOverride>,
    fact: SemanticOverride,
) -> Result<(), String> {
    let key = (
        fact.inventory_id.clone(),
        fact.function.clone(),
        fact.parameter.clone(),
    );
    if let Some(prior) = output.get(&key) {
        if prior.direction != fact.direction || prior.nullability != fact.nullability {
            return Err(format!(
                "conflicting reviewed semantics for {}::{}::{}",
                key.0, key.1, key.2
            ));
        }
        return Ok(());
    }
    output.insert(key, fact);
    Ok(())
}

fn propagate_identical_abi_facts(
    inventories: &[Inventory],
    output: &mut BTreeMap<(String, String, String), SemanticOverride>,
) -> Result<(), String> {
    let mut groups = BTreeMap::<(String, String, String), Vec<(&Inventory, &Entry)>>::new();
    for inventory in inventories {
        let backend = if inventory.inventory_id.contains("cuda")
            || inventory.inventory_id.starts_with("cudarc")
        {
            "cuda"
        } else {
            "hip"
        };
        for entry in &inventory.entries {
            if !matches!(entry.kind, ItemKind::Function | ItemKind::Callback) {
                continue;
            }
            let Some(abi) = &entry.abi else {
                continue;
            };
            groups
                .entry((backend.to_owned(), entry.name.clone(), abi_shape(abi)))
                .or_default()
                .push((inventory, entry));
        }
    }
    for ((_backend, function, shape), variants) in groups {
        if variants.len() < 2 {
            continue;
        }
        let parameter_count = group_parameter_count(&variants);
        for index in 0..parameter_count {
            for (target_inventory, target_entry) in &variants {
                let facts = collect_group_facts(
                    &function,
                    &shape,
                    &target_inventory.inventory_id,
                    &variants,
                    index,
                    output,
                )?;
                propagate_variant_facts(
                    &function,
                    target_inventory,
                    target_entry,
                    index,
                    &facts,
                    output,
                )?;
            }
        }
    }
    Ok(())
}

struct PropagatedFacts {
    direction: Option<Direction>,
    nullable: Option<bool>,
    evidence: Vec<String>,
}

fn group_parameter_count(variants: &[(&Inventory, &Entry)]) -> usize {
    variants
        .first()
        .and_then(|(_, entry)| entry.abi.as_ref())
        .map_or(0, |abi| abi.parameters.len())
}

fn collect_group_facts(
    function: &str,
    shape: &str,
    target_inventory_id: &str,
    variants: &[(&Inventory, &Entry)],
    index: usize,
    output: &BTreeMap<(String, String, String), SemanticOverride>,
) -> Result<PropagatedFacts, String> {
    let mut directions = BTreeMap::<u8, BTreeSet<Direction>>::new();
    let mut nullability = BTreeMap::<u8, BTreeSet<bool>>::new();
    let mut evidence = Vec::new();
    for (inventory, entry) in variants {
        let parameter = group_parameter(inventory, entry, function, index)?;
        let existing = output.get(&(
            inventory.inventory_id.clone(),
            function.to_owned(),
            parameter.name.clone(),
        ));
        let declaration_rank = declaration_authority(&inventory.inventory_id, target_inventory_id);
        if parameter.direction != Direction::Unknown {
            directions
                .entry(declaration_rank)
                .or_default()
                .insert(parameter.direction);
        }
        if let Some(nullable) = parameter.nullable {
            nullability
                .entry(declaration_rank)
                .or_default()
                .insert(nullable);
        }
        if let Some(existing) = existing {
            let authority = override_authority(&inventory.inventory_id, target_inventory_id);
            directions
                .entry(authority)
                .or_default()
                .insert(existing.direction);
            if let Some(nullable) = existing.nullability.as_bool() {
                nullability.entry(authority).or_default().insert(nullable);
            }
        }
        if parameter.direction != Direction::Unknown
            || parameter.nullable.is_some()
            || existing.is_some()
        {
            evidence.push(format!(
                "{}:{}",
                inventory.inventory_id, entry.signature_hash
            ));
        }
    }
    let direction = unique_highest_fact(&directions, function, shape, index, "direction")?;
    let nullable = unique_highest_fact(&nullability, function, shape, index, "nullability")?;
    evidence.sort();
    evidence.dedup();
    Ok(PropagatedFacts {
        direction,
        nullable,
        evidence,
    })
}

fn declaration_authority(source: &str, target: &str) -> u8 {
    if is_vendor_inventory(source) {
        if source == target {
            6
        } else if vendor_semantic_baseline_matches(source, target) {
            5
        } else {
            4
        }
    } else if source == target {
        3
    } else {
        1
    }
}

fn override_authority(source: &str, target: &str) -> u8 {
    if source == target {
        4
    } else if vendor_semantic_baseline_matches(source, target) {
        3
    } else {
        2
    }
}

fn is_vendor_inventory(inventory_id: &str) -> bool {
    inventory_id.starts_with("cuda-vendor-") || inventory_id.starts_with("hip-")
}

fn vendor_semantic_baseline_matches(source: &str, target: &str) -> bool {
    (source.starts_with("cuda-vendor-") && target == "cudarc-0.19.9")
        || (source == "hip-windows-7.2.0" && target == "rocmrc-0.5.0")
}

fn unique_highest_fact<T: Copy + Ord>(
    candidates: &BTreeMap<u8, BTreeSet<T>>,
    function: &str,
    shape: &str,
    index: usize,
    fact_name: &str,
) -> Result<Option<T>, String> {
    let Some((_, highest)) = candidates.last_key_value() else {
        return Ok(None);
    };
    if highest.len() != 1 {
        return Err(format!(
            "identical ABI shape {shape} for {function} has conflicting {fact_name} facts at parameter index {index}"
        ));
    }
    Ok(highest.iter().next().copied())
}

fn propagate_variant_facts(
    function: &str,
    inventory: &Inventory,
    entry: &Entry,
    index: usize,
    facts: &PropagatedFacts,
    output: &mut BTreeMap<(String, String, String), SemanticOverride>,
) -> Result<(), String> {
    if facts.direction.is_none() && facts.nullable.is_none() {
        return Ok(());
    }
    let parameter = group_parameter(inventory, entry, function, index)?;
    if parameter.pointer == PointerKind::Value {
        return Ok(());
    }
    let key = (
        inventory.inventory_id.clone(),
        function.to_owned(),
        parameter.name.clone(),
    );
    if let Some(existing) = output.get(&key) {
        match validate_propagated_conflict(inventory, function, parameter, facts, existing) {
            Ok(()) => return Ok(()),
            Err(error) if is_vendor_inventory(&inventory.inventory_id) => return Err(error),
            Err(_) => {
                output.remove(&key);
            }
        }
    }
    let direction = if parameter.direction == Direction::Unknown {
        facts.direction.unwrap_or_else(|| {
            if parameter.pointer == PointerKind::Callback {
                Direction::In
            } else {
                Direction::UnspecifiedBySource
            }
        })
    } else {
        parameter.direction
    };
    let nullability = match parameter.nullable.or(facts.nullable) {
        Some(nullable) => ReviewedNullability::from_bool(nullable),
        None if parameter.pointer == PointerKind::Callback => return Ok(()),
        None => ReviewedNullability::UnspecifiedBySource,
    };
    insert_override(
        output,
        SemanticOverride {
            inventory_id: inventory.inventory_id.clone(),
            function: function.to_owned(),
            parameter: parameter.name.clone(),
            direction,
            nullability,
            reason: "Propagated only across the same vendor spelling and an identical normalized ABI type graph; target-version authority is explicit and conflicting source facts are rejected.".to_owned(),
            provenance: format!(
                "oracle identical-ABI evidence [{}]",
                facts.evidence.join(", ")
            ),
        },
    )
}

fn apply_reviewed_callback_contracts(
    inventories: &[Inventory],
    output: &mut BTreeMap<(String, String, String), SemanticOverride>,
) -> Result<(), String> {
    for inventory in inventories {
        for entry in &inventory.entries {
            if !matches!(entry.kind, ItemKind::Function | ItemKind::Callback) {
                continue;
            }
            let Some(abi) = &entry.abi else {
                continue;
            };
            for parameter in &abi.parameters {
                if parameter.pointer != PointerKind::Callback || parameter.nullable.is_some() {
                    continue;
                }
                let nullable = reviewed_callback_nullability(&entry.name, &parameter.name)
                    .ok_or_else(|| {
                        format!(
                            "{}::{}::{} callback nullability lacks an exact reviewed contract",
                            inventory.inventory_id, entry.name, parameter.name
                        )
                    })?;
                insert_override(
                    output,
                    SemanticOverride {
                        inventory_id: inventory.inventory_id.clone(),
                        function: entry.name.clone(),
                        parameter: parameter.name.clone(),
                        direction: Direction::In,
                        nullability: ReviewedNullability::from_bool(nullable),
                        reason: callback_contract_reason(&entry.name, nullable).to_owned(),
                        provenance: format!(
                            "{}; exact callback contract review at #{}",
                            entry.provenance, entry.name
                        ),
                    },
                )?;
            }
        }
    }
    Ok(())
}

fn reviewed_callback_nullability(function: &str, parameter: &str) -> Option<bool> {
    match (function, parameter) {
        (
            "cuOccupancyMaxPotentialBlockSize" | "cuOccupancyMaxPotentialBlockSizeWithFlags",
            "blockSizeToDynamicSMemSize",
        ) => Some(true),
        (
            "cuDeviceRegisterAsyncNotification"
            | "cuLogsRegisterCallback"
            | "cuStreamBeginRecaptureToGraph",
            "callbackFunc",
        )
        | (
            "cuLaunchHostFunc"
            | "cuLaunchHostFunc_v2"
            | "hipLaunchHostFunc"
            | "hipLaunchHostFunc_spt",
            "fn",
        )
        | (
            "cuStreamAddCallback" | "hipStreamAddCallback" | "hipStreamAddCallback_spt",
            "callback",
        )
        | ("cuUserObjectCreate" | "hipUserObjectCreate", "destroy") => Some(false),
        _ => None,
    }
}

fn callback_contract_reason(function: &str, nullable: bool) -> &'static str {
    if nullable {
        "The exact vendor contract explicitly permits a null occupancy callback when dynamic shared-memory usage is constant or absent."
    } else if function.ends_with("_spt") {
        "The exact stream-per-thread declaration has the same callback ABI and required callable contract as its documented base entrypoint; only default-stream semantics differ."
    } else {
        "The exact vendor operation requires the supplied callable to register, enqueue, invoke, or destroy the documented object; null is not an optional callback mode."
    }
}

fn group_parameter<'a>(
    inventory: &Inventory,
    entry: &'a Entry,
    function: &str,
    index: usize,
) -> Result<&'a crate::model::Parameter, String> {
    entry
        .abi
        .as_ref()
        .and_then(|abi| abi.parameters.get(index))
        .ok_or_else(|| format!("{}::{function} lost ABI facts", inventory.inventory_id))
}

fn validate_propagated_conflict(
    inventory: &Inventory,
    function: &str,
    parameter: &crate::model::Parameter,
    facts: &PropagatedFacts,
    existing: &SemanticOverride,
) -> Result<(), String> {
    let direction_conflict = facts
        .direction
        .is_some_and(|direction| direction != existing.direction);
    let nullable_conflict = facts.nullable.is_some_and(|nullable| {
        existing
            .nullability
            .as_bool()
            .is_some_and(|value| value != nullable)
    });
    if direction_conflict || nullable_conflict {
        return Err(format!(
            "higher-authority identical-ABI evidence conflicts with reviewed semantics for {}::{function}::{}",
            inventory.inventory_id, parameter.name
        ));
    }
    Ok(())
}

fn record_source_unspecified_nullability(
    inventories: &[Inventory],
    output: &mut BTreeMap<(String, String, String), SemanticOverride>,
) -> Result<(), String> {
    for inventory in inventories {
        for entry in &inventory.entries {
            if !matches!(entry.kind, ItemKind::Function | ItemKind::Callback) {
                continue;
            }
            let Some(abi) = &entry.abi else {
                continue;
            };
            for parameter in &abi.parameters {
                if parameter.pointer == PointerKind::Value {
                    continue;
                }
                let key = (
                    inventory.inventory_id.clone(),
                    entry.name.clone(),
                    parameter.name.clone(),
                );
                if output.contains_key(&key)
                    || (parameter.direction != Direction::Unknown && parameter.nullable.is_some())
                    || (parameter.pointer == PointerKind::Callback && parameter.nullable.is_none())
                {
                    continue;
                }
                insert_override(
                    output,
                    SemanticOverride {
                        inventory_id: inventory.inventory_id.clone(),
                        function: entry.name.clone(),
                        parameter: parameter.name.clone(),
                        direction: if parameter.direction == Direction::Unknown {
                            if entry.kind == ItemKind::Callback {
                                Direction::In
                            } else {
                                Direction::UnspecifiedBySource
                            }
                        } else {
                            parameter.direction
                        },
                        nullability: parameter.nullable.map_or(
                            ReviewedNullability::UnspecifiedBySource,
                            ReviewedNullability::from_bool,
                        ),
                        reason: if entry.kind == ItemKind::Callback {
                            "The vendor invokes this callback argument as caller-provided input; the exact declaration states no ordinary-pointer nullability contract, so nullability is explicitly source-unspecified.".to_owned()
                        } else {
                            "The exact normalized declaration and extracted annotation/comment graph state no ordinary-pointer nullability contract; this is a final source-unspecified fact, not a non-null default.".to_owned()
                        },
                        provenance: format!(
                            "{}; normalized declaration {}",
                            entry.provenance, entry.signature_hash
                        ),
                    },
                )?;
            }
        }
    }
    Ok(())
}

fn abi_shape(abi: &Abi) -> String {
    let parameters = abi
        .parameters
        .iter()
        .map(|parameter| {
            format!(
                "{}:{:?}",
                canonical_type(&parameter.r#type),
                parameter.pointer
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{}({parameters})->{}",
        if matches!(abi.calling_convention.as_str(), "C" | "system") {
            "c64"
        } else {
            &abi.calling_convention
        },
        canonical_type(&abi.return_type)
    )
}

fn canonical_type(value: &str) -> String {
    let compact_source = value.split_whitespace().collect::<String>();
    if let Some(inner) = compact_source.strip_prefix("*mut") {
        return format!("mutptr({})", canonical_type(inner));
    }
    if let Some(inner) = compact_source.strip_prefix("*const") {
        return format!("constptr({})", canonical_type(inner));
    }
    if let Some(inner) = compact_source
        .strip_prefix("const")
        .and_then(|value| value.strip_suffix('*'))
    {
        return format!("constptr({})", canonical_type(inner));
    }
    if let Some(inner) = compact_source.strip_suffix('*') {
        return format!("mutptr({})", canonical_type(inner));
    }
    let compact = compact_source
        .replace("struct ", "")
        .replace("enum ", "")
        .split_whitespace()
        .collect::<String>()
        .to_ascii_lowercase()
        .replace("ocgpu", "");
    let leaf = compact.rsplit("::").next().unwrap_or(&compact);
    match leaf {
        "()" | "void" | "c_void" => "void".to_owned(),
        "char" | "c_char" => "char".to_owned(),
        "u8" | "unsignedchar" | "c_uchar" => "u8".to_owned(),
        "i8" | "signedchar" | "c_schar" => "i8".to_owned(),
        "u16" | "unsignedshort" | "c_ushort" => "u16".to_owned(),
        "i16" | "short" | "c_short" => "i16".to_owned(),
        "u32" | "unsignedint" | "c_uint" => "u32".to_owned(),
        "i32" | "int" | "c_int" | "cudevice" | "hipdevice_t" => "i32".to_owned(),
        "u64" | "unsignedlonglong" | "c_ulonglong" => "u64".to_owned(),
        "i64" | "longlong" | "c_longlong" => "i64".to_owned(),
        "usize" | "size_t" => "usize".to_owned(),
        "isize" | "ptrdiff_t" => "isize".to_owned(),
        _ => leaf.to_owned(),
    }
}
