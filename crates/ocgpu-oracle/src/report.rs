// SPDX-License-Identifier: CC0-1.0

use crate::model::{Classification, CoverageCatalog, Inventory, ItemKind};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

/// Deterministic coverage report built from committed inventories and reviewed decisions.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CoverageReport {
    /// Data format version.
    pub schema_version: u32,
    /// SPDX identifier for the generated report data.
    pub spdx_license_identifier: String,
    /// Individual, deliberately unblended metrics.
    pub metrics: Vec<Metric>,
    /// Counts for all classifications appearing in the catalog.
    pub classification_counts: BTreeMap<String, u64>,
    /// Item-level inventory for the SDK-free diagnostics CLI.
    pub symbols: Vec<CoverageSymbol>,
}

/// Compact item-level coverage fact embedded by the diagnostics CLI.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CoverageSymbol {
    /// Source inventory identifier.
    pub inventory_id: String,
    /// Exact source item name.
    pub name: String,
    /// Backend family (`cuda` or `hip`).
    pub backend: String,
    /// Normalized item kind.
    pub kind: String,
    /// Exclusive reviewed classification.
    pub classification: String,
    /// Applicable target triples.
    pub platforms: Vec<String>,
    /// Canonical manifest IDs accounting for the item.
    pub manifest_ids: Vec<String>,
    /// Runtime-resolution evidence flag.
    pub runtime_resolvable: bool,
    /// Bounded hardware-smoke evidence flag.
    pub hardware_smoke: bool,
}

/// A numerator and denominator with explicit semantics.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Metric {
    /// Stable machine-readable metric key.
    pub id: String,
    /// Human-readable metric name.
    pub label: String,
    /// Covered or exercised entries.
    pub numerator: u64,
    /// Applicable entries.
    pub denominator: u64,
    /// Exact counting rule.
    pub basis: String,
}

/// Builds every required metric without collapsing vendor, platform, API-level, and runtime
/// evidence into a misleading single percentage.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn build_report(inventories: &[Inventory], catalog: &CoverageCatalog) -> CoverageReport {
    let decisions: BTreeMap<(&str, ItemKind, &str), _> = catalog
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
        .collect();

    let mut metrics = vec![
        inventory_metric(
            inventories,
            &decisions,
            "cuda-vendor-13.3-13030",
            "cuda_logical_function_coverage",
            "CUDA logical-function coverage",
            &[ItemKind::Function],
            false,
        ),
        inventory_metric(
            inventories,
            &decisions,
            "cuda-vendor-13.3-13030",
            "cuda_exported_symbol_coverage",
            "CUDA exported-symbol coverage",
            &[ItemKind::Function, ItemKind::Alias],
            true,
        ),
        inventory_metric(
            inventories,
            &decisions,
            "cuda-vendor-13.3-13030",
            "cuda_type_constant_coverage",
            "CUDA type and constant coverage",
            &type_kinds(),
            false,
        ),
        inventory_metric(
            inventories,
            &decisions,
            "hip-general-7.14.60850",
            "hip_general_logical_function_coverage",
            "HIP general logical-function coverage",
            &[ItemKind::Function],
            false,
        ),
        inventory_metric(
            inventories,
            &decisions,
            "hip-windows-7.2.0",
            "hip_windows_logical_function_coverage",
            "HIP Windows logical-function coverage",
            &[ItemKind::Function],
            false,
        ),
        multi_inventory_metric(
            inventories,
            &decisions,
            &["hip-general-7.14.60850", "hip-windows-7.2.0"],
            "hip_type_constant_coverage",
            "HIP type and constant coverage",
            &type_kinds(),
        ),
        inventory_metric(
            inventories,
            &decisions,
            "cudarc-0.19.9",
            "cudarc_raw_surface_coverage",
            "cudarc raw-surface coverage",
            &all_kinds(),
            false,
        ),
        inventory_metric(
            inventories,
            &decisions,
            "rocmrc-0.5.0",
            "rocmrc_raw_surface_coverage",
            "rocmrc raw-surface coverage",
            &all_kinds(),
            false,
        ),
    ];

    let unique_manifest_items = |classification| {
        catalog
            .decisions
            .iter()
            .filter(|decision| decision.classification == classification)
            .flat_map(|decision| decision.manifest_ids.iter().cloned())
            .filter(|id| {
                !id.starts_with("raw.") && !id.starts_with("type.") && !id.starts_with("constant.")
            })
            .collect::<BTreeSet<_>>()
    };
    let exact = unique_manifest_items(Classification::CoveredExact);
    let adapter = unique_manifest_items(Classification::CoveredAdapter);
    let raw_only = catalog
        .decisions
        .iter()
        .filter(|decision| decision.classification == Classification::CoveredRawOnly)
        .flat_map(|decision| decision.manifest_ids.iter().cloned())
        .collect::<BTreeSet<_>>();
    let represented = exact.union(&adapter).cloned().collect::<BTreeSet<_>>();
    metrics.push(metric(
        "common_exact_coverage",
        "Common exact coverage",
        exact.len(),
        represented.len(),
        "unique common manifest IDs classified exact / all common exact-or-adapter IDs",
    ));
    metrics.push(metric(
        "common_adapter_coverage",
        "Common adapter coverage",
        adapter.len(),
        represented.len(),
        "unique common manifest IDs classified adapter / all common exact-or-adapter IDs",
    ));
    let exposed = represented.union(&raw_only).count();
    metrics.push(metric(
        "raw_only_coverage",
        "Raw-only coverage",
        raw_only.len(),
        exposed,
        "unique raw-only manifest IDs / all unique exact, adapter, or raw-only IDs",
    ));

    let applicable_callables = applicable_decisions(
        inventories,
        &decisions,
        &[ItemKind::Function, ItemKind::Alias],
    )
    .into_iter()
    .filter(|decision| {
        decision.item_kind == ItemKind::Function
            || decision
                .manifest_ids
                .iter()
                .any(|id| id.starts_with("raw."))
    })
    .collect::<Vec<_>>();
    metrics.push(metric(
        "runtime_resolvable_coverage",
        "Runtime-resolvable coverage",
        applicable_callables
            .iter()
            .filter(|decision| decision.runtime_resolvable)
            .count(),
        applicable_callables.len(),
        "representable applicable function and callable-alias entries recorded runtime-resolvable / representable applicable callable entries",
    ));
    let applicable_functions = applicable_decisions(inventories, &decisions, &[ItemKind::Function]);
    metrics.push(metric(
        "hardware_smoke_coverage",
        "Hardware-smoke coverage",
        applicable_functions
            .iter()
            .filter(|decision| decision.hardware_smoke)
            .count(),
        applicable_functions.len(),
        "applicable function entries exercised by the bounded hardware profile / applicable function entries",
    ));
    metrics.push(metric(
        "hardware_runner_execution_evidence",
        "Hardware-runner execution evidence",
        0,
        4,
        "source-committed successful runner attestations / four defined opt-in backend/platform jobs; generation does not claim local hardware execution",
    ));

    let mut record_total = 0_usize;
    let mut record_verified = 0_usize;
    for inventory in inventories {
        for entry in &inventory.entries {
            if !matches!(entry.kind, ItemKind::Struct | ItemKind::Union) {
                continue;
            }
            let Some(decision) = decisions.get(&(
                inventory.inventory_id.as_str(),
                entry.kind,
                entry.name.as_str(),
            )) else {
                continue;
            };
            if decision.classification == Classification::Unrepresentable
                || decision
                    .reason
                    .contains("incomplete vendor tag used only behind")
            {
                continue;
            }
            record_total += 1;
            if is_covered(decision.classification) {
                record_verified += 1;
            }
        }
    }
    metrics.push(metric(
        "layout_verified_coverage",
        "Layout-verified coverage",
        record_verified,
        record_total,
        "ABI-relevant record entries with exact generated per-target layout evidence and committed Rust/C assertions / ABI-relevant record entries (pointer-only incomplete tags excluded)",
    ));

    let mut classification_counts = BTreeMap::new();
    for decision in &catalog.decisions {
        let key = serde_json::to_value(decision.classification)
            .ok()
            .and_then(|value| value.as_str().map(str::to_owned))
            .unwrap_or_else(|| "unknown".to_owned());
        *classification_counts.entry(key).or_insert(0) += 1;
    }
    let mut symbols = inventories
        .iter()
        .flat_map(|inventory| {
            inventory.entries.iter().filter_map(|entry| {
                let decision = decisions.get(&(
                    inventory.inventory_id.as_str(),
                    entry.kind,
                    entry.name.as_str(),
                ))?;
                Some(CoverageSymbol {
                    inventory_id: inventory.inventory_id.clone(),
                    name: entry.name.clone(),
                    backend: if inventory.inventory_id.contains("cuda")
                        || inventory.inventory_id.starts_with("cudarc")
                    {
                        "cuda".to_owned()
                    } else {
                        "hip".to_owned()
                    },
                    kind: serde_json::to_value(entry.kind)
                        .ok()
                        .and_then(|value| value.as_str().map(str::to_owned))
                        .unwrap_or_else(|| "unknown".to_owned()),
                    classification: serde_json::to_value(decision.classification)
                        .ok()
                        .and_then(|value| value.as_str().map(str::to_owned))
                        .unwrap_or_else(|| "unknown".to_owned()),
                    platforms: entry.platforms.clone(),
                    manifest_ids: decision.manifest_ids.clone(),
                    runtime_resolvable: decision.runtime_resolvable,
                    hardware_smoke: decision.hardware_smoke,
                })
            })
        })
        .collect::<Vec<_>>();
    symbols.sort_by(|left, right| {
        (&left.inventory_id, &left.kind, &left.name).cmp(&(
            &right.inventory_id,
            &right.kind,
            &right.name,
        ))
    });
    CoverageReport {
        schema_version: 1,
        spdx_license_identifier: "CC0-1.0".to_owned(),
        metrics,
        classification_counts,
        symbols,
    }
}

fn inventory_metric(
    inventories: &[Inventory],
    decisions: &BTreeMap<(&str, ItemKind, &str), &crate::model::CoverageDecision>,
    inventory_id: &str,
    id: &str,
    label: &str,
    kinds: &[ItemKind],
    count_aliases: bool,
) -> Metric {
    let inventory = inventories
        .iter()
        .find(|value| value.inventory_id == inventory_id);
    let Some(inventory) = inventory else {
        return metric(id, label, 0, 0, "required inventory is absent");
    };
    let mut total = 0_usize;
    let mut covered = 0_usize;
    let mut logical_seen = BTreeSet::new();
    for entry in &inventory.entries {
        if !kinds.contains(&entry.kind) {
            continue;
        }
        let logical_name = if count_aliases {
            entry.name.as_str()
        } else {
            entry.alias_of.as_deref().unwrap_or(&entry.name)
        };
        if !logical_seen.insert(logical_name) {
            continue;
        }
        let Some(decision) = decisions.get(&(inventory_id, entry.kind, entry.name.as_str())) else {
            continue;
        };
        if decision.classification == Classification::PlatformUnavailable {
            continue;
        }
        total += 1;
        if is_covered(decision.classification) {
            covered += 1;
        }
    }
    metric(
        id,
        label,
        covered,
        total,
        if count_aliases {
            "covered applicable symbols / applicable symbols (aliases counted separately)"
        } else {
            "covered applicable logical entries / applicable logical entries (aliases collapsed)"
        },
    )
}

fn multi_inventory_metric(
    inventories: &[Inventory],
    decisions: &BTreeMap<(&str, ItemKind, &str), &crate::model::CoverageDecision>,
    inventory_ids: &[&str],
    id: &str,
    label: &str,
    kinds: &[ItemKind],
) -> Metric {
    let mut total = 0_usize;
    let mut covered = 0_usize;
    for inventory_id in inventory_ids {
        if let Some(inventory) = inventories
            .iter()
            .find(|value| &value.inventory_id == inventory_id)
        {
            for entry in &inventory.entries {
                if !kinds.contains(&entry.kind) {
                    continue;
                }
                if let Some(decision) =
                    decisions.get(&(*inventory_id, entry.kind, entry.name.as_str()))
                {
                    if decision.classification != Classification::PlatformUnavailable {
                        total += 1;
                        covered += usize::from(is_covered(decision.classification));
                    }
                }
            }
        }
    }
    metric(
        id,
        label,
        covered,
        total,
        "covered applicable type/constant entries / applicable type/constant entries, platform inventories separate",
    )
}

fn applicable_decisions<'a>(
    inventories: &'a [Inventory],
    decisions: &BTreeMap<(&str, ItemKind, &str), &'a crate::model::CoverageDecision>,
    kinds: &[ItemKind],
) -> Vec<&'a crate::model::CoverageDecision> {
    inventories
        .iter()
        .flat_map(|inventory| {
            inventory.entries.iter().filter_map(|entry| {
                kinds
                    .contains(&entry.kind)
                    .then(|| {
                        decisions.get(&(
                            inventory.inventory_id.as_str(),
                            entry.kind,
                            entry.name.as_str(),
                        ))
                    })
                    .flatten()
                    .copied()
            })
        })
        .filter(|decision| {
            !matches!(
                decision.classification,
                Classification::PlatformUnavailable | Classification::Unrepresentable
            )
        })
        .collect()
}

fn metric(id: &str, label: &str, numerator: usize, denominator: usize, basis: &str) -> Metric {
    Metric {
        id: id.to_owned(),
        label: label.to_owned(),
        numerator: u64::try_from(numerator).unwrap_or(u64::MAX),
        denominator: u64::try_from(denominator).unwrap_or(u64::MAX),
        basis: basis.to_owned(),
    }
}

fn is_covered(classification: Classification) -> bool {
    matches!(
        classification,
        Classification::CoveredExact
            | Classification::CoveredAdapter
            | Classification::CoveredRawOnly
            | Classification::DeprecatedCovered
    )
}

const fn type_kinds() -> [ItemKind; 8] {
    [
        ItemKind::Type,
        ItemKind::OpaqueHandle,
        ItemKind::Struct,
        ItemKind::Union,
        ItemKind::Callback,
        ItemKind::Constant,
        ItemKind::EnumValue,
        ItemKind::Flag,
    ]
}

const fn all_kinds() -> [ItemKind; 10] {
    [
        ItemKind::Function,
        ItemKind::Alias,
        ItemKind::Type,
        ItemKind::OpaqueHandle,
        ItemKind::Struct,
        ItemKind::Union,
        ItemKind::Callback,
        ItemKind::Constant,
        ItemKind::EnumValue,
        ItemKind::Flag,
    ]
}

/// Renders the report and the complete human-review reason ledger as Markdown.
#[must_use]
pub fn render_markdown(report: &CoverageReport, catalog: &CoverageCatalog) -> String {
    let mut output =
        String::from("<!-- SPDX-License-Identifier: CC0-1.0 -->\n\n# API coverage\n\n");
    output.push_str("Metrics are intentionally separate; no blended percentage is published. Runtime and hardware values are evidence dimensions, not substitutes for declaration coverage. Hardware-smoke coverage records the bounded profile's symbol breadth, while hardware-runner execution evidence separately records committed executions and therefore remains zero until a runner attestation is published.\n\n");
    output.push_str(
        "| Metric | Covered | Applicable | Percent | Counting basis |\n|---|---:|---:|---:|---|\n",
    );
    for metric in &report.metrics {
        let percent = if metric.denominator == 0 {
            "n/a".to_owned()
        } else {
            let basis_points =
                u128::from(metric.numerator) * 10_000 / u128::from(metric.denominator);
            format!("{}.{:02}%", basis_points / 100, basis_points % 100)
        };
        let _ = writeln!(
            &mut output,
            "| {} | {} | {} | {} | {} |",
            metric.label, metric.numerator, metric.denominator, percent, metric.basis
        );
    }
    output.push_str("\n## Human-reviewed classifications\n\n");
    output
        .push_str("| Inventory | Kind | Item | Classification | Reason |\n|---|---|---|---|---|\n");
    let mut decisions = catalog.decisions.iter().collect::<Vec<_>>();
    decisions.sort_by_key(|decision| {
        (
            &decision.inventory_id,
            decision.item_kind,
            &decision.item_name,
        )
    });
    for decision in decisions {
        let class = serde_json::to_value(decision.classification)
            .ok()
            .and_then(|value| value.as_str().map(str::to_owned))
            .unwrap_or_else(|| "unknown".to_owned());
        let kind = serde_json::to_value(decision.item_kind)
            .ok()
            .and_then(|value| value.as_str().map(str::to_owned))
            .unwrap_or_else(|| "unknown".to_owned());
        let _ = writeln!(
            &mut output,
            "| `{}` | `{}` | `{}` | `{}` | {} |",
            decision.inventory_id,
            kind,
            decision.item_name,
            class,
            decision.reason.replace('|', "\\|")
        );
    }
    output
}

#[cfg(test)]
mod tests {
    use super::{CoverageReport, render_markdown};
    use crate::model::CoverageCatalog;
    use std::collections::BTreeMap;

    #[test]
    fn empty_report_does_not_invent_a_percentage() {
        let report = CoverageReport {
            schema_version: 1,
            spdx_license_identifier: "CC0-1.0".to_owned(),
            metrics: Vec::new(),
            classification_counts: BTreeMap::new(),
            symbols: Vec::new(),
        };
        let catalog = CoverageCatalog {
            schema_version: 1,
            spdx_license_identifier: "CC0-1.0".to_owned(),
            decisions: Vec::new(),
        };
        assert!(render_markdown(&report, &catalog).contains("no blended percentage"));
    }
}
