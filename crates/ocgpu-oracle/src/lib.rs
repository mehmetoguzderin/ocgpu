// SPDX-License-Identifier: CC0-1.0

//! Offline validation and maintainer-only extraction for API coverage inventories.
//!
//! Normal builds never invoke this crate. The two GPU wrapper dependencies are exact-pinned
//! here so their source can be used as a coverage oracle without entering the shipping graph.

mod c_extract;
mod classification;
mod extract;
mod hash;
mod model;
mod report;
mod semantics;
mod validate;
mod vendor_union;

pub use extract::{ExtractRequest, extract_rust_inventory, locate_registry_package};
pub use model::{
    Abi, Classification, CoverageCatalog, CoverageDecision, CudaProcAddressCandidate,
    CudaProcAddressCatalog, CudaProcAddressVariant, Direction, Entry, Inventory, ItemKind, Layout,
    Parameter, PointerKind, ReviewedNullability, SemanticCatalog, SemanticOverride, SourceArtifact,
    VendorFunctionUnion, VendorUnionFunction, VendorUnionSource, VendorUnionVariant,
};
pub use report::{CoverageReport, CoverageSymbol, build_report, render_markdown};
pub use validate::{ValidationError, ValidationSummary, read_inputs, validate_repository};

use std::path::{Path, PathBuf};

/// Returns the repository root containing the oracle and coverage directories.
#[must_use]
pub fn repository_root() -> PathBuf {
    let manifest_directory = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest_directory
        .parent()
        .and_then(Path::parent)
        .map_or_else(|| manifest_directory.to_path_buf(), Path::to_path_buf)
}
pub use c_extract::{
    HeaderExtractRequest, HeaderExtractionError, HeaderSemanticEvidence, VendorFamily,
    extract_cuda_proc_address_catalog, extract_header_inventory,
};
pub use classification::build_seed_catalog;
pub use semantics::build_semantic_catalog;
pub use vendor_union::build_vendor_function_union;
