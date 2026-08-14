// SPDX-License-Identifier: CC0-1.0

use crate::hash::{sha256, sha256_bytes};
use crate::model::{
    Abi, CudaProcAddressCandidate, CudaProcAddressCatalog, CudaProcAddressVariant, Direction,
    Entry, Inventory, ItemKind, Layout, Parameter, PointerKind, SourceArtifact,
};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use thiserror::Error;

/// Vendor API family accepted by the public-header normalizer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VendorFamily {
    /// NVIDIA CUDA Driver API declarations beginning with `cu`/`CU`.
    Cuda,
    /// AMD HIP runtime and driver-shaped declarations beginning with `hip`.
    Hip,
}

/// Request for normalizing an authoritative public C/C++ header snapshot.
#[derive(Clone, Debug)]
pub struct HeaderExtractRequest {
    /// Vendor family.
    pub family: VendorFamily,
    /// Top-level public header.
    pub header: PathBuf,
    /// Include directories required by the snapshot.
    pub include_directories: Vec<PathBuf>,
    /// Stable committed inventory ID.
    pub inventory_id: String,
    /// Human source name.
    pub source_name: String,
    /// Exact source version.
    pub source_version: String,
    /// Canonical evidence URL and archive checksum or source commit.
    pub provenance: String,
    /// Exact fetched source artifacts used by this extraction.
    pub source_artifacts: Vec<SourceArtifact>,
    /// Applicable Rust target triples.
    pub platforms: Vec<String>,
    /// Optional exact-version annotated header used only to enrich parameter semantics.
    pub semantic_evidence: Option<HeaderSemanticEvidence>,
}

/// A second official header whose declaration annotations provide semantic evidence.
#[derive(Clone, Debug)]
pub struct HeaderSemanticEvidence {
    /// Exact annotated public header.
    pub header: PathBuf,
    /// Include directories needed to parse the annotated header.
    pub include_directories: Vec<PathBuf>,
    /// Stable official source locator for the annotations.
    pub provenance: String,
}

/// Public-header extraction failure.
#[derive(Debug, Error)]
pub enum HeaderExtractionError {
    /// An extraction request violates a command invariant.
    #[error("invalid extraction request: {0}")]
    InvalidRequest(String),
    /// An authoritative header could not be read for content hashing.
    #[error("failed to read authoritative header {path}: {source}")]
    HeaderRead {
        /// Header path.
        path: PathBuf,
        /// Filesystem error.
        #[source]
        source: std::io::Error,
    },
    /// The committed source artifact hash does not match the parsed header bytes.
    #[error("authoritative typedef header hash mismatch: expected {expected}, computed {actual}")]
    SourceHashMismatch {
        /// Hash recorded in the extraction request.
        expected: String,
        /// Hash of the parsed bytes.
        actual: String,
    },
    /// Clang could not be started.
    #[error("failed to run clang: {0}")]
    Clang(#[source] std::io::Error),
    /// Clang rejected the public header snapshot.
    #[error("clang rejected {header}: {diagnostic}")]
    ClangFailure {
        /// Header path.
        header: PathBuf,
        /// Compiler diagnostic.
        diagnostic: String,
    },
    /// Clang emitted malformed AST JSON.
    #[error("invalid clang AST JSON: {0}")]
    AstJson(#[from] serde_json::Error),
    /// A declaration changed between target passes of the same inventory.
    #[error("declaration {name} differs between {left} and {right}: {detail}")]
    PlatformDrift {
        /// Exact declaration name.
        name: String,
        /// First target triple.
        left: String,
        /// Second target triple.
        right: String,
        /// Fact that changed.
        detail: String,
    },
}

/// Uses Clang only as a parser over a downloaded public source snapshot. It neither runs a
/// vendor compiler nor links or installs a vendor SDK.
pub fn extract_header_inventory(
    request: &HeaderExtractRequest,
) -> Result<Inventory, HeaderExtractionError> {
    let mut entries = BTreeMap::new();
    for platform in &request.platforms {
        let mut pass = request.clone();
        pass.platforms = vec![platform.clone()];
        let mut pass_entries = BTreeMap::new();
        collect_ast(&clang_ast(&pass, platform)?, &pass, &mut pass_entries);
        if let Some(evidence) = &request.semantic_evidence {
            enrich_parameter_semantics(&mut pass_entries, &pass, evidence, platform)?;
        }
        collect_macros(&pass, platform, &mut pass_entries)?;
        collect_layouts(&pass, platform, &mut pass_entries)?;
        merge_entries(&mut entries, pass_entries)?;
    }
    reconcile_aliases(&mut entries);
    let mut platforms = request.platforms.clone();
    platforms.sort();
    platforms.dedup();
    let mut source_artifacts = request.source_artifacts.clone();
    source_artifacts.sort_by(|left, right| {
        (&left.role, &left.url, &left.path).cmp(&(&right.role, &right.url, &right.path))
    });
    source_artifacts.dedup_by(|left, right| {
        left.role == right.role && left.url == right.url && left.path == right.path
    });
    Ok(Inventory {
        schema_version: 1,
        spdx_license_identifier: "CC0-1.0".to_owned(),
        inventory_id: request.inventory_id.clone(),
        source_name: request.source_name.clone(),
        source_version: request.source_version.clone(),
        provenance: request.provenance.clone(),
        source_artifacts,
        platforms,
        entries: entries.into_values().collect(),
    })
}

/// Normalizes NVIDIA's versioned `PFN_*` declarations into an offline proc-address catalog.
pub fn extract_cuda_proc_address_catalog(
    request: &HeaderExtractRequest,
) -> Result<CudaProcAddressCatalog, HeaderExtractionError> {
    if request.family != VendorFamily::Cuda {
        return Err(HeaderExtractionError::InvalidRequest(
            "the CUDA proc-address catalog requires the cuda family".to_owned(),
        ));
    }
    let platform = request
        .platforms
        .iter()
        .find(|platform| platform.contains("windows"))
        .ok_or_else(|| {
            HeaderExtractionError::InvalidRequest(
                "the authoritative Windows CUDA header requires a Windows target pass".to_owned(),
            )
        })?;
    verify_proc_typedef_source_hash(request)?;
    let mut typedefs = BTreeMap::new();
    collect_cuda_proc_typedefs(&clang_ast(request, platform)?, request, &mut typedefs);
    if typedefs.is_empty() {
        return Err(HeaderExtractionError::InvalidRequest(
            "cudaTypedefs.h yielded no versioned PFN declarations".to_owned(),
        ));
    }
    let mut source_artifacts = request.source_artifacts.clone();
    source_artifacts.sort_by(|left, right| {
        (&left.role, &left.url, &left.path).cmp(&(&right.role, &right.url, &right.path))
    });
    source_artifacts.dedup_by(|left, right| {
        left.role == right.role && left.url == right.url && left.path == right.path
    });
    Ok(CudaProcAddressCatalog {
        schema_version: 1,
        spdx_license_identifier: "CC0-1.0".to_owned(),
        source_version: request.source_version.clone(),
        provenance: request.provenance.clone(),
        source_artifacts,
        typedefs: typedefs.into_values().collect(),
    })
}

fn verify_proc_typedef_source_hash(
    request: &HeaderExtractRequest,
) -> Result<(), HeaderExtractionError> {
    let bytes = fs::read(&request.header).map_err(|source| HeaderExtractionError::HeaderRead {
        path: request.header.clone(),
        source,
    })?;
    let actual = sha256_bytes(&bytes);
    let artifact = request
        .source_artifacts
        .iter()
        .find(|artifact| artifact.role == "authoritative-proc-address-typedef-header")
        .ok_or_else(|| {
            HeaderExtractionError::InvalidRequest(
                "missing authoritative-proc-address-typedef-header source artifact".to_owned(),
            )
        })?;
    if artifact.sha256 != actual {
        return Err(HeaderExtractionError::SourceHashMismatch {
            expected: artifact.sha256.clone(),
            actual,
        });
    }
    Ok(())
}

fn enrich_parameter_semantics(
    entries: &mut BTreeMap<(ItemKind, String), Entry>,
    request: &HeaderExtractRequest,
    evidence: &HeaderSemanticEvidence,
    platform: &str,
) -> Result<(), HeaderExtractionError> {
    let mut evidence_request = request.clone();
    evidence_request.header.clone_from(&evidence.header);
    evidence_request
        .include_directories
        .clone_from(&evidence.include_directories);
    evidence_request.provenance.clone_from(&evidence.provenance);
    evidence_request.semantic_evidence = None;
    let mut evidence_entries = BTreeMap::new();
    collect_ast(
        &clang_ast(&evidence_request, platform)?,
        &evidence_request,
        &mut evidence_entries,
    );
    for ((kind, name), evidence_entry) in evidence_entries {
        if kind != ItemKind::Function {
            continue;
        }
        let Some(entry) = entries.get_mut(&(kind, name.clone())) else {
            continue;
        };
        let (Some(abi), Some(evidence_abi)) = (entry.abi.as_mut(), evidence_entry.abi.as_ref())
        else {
            continue;
        };
        let same_shape = abi.calling_convention == evidence_abi.calling_convention
            && abi.return_type == evidence_abi.return_type
            && abi.parameters.len() == evidence_abi.parameters.len()
            && abi
                .parameters
                .iter()
                .zip(&evidence_abi.parameters)
                .all(|(left, right)| {
                    left.name == right.name
                        && left.r#type == right.r#type
                        && left.pointer == right.pointer
                });
        if !same_shape {
            return Err(HeaderExtractionError::PlatformDrift {
                name,
                left: request.provenance.clone(),
                right: evidence.provenance.clone(),
                detail: "semantic-evidence declaration has a different normalized ABI graph"
                    .to_owned(),
            });
        }
        for (parameter, evidence_parameter) in
            abi.parameters.iter_mut().zip(&evidence_abi.parameters)
        {
            if parameter.direction == Direction::Unknown
                && evidence_parameter.direction != Direction::Unknown
            {
                parameter.direction = evidence_parameter.direction;
            }
            if parameter.nullable.is_none() && evidence_parameter.nullable.is_some() {
                parameter.nullable = evidence_parameter.nullable;
            }
        }
        entry.normalized_signature = normalized_callable(&entry.name, abi);
        entry.signature_hash = sha256(&entry.normalized_signature);
        entry.provenance = format!(
            "{}; parameter annotations: {}#{}",
            entry.provenance, evidence.provenance, entry.name
        );
    }
    Ok(())
}

fn clang_ast(
    request: &HeaderExtractRequest,
    platform: &str,
) -> Result<Value, HeaderExtractionError> {
    let mut command = clang_command(request, platform);
    command.args(["-fparse-all-comments", "-Xclang", "-ast-dump=json"]);
    let output = command.output().map_err(HeaderExtractionError::Clang)?;
    if !output.status.success() {
        return Err(HeaderExtractionError::ClangFailure {
            header: request.header.clone(),
            diagnostic: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        });
    }
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn collect_layouts(
    request: &HeaderExtractRequest,
    platform: &str,
    entries: &mut BTreeMap<(ItemKind, String), Entry>,
) -> Result<(), HeaderExtractionError> {
    let mut command = clang_command(request, platform);
    command.args(["-Xclang", "-fdump-record-layouts-complete"]);
    let output = command.output().map_err(HeaderExtractionError::Clang)?;
    if !output.status.success() {
        return Err(HeaderExtractionError::ClangFailure {
            header: request.header.clone(),
            diagnostic: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        });
    }
    let dump = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    for block in dump.split("*** Dumping AST Record Layout") {
        let Some((kind, name, layout)) = parse_record_layout(block, platform, &request.provenance)
        else {
            continue;
        };
        if let Some(entry) = entries.get_mut(&(kind, name)) {
            entry.layouts.push(layout);
        }
    }
    Ok(())
}

fn parse_record_layout(
    block: &str,
    platform: &str,
    source: &str,
) -> Option<(ItemKind, String, Layout)> {
    let mut record = None;
    let mut size = None;
    let mut alignment = None;
    let mut field_offsets = BTreeMap::new();
    for line in block.lines() {
        let Some((offset, value)) = line.split_once('|') else {
            continue;
        };
        let value = value.trim_end();
        let trimmed = value.trim();
        if record.is_none() {
            let named = trimmed
                .strip_prefix("struct ")
                .map(|name| (ItemKind::Struct, name))
                .or_else(|| {
                    trimmed
                        .strip_prefix("union ")
                        .map(|name| (ItemKind::Union, name))
                });
            if let Some((kind, name)) = named {
                let name = name.split_whitespace().next()?;
                if name.contains("::") || name.starts_with("(unnamed") {
                    return None;
                }
                record = Some((kind, name.to_owned()));
                continue;
            }
        }
        if let Some(facts) = trimmed
            .strip_prefix("[sizeof=")
            .and_then(|facts| facts.strip_suffix(']'))
        {
            let (size_value, align_value) = facts.split_once(", align=")?;
            size = size_value.parse().ok();
            alignment = align_value.parse().ok();
            continue;
        }
        if !value.starts_with("   ") || value.starts_with("     ") {
            continue;
        }
        let byte_offset = offset
            .trim()
            .split_once(':')
            .map_or(offset.trim(), |(bytes, _)| bytes)
            .parse()
            .ok()?;
        if trimmed.contains("::(anonymous at") || trimmed.contains("::(unnamed at") {
            // Clang prints an anonymous member as `...::(anonymous at PATH:LINE)` and a
            // named field whose type is anonymous as `...::(unnamed at PATH:LINE) field`.
            // Preserve the declaration-level field name when one follows the closing `)`;
            // otherwise use the same stable `anonymous` spelling as the AST normalizer.
            let field = trimmed
                .rsplit_once(')')
                .map(|(_, suffix)| suffix.trim())
                .filter(|suffix| {
                    !suffix.is_empty()
                        && suffix
                            .chars()
                            .all(|character| character == '_' || character.is_ascii_alphanumeric())
                })
                .unwrap_or("anonymous");
            field_offsets.entry(field.to_owned()).or_insert(byte_offset);
            continue;
        }
        let field = trimmed.split_whitespace().last()?;
        if field.contains("::") || field.ends_with(')') {
            continue;
        }
        field_offsets.insert(field.to_owned(), byte_offset);
    }
    let (kind, name) = record?;
    Some((
        kind,
        name,
        Layout {
            target: platform.to_owned(),
            size: size?,
            alignment: alignment?,
            field_offsets,
            provenance: format!(
                "clang -fdump-record-layouts-complete --target={}; {source}",
                clang_target(platform)
            ),
        },
    ))
}

fn collect_macros(
    request: &HeaderExtractRequest,
    platform: &str,
    entries: &mut BTreeMap<(ItemKind, String), Entry>,
) -> Result<(), HeaderExtractionError> {
    let mut command = clang_command(request, platform);
    command.args(["-dM", "-E"]);
    let output = command.output().map_err(HeaderExtractionError::Clang)?;
    if !output.status.success() {
        return Err(HeaderExtractionError::ClangFailure {
            header: request.header.clone(),
            diagnostic: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        });
    }
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let Some(definition) = line.strip_prefix("#define ") else {
            continue;
        };
        let Some((name, replacement)) = definition.split_once(char::is_whitespace) else {
            continue;
        };
        if name.contains('(') || !is_vendor_constant(request.family, name) {
            continue;
        }
        let replacement = replacement.trim();
        let target = replacement
            .split(|character: char| !(character == '_' || character.is_ascii_alphanumeric()))
            .find(|token| {
                *token != name && entries.contains_key(&(ItemKind::Function, (*token).to_owned()))
            });
        if let Some(target) = target {
            let signature = format!("alias {name}={target}");
            entries.insert(
                (ItemKind::Alias, name.to_owned()),
                base_entry(
                    request,
                    ItemKind::Alias,
                    name,
                    signature,
                    None,
                    Some(target.to_owned()),
                ),
            );
        } else if looks_like_constant_expression(replacement) {
            let signature = format!("const {name}={}", normalize_c_type(replacement));
            entries
                .entry((ItemKind::Constant, name.to_owned()))
                .or_insert_with(|| {
                    base_entry(request, ItemKind::Constant, name, signature, None, None)
                });
        }
    }
    Ok(())
}

fn clang_command(request: &HeaderExtractRequest, platform: &str) -> Command {
    let mut command = Command::new("clang");
    command.args(["-x", "c", "-std=c11", "-ffreestanding", "-fsyntax-only"]);
    command.arg(format!("--target={}", clang_target(platform)));
    if request.family == VendorFamily::Hip {
        command.arg("-D__HIP_PLATFORM_AMD__=1");
    }
    if platform.contains("windows") {
        command.args(["-D_WIN32=1", "-D_WIN64=1"]);
    } else {
        command.args(["-U_WIN32", "-U_WIN64"]);
    }
    for directory in &request.include_directories {
        command.arg("-I").arg(directory);
    }
    command.arg(&request.header);
    command
}

fn clang_target(platform: &str) -> &str {
    match platform {
        "x86_64-pc-windows-msvc" => "x86_64-pc-windows-msvc",
        "x86_64-unknown-linux-gnu" => "x86_64-unknown-linux-gnu",
        "aarch64-unknown-linux-gnu" => "aarch64-unknown-linux-gnu",
        _ => platform,
    }
}

fn collect_ast(
    node: &Value,
    request: &HeaderExtractRequest,
    entries: &mut BTreeMap<(ItemKind, String), Entry>,
) {
    if let Some(kind) = node.get("kind").and_then(Value::as_str) {
        match kind {
            "FunctionDecl" => collect_function(node, request, entries),
            "TypedefDecl" => collect_typedef(node, request, entries),
            "RecordDecl" => collect_record(node, request, entries),
            "EnumDecl" => collect_enum(node, request, entries),
            "VarDecl" => collect_variable(node, request, entries),
            _ => {}
        }
    }
    if let Some(children) = node.get("inner").and_then(Value::as_array) {
        for child in children {
            collect_ast(child, request, entries);
        }
    }
}

fn collect_cuda_proc_typedefs(
    node: &Value,
    request: &HeaderExtractRequest,
    output: &mut BTreeMap<(String, u32, CudaProcAddressVariant, String), CudaProcAddressCandidate>,
) {
    if node.get("kind").and_then(Value::as_str) == Some("TypedefDecl") {
        if let Some(candidate) = cuda_proc_candidate(node, request) {
            output.insert(
                (
                    candidate.symbol.clone(),
                    candidate.api_version,
                    candidate.variant,
                    candidate.typedef_name.clone(),
                ),
                candidate,
            );
        }
    }
    if let Some(children) = node.get("inner").and_then(Value::as_array) {
        for child in children {
            collect_cuda_proc_typedefs(child, request, output);
        }
    }
}

fn cuda_proc_candidate(
    node: &Value,
    request: &HeaderExtractRequest,
) -> Option<CudaProcAddressCandidate> {
    let typedef_name = node.get("name").and_then(Value::as_str)?;
    let (symbol, api_version, variant) = parse_cuda_proc_typedef_name(typedef_name)?;
    let function = find_ast_kind(node, "FunctionProtoType")?;
    let children = function.get("inner").and_then(Value::as_array)?;
    let (return_node, parameter_nodes) = children.split_first()?;
    let return_type = ast_type_spelling(return_node);
    let parameters = parameter_nodes
        .iter()
        .enumerate()
        .map(|(index, parameter)| parameter_from_type_ast(index, parameter))
        .collect::<Vec<_>>();
    let normalized_signature = format!(
        "abi[calling_convention=system]({})->{}",
        parameter_nodes
            .iter()
            .map(normalized_type_graph)
            .collect::<Vec<_>>()
            .join(","),
        normalized_type_graph(return_node)
    );
    let (proc_address_flags, variant) = match variant {
        CudaProcAddressVariant::Legacy => (1, CudaProcAddressVariant::Legacy),
        CudaProcAddressVariant::Ptds => (2, CudaProcAddressVariant::Ptds),
        CudaProcAddressVariant::Ptsz => (2, CudaProcAddressVariant::Ptsz),
    };
    Some(CudaProcAddressCandidate {
        symbol,
        typedef_name: typedef_name.to_owned(),
        api_version,
        proc_address_flags,
        variant,
        signature_hash: sha256(&normalized_signature),
        normalized_signature,
        abi: Abi {
            calling_convention: "system".to_owned(),
            return_type,
            parameters,
        },
        provenance: proc_typedef_provenance(request, node, typedef_name),
    })
}

fn parse_cuda_proc_typedef_name(
    typedef_name: &str,
) -> Option<(String, u32, CudaProcAddressVariant)> {
    let declaration = typedef_name.strip_prefix("PFN_")?;
    let (symbol, encoded) = declaration.rsplit_once("_v")?;
    let digits = encoded.chars().take_while(char::is_ascii_digit).count();
    if digits == 0 {
        return None;
    }
    let api_version = encoded[..digits].parse().ok()?;
    let variant = match &encoded[digits..] {
        "" => CudaProcAddressVariant::Legacy,
        "_ptds" => CudaProcAddressVariant::Ptds,
        "_ptsz" => CudaProcAddressVariant::Ptsz,
        _ => return None,
    };
    Some((symbol.to_owned(), api_version, variant))
}

fn find_ast_kind<'a>(node: &'a Value, kind: &str) -> Option<&'a Value> {
    if node.get("kind").and_then(Value::as_str) == Some(kind) {
        return Some(node);
    }
    node.get("inner")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .find_map(|child| find_ast_kind(child, kind))
}

fn parameter_from_type_ast(index: usize, node: &Value) -> Parameter {
    let qualified = ast_type_spelling(node);
    let pointer = if ast_has_kind(node, "PointerType") {
        if ast_has_kind(node, "FunctionProtoType") {
            PointerKind::Callback
        } else if syntactic_const_pointer(&qualified) {
            PointerKind::Const
        } else {
            PointerKind::Mut
        }
    } else {
        PointerKind::Value
    };
    let syntactically_by_value = !qualified.contains('*');
    let direction = if syntactically_by_value
        || matches!(
            pointer,
            PointerKind::Value | PointerKind::Const | PointerKind::Callback
        ) {
        Direction::In
    } else {
        Direction::Unknown
    };
    Parameter {
        name: format!("arg{index}"),
        r#type: qualified,
        pointer,
        direction,
        nullable: (pointer == PointerKind::Value).then_some(false),
    }
}

fn syntactic_const_pointer(qualified: &str) -> bool {
    let before_pointer = qualified
        .split_once('*')
        .map_or(qualified, |(value, _)| value);
    before_pointer
        .split_whitespace()
        .any(|word| word == "const")
}

fn ast_type_spelling(node: &Value) -> String {
    normalize_c_type(
        node.pointer("/type/qualType")
            .and_then(Value::as_str)
            .unwrap_or("opaque"),
    )
}

fn normalized_type_graph(node: &Value) -> String {
    let kind = node.get("kind").and_then(Value::as_str).unwrap_or("Type");
    let children = node
        .get("inner")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    match kind {
        "AttributedType" | "ElaboratedType" | "ParenType" if children.len() == 1 => {
            normalized_type_graph(&children[0])
        }
        "TypedefType" => {
            let name = node
                .pointer("/decl/name")
                .and_then(Value::as_str)
                .map_or_else(|| ast_type_spelling(node), str::to_owned);
            let target = children
                .first()
                .map_or_else(|| "opaque".to_owned(), normalized_type_graph);
            format!("typedef({name}={target})")
        }
        "PointerType" => {
            let target = children
                .first()
                .map_or_else(|| "opaque".to_owned(), normalized_type_graph);
            let qualifier = if syntactic_const_pointer(&ast_type_spelling(node)) {
                "const"
            } else {
                "mut"
            };
            format!("ptr[{qualifier}]({target})")
        }
        "FunctionProtoType" => {
            let Some((return_type, parameters)) = children.split_first() else {
                return "fn()->opaque".to_owned();
            };
            format!(
                "fn({})->{}",
                parameters
                    .iter()
                    .map(normalized_type_graph)
                    .collect::<Vec<_>>()
                    .join(","),
                normalized_type_graph(return_type)
            )
        }
        "RecordType" => {
            let name = node
                .pointer("/decl/name")
                .and_then(Value::as_str)
                .map_or_else(|| ast_type_spelling(node), str::to_owned);
            format!("record({name})")
        }
        "EnumType" => {
            let name = node
                .pointer("/decl/name")
                .and_then(Value::as_str)
                .map_or_else(|| ast_type_spelling(node), str::to_owned);
            format!("enum({name})")
        }
        _ if children.len() == 1 => normalized_type_graph(&children[0]),
        _ => ast_type_spelling(node).to_ascii_lowercase(),
    }
}

fn proc_typedef_provenance(
    request: &HeaderExtractRequest,
    node: &Value,
    typedef_name: &str,
) -> String {
    let line = node.pointer("/loc/line").and_then(Value::as_u64);
    let artifact = request
        .source_artifacts
        .iter()
        .find(|artifact| artifact.role == "authoritative-proc-address-typedef-header");
    match (artifact, line) {
        (Some(artifact), Some(line)) => format!(
            "{}; archive member {} line {line} ({typedef_name}); {}",
            artifact.url, artifact.path, request.provenance
        ),
        _ => format!("{}#{typedef_name}", request.provenance),
    }
}

fn collect_function(
    node: &Value,
    request: &HeaderExtractRequest,
    entries: &mut BTreeMap<(ItemKind, String), Entry>,
) {
    let Some(name) = node.get("name").and_then(Value::as_str) else {
        return;
    };
    if !is_vendor_function(request.family, name)
        || node.get("isImplicit") == Some(&Value::Bool(true))
    {
        return;
    }
    let qualified = node
        .pointer("/type/qualType")
        .and_then(Value::as_str)
        .unwrap_or("void ()");
    let return_type = qualified
        .split_once(" (")
        .map_or("void", |(return_type, _)| return_type);
    let semantic_hints = documented_parameter_semantics(node);
    let parameters = node
        .get("inner")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|child| child.get("kind").and_then(Value::as_str) == Some("ParmVarDecl"))
        .enumerate()
        .map(|(index, parameter_node)| {
            let mut parameter = parameter_from_ast(index, parameter_node);
            if parameter.pointer != PointerKind::Value {
                if let Some(hint) = semantic_hints.get(&index) {
                    if parameter.direction == Direction::Unknown {
                        parameter.direction = hint.direction.unwrap_or(Direction::Unknown);
                    }
                    if let Some(nullable) = hint.nullable {
                        parameter.nullable = Some(nullable);
                    }
                }
            }
            parameter
        })
        .collect::<Vec<_>>();
    let abi = Abi {
        calling_convention: if request.family == VendorFamily::Cuda || qualified.contains("stdcall")
        {
            "system".to_owned()
        } else {
            "C".to_owned()
        },
        return_type: normalize_c_type(return_type),
        parameters,
    };
    let signature = normalized_callable(name, &abi);
    let mut entry = base_entry(
        request,
        ItemKind::Function,
        name,
        signature,
        Some(abi),
        None,
    );
    if ast_has_kind(node, "DeprecatedAttr") {
        entry.deprecated = Some(request.source_version.clone());
    }
    entries.insert((ItemKind::Function, name.to_owned()), entry);
}

#[derive(Clone, Copy, Default)]
struct ParameterSemanticHint {
    direction: Option<Direction>,
    nullable: Option<bool>,
}

fn documented_parameter_semantics(node: &Value) -> BTreeMap<usize, ParameterSemanticHint> {
    fn visit(node: &Value, output: &mut BTreeMap<usize, ParameterSemanticHint>) {
        if node.get("kind").and_then(Value::as_str) == Some("ParamCommandComment") {
            if let Some(index) = node
                .get("paramIdx")
                .and_then(Value::as_u64)
                .and_then(|index| usize::try_from(index).ok())
            {
                let explicit = node.get("explicit").and_then(Value::as_bool) == Some(true);
                let description = comment_text(node).to_ascii_lowercase();
                let direction = if explicit {
                    match node.get("direction").and_then(Value::as_str) {
                        Some("in") => Some(Direction::In),
                        Some("out") => Some(Direction::Out),
                        Some("inout" | "in,out") => Some(Direction::InOut),
                        _ => None,
                    }
                } else {
                    documented_direction(&description)
                };
                output.insert(
                    index,
                    ParameterSemanticHint {
                        direction,
                        nullable: documented_nullability(&description),
                    },
                );
            }
        }
        if let Some(children) = node.get("inner").and_then(Value::as_array) {
            for child in children {
                visit(child, output);
            }
        }
    }
    let mut output = BTreeMap::new();
    visit(node, &mut output);

    let paragraphs = comment_paragraphs(node);
    let names = node
        .get("inner")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|child| child.get("kind").and_then(Value::as_str) == Some("ParmVarDecl"))
        .enumerate()
        .filter_map(|(index, parameter)| {
            parameter
                .get("name")
                .and_then(Value::as_str)
                .map(|name| (index, name.to_ascii_lowercase()))
        });
    for (index, name) in names {
        let hint = output.entry(index).or_default();
        if hint.nullable.is_some() {
            continue;
        }
        for paragraph in &paragraphs {
            if !paragraph.contains(&name) || !paragraph.contains("null") {
                continue;
            }
            if let Some(nullable) = documented_named_nullability(paragraph, &name) {
                hint.nullable = Some(nullable);
                break;
            }
        }
    }
    output
}

fn documented_direction(description: &str) -> Option<Direction> {
    // Clang preserves the conventional Doxygen ` - description` separator in the
    // paragraph text. It is punctuation, not semantic evidence.
    let normalized = description
        .trim()
        .trim_start_matches('-')
        .trim()
        .to_ascii_lowercase();
    let description = normalized.as_str();
    if description.contains("input and output")
        || description.contains("input/output")
        || description.contains("in/out")
        || description.contains("updated with")
        || description.contains("would be or should be created")
    {
        return Some(Direction::InOut);
    }

    let output_prefix = [
        "return ",
        "returned ",
        "returns ",
        "retrieved ",
        "output ",
        "optionally returns ",
        "location to store ",
        "location to return ",
        "pointer to store ",
        "pointer to return ",
        "pointer used to return ",
        "pointer for the output ",
        "number of kernels found ",
        "number of functions found ",
    ];
    if output_prefix
        .iter()
        .any(|prefix| description.starts_with(prefix))
        || description.contains(" where the result")
        || description.contains("remainder is placed in here")
    {
        return Some(Direction::Out);
    }

    let input_prefix = [
        "optional checkpoint operation arguments",
        "optional lock operation arguments",
        "optional restore operation arguments",
        "optional unlock operation arguments",
        "user data ",
        "user-specified data ",
        "user specified data ",
        "a generic pointer to user data",
        "array of pointers to be ",
        "array of indices to ",
        "array of locations to ",
        "array of sizes for memory ",
        "array of pointers to kernel parameters",
        "an array of attributes to query",
        "parameters to copy",
        "updated parameters to set",
        "parameters for the node",
        "options for ",
        "option values for ",
        "extra options",
        "operation to perform",
        "attributes for the copy",
        "resources to map ",
        "array of resources to be included ",
        "starting address of memory region ",
        "pointer to memory to free",
        "host pointer",
        "the pointer to pass ",
        "pointer to value to set",
        "the pool being modified",
        "the location accessing the pool",
        "instantiation parameters",
        "cig capture parameters",
        "controls the interaction ",
    ];
    input_prefix
        .iter()
        .any(|prefix| description.starts_with(prefix))
        .then_some(Direction::In)
}

fn documented_named_nullability(description: &str, name: &str) -> Option<bool> {
    let named_null_condition = description.contains(&format!("if {name} is null"));
    if description.contains("must not be null")
        || description.contains("cannot be null")
        || (named_null_condition
            && (description.contains("error") || description.contains("invalid")))
    {
        Some(false)
    } else if description.contains("may be null")
        || description.contains("can be null")
        || description.contains("optional")
    {
        Some(true)
    } else {
        None
    }
}

fn documented_nullability(description: &str) -> Option<bool> {
    if description.contains("must not be null") || description.contains("cannot be null") {
        Some(false)
    } else if description.contains("may be null")
        || description.contains("can be null")
        || description.contains("optional")
    {
        Some(true)
    } else {
        None
    }
}

fn comment_paragraphs(node: &Value) -> Vec<String> {
    fn visit(node: &Value, paragraphs: &mut Vec<String>) {
        if node.get("kind").and_then(Value::as_str) == Some("ParagraphComment") {
            let text = comment_text(node).to_ascii_lowercase();
            if !text.trim().is_empty() {
                paragraphs.push(text);
            }
        }
        if let Some(children) = node.get("inner").and_then(Value::as_array) {
            for child in children {
                visit(child, paragraphs);
            }
        }
    }
    let mut paragraphs = Vec::new();
    visit(node, &mut paragraphs);
    paragraphs
}

fn comment_text(node: &Value) -> String {
    fn visit(node: &Value, parts: &mut Vec<String>) {
        if let Some(text) = node.get("text").and_then(Value::as_str) {
            parts.push(text.to_owned());
        }
        if let Some(arguments) = node.get("args").and_then(Value::as_array) {
            parts.extend(
                arguments
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned),
            );
        }
        if let Some(children) = node.get("inner").and_then(Value::as_array) {
            for child in children {
                visit(child, parts);
            }
        }
    }
    let mut parts = Vec::new();
    visit(node, &mut parts);
    parts
        .join(" ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn collect_typedef(
    node: &Value,
    request: &HeaderExtractRequest,
    entries: &mut BTreeMap<(ItemKind, String), Entry>,
) {
    let Some(name) = node.get("name").and_then(Value::as_str) else {
        return;
    };
    if !is_vendor_type(request.family, name) || node.get("isImplicit") == Some(&Value::Bool(true)) {
        return;
    }
    let qualified = node
        .pointer("/type/qualType")
        .and_then(Value::as_str)
        .unwrap_or("opaque");
    let (kind, abi) =
        if ast_has_kind(node, "PointerType") && ast_has_kind(node, "FunctionProtoType") {
            (ItemKind::Callback, callback_abi(qualified, request.family))
        } else if qualified.contains('*') && name.to_ascii_lowercase().ends_with("_t") {
            (ItemKind::OpaqueHandle, None)
        } else {
            (ItemKind::Type, None)
        };
    let signature = if let Some(abi) = &abi {
        normalized_callable(name, abi)
    } else {
        format!("type {name}={}", normalize_c_type(qualified))
    };
    entries.insert(
        (kind, name.to_owned()),
        base_entry(request, kind, name, signature, abi, None),
    );
}

fn collect_record(
    node: &Value,
    request: &HeaderExtractRequest,
    entries: &mut BTreeMap<(ItemKind, String), Entry>,
) {
    let Some(name) = node.get("name").and_then(Value::as_str) else {
        return;
    };
    if !is_vendor_type(request.family, name)
        || node.get("completeDefinition") != Some(&Value::Bool(true))
    {
        return;
    }
    let kind = if node.get("tagUsed").and_then(Value::as_str) == Some("union") {
        ItemKind::Union
    } else {
        ItemKind::Struct
    };
    let fields = node
        .get("inner")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|child| child.get("kind").and_then(Value::as_str) == Some("FieldDecl"))
        .map(|field| {
            format!(
                "{}:{}",
                field
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("anonymous"),
                normalize_c_type(
                    field
                        .pointer("/type/qualType")
                        .and_then(Value::as_str)
                        .unwrap_or("opaque")
                )
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let label = if kind == ItemKind::Union {
        "union"
    } else {
        "struct"
    };
    let signature = format!("{label} {name}:{{{fields}}}");
    entries.insert(
        (kind, name.to_owned()),
        base_entry(request, kind, name, signature, None, None),
    );
}

fn collect_enum(
    node: &Value,
    request: &HeaderExtractRequest,
    entries: &mut BTreeMap<(ItemKind, String), Entry>,
) {
    if let Some(name) = node.get("name").and_then(Value::as_str) {
        if is_vendor_type(request.family, name) {
            let signature = format!("enum {name}");
            entries.insert(
                (ItemKind::Type, name.to_owned()),
                base_entry(request, ItemKind::Type, name, signature, None, None),
            );
        }
    }
    let Some(children) = node.get("inner").and_then(Value::as_array) else {
        return;
    };
    let mut previous_numeric_value = None;
    for variant in children {
        if variant.get("kind").and_then(Value::as_str) != Some("EnumConstantDecl") {
            continue;
        }
        let Some(name) = variant.get("name").and_then(Value::as_str) else {
            continue;
        };
        if !is_vendor_constant(request.family, name) {
            continue;
        }
        let explicit_value = find_ast_value(variant);
        let numeric_value = explicit_value
            .as_deref()
            .and_then(|value| value.parse::<i64>().ok())
            .or_else(|| {
                explicit_value
                    .is_none()
                    .then(|| previous_numeric_value?.checked_add(1))
                    .flatten()
            });
        let value = explicit_value.unwrap_or_else(|| "implicit".to_owned());
        let signature = format!("enum-value {name}={value}");
        let mut entry = base_entry(request, ItemKind::EnumValue, name, signature, None, None);
        entry.numeric_value = numeric_value;
        entries.insert(
            (ItemKind::EnumValue, name.to_owned()),
            entry,
        );
        previous_numeric_value = numeric_value;
    }
}

fn collect_variable(
    node: &Value,
    request: &HeaderExtractRequest,
    entries: &mut BTreeMap<(ItemKind, String), Entry>,
) {
    let Some(name) = node.get("name").and_then(Value::as_str) else {
        return;
    };
    if !is_vendor_constant(request.family, name) || node.get("storageClass").is_some() {
        return;
    }
    let qualified = node
        .pointer("/type/qualType")
        .and_then(Value::as_str)
        .unwrap_or("opaque");
    let signature = format!("constant {name}:{}", normalize_c_type(qualified));
    entries.insert(
        (ItemKind::Constant, name.to_owned()),
        base_entry(request, ItemKind::Constant, name, signature, None, None),
    );
}

fn parameter_from_ast(index: usize, node: &Value) -> Parameter {
    let name = node
        .get("name")
        .and_then(Value::as_str)
        .map_or_else(|| format!("arg{index}"), str::to_owned);
    let qualified = node
        .pointer("/type/qualType")
        .and_then(Value::as_str)
        .unwrap_or("opaque");
    let desugared = node
        .pointer("/type/desugaredQualType")
        .and_then(Value::as_str)
        .unwrap_or(qualified);
    let source_star_count = qualified
        .chars()
        .filter(|character| *character == '*')
        .count();
    let semantic_star_count = desugared
        .chars()
        .filter(|character| *character == '*')
        .count();
    let alias_pointer = source_star_count == 0 && semantic_star_count > 0;
    let pointer = if desugared.contains("(*)") || desugared.contains("(* ") {
        PointerKind::Callback
    } else if semantic_star_count == 0 {
        PointerKind::Value
    } else if semantic_star_count == 1 && desugared.trim_start().starts_with("const ") {
        PointerKind::Const
    } else {
        PointerKind::Mut
    };
    let direction = match (pointer, alias_pointer) {
        (_, true) | (PointerKind::Value | PointerKind::Const | PointerKind::Callback, false) => {
            Direction::In
        }
        (PointerKind::Mut, false) => Direction::Unknown,
    };
    Parameter {
        name,
        r#type: normalize_c_type(qualified),
        pointer,
        direction,
        nullable: (pointer == PointerKind::Value).then_some(false),
    }
}

fn callback_abi(qualified: &str, family: VendorFamily) -> Option<Abi> {
    let callable = qualified
        .split_once(" __attribute__")
        .map_or(qualified, |(callable, _)| callable);
    let open = callable.find("(*")?;
    let close = callable.rfind(')')?;
    let return_type = normalize_c_type(&callable[..open]);
    let arguments_start = callable[open..].find(")(")? + open + 2;
    let arguments = &callable[arguments_start..close];
    let parameters = if arguments.trim().is_empty() || arguments.trim() == "void" {
        Vec::new()
    } else {
        split_c_arguments(arguments)
            .into_iter()
            .enumerate()
            .map(|(index, argument)| {
                let synthetic = serde_json::json!({
                    "name": format!("arg{index}"),
                    "type": { "qualType": argument }
                });
                parameter_from_ast(index, &synthetic)
            })
            .collect()
    };
    Some(Abi {
        calling_convention: if family == VendorFamily::Cuda || qualified.contains("stdcall") {
            "system".to_owned()
        } else {
            "C".to_owned()
        },
        return_type,
        parameters,
    })
}

fn split_c_arguments(arguments: &str) -> Vec<String> {
    let mut depth = 0_i32;
    let mut start = 0_usize;
    let mut output = Vec::new();
    for (index, character) in arguments.char_indices() {
        match character {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            ',' if depth == 0 => {
                output.push(normalize_c_type(&arguments[start..index]));
                start = index + 1;
            }
            _ => {}
        }
    }
    output.push(normalize_c_type(&arguments[start..]));
    output
}

fn base_entry(
    request: &HeaderExtractRequest,
    kind: ItemKind,
    name: &str,
    normalized_signature: String,
    abi: Option<Abi>,
    alias_of: Option<String>,
) -> Entry {
    Entry {
        kind,
        name: name.to_owned(),
        signature_hash: sha256(&normalized_signature),
        normalized_signature,
        numeric_value: None,
        abi,
        aliases: Vec::new(),
        alias_of,
        platforms: platforms_for_name(request, name),
        introduced: None,
        deprecated: None,
        layouts: Vec::new(),
        provenance: format!("{}#{name}", request.provenance),
    }
}

fn platforms_for_name(request: &HeaderExtractRequest, name: &str) -> Vec<String> {
    if request.family != VendorFamily::Cuda {
        return request.platforms.clone();
    }
    let uppercase = name.to_ascii_uppercase();
    if uppercase.contains("D3D9") || uppercase.contains("D3D10") || uppercase.contains("D3D11") {
        return request
            .platforms
            .iter()
            .filter(|platform| platform.contains("windows"))
            .cloned()
            .collect();
    }
    if uppercase.contains("VDPAU") || uppercase.contains("EGL") {
        return request
            .platforms
            .iter()
            .filter(|platform| platform.contains("linux"))
            .cloned()
            .collect();
    }
    request.platforms.clone()
}

fn reconcile_aliases(entries: &mut BTreeMap<(ItemKind, String), Entry>) {
    let aliases = entries
        .values()
        .filter_map(|entry| {
            entry
                .alias_of
                .as_ref()
                .map(|target| (target.clone(), entry.name.clone()))
        })
        .collect::<Vec<_>>();
    for (target, alias) in aliases {
        if let Some(entry) = entries.get_mut(&(ItemKind::Function, target)) {
            entry.aliases.push(alias);
            entry.aliases.sort();
            entry.aliases.dedup();
        }
    }
}

fn merge_entries(
    destination: &mut BTreeMap<(ItemKind, String), Entry>,
    source: BTreeMap<(ItemKind, String), Entry>,
) -> Result<(), HeaderExtractionError> {
    for (key, mut entry) in source {
        if let Some(existing) = destination.get_mut(&key) {
            if existing.normalized_signature != entry.normalized_signature
                || existing.signature_hash != entry.signature_hash
                || existing.numeric_value != entry.numeric_value
            {
                return Err(HeaderExtractionError::PlatformDrift {
                    name: entry.name,
                    left: existing.platforms.join(","),
                    right: entry.platforms.join(","),
                    detail: "normalized signature or signature hash changed".to_owned(),
                });
            }
            existing.platforms.append(&mut entry.platforms);
            existing.platforms.sort();
            existing.platforms.dedup();
            existing.aliases.append(&mut entry.aliases);
            existing.aliases.sort();
            existing.aliases.dedup();
            for layout in entry.layouts {
                if let Some(prior) = existing
                    .layouts
                    .iter()
                    .find(|prior| prior.target == layout.target)
                {
                    if prior.size != layout.size
                        || prior.alignment != layout.alignment
                        || prior.field_offsets != layout.field_offsets
                    {
                        return Err(HeaderExtractionError::PlatformDrift {
                            name: existing.name.clone(),
                            left: prior.target.clone(),
                            right: layout.target,
                            detail: "record layout changed for the same target".to_owned(),
                        });
                    }
                } else {
                    existing.layouts.push(layout);
                }
            }
            existing
                .layouts
                .sort_by(|left, right| left.target.cmp(&right.target));
        } else {
            destination.insert(key, entry);
        }
    }
    Ok(())
}

fn normalized_callable(name: &str, abi: &Abi) -> String {
    let parameters = abi
        .parameters
        .iter()
        .map(|parameter| {
            format!(
                "{}:{}:{:?}:{:?}:nullable={}",
                parameter.name,
                parameter.r#type,
                parameter.pointer,
                parameter.direction,
                parameter
                    .nullable
                    .map_or_else(|| "unknown".to_owned(), |value| value.to_string())
            )
            .to_lowercase()
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "fn {name}[abi={}]({parameters})->{}",
        abi.calling_convention, abi.return_type
    )
}

fn normalize_c_type(value: &str) -> String {
    erase_clang_source_locations(value)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .replace(" *", "*")
        .replace("[ ", "[")
        .replace(" ]", "]")
        .replace(" ,", ",")
        .trim()
        .to_owned()
}

fn erase_clang_source_locations(value: &str) -> String {
    let mut output = value.to_owned();
    while let Some(start) = output.find("(unnamed ") {
        let Some(end_offset) = output[start..].find(')') else {
            break;
        };
        let end = start + end_offset;
        let descriptor = &output[start + "(unnamed ".len()..end];
        let kind = descriptor
            .split_once(" at ")
            .map_or(descriptor, |(kind, _)| kind);
        let replacement = format!("(anonymous {kind})");
        output.replace_range(start..=end, &replacement);
    }
    while let Some(start) = output.find("(anonymous at ") {
        let Some(end_offset) = output[start..].find(')') else {
            break;
        };
        let end = start + end_offset;
        output.replace_range(start..=end, "(anonymous)");
    }
    output
}

fn is_vendor_function(family: VendorFamily, name: &str) -> bool {
    match family {
        VendorFamily::Cuda => name.strip_prefix("cu").is_some_and(|suffix| {
            suffix
                .chars()
                .next()
                .is_some_and(|character| character.is_ascii_uppercase())
        }),
        VendorFamily::Hip => {
            name.strip_prefix("hip").is_some_and(|suffix| {
                suffix
                    .chars()
                    .next()
                    .is_some_and(|character| character.is_ascii_uppercase())
            }) && !name.starts_with("hiprtc")
        }
    }
}

fn is_vendor_type(family: VendorFamily, name: &str) -> bool {
    match family {
        VendorFamily::Cuda => {
            name.starts_with("CU")
                && name
                    .chars()
                    .nth(2)
                    .is_some_and(|character| character == '_' || character.is_ascii_alphabetic())
        }
        VendorFamily::Hip => name.strip_prefix("hip").is_some_and(|suffix| {
            suffix
                .chars()
                .next()
                .is_some_and(|character| character.is_ascii_uppercase())
        }),
    }
}

fn is_vendor_constant(family: VendorFamily, name: &str) -> bool {
    match family {
        VendorFamily::Cuda => {
            name.starts_with("CU_") || name.starts_with("CUDA_") || is_vendor_function(family, name)
        }
        VendorFamily::Hip => name.starts_with("HIP_") || is_vendor_function(family, name),
    }
}

fn looks_like_constant_expression(replacement: &str) -> bool {
    !replacement.is_empty()
        && replacement.len() < 256
        && !replacement.contains("__")
        && replacement.chars().all(|character| {
            character.is_ascii_alphanumeric() || "_()+-*/~<>&|.xXuUlL ".contains(character)
        })
}

fn ast_has_kind(node: &Value, sought: &str) -> bool {
    if node.get("kind").and_then(Value::as_str) == Some(sought) {
        return true;
    }
    node.get("inner")
        .and_then(Value::as_array)
        .is_some_and(|children| children.iter().any(|child| ast_has_kind(child, sought)))
}

fn find_ast_value(node: &Value) -> Option<String> {
    if let Some(value) = node.get("value").and_then(Value::as_str) {
        return Some(value.to_owned());
    }
    node.get("inner")
        .and_then(Value::as_array)
        .and_then(|children| children.iter().find_map(find_ast_value))
}

#[cfg(test)]
mod tests {
    use super::{
        CudaProcAddressVariant, Direction, VendorFamily, documented_direction,
        documented_named_nullability, documented_nullability, is_vendor_function, normalize_c_type,
        parse_cuda_proc_typedef_name, parse_record_layout, split_c_arguments,
    };

    #[test]
    fn vendor_names_exclude_runtime_compilers() {
        assert!(is_vendor_function(VendorFamily::Cuda, "cuInit"));
        assert!(!is_vendor_function(VendorFamily::Cuda, "cudaMalloc"));
        assert!(is_vendor_function(VendorFamily::Hip, "hipInit"));
        assert!(!is_vendor_function(VendorFamily::Hip, "hiprtcVersion"));
    }

    #[test]
    fn argument_split_preserves_nested_callbacks() {
        assert_eq!(
            split_c_arguments("int, void (*)(int, int), const char *"),
            ["int", "void (*)(int, int)", "const char*"]
        );
    }

    #[test]
    fn record_layout_distinguishes_anonymous_members_from_named_anonymous_types() {
        let block = r"
         0 | struct sample
         0 |   union sample::(anonymous at sample.h:2:3)
         0 |     unsigned int first
         8 |   struct sample::(unnamed at sample.h:5:3) named
         8 |     unsigned int nested
           | [sizeof=16, align=8]
";
        let (_, _, layout) = parse_record_layout(
            block,
            "x86_64-unknown-linux-gnu",
            "https://example.invalid/sample.h",
        )
        .expect("the complete fixture contains a parseable record layout");
        assert_eq!(layout.field_offsets.get("anonymous"), Some(&0));
        assert_eq!(layout.field_offsets.get("named"), Some(&8));
    }

    #[test]
    fn normalized_types_never_embed_extractor_paths() {
        assert_eq!(
            normalize_c_type(
                "union (unnamed union at C:\\work\\cuda.h:12:3) struct_name::(anonymous at /tmp/hip.h:4:2)"
            ),
            "union (anonymous union) struct_name::(anonymous)"
        );
    }

    #[test]
    fn cuda_proc_typedef_name_preserves_stream_variant() {
        assert_eq!(
            parse_cuda_proc_typedef_name("PFN_cuMemcpyHtoDAsync_v3020_ptsz"),
            Some((
                "cuMemcpyHtoDAsync".to_owned(),
                3020,
                CudaProcAddressVariant::Ptsz
            ))
        );
    }

    #[test]
    fn generic_invalid_value_text_does_not_invent_non_nullability() {
        assert_eq!(
            documented_nullability(
                "returns invalid value when both kernelparams and extra are non-null"
            ),
            None
        );
        assert_eq!(
            documented_nullability("the callback must not be null"),
            Some(false)
        );
        assert_eq!(
            documented_named_nullability(
                "this function returns an invalid-value error if driverversion is null",
                "driverversion"
            ),
            Some(false)
        );
        assert_eq!(
            documented_named_nullability(
                "an error is returned if both kernelparams and extra are non-null",
                "kernelparams"
            ),
            None
        );
    }

    #[test]
    fn documented_direction_requires_semantic_language() {
        assert_eq!(
            documented_direction("location to store the callback handle"),
            Some(Direction::Out)
        );
        assert_eq!(
            documented_direction(" - Returned array"),
            Some(Direction::Out)
        );
        assert_eq!(
            documented_direction("array of pointers to be discarded"),
            Some(Direction::In)
        );
        assert_eq!(
            documented_direction("number of groups that would be or should be created"),
            Some(Direction::InOut)
        );
        assert_eq!(documented_direction("resource descriptor"), None);
        assert_eq!(documented_direction("pointer to value"), None);
    }
}
