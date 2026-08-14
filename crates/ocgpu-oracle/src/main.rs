// SPDX-License-Identifier: CC0-1.0

//! Maintainer CLI for reproducible oracle extraction and validation.

use ocgpu_oracle::{
    CudaProcAddressCatalog, ExtractRequest, HeaderExtractRequest, HeaderSemanticEvidence,
    Inventory, SourceArtifact, VendorFamily, build_report, build_seed_catalog,
    build_semantic_catalog, build_vendor_function_union, extract_cuda_proc_address_catalog,
    extract_header_inventory, extract_rust_inventory, read_inputs, render_markdown,
    repository_root, validate_repository,
};
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    if let Err(error) = run() {
        eprintln!("ocgpu-oracle: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args().skip(1).collect::<Vec<_>>();
    let root = take_option(&mut arguments, "--root").map_or_else(repository_root, PathBuf::from);
    let Some(command) = arguments.first().map(String::as_str) else {
        return Err(usage().into());
    };
    match command {
        "validate" if arguments.len() == 1 => {
            let summary = validate_repository(&root)?;
            println!(
                "validated {} inventories, {} entries, {} decisions, {} manifest entries, {} exports, and HIP profiles {}/common {}/attributes {}",
                summary.inventories,
                summary.entries,
                summary.decisions,
                summary.manifest_entries,
                summary.exports,
                summary.hip_runtime_profiles,
                summary.hip_common_functions,
                summary.hip_device_attributes
            );
        }
        "report" => {
            let check = take_flag(&mut arguments, "--check");
            if arguments.len() != 1 {
                return Err(usage().into());
            }
            if check {
                validate_repository(&root)?;
            }
            report(&root, check)?;
        }
        "classify" => {
            let check = take_flag(&mut arguments, "--check");
            if arguments.len() != 1 {
                return Err(usage().into());
            }
            classify(&root, check)?;
        }
        "semantics" => {
            let check = take_flag(&mut arguments, "--check");
            if arguments.len() != 1 {
                return Err(usage().into());
            }
            semantics(&root, check)?;
        }
        "vendor-union" => {
            let check = take_flag(&mut arguments, "--check");
            if arguments.len() != 1 {
                return Err(usage().into());
            }
            vendor_union(&root, check)?;
        }
        "check" if arguments.len() == 1 => {
            vendor_union(&root, true)?;
            semantics(&root, true)?;
            classify(&root, true)?;
            let summary = validate_repository(&root)?;
            report(&root, true)?;
            println!(
                "oracle check passed: {} entries, {} explicit decisions, and HIP profiles {}/common {}/attributes {}",
                summary.entries,
                summary.decisions,
                summary.hip_runtime_profiles,
                summary.hip_common_functions,
                summary.hip_device_attributes
            );
        }
        "extract-rust" => extract(&root, &arguments[1..])?,
        "extract-vendor" => extract_vendor(&arguments[1..])?,
        "extract-cuda-proc-typedefs" => extract_cuda_proc_typedefs(&arguments[1..])?,
        _ => return Err(usage().into()),
    }
    Ok(())
}

fn vendor_union(root: &Path, check: bool) -> Result<(), Box<dyn Error>> {
    let inventories = load_inventories(root)?;
    let catalog: CudaProcAddressCatalog = serde_json::from_str(&fs::read_to_string(
        root.join("oracle/vendor/cuda/13.3-13030-proc-address.json"),
    )?)?;
    let union = build_vendor_function_union(&inventories, &catalog);
    let bytes = format!("{}\n", serde_json::to_string_pretty(&union)?).into_bytes();
    let path = root.join("oracle/vendor/function-union.json");
    if check {
        check_file(&path, &bytes)?;
    } else {
        fs::create_dir_all(root.join("oracle/vendor"))?;
        fs::write(&path, bytes)?;
        println!(
            "wrote {} ({} unique vendor callables)",
            path.display(),
            union.functions.len()
        );
    }
    Ok(())
}

fn extract_cuda_proc_typedefs(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let mut remaining = arguments.to_vec();
    let header = PathBuf::from(required_option(&mut remaining, "--header")?);
    let source_version = required_option(&mut remaining, "--source-version")?;
    let provenance = required_option(&mut remaining, "--provenance")?;
    let output = PathBuf::from(required_option(&mut remaining, "--output")?);
    let include_directories = take_all_options(&mut remaining, "--include")
        .into_iter()
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    let source_artifacts = take_all_options(&mut remaining, "--artifact")
        .into_iter()
        .map(|value| parse_source_artifact(&value))
        .collect::<Result<Vec<_>, _>>()?;
    if source_artifacts.is_empty() {
        return Err(
            "extract-cuda-proc-typedefs requires archive and exact header --artifact values".into(),
        );
    }
    if !remaining.is_empty() {
        return Err(usage().into());
    }
    let includes = if include_directories.is_empty() {
        vec![
            header
                .parent()
                .ok_or("CUDA typedef header must have a parent directory")?
                .to_path_buf(),
        ]
    } else {
        include_directories
    };
    let catalog = extract_cuda_proc_address_catalog(&HeaderExtractRequest {
        family: VendorFamily::Cuda,
        header,
        include_directories: includes,
        inventory_id: "cuda-proc-address-13.3-13030".to_owned(),
        source_name: "NVIDIA CUDA Driver API cudaTypedefs.h".to_owned(),
        source_version,
        provenance,
        source_artifacts,
        platforms: vec!["x86_64-pc-windows-msvc".to_owned()],
        semantic_evidence: None,
    })?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(
        &output,
        format!("{}\n", serde_json::to_string_pretty(&catalog)?),
    )?;
    println!(
        "wrote {} ({} CUDA proc-address typedefs)",
        output.display(),
        catalog.typedefs.len()
    );
    Ok(())
}

fn semantics(root: &Path, check: bool) -> Result<(), Box<dyn Error>> {
    let inventories = load_inventories(root)?;
    let manifest_path = root.join("api/ocgpu-api.toml");
    let manifest = toml::from_str(&fs::read_to_string(&manifest_path)?)?;
    let catalog = build_semantic_catalog(&inventories, &manifest)?;
    let bytes = format!("{}\n", serde_json::to_string_pretty(&catalog)?).into_bytes();
    let path = root.join("oracle/semantic-overrides.json");
    if check {
        check_file(&path, &bytes)?;
    } else {
        fs::write(&path, bytes)?;
        println!("wrote {}", path.display());
    }
    Ok(())
}

fn classify(root: &Path, check: bool) -> Result<(), Box<dyn Error>> {
    let inventories = load_inventories(root)?;
    let generated_path = root.join("crates/ocgpu-codegen/generated/api-inventory.json");
    let generated = serde_json::from_str(&fs::read_to_string(generated_path)?)?;
    let catalog = build_seed_catalog(&inventories, &generated);
    let bytes = format!("{}\n", serde_json::to_string_pretty(&catalog)?).into_bytes();
    let path = root.join("coverage/classifications.json");
    if check {
        check_file(&path, &bytes)?;
    } else {
        fs::create_dir_all(root.join("coverage"))?;
        fs::write(&path, bytes)?;
        println!("wrote {}", path.display());
    }
    Ok(())
}

fn load_inventories(root: &Path) -> Result<Vec<Inventory>, Box<dyn Error>> {
    [
        "oracle/vendor/cuda/13.3-13030.json",
        "oracle/vendor/hip/general-7.14.60850.json",
        "oracle/vendor/hip/windows-7.2.0.json",
        "oracle/rust/cudarc-0.19.9.json",
        "oracle/rust/rocmrc-0.5.0.json",
    ]
    .iter()
    .map(|path| {
        let path = root.join(path);
        let source = fs::read_to_string(&path)?;
        serde_json::from_str::<Inventory>(&source).map_err(Into::into)
    })
    .collect()
}

fn extract_vendor(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    if arguments.is_empty() {
        return Err(usage().into());
    }
    let mut remaining = arguments.to_vec();
    let family = match remaining.remove(0).as_str() {
        "cuda" => VendorFamily::Cuda,
        "hip" => VendorFamily::Hip,
        _ => return Err("extract-vendor family must be cuda or hip".into()),
    };
    let header = required_option(&mut remaining, "--header")?;
    let inventory_id = required_option(&mut remaining, "--inventory-id")?;
    let source_name = required_option(&mut remaining, "--source-name")?;
    let source_version = required_option(&mut remaining, "--source-version")?;
    let provenance = required_option(&mut remaining, "--provenance")?;
    let platforms = required_option(&mut remaining, "--platforms")?
        .split(',')
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let output = PathBuf::from(required_option(&mut remaining, "--output")?);
    let include_directories = take_all_options(&mut remaining, "--include")
        .into_iter()
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    let semantic_header = take_option(&mut remaining, "--semantic-header").map(PathBuf::from);
    let semantic_provenance = take_option(&mut remaining, "--semantic-provenance");
    let semantic_include_directories = take_all_options(&mut remaining, "--semantic-include")
        .into_iter()
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    let source_artifacts = take_all_options(&mut remaining, "--artifact")
        .into_iter()
        .map(|value| parse_source_artifact(&value))
        .collect::<Result<Vec<_>, _>>()?;
    if source_artifacts.is_empty() {
        return Err(
            "extract-vendor requires at least one --artifact ROLE|URL|SHA256|REVISION|PATH".into(),
        );
    }
    if !remaining.is_empty() {
        return Err(usage().into());
    }
    let header = PathBuf::from(header);
    let includes = if include_directories.is_empty() {
        vec![
            header
                .parent()
                .ok_or("vendor header must have a parent directory")?
                .to_path_buf(),
        ]
    } else {
        include_directories
    };
    let request = HeaderExtractRequest {
        family,
        header,
        include_directories: includes,
        inventory_id,
        source_name,
        source_version,
        provenance,
        source_artifacts,
        platforms,
        semantic_evidence: match (semantic_header, semantic_provenance) {
            (Some(header), Some(provenance)) if !semantic_include_directories.is_empty() => {
                Some(HeaderSemanticEvidence {
                    header,
                    include_directories: semantic_include_directories,
                    provenance,
                })
            }
            (None, None) if semantic_include_directories.is_empty() => None,
            _ => {
                return Err("semantic evidence requires --semantic-header, --semantic-provenance, and at least one --semantic-include".into());
            }
        },
    };
    let inventory = extract_header_inventory(&request)?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(
        &output,
        format!("{}\n", serde_json::to_string_pretty(&inventory)?),
    )?;
    println!("wrote {}", output.display());
    Ok(())
}

fn report(root: &Path, check: bool) -> Result<(), Box<dyn Error>> {
    let (inventories, catalog) = read_inputs(root)?;
    let report = build_report(&inventories, &catalog);
    let canonical_json = serde_json::to_vec(&report)?;
    let mut json = canonical_json.clone();
    json.push(b'\n');
    let compressed = miniz_oxide::deflate::compress_to_vec(&canonical_json, 10);
    let markdown = render_markdown(&report, &catalog);
    let json_path = root.join("coverage/coverage.json");
    let compressed_path = root.join("coverage/coverage.json.deflate");
    let markdown_path = root.join("coverage/coverage.md");
    if check {
        check_file(&json_path, &json)?;
        check_file(&compressed_path, &compressed)?;
        check_file(&markdown_path, markdown.as_bytes())?;
    } else {
        fs::create_dir_all(root.join("coverage"))?;
        fs::write(&json_path, &json)?;
        fs::write(&compressed_path, compressed)?;
        fs::write(&markdown_path, markdown)?;
        println!(
            "wrote {}, {}, and {}",
            json_path.display(),
            compressed_path.display(),
            markdown_path.display()
        );
    }
    Ok(())
}

fn extract(root: &Path, arguments: &[String]) -> Result<(), Box<dyn Error>> {
    if arguments.is_empty() {
        return Err(usage().into());
    }
    let mut remaining = arguments.to_vec();
    let package = remaining.remove(0);
    let output = take_option(&mut remaining, "--output").map(PathBuf::from);
    if !remaining.is_empty() {
        return Err(usage().into());
    }
    let request = match package.as_str() {
        "cudarc" => ExtractRequest {
            workspace_root: root.to_path_buf(),
            package_name: "cudarc".to_owned(),
            package_version: "0.19.9".to_owned(),
            module_path: PathBuf::from("src/driver/sys/mod.rs"),
            inventory_id: "cudarc-0.19.9".to_owned(),
            source_name: "cudarc::driver::sys".to_owned(),
            platforms: vec![
                "aarch64-unknown-linux-gnu".to_owned(),
                "x86_64-pc-windows-msvc".to_owned(),
                "x86_64-unknown-linux-gnu".to_owned(),
            ],
            provenance: "crates.io:cudarc@0.19.9/src/driver/sys/mod.rs".to_owned(),
            source_artifacts: vec![SourceArtifact {
                role: "authoritative-crate-archive".to_owned(),
                url: "https://crates.io/api/v1/crates/cudarc/0.19.9/download".to_owned(),
                sha256: "sha256:804764d10e844da09765a7b2ca9641a0851523d1702efb0d7299d73e31b86e80"
                    .to_owned(),
                revision: "0.19.9".to_owned(),
                path: "src/driver/sys/mod.rs and recursively referenced modules".to_owned(),
            }],
        },
        "rocmrc" => ExtractRequest {
            workspace_root: root.to_path_buf(),
            package_name: "rocmrc".to_owned(),
            package_version: "0.5.0".to_owned(),
            module_path: PathBuf::from("src/hip/sys/mod.rs"),
            inventory_id: "rocmrc-0.5.0".to_owned(),
            source_name: "rocmrc::hip::sys".to_owned(),
            platforms: vec![
                "aarch64-unknown-linux-gnu".to_owned(),
                "x86_64-pc-windows-msvc".to_owned(),
                "x86_64-unknown-linux-gnu".to_owned(),
            ],
            provenance: "crates.io:rocmrc@0.5.0/src/hip/sys/mod.rs".to_owned(),
            source_artifacts: vec![SourceArtifact {
                role: "authoritative-crate-archive".to_owned(),
                url: "https://crates.io/api/v1/crates/rocmrc/0.5.0/download".to_owned(),
                sha256: "sha256:766806566f7d4fffd7f53fe065c86ae935a1296ff148395ca4cdf69d9a41cc18"
                    .to_owned(),
                revision: "0.5.0".to_owned(),
                path: "src/hip/sys/mod.rs and recursively referenced modules".to_owned(),
            }],
        },
        _ => return Err("extract-rust package must be cudarc or rocmrc".into()),
    };
    let inventory = extract_rust_inventory(&request)?;
    let json = format!("{}\n", serde_json::to_string_pretty(&inventory)?);
    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, json)?;
        println!("wrote {}", path.display());
    } else {
        print!("{json}");
    }
    Ok(())
}

fn check_file(path: &Path, expected: &[u8]) -> Result<(), Box<dyn Error>> {
    let actual = fs::read(path)?;
    if actual == expected {
        Ok(())
    } else {
        Err(format!(
            "{} is stale; regenerate its oracle artifact",
            path.display()
        )
        .into())
    }
}

fn take_option(arguments: &mut Vec<String>, name: &str) -> Option<String> {
    let index = arguments.iter().position(|argument| argument == name)?;
    if index + 1 >= arguments.len() {
        return None;
    }
    arguments.remove(index);
    Some(arguments.remove(index))
}

fn required_option(arguments: &mut Vec<String>, name: &str) -> Result<String, Box<dyn Error>> {
    take_option(arguments, name).ok_or_else(|| format!("missing required option {name}").into())
}

fn take_all_options(arguments: &mut Vec<String>, name: &str) -> Vec<String> {
    let mut values = Vec::new();
    while let Some(value) = take_option(arguments, name) {
        values.push(value);
    }
    values
}

fn take_flag(arguments: &mut Vec<String>, name: &str) -> bool {
    if let Some(index) = arguments.iter().position(|argument| argument == name) {
        arguments.remove(index);
        true
    } else {
        false
    }
}

fn parse_source_artifact(value: &str) -> Result<SourceArtifact, Box<dyn Error>> {
    let fields = value.split('|').collect::<Vec<_>>();
    if fields.len() != 5 || fields.iter().any(|field| field.trim().is_empty()) {
        return Err("--artifact must be ROLE|URL|SHA256|REVISION|PATH with no empty fields".into());
    }
    let hash = fields[2]
        .strip_prefix("sha256:")
        .unwrap_or(fields[2])
        .to_ascii_lowercase();
    if hash.len() != 64 || !hash.chars().all(|character| character.is_ascii_hexdigit()) {
        return Err("--artifact SHA256 must contain exactly 64 hexadecimal digits".into());
    }
    Ok(SourceArtifact {
        role: fields[0].to_owned(),
        url: fields[1].to_owned(),
        sha256: format!("sha256:{hash}"),
        revision: fields[3].to_owned(),
        path: fields[4].to_owned(),
    })
}

fn usage() -> &'static str {
    "usage: ocgpu-oracle [--root PATH] <validate|check|classify [--check]|semantics [--check]|vendor-union [--check]|report [--check]|extract-rust <cudarc|rocmrc> [--output PATH]|extract-cuda-proc-typedefs --header PATH --source-version VERSION --provenance URI --output PATH --artifact 'ROLE|URL|SHA256|REVISION|PATH' [--artifact ...] [--include PATH]...|extract-vendor <cuda|hip> --header PATH --inventory-id ID --source-name NAME --source-version VERSION --provenance URI --platforms CSV --output PATH --artifact 'ROLE|URL|SHA256|REVISION|PATH' [--artifact ...] [--include PATH]... [--semantic-header PATH --semantic-provenance URI --semantic-include PATH]...>"
}
