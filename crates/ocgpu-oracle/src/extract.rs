// SPDX-License-Identifier: CC0-1.0

use crate::hash::sha256;
use crate::model::{
    Abi, Direction, Entry, Inventory, ItemKind, Parameter, PointerKind, SourceArtifact,
};
use quote::ToTokens;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use syn::{
    Attribute, Expr, ExprLit, File, FnArg, ForeignItem, GenericArgument, Item, ItemUse, Lit, Macro,
    Meta, Pat, PathArguments, ReturnType, Type, UseTree, Visibility,
};
use thiserror::Error;

/// Exact request for extracting a Rust crate's raw binding module.
#[derive(Clone, Debug)]
pub struct ExtractRequest {
    /// Workspace root used for `cargo metadata`.
    pub workspace_root: PathBuf,
    /// Exact crates.io package name.
    pub package_name: String,
    /// Exact crates.io package version.
    pub package_version: String,
    /// Module path relative to the package root, such as `src/driver/sys/mod.rs`.
    pub module_path: PathBuf,
    /// Stable inventory identifier.
    pub inventory_id: String,
    /// Source display name.
    pub source_name: String,
    /// Platforms to attach when no narrower `cfg` gate is present.
    pub platforms: Vec<String>,
    /// Evidence locator stored on extracted entries.
    pub provenance: String,
    /// Exact crate archive and lockfile checksum.
    pub source_artifacts: Vec<SourceArtifact>,
}

/// Extraction failures include enough context for a maintainer to reproduce them.
#[derive(Debug, Error)]
pub enum ExtractionError {
    /// `cargo metadata` could not run.
    #[error("failed to run cargo metadata: {0}")]
    MetadataCommand(#[source] std::io::Error),
    /// `cargo metadata` returned a failure.
    #[error("cargo metadata failed: {0}")]
    MetadataFailure(String),
    /// Metadata JSON was invalid.
    #[error("invalid cargo metadata JSON: {0}")]
    MetadataJson(#[from] serde_json::Error),
    /// The requested exact registry package was absent.
    #[error("cargo metadata did not contain registry package {name} {version}")]
    PackageNotFound {
        /// Package name.
        name: String,
        /// Package version.
        version: String,
    },
    /// A source file could not be read.
    #[error("failed to read {path}: {source}")]
    Read {
        /// File path.
        path: PathBuf,
        /// Filesystem error.
        source: std::io::Error,
    },
    /// Rust source was not syntactically valid.
    #[error("failed to parse {path}: {source}")]
    Parse {
        /// File path.
        path: PathBuf,
        /// Parser error.
        source: syn::Error,
    },
    /// An include expression could not be resolved without executing a build script.
    #[error("cannot resolve include expression `{expression}` in {path}")]
    DynamicInclude {
        /// File containing the include.
        path: PathBuf,
        /// Rendered expression.
        expression: String,
    },
    /// An external module declaration did not resolve to a file.
    #[error("cannot resolve module `{module}` declared in {path}")]
    ModuleNotFound {
        /// Declaring file.
        path: PathBuf,
        /// Module identifier.
        module: String,
    },
}

#[derive(Debug, Deserialize)]
struct Metadata {
    packages: Vec<MetadataPackage>,
}

#[derive(Debug, Deserialize)]
struct MetadataPackage {
    name: String,
    version: String,
    manifest_path: PathBuf,
    source: Option<String>,
}

/// Locates an exact registry package using Cargo's own resolved metadata.
pub fn locate_registry_package(
    workspace_root: &Path,
    name: &str,
    version: &str,
) -> Result<PathBuf, ExtractionError> {
    let output = Command::new("cargo")
        .args(["metadata", "--format-version", "1"])
        .arg("--manifest-path")
        .arg(workspace_root.join("Cargo.toml"))
        .output()
        .map_err(ExtractionError::MetadataCommand)?;
    if !output.status.success() {
        return Err(ExtractionError::MetadataFailure(
            String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        ));
    }
    let metadata: Metadata = serde_json::from_slice(&output.stdout)?;
    let package = metadata.packages.into_iter().find(|package| {
        package.name == name
            && package.version == version
            && package
                .source
                .as_deref()
                .is_some_and(|source| source.starts_with("registry+"))
    });
    package
        .and_then(|package| package.manifest_path.parent().map(Path::to_path_buf))
        .ok_or_else(|| ExtractionError::PackageNotFound {
            name: name.to_owned(),
            version: version.to_owned(),
        })
}

/// Parses an exact dependency's source using `syn`, recursively following source modules and
/// statically resolvable `include!` expressions.
pub fn extract_rust_inventory(request: &ExtractRequest) -> Result<Inventory, ExtractionError> {
    let package_root = locate_registry_package(
        &request.workspace_root,
        &request.package_name,
        &request.package_version,
    )?;
    let root = package_root.join(&request.module_path);
    let mut collector = Collector {
        package_root,
        platforms: &request.platforms,
        provenance: &request.provenance,
        visited: BTreeSet::new(),
        entries: BTreeMap::new(),
    };
    collector.parse_path(&root)?;
    refine_alias_parameter_facts(&mut collector.entries);
    Ok(Inventory {
        schema_version: 1,
        spdx_license_identifier: "CC0-1.0".to_owned(),
        inventory_id: request.inventory_id.clone(),
        source_name: request.source_name.clone(),
        source_version: request.package_version.clone(),
        provenance: request.provenance.clone(),
        source_artifacts: request.source_artifacts.clone(),
        platforms: request.platforms.clone(),
        entries: collector.entries.into_values().collect(),
    })
}

#[derive(Clone, Copy)]
struct AliasParameterFact {
    pointer: PointerKind,
    nullable: Option<bool>,
}

fn refine_alias_parameter_facts(entries: &mut BTreeMap<(ItemKind, String), Entry>) {
    let facts = entries
        .values()
        .filter_map(|entry| {
            alias_parameter_fact(&entry.name, entries, &mut BTreeSet::new())
                .map(|fact| (entry.name.clone(), fact))
        })
        .collect::<BTreeMap<_, _>>();
    for entry in entries.values_mut() {
        let Some(abi) = entry.abi.as_mut() else {
            continue;
        };
        for parameter in &mut abi.parameters {
            if parameter.pointer != PointerKind::Value {
                continue;
            }
            let (type_name, option_wrapped) = path_type_name(&parameter.r#type);
            let Some(fact) = type_name.and_then(|name| facts.get(name)) else {
                continue;
            };
            parameter.pointer = fact.pointer;
            parameter.direction = Direction::In;
            parameter.nullable = if option_wrapped {
                Some(true)
            } else {
                fact.nullable
            };
        }
        entry.normalized_signature = normalized_callable(&entry.name, abi);
        entry.signature_hash = sha256(&entry.normalized_signature);
    }
}

fn alias_parameter_fact(
    name: &str,
    entries: &BTreeMap<(ItemKind, String), Entry>,
    visiting: &mut BTreeSet<String>,
) -> Option<AliasParameterFact> {
    if !visiting.insert(name.to_owned()) {
        return None;
    }
    let entry = entries.iter().find_map(|((kind, candidate), entry)| {
        (candidate == name
            && matches!(
                kind,
                ItemKind::Type | ItemKind::OpaqueHandle | ItemKind::Callback | ItemKind::Alias
            ))
        .then_some(entry)
    })?;
    let direct = match entry.kind {
        ItemKind::OpaqueHandle => Some(AliasParameterFact {
            pointer: PointerKind::Mut,
            nullable: None,
        }),
        ItemKind::Callback => Some(AliasParameterFact {
            pointer: PointerKind::Callback,
            nullable: None,
        }),
        ItemKind::Alias => entry.alias_of.as_deref().and_then(|target| {
            alias_parameter_fact(
                target.rsplit("::").next().unwrap_or(target),
                entries,
                visiting,
            )
        }),
        ItemKind::Type => entry
            .normalized_signature
            .split_once('=')
            .and_then(|(_, target)| {
                let target = target.trim();
                if target.starts_with("* mut ") {
                    Some(AliasParameterFact {
                        pointer: PointerKind::Mut,
                        nullable: None,
                    })
                } else if target.starts_with("* const ") {
                    Some(AliasParameterFact {
                        pointer: PointerKind::Const,
                        nullable: None,
                    })
                } else {
                    let (target, option_wrapped) = path_type_name(target);
                    target.and_then(|target| {
                        alias_parameter_fact(target, entries, visiting).map(|mut fact| {
                            if option_wrapped {
                                fact.nullable = Some(true);
                            }
                            fact
                        })
                    })
                }
            }),
        _ => None,
    };
    visiting.remove(name);
    direct
}

fn path_type_name(value: &str) -> (Option<&str>, bool) {
    let value = value.trim();
    let (value, option_wrapped) = value
        .strip_prefix("Option < ")
        .and_then(|inner| inner.strip_suffix(" >"))
        .map_or((value, false), |inner| (inner.trim(), true));
    if value.is_empty()
        || !value
            .chars()
            .all(|character| character == '_' || character == ':' || character.is_alphanumeric())
    {
        return (None, option_wrapped);
    }
    (value.rsplit("::").next(), option_wrapped)
}

struct Collector<'a> {
    package_root: PathBuf,
    platforms: &'a [String],
    provenance: &'a str,
    visited: BTreeSet<PathBuf>,
    entries: BTreeMap<(ItemKind, String), Entry>,
}

impl Collector<'_> {
    fn parse_path(&mut self, path: &Path) -> Result<(), ExtractionError> {
        let canonical = path
            .canonicalize()
            .map_err(|source| ExtractionError::Read {
                path: path.to_path_buf(),
                source,
            })?;
        if !self.visited.insert(canonical.clone()) {
            return Ok(());
        }
        let source = fs::read_to_string(&canonical).map_err(|source| ExtractionError::Read {
            path: canonical.clone(),
            source,
        })?;
        let file = syn::parse_file(&source).map_err(|source| ExtractionError::Parse {
            path: canonical.clone(),
            source,
        })?;
        self.collect_file(&file, &canonical)
    }

    fn collect_file(&mut self, file: &File, path: &Path) -> Result<(), ExtractionError> {
        self.collect_items(&file.items, path)
    }

    #[allow(clippy::too_many_lines)]
    fn collect_items(&mut self, items: &[Item], path: &Path) -> Result<(), ExtractionError> {
        for item in items {
            match item {
                Item::Const(item) if is_public(&item.vis) => {
                    let signature = format!(
                        "const {}:{}={}",
                        item.ident,
                        normalized_tokens(&item.ty),
                        normalized_tokens(&item.expr)
                    );
                    self.insert(
                        ItemKind::Constant,
                        item.ident.to_string(),
                        signature,
                        None,
                        &item.attrs,
                    );
                }
                Item::Enum(item) if is_public(&item.vis) => {
                    let name = item.ident.to_string();
                    let signature = format!("enum {name}:{}", normalized_tokens(item));
                    self.insert(ItemKind::Type, name, signature, None, &item.attrs);
                    for variant in &item.variants {
                        let variant_name = variant.ident.to_string();
                        let value = variant.discriminant.as_ref().map_or_else(
                            || "implicit".to_owned(),
                            |(_, expression)| normalized_tokens(expression),
                        );
                        self.insert(
                            ItemKind::EnumValue,
                            variant_name.clone(),
                            format!("enum-value {variant_name}={value}"),
                            None,
                            &variant.attrs,
                        );
                    }
                }
                Item::Fn(item) if is_public(&item.vis) => {
                    let abi = abi_from_signature(&item.sig);
                    let signature = normalized_callable(&item.sig.ident.to_string(), &abi);
                    self.insert(
                        ItemKind::Function,
                        item.sig.ident.to_string(),
                        signature,
                        Some(abi),
                        &item.attrs,
                    );
                }
                Item::ForeignMod(item) => {
                    for foreign in &item.items {
                        if let ForeignItem::Fn(function) = foreign {
                            if !is_public(&function.vis) {
                                continue;
                            }
                            let mut abi = abi_from_signature(&function.sig);
                            abi.calling_convention = item
                                .abi
                                .name
                                .as_ref()
                                .map_or_else(|| "C".to_owned(), syn::LitStr::value);
                            let signature =
                                normalized_callable(&function.sig.ident.to_string(), &abi);
                            self.insert(
                                ItemKind::Function,
                                function.sig.ident.to_string(),
                                signature,
                                Some(abi),
                                &function.attrs,
                            );
                        }
                    }
                }
                Item::Macro(item) if item.mac.path.is_ident("include") => {
                    let included = resolve_include(&item.mac, path, &self.package_root)?;
                    self.parse_path(&included)?;
                }
                Item::Mod(item) => {
                    if let Some((_, nested)) = &item.content {
                        self.collect_items(nested, path)?;
                    } else {
                        let module_path = resolve_module(item, path)?;
                        self.parse_path(&module_path)?;
                    }
                }
                Item::Static(item) if is_public(&item.vis) => {
                    let signature =
                        format!("static {}:{}", item.ident, normalized_tokens(&item.ty));
                    self.insert(
                        ItemKind::Constant,
                        item.ident.to_string(),
                        signature,
                        None,
                        &item.attrs,
                    );
                }
                Item::Struct(item) if is_public(&item.vis) => {
                    let name = item.ident.to_string();
                    self.insert(
                        ItemKind::Struct,
                        name.clone(),
                        format!("struct {name}:{}", normalized_tokens(&item.fields)),
                        None,
                        &item.attrs,
                    );
                }
                Item::Type(item) if is_public(&item.vis) => {
                    let kind = match item.ty.as_ref() {
                        Type::BareFn(_) => ItemKind::Callback,
                        Type::Ptr(_) if looks_like_handle(&item.ident.to_string()) => {
                            ItemKind::OpaqueHandle
                        }
                        _ => ItemKind::Type,
                    };
                    let name = item.ident.to_string();
                    let abi = match item.ty.as_ref() {
                        Type::BareFn(function) => Some(abi_from_bare_fn(function)),
                        _ => None,
                    };
                    let signature = if let Some(callable) = &abi {
                        normalized_callable(&name, callable)
                    } else {
                        format!("type {name}={}", normalized_tokens(&item.ty))
                    };
                    self.insert(kind, name, signature, abi, &item.attrs);
                }
                Item::Union(item) if is_public(&item.vis) => {
                    let name = item.ident.to_string();
                    self.insert(
                        ItemKind::Union,
                        name.clone(),
                        format!("union {name}:{}", normalized_tokens(&item.fields)),
                        None,
                        &item.attrs,
                    );
                }
                Item::Use(item) if is_public(&item.vis) => self.collect_use(item),
                _ => {}
            }
        }
        Ok(())
    }

    fn collect_use(&mut self, item: &ItemUse) {
        let mut aliases = Vec::new();
        flatten_use(&item.tree, "", &mut aliases);
        for (target, alias) in aliases {
            if target == alias || alias == "self" || alias == "*" {
                continue;
            }
            let signature = format!("alias {alias}={target}");
            let mut entry = self.new_entry(ItemKind::Alias, alias, signature, None, &item.attrs);
            entry.alias_of = Some(target);
            self.entries.insert((entry.kind, entry.name.clone()), entry);
        }
    }

    fn insert(
        &mut self,
        kind: ItemKind,
        name: String,
        signature: String,
        abi: Option<Abi>,
        attrs: &[Attribute],
    ) {
        let entry = self.new_entry(kind, name, signature, abi, attrs);
        self.entries.insert((entry.kind, entry.name.clone()), entry);
    }

    fn new_entry(
        &self,
        kind: ItemKind,
        name: String,
        signature: String,
        abi: Option<Abi>,
        attrs: &[Attribute],
    ) -> Entry {
        let platforms = narrowed_platforms(attrs, self.platforms);
        Entry {
            kind,
            name,
            signature_hash: sha256(&signature),
            normalized_signature: signature,
            numeric_value: None,
            abi,
            aliases: Vec::new(),
            alias_of: None,
            platforms,
            introduced: None,
            deprecated: attrs
                .iter()
                .any(|attribute| attribute.path().is_ident("deprecated"))
                .then(|| "upstream".to_owned()),
            layouts: Vec::new(),
            provenance: self.provenance.to_owned(),
        }
    }
}

fn is_public(visibility: &Visibility) -> bool {
    matches!(visibility, Visibility::Public(_))
}

fn normalized_tokens(value: &impl ToTokens) -> String {
    value
        .to_token_stream()
        .to_string()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .replace(" :: ", "::")
        .replace(" * ", "*")
        .replace(" & ", "&")
        .replace(" ,", ",")
        .replace(" ;", ";")
}

fn abi_from_signature(signature: &syn::Signature) -> Abi {
    let parameters = signature
        .inputs
        .iter()
        .filter_map(|argument| match argument {
            FnArg::Receiver(_) => None,
            FnArg::Typed(typed) => {
                let name = match typed.pat.as_ref() {
                    Pat::Ident(identifier) => identifier.ident.to_string(),
                    _ => normalized_tokens(&typed.pat),
                };
                Some(parameter(name, &typed.ty))
            }
        })
        .collect();
    Abi {
        calling_convention: signature
            .abi
            .as_ref()
            .and_then(|abi| abi.name.as_ref())
            .map_or_else(|| "C".to_owned(), syn::LitStr::value),
        return_type: match &signature.output {
            ReturnType::Default => "()".to_owned(),
            ReturnType::Type(_, ty) => normalized_tokens(ty),
        },
        parameters,
    }
}

fn abi_from_bare_fn(function: &syn::TypeBareFn) -> Abi {
    Abi {
        calling_convention: function
            .abi
            .as_ref()
            .and_then(|abi| abi.name.as_ref())
            .map_or_else(|| "C".to_owned(), syn::LitStr::value),
        return_type: match &function.output {
            ReturnType::Default => "()".to_owned(),
            ReturnType::Type(_, ty) => normalized_tokens(ty),
        },
        parameters: function
            .inputs
            .iter()
            .enumerate()
            .map(|(index, argument)| {
                let name = argument.name.as_ref().map_or_else(
                    || format!("arg{index}"),
                    |(identifier, _)| identifier.to_string(),
                );
                parameter(name, &argument.ty)
            })
            .collect(),
    }
}

fn parameter(name: String, ty: &Type) -> Parameter {
    let (pointer, direction, nullable) = match ty {
        Type::Ptr(pointer) if pointer.mutability.is_some() => {
            (PointerKind::Mut, Direction::Unknown, None)
        }
        Type::Ptr(_) => (PointerKind::Const, Direction::In, None),
        Type::Reference(reference) if reference.mutability.is_some() => {
            (PointerKind::Mut, Direction::Unknown, Some(false))
        }
        Type::Reference(_) => (PointerKind::Const, Direction::In, Some(false)),
        Type::BareFn(_) => (PointerKind::Callback, Direction::In, Some(false)),
        Type::Path(path) if option_contains_bare_fn(path) => {
            (PointerKind::Callback, Direction::In, Some(true))
        }
        _ => (PointerKind::Value, Direction::In, Some(false)),
    };
    Parameter {
        name,
        r#type: normalized_tokens(ty),
        pointer,
        direction,
        nullable,
    }
}

fn option_contains_bare_fn(path: &syn::TypePath) -> bool {
    path.path.segments.last().is_some_and(|segment| {
        segment.ident == "Option"
            && matches!(
                &segment.arguments,
                PathArguments::AngleBracketed(arguments)
                    if arguments.args.iter().any(|argument| matches!(argument, GenericArgument::Type(Type::BareFn(_))))
            )
    })
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

fn looks_like_handle(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.ends_with("_t")
        && [
            "context", "stream", "event", "module", "function", "array", "graph",
        ]
        .iter()
        .any(|fragment| lower.contains(fragment))
}

fn narrowed_platforms(attrs: &[Attribute], defaults: &[String]) -> Vec<String> {
    let cfg = attrs
        .iter()
        .filter(|attribute| attribute.path().is_ident("cfg"))
        .map(normalized_tokens)
        .collect::<Vec<_>>()
        .join(" ");
    if cfg.is_empty() {
        return defaults.to_vec();
    }
    let mentions_windows = cfg.contains("windows");
    let mentions_linux = cfg.contains("linux") || cfg.contains("unix");
    if mentions_windows == mentions_linux {
        return defaults.to_vec();
    }
    defaults
        .iter()
        .filter(|target| {
            (mentions_windows && target.contains("windows"))
                || (mentions_linux && target.contains("linux"))
        })
        .cloned()
        .collect()
}

fn resolve_include(
    include: &Macro,
    current_file: &Path,
    package_root: &Path,
) -> Result<PathBuf, ExtractionError> {
    let expression: Expr =
        syn::parse2(include.tokens.clone()).map_err(|source| ExtractionError::Parse {
            path: current_file.to_path_buf(),
            source,
        })?;
    if let Expr::Lit(ExprLit {
        lit: Lit::Str(path),
        ..
    }) = &expression
    {
        return Ok(current_file
            .parent()
            .unwrap_or(package_root)
            .join(path.value()));
    }
    if let Some(path) = resolve_concat(&expression, package_root) {
        return Ok(path);
    }
    Err(ExtractionError::DynamicInclude {
        path: current_file.to_path_buf(),
        expression: normalized_tokens(&expression),
    })
}

fn resolve_concat(expression: &Expr, package_root: &Path) -> Option<PathBuf> {
    let Expr::Macro(expression) = expression else {
        return None;
    };
    if !expression.mac.path.is_ident("concat") {
        return None;
    }
    let arguments = expression
        .mac
        .parse_body_with(syn::punctuated::Punctuated::<Expr, syn::Token![,]>::parse_terminated)
        .ok()?;
    let mut result = PathBuf::new();
    for argument in arguments {
        match argument {
            Expr::Lit(ExprLit {
                lit: Lit::Str(value),
                ..
            }) => result.push(value.value()),
            Expr::Macro(value) if value.mac.path.is_ident("env") => {
                let variable: syn::LitStr = value.mac.parse_body().ok()?;
                if variable.value() != "CARGO_MANIFEST_DIR" {
                    return None;
                }
                result.push(package_root);
            }
            _ => return None,
        }
    }
    Some(result)
}

fn resolve_module(module: &syn::ItemMod, current_file: &Path) -> Result<PathBuf, ExtractionError> {
    let parent = current_file.parent().unwrap_or_else(|| Path::new("."));
    if let Some(path) = module.attrs.iter().find_map(|attribute| {
        if !attribute.path().is_ident("path") {
            return None;
        }
        match &attribute.meta {
            Meta::NameValue(value) => match &value.value {
                Expr::Lit(ExprLit {
                    lit: Lit::Str(path),
                    ..
                }) => Some(path.value()),
                _ => None,
            },
            _ => None,
        }
    }) {
        return Ok(parent.join(path));
    }
    let stem = current_file
        .file_stem()
        .and_then(OsStr::to_str)
        .unwrap_or("mod");
    let module_root = if stem == "mod" || stem == "lib" {
        parent.to_path_buf()
    } else {
        parent.join(stem)
    };
    let flat = module_root.join(format!("{}.rs", module.ident));
    if flat.is_file() {
        return Ok(flat);
    }
    let nested = module_root.join(module.ident.to_string()).join("mod.rs");
    if nested.is_file() {
        return Ok(nested);
    }
    Err(ExtractionError::ModuleNotFound {
        path: current_file.to_path_buf(),
        module: module.ident.to_string(),
    })
}

fn flatten_use(tree: &UseTree, prefix: &str, output: &mut Vec<(String, String)>) {
    match tree {
        UseTree::Name(name) => {
            let target = join_path(prefix, &name.ident.to_string());
            output.push((target, name.ident.to_string()));
        }
        UseTree::Rename(rename) => {
            output.push((
                join_path(prefix, &rename.ident.to_string()),
                rename.rename.to_string(),
            ));
        }
        UseTree::Path(path) => {
            let nested_prefix = join_path(prefix, &path.ident.to_string());
            flatten_use(&path.tree, &nested_prefix, output);
        }
        UseTree::Group(group) => {
            for nested in &group.items {
                flatten_use(nested, prefix, output);
            }
        }
        UseTree::Glob(_) => output.push((join_path(prefix, "*"), "*".to_owned())),
    }
}

fn join_path(prefix: &str, suffix: &str) -> String {
    if prefix.is_empty() {
        suffix.to_owned()
    } else {
        format!("{prefix}::{suffix}")
    }
}

#[cfg(test)]
mod tests {
    use super::{abi_from_signature, normalized_callable};

    #[test]
    fn callable_normalization_retains_pointer_semantics() {
        let function: syn::ItemFn = syn::parse_quote! {
            pub unsafe extern "C" fn sample(output: *mut i32, input: *const u8) -> i32 { 0 }
        };
        let abi = abi_from_signature(&function.sig);
        assert_eq!(abi.parameters.len(), 2);
        assert!(normalized_callable("sample", &abi).contains("mut:unknown"));
        assert!(normalized_callable("sample", &abi).contains("const:in"));
    }
}
