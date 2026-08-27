// SPDX-License-Identifier: CC0-1.0

//! Command-line diagnostics for installed CUDA and HIP runtimes, the ocgpu
//! ABI, committed coverage data, and precompiled device modules.

mod args;
mod inspect;

use args::{BackendChoice, Command, SymbolFilter};
use miniz_oxide::inflate::decompress_to_vec;
use ocgpu::{AnyDriver, BackendDiagnostics, BackendKind, SymbolResolution};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::env;
use std::fmt;
use std::path::Path;
use std::process::ExitCode;

const HELP: &str = "\
ocgpu — SDK-free CUDA/HIP runtime diagnostics

USAGE:
  ocgpu backends [--json]
  ocgpu devices [--backend cuda|hip|all] [--json]
  ocgpu doctor [--strict] [--json]
  ocgpu symbols --backend cuda|hip [--available|--missing|--all] [--json]
  ocgpu abi [--json]
  ocgpu coverage [--json]
  ocgpu module inspect FILE [--json]
  ocgpu help
  ocgpu version
";

const COVERAGE_DEFLATE: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../coverage/coverage.json.deflate"
));

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
const CURRENT_TARGET: &str = "x86_64-unknown-linux-gnu";
#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
const CURRENT_TARGET: &str = "aarch64-unknown-linux-gnu";
#[cfg(all(target_arch = "x86_64", target_os = "windows"))]
const CURRENT_TARGET: &str = "x86_64-pc-windows-msvc";
#[cfg(not(any(
    all(target_arch = "x86_64", target_os = "linux"),
    all(target_arch = "aarch64", target_os = "linux"),
    all(target_arch = "x86_64", target_os = "windows")
)))]
const CURRENT_TARGET: &str = "unsupported";

fn main() -> ExitCode {
    match args::parse(env::args_os().skip(1)) {
        Ok(command) => match execute(command) {
            Ok(code) => code,
            Err(error) => {
                eprintln!("ocgpu: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("ocgpu: {error}");
            eprintln!("Try `ocgpu help` for usage.");
            ExitCode::from(2)
        }
    }
}

fn execute(command: Command) -> Result<ExitCode, CliError> {
    match command {
        Command::Backends { json } => {
            let reports = backend_reports();
            if json {
                print_json(&reports)?;
            } else {
                print_backend_reports(&reports);
            }
            Ok(ExitCode::SUCCESS)
        }
        Command::Devices { backend, json } => devices(backend, json),
        Command::Doctor { strict, json } => doctor(strict, json),
        Command::Symbols {
            backend,
            filter,
            json,
        } => symbols(backend, filter, json),
        Command::Abi { json } => abi(json),
        Command::Coverage { json } => coverage(json),
        Command::ModuleInspect { path, json } => module_inspect(&path, json),
        Command::Help => {
            print!("{HELP}");
            Ok(ExitCode::SUCCESS)
        }
        Command::Version => {
            println!("ocgpu {}", env!("CARGO_PKG_VERSION"));
            Ok(ExitCode::SUCCESS)
        }
    }
}

#[derive(Debug)]
struct CliError(String);

impl fmt::Display for CliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for CliError {}

impl From<std::io::Error> for CliError {
    fn from(error: std::io::Error) -> Self {
        Self(error.to_string())
    }
}

impl From<serde_json::Error> for CliError {
    fn from(error: serde_json::Error) -> Self {
        Self(format!("JSON error: {error}"))
    }
}

#[derive(Debug, Serialize)]
struct BackendReport {
    backend: &'static str,
    available: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    library_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    runtime_version: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    driver_version: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    compiled_api_version: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    runtime_profile: Option<&'static str>,
    #[serde(skip)]
    runtime_profile_display: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    proc_address_support: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    proc_address_variant: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    device_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    missing_core_functions: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    missing_optional_functions: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    profile_specific_omissions: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    platform_specific_omissions: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    loaded_architecture: Option<&'static str>,
    abi_version: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

fn backend_reports() -> Vec<BackendReport> {
    [BackendKind::Cuda, BackendKind::Hip]
        .into_iter()
        .map(backend_report)
        .collect()
}

fn backend_report(backend: BackendKind) -> BackendReport {
    let diagnostics = ocgpu::backend_diagnostics(backend);
    let driver = AnyDriver::load(backend);
    match (diagnostics, driver) {
        (Ok(diagnostics), Ok(driver)) => {
            let device_count = driver.device_count();
            report_from_diagnostics(
                &diagnostics,
                device_count.as_ref().ok().copied(),
                device_count.err().map(|error| error.to_string()),
            )
        }
        (Ok(diagnostics), Err(error)) => {
            report_from_diagnostics(&diagnostics, None, Some(error.to_string()))
        }
        (Err(diagnostic_error), Ok(driver)) => {
            let device_count = driver.device_count();
            let driver_version = driver.driver_version().ok();
            BackendReport {
                backend: backend.as_str(),
                available: device_count.is_ok(),
                library_path: None,
                runtime_version: driver_version,
                driver_version,
                compiled_api_version: None,
                runtime_profile: None,
                runtime_profile_display: None,
                proc_address_support: None,
                proc_address_variant: None,
                device_count: device_count.ok(),
                missing_core_functions: None,
                missing_optional_functions: None,
                profile_specific_omissions: None,
                platform_specific_omissions: None,
                loaded_architecture: None,
                abi_version: driver.metadata().abi_version,
                error: Some(diagnostic_error.to_string()),
            }
        }
        (Err(diagnostic_error), Err(load_error)) => BackendReport {
            backend: backend.as_str(),
            available: false,
            library_path: None,
            runtime_version: None,
            driver_version: None,
            compiled_api_version: None,
            runtime_profile: None,
            runtime_profile_display: None,
            proc_address_support: None,
            proc_address_variant: None,
            device_count: None,
            missing_core_functions: None,
            missing_optional_functions: None,
            profile_specific_omissions: None,
            platform_specific_omissions: None,
            loaded_architecture: None,
            abi_version: ocgpu::sys::OCGPU_ABI_VERSION_1,
            error: Some(if diagnostic_error == load_error {
                load_error.to_string()
            } else {
                format!("diagnostics: {diagnostic_error}; validation: {load_error}")
            }),
        },
    }
}

fn report_from_diagnostics(
    diagnostics: &BackendDiagnostics,
    device_count: Option<usize>,
    operation_error: Option<String>,
) -> BackendReport {
    let missing_core_functions = diagnostics.missing_required_symbols().count();
    let missing_optional_functions = diagnostics.missing_optional_symbols().count();
    let profile_specific_omissions = diagnostics.profile_omissions().count();
    let platform_specific_omissions = diagnostics.platform_omissions().count();
    BackendReport {
        backend: diagnostics.backend.as_str(),
        available: operation_error.is_none() && missing_core_functions == 0,
        library_path: Some(diagnostics.library_path.display().to_string()),
        runtime_version: diagnostics.runtime_version,
        driver_version: diagnostics.driver_version,
        compiled_api_version: Some(diagnostics.compiled_api_version),
        runtime_profile: diagnostics
            .hip_runtime_profile
            .map(ocgpu::HipRuntimeProfile::as_str),
        runtime_profile_display: diagnostics
            .hip_runtime_profile
            .map(ocgpu::HipRuntimeProfile::display_name),
        proc_address_support: Some(diagnostics.proc_address_support),
        proc_address_variant: diagnostics
            .proc_address_variant
            .map(ocgpu::ProcAddressVariant::as_str),
        device_count,
        missing_core_functions: Some(missing_core_functions),
        missing_optional_functions: Some(missing_optional_functions),
        profile_specific_omissions: Some(profile_specific_omissions),
        platform_specific_omissions: Some(platform_specific_omissions),
        loaded_architecture: Some(diagnostics.loaded_architecture),
        abi_version: ocgpu::sys::OCGPU_ABI_VERSION_1,
        error: operation_error,
    }
}

fn print_backend_reports(reports: &[BackendReport]) {
    for report in reports {
        println!(
            "{}: {}",
            report.backend,
            if report.available {
                "available"
            } else {
                "unavailable"
            }
        );
        if let Some(path) = &report.library_path {
            println!("  library: {path}");
        }
        if let Some(version) = report.runtime_version {
            println!("  runtime version: {version}");
        }
        if let Some(version) = report.driver_version {
            println!("  driver version: {version}");
        }
        if let Some(version) = report.compiled_api_version {
            println!("  compiled API version: {version}");
        }
        if let Some(profile) = report.runtime_profile_display {
            println!("  runtime profile: {profile}");
        }
        if let Some(supported) = report.proc_address_support {
            println!(
                "  proc-address resolution enabled: {}",
                if supported { "yes" } else { "no" }
            );
        }
        if let Some(variant) = report.proc_address_variant {
            println!("  proc-address ABI: {variant}");
        }
        if let Some(architecture) = report.loaded_architecture {
            println!("  loaded architecture: {architecture}");
        }
        println!("  ABI version: 0x{:08x}", report.abi_version);
        if let Some(count) = report.device_count {
            println!("  devices: {count}");
        }
        if let Some(count) = report.missing_core_functions {
            println!("  missing core functions: {count}");
        }
        if let Some(count) = report.missing_optional_functions {
            println!("  missing optional functions: {count}");
        }
        if let Some(count) = report.profile_specific_omissions {
            println!("  profile-specific omissions: {count}");
        }
        if let Some(count) = report.platform_specific_omissions {
            println!("  platform-specific omissions: {count}");
        }
        if let Some(error) = &report.error {
            println!("  error: {error}");
        }
    }
}

#[derive(Debug, Serialize)]
struct DeviceReport {
    backend: &'static str,
    ordinal: usize,
    name: String,
}

#[derive(Debug, Serialize)]
struct DeviceOutput {
    devices: Vec<DeviceReport>,
    errors: Vec<CommandBackendError>,
}

#[derive(Debug, Serialize)]
struct CommandBackendError {
    backend: &'static str,
    error: String,
}

fn devices(choice: BackendChoice, json: bool) -> Result<ExitCode, CliError> {
    let backends = selected_backends(choice);
    let mut output = DeviceOutput {
        devices: Vec::new(),
        errors: Vec::new(),
    };
    for backend in backends {
        match AnyDriver::load(backend).and_then(|driver| driver.device_summaries()) {
            Ok(devices) => output
                .devices
                .extend(devices.into_iter().map(|device| DeviceReport {
                    backend: device.backend.as_str(),
                    ordinal: device.ordinal,
                    name: device.name,
                })),
            Err(error) => output.errors.push(CommandBackendError {
                backend: backend.as_str(),
                error: error.to_string(),
            }),
        }
    }
    if json {
        print_json(&output)?;
    } else {
        for device in &output.devices {
            println!("{}:{} {}", device.backend, device.ordinal, device.name);
        }
        for error in &output.errors {
            eprintln!("{}: {}", error.backend, error.error);
        }
    }
    let requested_one = !matches!(choice, BackendChoice::All);
    Ok(if requested_one && !output.errors.is_empty() {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    })
}

#[derive(Debug, Serialize)]
struct DoctorReport {
    healthy: bool,
    strict: bool,
    abi_version: u32,
    inventory_schema: u32,
    backends: Vec<BackendReport>,
}

fn doctor(strict: bool, json: bool) -> Result<ExitCode, CliError> {
    let coverage = load_coverage()?;
    let reports = backend_reports();
    let healthy = reports.iter().all(|report| report.available);
    let report = DoctorReport {
        healthy,
        strict,
        abi_version: ocgpu::sys::OCGPU_ABI_VERSION_1,
        inventory_schema: coverage.schema_version,
        backends: reports,
    };
    if json {
        print_json(&report)?;
    } else {
        println!("ocgpu ABI: 0x{:08x}", report.abi_version);
        println!("coverage schema: {}", report.inventory_schema);
        print_backend_reports(&report.backends);
        println!("health: {}", if healthy { "healthy" } else { "degraded" });
    }
    Ok(if strict && !healthy {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    })
}

#[derive(Debug, Deserialize)]
struct CoverageDocument {
    schema_version: u32,
    metrics: Vec<CoverageMetric>,
    classification_counts: BTreeMap<String, u64>,
    #[serde(default)]
    symbols: Vec<CoverageSymbol>,
}

#[derive(Debug, Deserialize)]
struct CoverageMetric {
    id: String,
    label: String,
    numerator: u64,
    denominator: u64,
    basis: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CoverageSymbol {
    inventory_id: String,
    name: String,
    backend: String,
    kind: String,
    classification: String,
    platforms: Vec<String>,
    manifest_ids: Vec<String>,
    runtime_resolvable: bool,
    hardware_smoke: bool,
}

#[derive(Debug, Serialize)]
struct SymbolOutput {
    backend: &'static str,
    target: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    runtime_profile: Option<&'static str>,
    #[serde(skip)]
    runtime_profile_display: Option<&'static str>,
    library_error: Option<String>,
    symbols: Vec<DisplayedSymbol>,
}

#[derive(Debug, Serialize)]
struct DisplayedSymbol {
    #[serde(flatten)]
    coverage: RuntimeCoverageSymbol,
    runtime_available: bool,
    runtime_resolution: &'static str,
    resolved_name: Option<&'static str>,
    runtime_required: bool,
    runtime_applicable: bool,
}

#[derive(Debug, Serialize)]
struct RuntimeCoverageSymbol {
    inventory_ids: Vec<String>,
    name: String,
    backend: String,
    kind: String,
    classification: String,
    platforms: Vec<String>,
    manifest_ids: Vec<String>,
    runtime_resolvable: bool,
    hardware_smoke: bool,
}

impl RuntimeCoverageSymbol {
    fn from_coverage(symbol: CoverageSymbol) -> Self {
        Self {
            inventory_ids: vec![symbol.inventory_id],
            name: symbol.name,
            backend: symbol.backend,
            kind: symbol.kind,
            classification: symbol.classification,
            platforms: symbol.platforms,
            manifest_ids: symbol.manifest_ids,
            runtime_resolvable: symbol.runtime_resolvable,
            hardware_smoke: symbol.hardware_smoke,
        }
    }

    fn merge(&mut self, symbol: CoverageSymbol) -> Result<(), CliError> {
        let merged_classification =
            merge_runtime_classification(&self.classification, &symbol.classification);
        if self.name != symbol.name
            || self.backend != symbol.backend
            || self.kind != symbol.kind
            || merged_classification.is_none()
            || self.runtime_resolvable != symbol.runtime_resolvable
        {
            return Err(CliError(format!(
                "coverage rows disagree for runtime symbol {}",
                self.name
            )));
        }
        self.classification =
            merged_classification.expect("classification compatibility was checked");
        self.inventory_ids.push(symbol.inventory_id);
        self.inventory_ids.sort();
        self.inventory_ids.dedup();
        self.platforms.extend(symbol.platforms);
        self.platforms.sort();
        self.platforms.dedup();
        self.manifest_ids.extend(symbol.manifest_ids);
        self.manifest_ids.sort();
        self.manifest_ids.dedup();
        self.hardware_smoke |= symbol.hardware_smoke;
        Ok(())
    }
}

fn merge_runtime_classification(left: &str, right: &str) -> Option<String> {
    if left == right {
        return Some(left.to_owned());
    }
    matches!(
        (left, right),
        ("deprecated_covered", "covered_raw_only") | ("covered_raw_only", "deprecated_covered")
    )
    .then(|| "deprecated_covered".to_owned())
}

fn symbols(choice: BackendChoice, filter: SymbolFilter, json: bool) -> Result<ExitCode, CliError> {
    let backend = match choice {
        BackendChoice::Cuda => BackendKind::Cuda,
        BackendChoice::Hip => BackendKind::Hip,
        BackendChoice::All => {
            return Err(CliError("symbols requires one concrete backend".to_owned()));
        }
    };
    let (statuses, runtime_profile, runtime_profile_display, library_error) =
        match ocgpu::backend_diagnostics(backend) {
            Ok(diagnostics) => (
                diagnostics
                    .symbols
                    .into_iter()
                    .map(|status| (status.name, status))
                    .collect::<HashMap<_, _>>(),
                diagnostics
                    .hip_runtime_profile
                    .map(ocgpu::HipRuntimeProfile::as_str),
                diagnostics
                    .hip_runtime_profile
                    .map(ocgpu::HipRuntimeProfile::display_name),
                None,
            ),
            Err(error) => (HashMap::new(), None, None, Some(error.to_string())),
        };
    let coverage = load_coverage()?;
    let runtime_symbols = consolidated_runtime_symbols(coverage.symbols, backend)?;
    let mut displayed = Vec::new();
    for symbol in runtime_symbols {
        let status = statuses.get(symbol.name.as_str());
        let inventory_applicable = applies_to_target(&symbol.platforms, CURRENT_TARGET);
        let runtime_applicable =
            inventory_applicable && status.is_none_or(|status| status.applicable);
        let available =
            runtime_applicable && status.is_some_and(ocgpu::RuntimeSymbolStatus::available);
        if matches!(filter, SymbolFilter::Available) && !available {
            continue;
        }
        if matches!(filter, SymbolFilter::Missing) && (!runtime_applicable || available) {
            continue;
        }
        let (runtime_resolution, resolved_name) = if inventory_applicable {
            status.map_or(("not_resolved", None), |status| {
                (
                    symbol_resolution_name(status.resolution),
                    status.resolved_name,
                )
            })
        } else {
            ("platform_unavailable", None)
        };
        displayed.push(DisplayedSymbol {
            coverage: symbol,
            runtime_available: available,
            runtime_resolution,
            resolved_name,
            runtime_required: runtime_applicable && status.is_some_and(|status| status.required),
            runtime_applicable,
        });
    }
    let output = SymbolOutput {
        backend: backend.as_str(),
        target: CURRENT_TARGET,
        runtime_profile,
        runtime_profile_display,
        library_error,
        symbols: displayed,
    };
    if json {
        print_json(&output)?;
    } else {
        if let Some(profile) = output.runtime_profile_display {
            println!("{} runtime profile: {profile}", output.backend);
        }
        if let Some(error) = &output.library_error {
            eprintln!("{}: {error}", output.backend);
        }
        for symbol in &output.symbols {
            println!(
                "{} {:<12} {}",
                if symbol.runtime_available { "+" } else { "-" },
                symbol.runtime_resolution,
                symbol.coverage.name
            );
        }
    }
    Ok(ExitCode::SUCCESS)
}

const fn symbol_resolution_name(resolution: SymbolResolution) -> &'static str {
    match resolution {
        SymbolResolution::ProcAddress => "proc_address",
        SymbolResolution::Direct => "direct",
        SymbolResolution::DirectAdapter => "direct_adapter",
        SymbolResolution::Missing => "missing",
        SymbolResolution::ProfileUnavailable => "profile_unavailable",
        SymbolResolution::PlatformUnavailable => "platform_unavailable",
    }
}

fn is_runtime_callable_kind(kind: &str) -> bool {
    matches!(kind, "function" | "alias")
}

fn consolidated_runtime_symbols(
    symbols: Vec<CoverageSymbol>,
    backend: BackendKind,
) -> Result<Vec<RuntimeCoverageSymbol>, CliError> {
    let mut by_name = BTreeMap::<String, RuntimeCoverageSymbol>::new();
    for symbol in symbols.into_iter().filter(|symbol| {
        symbol.backend.eq_ignore_ascii_case(backend.as_str())
            && is_runtime_callable_kind(&symbol.kind)
            && symbol.runtime_resolvable
    }) {
        if let Some(existing) = by_name.get_mut(&symbol.name) {
            existing.merge(symbol)?;
        } else {
            by_name.insert(
                symbol.name.clone(),
                RuntimeCoverageSymbol::from_coverage(symbol),
            );
        }
    }
    Ok(by_name.into_values().collect())
}

fn applies_to_target(platforms: &[String], target: &str) -> bool {
    platforms.iter().any(|platform| platform == target)
}

#[derive(Debug, Serialize)]
struct AbiReport {
    abi_version: u32,
    pointer_width: u32,
    common_table_size: usize,
    cuda_table_size: usize,
    hip_table_size: usize,
    result_size: usize,
    backend_size: usize,
    device_size: usize,
    device_pointer_size: usize,
}

fn abi(json: bool) -> Result<ExitCode, CliError> {
    let report = AbiReport {
        abi_version: ocgpu::sys::OCGPU_ABI_VERSION_1,
        pointer_width: usize::BITS,
        common_table_size: size_of::<ocgpu::sys::ocgpuApi_v1>(),
        cuda_table_size: size_of::<ocgpu::sys::ocgpuCuApi_v1>(),
        hip_table_size: size_of::<ocgpu::sys::ocgpuHipApi_v1>(),
        result_size: size_of::<ocgpu::sys::ocgpuResult>(),
        backend_size: size_of::<ocgpu::sys::ocgpuBackend>(),
        device_size: size_of::<ocgpu::sys::ocgpuDevice>(),
        device_pointer_size: size_of::<ocgpu::sys::ocgpuDeviceptr>(),
    };
    if json {
        print_json(&report)?;
    } else {
        println!("ABI version: 0x{:08x}", report.abi_version);
        println!("pointer width: {}", report.pointer_width);
        println!("common table: {} bytes", report.common_table_size);
        println!("CUDA raw table: {} bytes", report.cuda_table_size);
        println!("HIP raw table: {} bytes", report.hip_table_size);
    }
    Ok(ExitCode::SUCCESS)
}

fn coverage(json: bool) -> Result<ExitCode, CliError> {
    let bytes = coverage_bytes()?;
    if json {
        let text = std::str::from_utf8(&bytes)
            .map_err(|error| CliError(format!("embedded coverage is not UTF-8: {error}")))?;
        println!("{text}");
    } else {
        let document: CoverageDocument = serde_json::from_slice(&bytes)?;
        println!("coverage schema: {}", document.schema_version);
        for metric in document.metrics {
            println!(
                "{}: {}/{} ({})",
                metric.label, metric.numerator, metric.denominator, metric.id
            );
            println!("  basis: {}", metric.basis);
        }
        println!("classifications:");
        for (classification, count) in document.classification_counts {
            println!("  {classification}: {count}");
        }
    }
    Ok(ExitCode::SUCCESS)
}

fn module_inspect(path: &Path, json: bool) -> Result<ExitCode, CliError> {
    let info = inspect::inspect(path).map_err(|error| CliError(error.to_string()))?;
    if json {
        print_json(&info)?;
    } else {
        println!("format: {}", info.format);
        println!("size: {} bytes", info.size_bytes);
        if let Some(width) = info.pointer_width {
            println!("pointer width: {width}");
        }
        if let Some(machine) = info.machine {
            println!("ELF machine: {machine}");
        }
        if let Some(version) = &info.ptx_version {
            println!("PTX version: {version}");
        }
        if let Some(target) = &info.ptx_target {
            println!("PTX target: {target}");
        }
        for target in &info.amdgpu_target_ids {
            println!("AMDGPU target ID: {target}");
        }
        for architecture in &info.amdgpu_architectures {
            println!("AMDGPU architecture: {architecture}");
        }
    }
    Ok(ExitCode::SUCCESS)
}

fn selected_backends(choice: BackendChoice) -> Vec<BackendKind> {
    match choice {
        BackendChoice::Cuda => vec![BackendKind::Cuda],
        BackendChoice::Hip => vec![BackendKind::Hip],
        BackendChoice::All => vec![BackendKind::Cuda, BackendKind::Hip],
    }
}

fn load_coverage() -> Result<CoverageDocument, CliError> {
    Ok(serde_json::from_slice(&coverage_bytes()?)?)
}

fn coverage_bytes() -> Result<Vec<u8>, CliError> {
    decompress_to_vec(COVERAGE_DEFLATE)
        .map_err(|error| CliError(format!("embedded coverage decompression failed: {error:?}")))
}

fn print_json(value: &impl Serialize) -> Result<(), CliError> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        applies_to_target, consolidated_runtime_symbols, coverage_bytes, is_runtime_callable_kind,
        load_coverage, report_from_diagnostics, selected_backends, symbol_resolution_name,
    };
    use crate::args::BackendChoice;
    use ocgpu::{
        BackendDiagnostics, BackendKind, HipRuntimeProfile, RuntimeSymbolStatus, SymbolResolution,
    };
    use std::ffi::OsString;
    use std::path::PathBuf;

    #[test]
    fn embedded_coverage_is_valid_and_versioned() {
        assert!(!coverage_bytes().expect("decompresses").is_empty());
        assert_eq!(load_coverage().expect("valid JSON").schema_version, 1);
    }

    #[test]
    fn all_backend_order_is_deterministic() {
        assert_eq!(
            selected_backends(BackendChoice::All),
            vec![BackendKind::Cuda, BackendKind::Hip]
        );
    }

    #[test]
    fn hip_profile_is_machine_readable_without_loading_a_runtime() {
        let diagnostics = BackendDiagnostics {
            backend: BackendKind::Hip,
            library_path: PathBuf::from("mock-amdhip64_6"),
            runtime_version: Some(60_400_123),
            driver_version: Some(60_400_123),
            compiled_api_version: 70_200_000,
            hip_runtime_profile: Some(HipRuntimeProfile::Hip6),
            proc_address_support: false,
            proc_address_variant: None,
            loaded_architecture: "x86_64",
            symbols: Vec::new(),
        };
        let report = report_from_diagnostics(&diagnostics, Some(1), None);
        assert_eq!(report.runtime_profile, Some("hip_6"));
        assert_eq!(report.runtime_profile_display, Some("HIP 6"));
        assert_eq!(report.profile_specific_omissions, Some(0));
        let json = serde_json::to_value(report).expect("backend report serializes");
        assert_eq!(json["runtime_profile"], "hip_6");
        assert!(json.get("runtime_profile_display").is_none());
    }

    #[test]
    fn profile_unavailable_has_a_distinct_stable_cli_spelling() {
        assert_eq!(
            symbol_resolution_name(SymbolResolution::ProfileUnavailable),
            "profile_unavailable"
        );
        assert_ne!(
            symbol_resolution_name(SymbolResolution::ProfileUnavailable),
            symbol_resolution_name(SymbolResolution::Missing)
        );
    }

    #[test]
    fn direct_adapter_has_a_distinct_stable_cli_spelling() {
        let status = RuntimeSymbolStatus {
            name: "hipMemcpyHtoD",
            resolved_name: Some("hipMemcpyHtoD"),
            resolution: SymbolResolution::DirectAdapter,
            proc_attempts: 0,
            required: true,
            applicable: true,
        };
        assert!(
            !status.available(),
            "the symbols command must classify its null raw slot as unavailable"
        );
        assert_eq!(
            symbol_resolution_name(SymbolResolution::DirectAdapter),
            "direct_adapter"
        );
        assert_ne!(
            symbol_resolution_name(SymbolResolution::DirectAdapter),
            symbol_resolution_name(SymbolResolution::Direct)
        );
    }

    #[test]
    fn platform_specific_oracle_rows_do_not_leak_across_targets() {
        let linux = vec!["x86_64-unknown-linux-gnu".to_owned()];
        assert!(applies_to_target(&linux, "x86_64-unknown-linux-gnu"));
        assert!(!applies_to_target(&linux, "x86_64-pc-windows-msvc"));
    }

    #[test]
    fn functions_and_callable_aliases_are_runtime_symbols() {
        assert!(is_runtime_callable_kind("function"));
        assert!(is_runtime_callable_kind("alias"));
        assert!(!is_runtime_callable_kind("constant"));
        assert!(!is_runtime_callable_kind("type"));
    }

    #[test]
    fn runtime_symbol_inventory_is_unique_and_complete() {
        let cuda = consolidated_runtime_symbols(
            load_coverage().expect("valid coverage").symbols,
            BackendKind::Cuda,
        )
        .expect("consistent CUDA rows");
        assert_eq!(cuda.len(), 573);
        assert!(cuda.windows(2).all(|pair| pair[0].name < pair[1].name));
        assert!(cuda.iter().any(|symbol| symbol.kind == "alias"));

        let hip = consolidated_runtime_symbols(
            load_coverage().expect("valid coverage").symbols,
            BackendKind::Hip,
        )
        .expect("consistent HIP rows");
        assert_eq!(hip.len(), 535);
        assert!(hip.windows(2).all(|pair| pair[0].name < pair[1].name));
        assert!(hip.iter().any(|symbol| symbol.kind == "alias"));
    }

    #[test]
    fn runtime_merge_preserves_reviewed_vendor_deprecation() {
        assert_eq!(
            super::merge_runtime_classification("covered_raw_only", "deprecated_covered")
                .as_deref(),
            Some("deprecated_covered")
        );
        assert_eq!(
            super::merge_runtime_classification("deprecated_covered", "covered_raw_only")
                .as_deref(),
            Some("deprecated_covered")
        );
        assert!(
            super::merge_runtime_classification("covered_adapter", "covered_raw_only").is_none()
        );
    }

    #[test]
    fn help_mentions_every_command() {
        for command in [
            "backends",
            "devices",
            "doctor",
            "symbols",
            "abi",
            "coverage",
            "module inspect",
        ] {
            assert!(super::HELP.contains(command));
        }
    }

    #[test]
    fn os_string_import_remains_supported_by_parser_contract() {
        let _: OsString = "backends".into();
    }
}
