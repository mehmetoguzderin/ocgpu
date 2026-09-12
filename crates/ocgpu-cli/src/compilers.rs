// SPDX-License-Identifier: CC0-1.0

//! Independent runtime-compiler diagnostics without driver initialization.

use crate::args::BackendChoice;
use crate::{CliError, print_json};
use ocgpu::rtc::{Compiler, Hiprtc, Nvrtc, RtcBackend};
use serde::Serialize;
use std::process::ExitCode;

#[derive(Debug, Serialize)]
struct CompilerReport {
    compiler: &'static str,
    backend: &'static str,
    available: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    library_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    version: Option<[i32; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

fn report<B: RtcBackend>(
    compiler: &'static str,
    backend: &'static str,
    loaded: Result<Compiler<B>, ocgpu::rtc::Error>,
) -> CompilerReport {
    let mut report = CompilerReport {
        compiler,
        backend,
        available: false,
        library_path: None,
        version: None,
        error: None,
    };
    match loaded {
        Ok(loaded) => {
            report.library_path = Some(loaded.loaded_path().display().to_string());
            match loaded.version() {
                Ok((major, minor)) => {
                    report.version = Some([major, minor]);
                    report.available = true;
                }
                Err(error) => report.error = Some(error.to_string()),
            }
        }
        Err(error) => report.error = Some(error.to_string()),
    }
    report
}

pub fn run(backend: BackendChoice, json: bool) -> Result<ExitCode, CliError> {
    let mut reports = Vec::new();
    if matches!(backend, BackendChoice::Cuda | BackendChoice::All) {
        reports.push(report("nvrtc", "cuda", Compiler::<Nvrtc>::load()));
    }
    if matches!(backend, BackendChoice::Hip | BackendChoice::All) {
        reports.push(report("hiprtc", "hip", Compiler::<Hiprtc>::load()));
    }
    if json {
        print_json(&reports)?;
    } else {
        for report in &reports {
            let status = if report.available {
                "available"
            } else {
                "unavailable"
            };
            println!("{} ({}): {status}", report.compiler, report.backend);
            if let Some(path) = &report.library_path {
                println!("  library: {path}");
            }
            if let Some([major, minor]) = report.version {
                println!("  version: {major}.{minor}");
            }
            if let Some(error) = &report.error {
                println!("  error: {error}");
            }
        }
    }
    Ok(ExitCode::SUCCESS)
}

#[cfg(test)]
mod tests {
    use super::report;
    use ocgpu::rtc::{CompilerKind, Error, Hiprtc, Nvrtc, RtcFailure};
    use serde_json::json;

    #[test]
    fn unavailable_compiler_json_omits_unobserved_library_and_version() {
        let diagnostic = report::<Nvrtc>(
            "nvrtc",
            "cuda",
            Err(Error::MissingRequiredSymbols {
                compiler: CompilerKind::Nvrtc,
                symbols: vec!["nvrtcCreateProgram"],
            }),
        );
        assert_eq!(
            serde_json::to_value(diagnostic).expect("compiler report serializes"),
            json!({
                "compiler": "nvrtc",
                "backend": "cuda",
                "available": false,
                "error": "NVRTC is missing mandatory symbols: nvrtcCreateProgram",
            })
        );
    }

    #[test]
    fn independent_failures_preserve_both_compiler_identities_and_diagnostics() {
        let nvrtc = report::<Nvrtc>(
            "nvrtc",
            "cuda",
            Err(Error::MissingRequiredSymbols {
                compiler: CompilerKind::Nvrtc,
                symbols: vec!["nvrtcGetPTX"],
            }),
        );
        let hiprtc = report::<Hiprtc>(
            "hiprtc",
            "hip",
            Err(Error::Rtc(RtcFailure {
                compiler: CompilerKind::Hiprtc,
                operation: "Version",
                result: 11,
                message: "HIPRTC_ERROR_INTERNAL_ERROR".to_owned(),
            })),
        );
        let reports = serde_json::to_value([nvrtc, hiprtc]).expect("both reports serialize");
        assert_eq!(reports.as_array().expect("report array").len(), 2);
        assert_eq!(reports[0]["compiler"], "nvrtc");
        assert_eq!(reports[0]["backend"], "cuda");
        assert_eq!(reports[0]["available"], false);
        assert_eq!(
            reports[0]["error"],
            "NVRTC is missing mandatory symbols: nvrtcGetPTX"
        );
        assert_eq!(reports[1]["compiler"], "hiprtc");
        assert_eq!(reports[1]["backend"], "hip");
        assert_eq!(reports[1]["available"], false);
        assert_eq!(
            reports[1]["error"],
            "HIPRTC Version failed with 11: HIPRTC_ERROR_INTERNAL_ERROR"
        );
    }
}
