// SPDX-License-Identifier: CC0-1.0

//! Command-line entry point for deterministic ocgpu artifact generation.

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use ocgpu_codegen::{Mode, run, sync_rust_oracles};

enum Action {
    Generate(Mode),
    SyncRustOracles,
}

fn usage() -> &'static str {
    "usage: ocgpu-codegen <generate [--check] | check | sync-rust-oracles> [--workspace-root PATH]"
}

fn parse_args() -> Result<(Action, PathBuf), String> {
    let mut args = env::args().skip(1);
    let command = args.next().ok_or_else(|| usage().to_owned())?;
    let mut action = match command.as_str() {
        "generate" => Action::Generate(Mode::Generate),
        "check" => Action::Generate(Mode::Check),
        "sync-rust-oracles" => Action::SyncRustOracles,
        _ => return Err(format!("unknown command {command:?}\n{}", usage())),
    };
    let mut workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .expect("crate is nested below the workspace root")
        .to_path_buf();

    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--check" if command == "generate" => action = Action::Generate(Mode::Check),
            "--workspace-root" => {
                workspace_root = args
                    .next()
                    .map(PathBuf::from)
                    .ok_or_else(|| "--workspace-root requires a path".to_owned())?;
            }
            _ => return Err(format!("unknown argument {argument:?}\n{}", usage())),
        }
    }

    Ok((action, workspace_root))
}

fn main() -> ExitCode {
    let result = parse_args().and_then(|(action, root)| match action {
        Action::Generate(mode) => run(&root, mode)
            .map(|report| report.summary())
            .map_err(|error| error.to_string()),
        Action::SyncRustOracles => sync_rust_oracles(&root)
            .map(|report| report.summary())
            .map_err(|error| error.to_string()),
    });
    match result {
        Ok(summary) => {
            println!("{summary}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("ocgpu-codegen: {error}");
            ExitCode::FAILURE
        }
    }
}
