// SPDX-License-Identifier: CC0-1.0

use std::ffi::OsString;
use std::fmt;
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendChoice {
    Cuda,
    Hip,
    All,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SymbolFilter {
    Available,
    Missing,
    All,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Command {
    Backends {
        json: bool,
    },
    Devices {
        backend: BackendChoice,
        json: bool,
    },
    Doctor {
        strict: bool,
        json: bool,
    },
    Symbols {
        backend: BackendChoice,
        filter: SymbolFilter,
        json: bool,
    },
    Abi {
        json: bool,
    },
    Coverage {
        json: bool,
    },
    ModuleInspect {
        path: PathBuf,
        json: bool,
    },
    Help,
    Version,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParseError {
    message: String,
}

impl ParseError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ParseError {}

pub fn parse<I>(arguments: I) -> Result<Command, ParseError>
where
    I: IntoIterator<Item = OsString>,
{
    let mut arguments = arguments.into_iter();
    let Some(command) = arguments.next() else {
        return Ok(Command::Help);
    };
    let command = command
        .into_string()
        .map_err(|_| ParseError::new("command is not valid Unicode"))?;
    let rest: Vec<OsString> = arguments.collect();

    match command.as_str() {
        "backends" => parse_backends(&rest),
        "devices" => parse_devices(&rest),
        "doctor" => parse_doctor(&rest),
        "symbols" => parse_symbols(&rest),
        "abi" => parse_json_only("abi", &rest).map(|json| Command::Abi { json }),
        "coverage" => parse_json_only("coverage", &rest).map(|json| Command::Coverage { json }),
        "module" => parse_module(rest),
        "help" | "--help" | "-h" => no_arguments("help", &rest, Command::Help),
        "version" | "--version" | "-V" => no_arguments("version", &rest, Command::Version),
        unknown => Err(ParseError::new(format!(
            "unknown command `{unknown}`; run `ocgpu help` for usage"
        ))),
    }
}

fn parse_backends(arguments: &[OsString]) -> Result<Command, ParseError> {
    parse_json_only("backends", arguments).map(|json| Command::Backends { json })
}

fn parse_devices(arguments: &[OsString]) -> Result<Command, ParseError> {
    let mut json = false;
    let mut backend = BackendChoice::All;
    let mut backend_seen = false;
    let mut index = 0;
    while index < arguments.len() {
        let argument = unicode(&arguments[index], "devices option")?;
        match argument {
            "--json" => set_once(&mut json, "--json")?,
            "--backend" => {
                if backend_seen {
                    return Err(ParseError::new("--backend may be supplied only once"));
                }
                index += 1;
                let value = arguments
                    .get(index)
                    .ok_or_else(|| ParseError::new("--backend requires cuda, hip, or all"))?;
                backend = parse_backend(unicode(value, "backend")?, true)?;
                backend_seen = true;
            }
            unknown => return Err(unexpected("devices", unknown)),
        }
        index += 1;
    }
    Ok(Command::Devices { backend, json })
}

fn parse_doctor(arguments: &[OsString]) -> Result<Command, ParseError> {
    let mut json = false;
    let mut strict = false;
    for argument in arguments {
        match unicode(argument, "doctor option")? {
            "--json" => set_once(&mut json, "--json")?,
            "--strict" => set_once(&mut strict, "--strict")?,
            unknown => return Err(unexpected("doctor", unknown)),
        }
    }
    Ok(Command::Doctor { strict, json })
}

fn parse_symbols(arguments: &[OsString]) -> Result<Command, ParseError> {
    let mut backend = None;
    let mut filter = SymbolFilter::All;
    let mut filter_seen = false;
    let mut json = false;
    let mut index = 0;
    while index < arguments.len() {
        let argument = unicode(&arguments[index], "symbols option")?;
        match argument {
            "--backend" => {
                if backend.is_some() {
                    return Err(ParseError::new("--backend may be supplied only once"));
                }
                index += 1;
                let value = arguments
                    .get(index)
                    .ok_or_else(|| ParseError::new("--backend requires cuda or hip"))?;
                backend = Some(parse_backend(unicode(value, "backend")?, false)?);
            }
            "--available" => set_filter(
                &mut filter,
                &mut filter_seen,
                SymbolFilter::Available,
                "--available",
            )?,
            "--missing" => set_filter(
                &mut filter,
                &mut filter_seen,
                SymbolFilter::Missing,
                "--missing",
            )?,
            "--all" => set_filter(&mut filter, &mut filter_seen, SymbolFilter::All, "--all")?,
            "--json" => set_once(&mut json, "--json")?,
            unknown => return Err(unexpected("symbols", unknown)),
        }
        index += 1;
    }
    let backend = backend
        .ok_or_else(|| ParseError::new("symbols requires exactly one --backend cuda|hip option"))?;
    Ok(Command::Symbols {
        backend,
        filter,
        json,
    })
}

fn parse_module(arguments: Vec<OsString>) -> Result<Command, ParseError> {
    let mut arguments = arguments.into_iter();
    let subcommand = arguments
        .next()
        .ok_or_else(|| ParseError::new("module requires the `inspect` subcommand"))?;
    if unicode(&subcommand, "module subcommand")? != "inspect" {
        return Err(ParseError::new(
            "module supports only `ocgpu module inspect FILE [--json]`",
        ));
    }

    let mut path = None;
    let mut json = false;
    for argument in arguments {
        let value = unicode(&argument, "module inspect argument")?;
        if value == "--json" {
            set_once(&mut json, "--json")?;
        } else if value.starts_with('-') {
            return Err(unexpected("module inspect", value));
        } else if path.replace(PathBuf::from(argument)).is_some() {
            return Err(ParseError::new(
                "module inspect accepts exactly one input file",
            ));
        }
    }
    let path = path.ok_or_else(|| ParseError::new("module inspect requires an input file"))?;
    Ok(Command::ModuleInspect { path, json })
}

fn parse_json_only(command: &str, arguments: &[OsString]) -> Result<bool, ParseError> {
    let mut json = false;
    for argument in arguments {
        match unicode(argument, "option")? {
            "--json" => set_once(&mut json, "--json")?,
            unknown => return Err(unexpected(command, unknown)),
        }
    }
    Ok(json)
}

fn no_arguments(
    command: &str,
    arguments: &[OsString],
    result: Command,
) -> Result<Command, ParseError> {
    if arguments.is_empty() {
        Ok(result)
    } else {
        Err(ParseError::new(format!(
            "{command} does not accept arguments"
        )))
    }
}

fn parse_backend(value: &str, allow_all: bool) -> Result<BackendChoice, ParseError> {
    match value {
        "cuda" => Ok(BackendChoice::Cuda),
        "hip" => Ok(BackendChoice::Hip),
        "all" if allow_all => Ok(BackendChoice::All),
        _ if allow_all => Err(ParseError::new("backend must be cuda, hip, or all")),
        _ => Err(ParseError::new("backend must be cuda or hip")),
    }
}

fn set_once(value: &mut bool, option: &str) -> Result<(), ParseError> {
    if *value {
        Err(ParseError::new(format!(
            "{option} may be supplied only once"
        )))
    } else {
        *value = true;
        Ok(())
    }
}

fn set_filter(
    filter: &mut SymbolFilter,
    seen: &mut bool,
    requested: SymbolFilter,
    option: &str,
) -> Result<(), ParseError> {
    if *seen {
        Err(ParseError::new(format!(
            "{option} conflicts with the previously supplied symbol filter"
        )))
    } else {
        *filter = requested;
        *seen = true;
        Ok(())
    }
}

fn unicode<'a>(value: &'a OsString, context: &str) -> Result<&'a str, ParseError> {
    value
        .to_str()
        .ok_or_else(|| ParseError::new(format!("{context} is not valid Unicode")))
}

fn unexpected(command: &str, argument: &str) -> ParseError {
    ParseError::new(format!("unexpected {command} argument `{argument}`"))
}

#[cfg(test)]
mod tests {
    use super::{BackendChoice, Command, SymbolFilter, parse};
    use std::ffi::OsString;

    fn args(values: &[&str]) -> Vec<OsString> {
        values.iter().map(OsString::from).collect()
    }

    #[test]
    fn no_command_displays_help() {
        assert_eq!(parse(Vec::new()).expect("valid"), Command::Help);
    }

    #[test]
    fn parses_every_required_command() {
        assert_eq!(
            parse(args(&["backends", "--json"])).expect("valid"),
            Command::Backends { json: true }
        );
        assert_eq!(
            parse(args(&["devices", "--backend", "hip"])).expect("valid"),
            Command::Devices {
                backend: BackendChoice::Hip,
                json: false,
            }
        );
        assert_eq!(
            parse(args(&["doctor", "--json", "--strict"])).expect("valid"),
            Command::Doctor {
                strict: true,
                json: true,
            }
        );
        assert_eq!(
            parse(args(&[
                "symbols",
                "--missing",
                "--backend",
                "cuda",
                "--json",
            ]))
            .expect("valid"),
            Command::Symbols {
                backend: BackendChoice::Cuda,
                filter: SymbolFilter::Missing,
                json: true,
            }
        );
        assert_eq!(
            parse(args(&["abi"])).expect("valid"),
            Command::Abi { json: false }
        );
        assert_eq!(
            parse(args(&["coverage", "--json"])).expect("valid"),
            Command::Coverage { json: true }
        );
        assert_eq!(
            parse(args(&["module", "inspect", "kernel.ptx", "--json"])).expect("valid"),
            Command::ModuleInspect {
                path: "kernel.ptx".into(),
                json: true,
            }
        );
    }

    #[test]
    fn devices_defaults_to_all_backends() {
        assert_eq!(
            parse(args(&["devices"])).expect("valid"),
            Command::Devices {
                backend: BackendChoice::All,
                json: false,
            }
        );
    }

    #[test]
    fn symbols_requires_one_specific_backend() {
        let missing = parse(args(&["symbols"])).expect_err("backend is required");
        assert!(missing.to_string().contains("requires exactly one"));
        let all = parse(args(&["symbols", "--backend", "all"]))
            .expect_err("all is not a concrete backend");
        assert!(all.to_string().contains("cuda or hip"));
    }

    #[test]
    fn conflicting_filters_are_rejected() {
        let error = parse(args(&[
            "symbols",
            "--backend",
            "cuda",
            "--available",
            "--missing",
        ]))
        .expect_err("filters conflict");
        assert!(error.to_string().contains("conflicts"));
    }

    #[test]
    fn duplicate_boolean_options_are_rejected() {
        let error = parse(args(&["doctor", "--strict", "--strict"])).expect_err("duplicate option");
        assert!(error.to_string().contains("only once"));
    }

    #[test]
    fn inspect_requires_exactly_one_file() {
        assert!(parse(args(&["module", "inspect"])).is_err());
        assert!(parse(args(&["module", "inspect", "a", "b"])).is_err());
    }
}
