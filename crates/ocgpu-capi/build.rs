// SPDX-License-Identifier: CC0-1.0

//! Selects the generated Windows export definition that matches this build's
//! public feature surface.

use std::env;
use std::error::Error;
use std::io;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn Error>> {
    let manifest_directory = env::var_os("CARGO_MANIFEST_DIR").ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "Cargo did not provide CARGO_MANIFEST_DIR",
        )
    })?;
    let export_directory = PathBuf::from(manifest_directory).join("../../exports");
    let flat = env::var_os("CARGO_FEATURE_FLAT_C_EXPORTS").is_some();
    let definition_name = if flat { "ocgpu-flat.def" } else { "ocgpu.def" };
    let map_name = if flat { "ocgpu-flat.map" } else { "ocgpu.map" };
    let definition = export_directory.join(definition_name);
    let version_map = export_directory.join(map_name);

    println!("cargo:rerun-if-changed={}", definition.display());
    println!("cargo:rerun-if-changed={}", version_map.display());

    if env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        if !definition.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "generated export definition is missing: {}",
                    definition.display()
                ),
            )
            .into());
        }
        println!("cargo:rustc-cdylib-link-arg=/DEF:{}", definition.display());
    }

    Ok(())
}
