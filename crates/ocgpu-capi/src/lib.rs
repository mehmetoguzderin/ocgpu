// SPDX-License-Identifier: CC0-1.0

//! Generated C exports backed by panic-contained Rust delegates.

#![allow(non_snake_case)]

mod implementation;

include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/generated_exports.rs"
));
