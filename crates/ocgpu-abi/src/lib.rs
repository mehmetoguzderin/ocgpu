// SPDX-License-Identifier: CC0-1.0

#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(non_camel_case_types, non_snake_case)]
#![doc = "Generated, SDK-independent C ABI declarations for ocgpu."]

#[cfg(not(target_pointer_width = "64"))]
compile_error!("ocgpu ABI version 1 supports only 64-bit targets");

mod generated;
mod rtc;

pub use generated::*;
pub use rtc::*;
