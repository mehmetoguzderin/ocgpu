// SPDX-License-Identifier: CC0-1.0

use serde::Serialize;
use std::fmt;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

const INSPECTION_PREFIX_LIMIT: u64 = 64 * 1024;
const ELF_MACHINE_CUDA: u16 = 190;
const ELF_MACHINE_AMDGPU: u16 = 224;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModuleFormat {
    Ptx,
    Cubin,
    CudaFatBinary,
    Hsaco,
    HipFatBinary,
    Elf,
    Unknown,
}

impl fmt::Display for ModuleFormat {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Ptx => "PTX",
            Self::Cubin => "CUDA cubin",
            Self::CudaFatBinary => "CUDA fat binary",
            Self::Hsaco => "AMDGPU code object (HSACO)",
            Self::HipFatBinary => "HIP/Clang offload bundle",
            Self::Elf => "ELF",
            Self::Unknown => "unknown",
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ModuleInfo {
    pub format: ModuleFormat,
    pub size_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pointer_width: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endianness: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub machine: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ptx_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ptx_target: Option<String>,
}

#[derive(Debug)]
pub enum InspectionError {
    Io(io::Error),
    Empty,
}

impl fmt::Display for InspectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "module inspection failed: {error}"),
            Self::Empty => formatter.write_str("module file is empty"),
        }
    }
}

impl std::error::Error for InspectionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Empty => None,
        }
    }
}

impl From<io::Error> for InspectionError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

pub fn inspect(path: &Path) -> Result<ModuleInfo, InspectionError> {
    let file = File::open(path)?;
    let size = file.metadata()?.len();
    if size == 0 {
        return Err(InspectionError::Empty);
    }
    let mut prefix =
        Vec::with_capacity(usize::try_from(size.min(INSPECTION_PREFIX_LIMIT)).unwrap_or(64 * 1024));
    file.take(INSPECTION_PREFIX_LIMIT)
        .read_to_end(&mut prefix)?;
    Ok(inspect_prefix(&prefix, size))
}

fn inspect_prefix(prefix: &[u8], size: u64) -> ModuleInfo {
    if prefix.starts_with(b"\x7fELF") && prefix.len() >= 20 {
        return inspect_elf(prefix, size);
    }
    if prefix.starts_with(&[0xb1, 0x43, 0x62, 0x46]) {
        return basic(ModuleFormat::CudaFatBinary, size);
    }
    if prefix.starts_with(b"__CLANG_OFFLOAD_BUNDLE__")
        || prefix.windows(10).any(|window| window == b"hip-fatbin")
    {
        return basic(ModuleFormat::HipFatBinary, size);
    }
    if let Ok(text) = std::str::from_utf8(prefix) {
        if text
            .lines()
            .any(|line| line.trim_start().starts_with(".version"))
        {
            return inspect_ptx(text, size);
        }
    }
    basic(ModuleFormat::Unknown, size)
}

fn inspect_elf(prefix: &[u8], size: u64) -> ModuleInfo {
    let pointer_width = match prefix[4] {
        1 => Some(32),
        2 => Some(64),
        _ => None,
    };
    let (endianness, machine) = match prefix[5] {
        1 => (
            Some("little"),
            Some(u16::from_le_bytes([prefix[18], prefix[19]])),
        ),
        2 => (
            Some("big"),
            Some(u16::from_be_bytes([prefix[18], prefix[19]])),
        ),
        _ => (None, None),
    };
    let format = match machine {
        Some(ELF_MACHINE_CUDA) => ModuleFormat::Cubin,
        Some(ELF_MACHINE_AMDGPU) => ModuleFormat::Hsaco,
        _ => ModuleFormat::Elf,
    };
    ModuleInfo {
        format,
        size_bytes: size,
        pointer_width,
        endianness,
        machine,
        ptx_version: None,
        ptx_target: None,
    }
}

fn inspect_ptx(text: &str, size: u64) -> ModuleInfo {
    let mut version = None;
    let mut target = None;
    let mut pointer_width = None;
    for line in text.lines() {
        let line = line.trim();
        if let Some(value) = line.strip_prefix(".version") {
            version = Some(value.trim().to_owned());
        } else if let Some(value) = line.strip_prefix(".target") {
            target = Some(value.trim().to_owned());
        } else if let Some(value) = line.strip_prefix(".address_size") {
            pointer_width = value.trim().parse::<u8>().ok();
        }
    }
    ModuleInfo {
        format: ModuleFormat::Ptx,
        size_bytes: size,
        pointer_width,
        endianness: None,
        machine: None,
        ptx_version: version,
        ptx_target: target,
    }
}

const fn basic(format: ModuleFormat, size_bytes: u64) -> ModuleInfo {
    ModuleInfo {
        format,
        size_bytes,
        pointer_width: None,
        endianness: None,
        machine: None,
        ptx_version: None,
        ptx_target: None,
    }
}

#[cfg(test)]
mod tests {
    use super::{ModuleFormat, inspect_prefix};

    #[test]
    fn recognizes_ptx_metadata() {
        let input = b"// fixture\n.version 8.0\n.target sm_75\n.address_size 64\n";
        let info = inspect_prefix(input, input.len() as u64);
        assert_eq!(info.format, ModuleFormat::Ptx);
        assert_eq!(info.ptx_version.as_deref(), Some("8.0"));
        assert_eq!(info.ptx_target.as_deref(), Some("sm_75"));
        assert_eq!(info.pointer_width, Some(64));
    }

    #[test]
    fn distinguishes_cuda_and_amdgpu_elf_machines() {
        let mut elf = [0_u8; 20];
        elf[..4].copy_from_slice(b"\x7fELF");
        elf[4] = 2;
        elf[5] = 1;
        elf[18..20].copy_from_slice(&190_u16.to_le_bytes());
        assert_eq!(inspect_prefix(&elf, 20).format, ModuleFormat::Cubin);
        elf[18..20].copy_from_slice(&224_u16.to_le_bytes());
        assert_eq!(inspect_prefix(&elf, 20).format, ModuleFormat::Hsaco);
    }

    #[test]
    fn recognizes_cuda_fat_binary_magic() {
        assert_eq!(
            inspect_prefix(&[0xb1, 0x43, 0x62, 0x46], 4).format,
            ModuleFormat::CudaFatBinary
        );
    }
}
