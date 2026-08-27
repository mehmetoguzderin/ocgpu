// SPDX-License-Identifier: CC0-1.0

use serde::Serialize;
use std::collections::BTreeSet;
use std::fmt;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

const INSPECTION_PREFIX_LIMIT: u64 = 64 * 1024;
const ELF_MACHINE_CUDA: u16 = 190;
const ELF_MACHINE_AMDGPU: u16 = 224;
const ELF_MACHINE_FLAGS_MASK_AMDGPU: u32 = 0xff;
const CLANG_BUNDLE_MAGIC: &[u8] = b"__CLANG_OFFLOAD_BUNDLE__";
const CLANG_BUNDLE_DESCRIPTOR_BYTES: u64 = 24;
const MAX_CLANG_BUNDLE_ENTRIES: u64 = 256;
const MAX_CLANG_BUNDLE_ENTRY_ID_BYTES: u64 = 1024;
const MAX_AMDGPU_TARGET_ID_BYTES: usize = 128;

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
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub amdgpu_target_ids: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub amdgpu_architectures: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ValidationError {
    TruncatedHeader {
        field: &'static str,
    },
    InspectionLimitExceeded {
        field: &'static str,
        limit: u64,
    },
    InvalidBundleCount {
        count: u64,
        maximum: u64,
    },
    InvalidBundleEntryId {
        index: u64,
        reason: &'static str,
    },
    IntegerOverflow {
        field: &'static str,
    },
    BundlePayloadOutOfBounds {
        index: u64,
        offset: u64,
        size: u64,
        file_size: u64,
    },
    BundlePayloadOverlapsHeader {
        index: u64,
        offset: u64,
        header_size: u64,
    },
    BundlePayloadsOverlap {
        first_index: u64,
        second_index: u64,
    },
}

impl fmt::Display for ValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TruncatedHeader { field } => {
                write!(formatter, "truncated header while reading {field}")
            }
            Self::InspectionLimitExceeded { field, limit } => write!(
                formatter,
                "{field} extends beyond the {limit}-byte inspection prefix limit"
            ),
            Self::InvalidBundleCount { count, maximum } => write!(
                formatter,
                "invalid bundle entry count {count}; expected 1..={maximum}"
            ),
            Self::InvalidBundleEntryId { index, reason } => {
                write!(formatter, "invalid bundle entry {index} ID: {reason}")
            }
            Self::IntegerOverflow { field } => {
                write!(formatter, "integer overflow while validating {field}")
            }
            Self::BundlePayloadOutOfBounds {
                index,
                offset,
                size,
                file_size,
            } => write!(
                formatter,
                "bundle entry {index} payload ({offset}+{size}) exceeds file size {file_size}"
            ),
            Self::BundlePayloadOverlapsHeader {
                index,
                offset,
                header_size,
            } => write!(
                formatter,
                "bundle entry {index} payload offset {offset} precedes header end {header_size}"
            ),
            Self::BundlePayloadsOverlap {
                first_index,
                second_index,
            } => write!(
                formatter,
                "bundle entry payloads {first_index} and {second_index} overlap"
            ),
        }
    }
}

#[derive(Debug)]
pub enum InspectionError {
    Io(io::Error),
    Empty,
    Validation {
        format: ModuleFormat,
        error: ValidationError,
    },
}

impl fmt::Display for InspectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "module inspection failed: {error}"),
            Self::Empty => formatter.write_str("module file is empty"),
            Self::Validation { format, error } => {
                write!(formatter, "invalid {format}: {error}")
            }
        }
    }
}

impl std::error::Error for InspectionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Empty | Self::Validation { .. } => None,
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
    inspect_prefix(&prefix, size)
}

fn inspect_prefix(prefix: &[u8], size: u64) -> Result<ModuleInfo, InspectionError> {
    if prefix.starts_with(b"\x7fELF") && prefix.len() >= 20 {
        return Ok(inspect_elf(prefix, size));
    }
    if prefix.starts_with(&[0xb1, 0x43, 0x62, 0x46]) {
        return Ok(basic(ModuleFormat::CudaFatBinary, size));
    }
    if prefix.starts_with(CLANG_BUNDLE_MAGIC) {
        return inspect_clang_bundle(prefix, size);
    }
    if prefix.len() >= 8
        && prefix.len() < CLANG_BUNDLE_MAGIC.len()
        && CLANG_BUNDLE_MAGIC.starts_with(prefix)
    {
        return Err(validation(
            ModuleFormat::HipFatBinary,
            ValidationError::TruncatedHeader {
                field: "Clang bundle magic",
            },
        ));
    }
    if prefix.windows(10).any(|window| window == b"hip-fatbin") {
        return Ok(basic(ModuleFormat::HipFatBinary, size));
    }
    if let Ok(text) = std::str::from_utf8(prefix) {
        if text
            .lines()
            .any(|line| line.trim_start().starts_with(".version"))
        {
            return Ok(inspect_ptx(text, size));
        }
    }
    Ok(basic(ModuleFormat::Unknown, size))
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
    let mut amdgpu_target_ids = Vec::new();
    let mut amdgpu_architectures = Vec::new();
    if format == ModuleFormat::Hsaco {
        amdgpu_target_ids = amdgpu_metadata_target_ids(prefix);
        amdgpu_architectures.extend(
            amdgpu_target_ids
                .iter()
                .filter_map(|target| amdgpu_architecture(target).map(ToOwned::to_owned)),
        );
        if let Some(architecture) = amdgpu_elf_architecture(prefix, pointer_width, endianness) {
            amdgpu_architectures.push(architecture.to_owned());
        }
        sort_deduplicate(&mut amdgpu_target_ids);
        sort_deduplicate(&mut amdgpu_architectures);
    }
    ModuleInfo {
        format,
        size_bytes: size,
        pointer_width,
        endianness,
        machine,
        ptx_version: None,
        ptx_target: None,
        amdgpu_target_ids,
        amdgpu_architectures,
    }
}

fn inspect_clang_bundle(prefix: &[u8], size: u64) -> Result<ModuleInfo, InspectionError> {
    let format = ModuleFormat::HipFatBinary;
    let mut cursor = CLANG_BUNDLE_MAGIC.len();
    let entry_count = read_bundle_u64(prefix, size, &mut cursor, "bundle entry count")?;
    if entry_count == 0 || entry_count > MAX_CLANG_BUNDLE_ENTRIES {
        return Err(validation(
            format,
            ValidationError::InvalidBundleCount {
                count: entry_count,
                maximum: MAX_CLANG_BUNDLE_ENTRIES,
            },
        ));
    }
    let descriptor_bytes = entry_count
        .checked_mul(CLANG_BUNDLE_DESCRIPTOR_BYTES)
        .ok_or_else(|| {
            validation(
                format,
                ValidationError::IntegerOverflow {
                    field: "bundle descriptor table size",
                },
            )
        })?;
    let minimum_header_size = u64::try_from(cursor)
        .unwrap_or(u64::MAX)
        .checked_add(descriptor_bytes)
        .ok_or_else(|| {
            validation(
                format,
                ValidationError::IntegerOverflow {
                    field: "bundle header size",
                },
            )
        })?;
    if minimum_header_size > size {
        return Err(validation(
            format,
            ValidationError::TruncatedHeader {
                field: "bundle descriptor table",
            },
        ));
    }
    if minimum_header_size > INSPECTION_PREFIX_LIMIT {
        return Err(validation(
            format,
            ValidationError::InspectionLimitExceeded {
                field: "bundle descriptor table",
                limit: INSPECTION_PREFIX_LIMIT,
            },
        ));
    }

    let mut entries = Vec::with_capacity(usize::try_from(entry_count).unwrap_or(0));
    let mut entry_ids = BTreeSet::new();
    let mut target_ids = Vec::new();
    for index in 0..entry_count {
        let (offset, payload_size, id) = read_bundle_entry(prefix, size, &mut cursor, index)?;
        if !entry_ids.insert(id.to_owned()) {
            return Err(validation(
                format,
                ValidationError::InvalidBundleEntryId {
                    index,
                    reason: "value duplicates an earlier entry ID",
                },
            ));
        }
        if let Some(target_id) = amdgpu_target_from_bundle_id(id) {
            target_ids.push(target_id.to_owned());
        }
        entries.push((index, offset, payload_size));
    }

    let header_size = u64::try_from(cursor).unwrap_or(u64::MAX);
    validate_bundle_payloads(&entries, header_size, size)?;
    sort_deduplicate(&mut target_ids);
    let mut architectures = target_ids
        .iter()
        .filter_map(|target| amdgpu_architecture(target).map(ToOwned::to_owned))
        .collect::<Vec<_>>();
    sort_deduplicate(&mut architectures);
    let mut info = basic(format, size);
    info.amdgpu_target_ids = target_ids;
    info.amdgpu_architectures = architectures;
    Ok(info)
}

fn read_bundle_entry<'a>(
    prefix: &'a [u8],
    file_size: u64,
    cursor: &mut usize,
    index: u64,
) -> Result<(u64, u64, &'a str), InspectionError> {
    let format = ModuleFormat::HipFatBinary;
    let offset = read_bundle_u64(prefix, file_size, cursor, "bundle payload offset")?;
    let payload_size = read_bundle_u64(prefix, file_size, cursor, "bundle payload size")?;
    let id_size = read_bundle_u64(prefix, file_size, cursor, "bundle entry ID length")?;
    if id_size == 0 || id_size > MAX_CLANG_BUNDLE_ENTRY_ID_BYTES {
        return Err(validation(
            format,
            ValidationError::InvalidBundleEntryId {
                index,
                reason: "length is outside the supported 1..=1024 byte range",
            },
        ));
    }
    let id = read_bundle_bytes(prefix, file_size, cursor, id_size, "bundle entry ID")?;
    let id = std::str::from_utf8(id).map_err(|_| {
        validation(
            format,
            ValidationError::InvalidBundleEntryId {
                index,
                reason: "value is not UTF-8",
            },
        )
    })?;
    if !id.bytes().all(|byte| byte.is_ascii_graphic()) {
        return Err(validation(
            format,
            ValidationError::InvalidBundleEntryId {
                index,
                reason: "value contains non-graphic ASCII characters",
            },
        ));
    }
    Ok((offset, payload_size, id))
}

fn read_bundle_u64(
    prefix: &[u8],
    file_size: u64,
    cursor: &mut usize,
    field: &'static str,
) -> Result<u64, InspectionError> {
    let bytes = read_bundle_bytes(prefix, file_size, cursor, 8, field)?;
    let bytes = <[u8; 8]>::try_from(bytes).map_err(|_| {
        validation(
            ModuleFormat::HipFatBinary,
            ValidationError::TruncatedHeader { field },
        )
    })?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_bundle_bytes<'a>(
    prefix: &'a [u8],
    file_size: u64,
    cursor: &mut usize,
    length: u64,
    field: &'static str,
) -> Result<&'a [u8], InspectionError> {
    let start = u64::try_from(*cursor).unwrap_or(u64::MAX);
    let end = start.checked_add(length).ok_or_else(|| {
        validation(
            ModuleFormat::HipFatBinary,
            ValidationError::IntegerOverflow { field },
        )
    })?;
    if end > file_size {
        return Err(validation(
            ModuleFormat::HipFatBinary,
            ValidationError::TruncatedHeader { field },
        ));
    }
    if end > u64::try_from(prefix.len()).unwrap_or(u64::MAX) {
        return Err(validation(
            ModuleFormat::HipFatBinary,
            ValidationError::InspectionLimitExceeded {
                field,
                limit: INSPECTION_PREFIX_LIMIT,
            },
        ));
    }
    let end = usize::try_from(end).map_err(|_| {
        validation(
            ModuleFormat::HipFatBinary,
            ValidationError::InspectionLimitExceeded {
                field,
                limit: INSPECTION_PREFIX_LIMIT,
            },
        )
    })?;
    let bytes = &prefix[*cursor..end];
    *cursor = end;
    Ok(bytes)
}

fn validate_bundle_payloads(
    entries: &[(u64, u64, u64)],
    header_size: u64,
    file_size: u64,
) -> Result<(), InspectionError> {
    let format = ModuleFormat::HipFatBinary;
    let mut ranges = Vec::with_capacity(entries.len());
    for &(index, offset, size) in entries {
        if offset < header_size {
            return Err(validation(
                format,
                ValidationError::BundlePayloadOverlapsHeader {
                    index,
                    offset,
                    header_size,
                },
            ));
        }
        let end = offset.checked_add(size).ok_or_else(|| {
            validation(
                format,
                ValidationError::IntegerOverflow {
                    field: "bundle payload range",
                },
            )
        })?;
        if end > file_size {
            return Err(validation(
                format,
                ValidationError::BundlePayloadOutOfBounds {
                    index,
                    offset,
                    size,
                    file_size,
                },
            ));
        }
        if size != 0 {
            ranges.push((offset, end, index));
        }
    }
    ranges.sort_unstable_by_key(|&(offset, _, _)| offset);
    for pair in ranges.windows(2) {
        if pair[1].0 < pair[0].1 {
            return Err(validation(
                format,
                ValidationError::BundlePayloadsOverlap {
                    first_index: pair[0].2,
                    second_index: pair[1].2,
                },
            ));
        }
    }
    Ok(())
}

const fn validation(format: ModuleFormat, error: ValidationError) -> InspectionError {
    InspectionError::Validation { format, error }
}

fn amdgpu_metadata_target_ids(prefix: &[u8]) -> Vec<String> {
    const KEY: &[u8] = b"amdhsa.target";
    let mut targets = Vec::new();
    let mut search_start = 0;
    while let Some(relative_start) = find_bytes(&prefix[search_start..], KEY) {
        let key_start = search_start + relative_start;
        let value_start = key_start + KEY.len();
        if msgpack_string_header_matches(prefix, key_start, KEY.len()) {
            if let Some(value) = read_msgpack_string(prefix, value_start) {
                if let Ok(value) = std::str::from_utf8(value) {
                    if let Some(target) = amdgpu_target_from_triple(value) {
                        targets.push(target.to_owned());
                    }
                }
            }
        }
        search_start = value_start;
    }
    sort_deduplicate(&mut targets);
    targets
}

fn msgpack_string_header_matches(bytes: &[u8], start: usize, length: usize) -> bool {
    let Ok(length_u8) = u8::try_from(length) else {
        return false;
    };
    if length <= 31 && start >= 1 && bytes[start - 1] == 0xa0 | length_u8 {
        return true;
    }
    if start >= 2 && bytes[start - 2] == 0xd9 && bytes[start - 1] == length_u8 {
        return true;
    }
    if let Ok(length_u16) = u16::try_from(length) {
        if start >= 3
            && bytes[start - 3] == 0xda
            && bytes[start - 2..start] == length_u16.to_be_bytes()
        {
            return true;
        }
    }
    false
}

fn read_msgpack_string(bytes: &[u8], start: usize) -> Option<&[u8]> {
    let marker = *bytes.get(start)?;
    let (header_size, length) = match marker {
        0xa0..=0xbf => (1, usize::from(marker & 0x1f)),
        0xd9 => (2, usize::from(*bytes.get(start + 1)?)),
        0xda => {
            let length = u16::from_be_bytes([*bytes.get(start + 1)?, *bytes.get(start + 2)?]);
            (3, usize::from(length))
        }
        0xdb => {
            let length = u32::from_be_bytes([
                *bytes.get(start + 1)?,
                *bytes.get(start + 2)?,
                *bytes.get(start + 3)?,
                *bytes.get(start + 4)?,
            ]);
            (5, usize::try_from(length).ok()?)
        }
        _ => return None,
    };
    if length > MAX_AMDGPU_TARGET_ID_BYTES {
        return None;
    }
    let value_start = start.checked_add(header_size)?;
    let value_end = value_start.checked_add(length)?;
    bytes.get(value_start..value_end)
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn amdgpu_target_from_bundle_id(bundle_id: &str) -> Option<&str> {
    if !(bundle_id.contains("-amdgcn-amd-amdhsa-") || bundle_id.contains("-amdgpu-amd-amdhsa-")) {
        return None;
    }
    bundle_id
        .find("-gfx")
        .and_then(|start| valid_amdgpu_target_id(&bundle_id[start + 1..]))
}

fn amdgpu_target_from_triple(target: &str) -> Option<&str> {
    ["amdgcn-amd-amdhsa--", "amdgpu-amd-amdhsa--"]
        .into_iter()
        .find_map(|prefix| target.strip_prefix(prefix))
        .and_then(valid_amdgpu_target_id)
}

fn valid_amdgpu_target_id(target: &str) -> Option<&str> {
    if target.is_empty()
        || target.len() > MAX_AMDGPU_TARGET_ID_BYTES
        || !target.starts_with("gfx")
        || !target.as_bytes().get(3).is_some_and(u8::is_ascii_digit)
        || !target
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b':' | b'+' | b'-'))
    {
        return None;
    }
    Some(target)
}

fn amdgpu_architecture(target: &str) -> Option<&str> {
    let end = target.find([':', '+']).unwrap_or(target.len());
    valid_amdgpu_target_id(&target[..end])
}

fn amdgpu_elf_architecture(
    prefix: &[u8],
    pointer_width: Option<u8>,
    endianness: Option<&str>,
) -> Option<&'static str> {
    let flags_offset = match pointer_width {
        Some(32) => 36,
        Some(64) => 48,
        _ => return None,
    };
    let flags = <[u8; 4]>::try_from(prefix.get(flags_offset..flags_offset + 4)?).ok()?;
    let flags = match endianness {
        Some("little") => u32::from_le_bytes(flags),
        Some("big") => u32::from_be_bytes(flags),
        _ => return None,
    };
    amdgpu_machine_architecture(flags & ELF_MACHINE_FLAGS_MASK_AMDGPU)
}

const fn amdgpu_machine_architecture(machine: u32) -> Option<&'static str> {
    Some(match machine {
        0x020 => "gfx600",
        0x021 => "gfx601",
        0x022 => "gfx700",
        0x023 => "gfx701",
        0x024 => "gfx702",
        0x025 => "gfx703",
        0x026 => "gfx704",
        0x028 => "gfx801",
        0x029 => "gfx802",
        0x02a => "gfx803",
        0x02b => "gfx810",
        0x02c => "gfx900",
        0x02d => "gfx902",
        0x02e => "gfx904",
        0x02f => "gfx906",
        0x030 => "gfx908",
        0x031 => "gfx909",
        0x032 => "gfx90c",
        0x033 => "gfx1010",
        0x034 => "gfx1011",
        0x035 => "gfx1012",
        0x036 => "gfx1030",
        0x037 => "gfx1031",
        0x038 => "gfx1032",
        0x039 => "gfx1033",
        0x03a => "gfx602",
        0x03b => "gfx705",
        0x03c => "gfx805",
        0x03d => "gfx1035",
        0x03e => "gfx1034",
        0x03f => "gfx90a",
        0x041 => "gfx1100",
        0x042 => "gfx1013",
        0x043 => "gfx1150",
        0x044 => "gfx1103",
        0x045 => "gfx1036",
        0x046 => "gfx1101",
        0x047 => "gfx1102",
        0x048 => "gfx1200",
        0x049 => "gfx1250",
        0x04a => "gfx1151",
        0x04c => "gfx942",
        0x04e => "gfx1201",
        0x04f => "gfx950",
        0x050 => "gfx1310",
        0x051 => "gfx9-generic",
        0x052 => "gfx10-1-generic",
        0x053 => "gfx10-3-generic",
        0x054 => "gfx11-generic",
        0x055 => "gfx1152",
        0x057 => "gfx1154",
        0x058 => "gfx1153",
        0x059 => "gfx12-generic",
        0x05a => "gfx1251",
        0x05b => "gfx12-5-generic",
        0x05c => "gfx1172",
        0x05d => "gfx1170",
        0x05e => "gfx1171",
        0x05f => "gfx9-4-generic",
        0x062 => "gfx11-7-generic",
        0x063 => "gfx13-generic",
        0x0eb => "gfx1250-strict",
        _ => return None,
    })
}

fn sort_deduplicate(values: &mut Vec<String>) {
    values.sort();
    values.dedup();
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
        amdgpu_target_ids: Vec::new(),
        amdgpu_architectures: Vec::new(),
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
        amdgpu_target_ids: Vec::new(),
        amdgpu_architectures: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CLANG_BUNDLE_MAGIC, InspectionError, ModuleFormat, ValidationError, inspect_prefix,
    };

    fn hsaco(architecture: u32, metadata_target: Option<&str>) -> Vec<u8> {
        let mut elf = vec![0_u8; 64];
        elf[..4].copy_from_slice(b"\x7fELF");
        elf[4] = 2;
        elf[5] = 1;
        elf[18..20].copy_from_slice(&224_u16.to_le_bytes());
        elf[48..52].copy_from_slice(&architecture.to_le_bytes());
        if let Some(target) = metadata_target {
            let key = b"amdhsa.target";
            elf.push(0xa0 | u8::try_from(key.len()).expect("small key"));
            elf.extend_from_slice(key);
            elf.push(0xd9);
            elf.push(u8::try_from(target.len()).expect("small target"));
            elf.extend_from_slice(target.as_bytes());
        }
        elf
    }

    fn clang_bundle(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let header_size = CLANG_BUNDLE_MAGIC.len()
            + 8
            + entries.iter().map(|(id, _)| 24 + id.len()).sum::<usize>();
        let mut offsets = Vec::with_capacity(entries.len());
        let mut next_offset = header_size;
        for (_, payload) in entries {
            offsets.push(next_offset);
            next_offset += payload.len();
        }
        let mut bytes = CLANG_BUNDLE_MAGIC.to_vec();
        bytes.extend_from_slice(&(entries.len() as u64).to_le_bytes());
        for ((id, payload), offset) in entries.iter().zip(offsets) {
            bytes.extend_from_slice(&(offset as u64).to_le_bytes());
            bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
            bytes.extend_from_slice(&(id.len() as u64).to_le_bytes());
            bytes.extend_from_slice(id.as_bytes());
        }
        for (_, payload) in entries {
            bytes.extend_from_slice(payload);
        }
        bytes
    }

    #[test]
    fn recognizes_ptx_metadata() {
        let input = b"// fixture\n.version 8.0\n.target sm_75\n.address_size 64\n";
        let info = inspect_prefix(input, input.len() as u64).expect("valid PTX");
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
        assert_eq!(
            inspect_prefix(&elf, 20).expect("valid cubin").format,
            ModuleFormat::Cubin
        );
        elf[18..20].copy_from_slice(&224_u16.to_le_bytes());
        assert_eq!(
            inspect_prefix(&elf, 20).expect("valid HSACO").format,
            ModuleFormat::Hsaco
        );
    }

    #[test]
    fn reports_hsaco_metadata_target_and_elf_architecture() {
        let input = hsaco(0x032, Some("amdgcn-amd-amdhsa--gfx90c:xnack-"));
        let info = inspect_prefix(&input, input.len() as u64).expect("valid HSACO");
        assert_eq!(info.format, ModuleFormat::Hsaco);
        assert_eq!(info.amdgpu_target_ids, ["gfx90c:xnack-"]);
        assert_eq!(info.amdgpu_architectures, ["gfx90c"]);
    }

    #[test]
    fn reports_and_deduplicates_amdgpu_bundle_targets() {
        let input = clang_bundle(&[
            ("host-x86_64-unknown-linux-gnu", b"host"),
            ("hipv4-amdgcn-amd-amdhsa--gfx90c", b"amd-one"),
            ("hip-amdgcn-amd-amdhsa--gfx90c:xnack-", b"amd-two"),
        ]);
        let info = inspect_prefix(&input, input.len() as u64).expect("valid bundle");
        assert_eq!(info.format, ModuleFormat::HipFatBinary);
        assert_eq!(info.amdgpu_target_ids, ["gfx90c", "gfx90c:xnack-"]);
        assert_eq!(info.amdgpu_architectures, ["gfx90c"]);
    }

    #[test]
    fn rejects_truncated_bundle_descriptor_without_panicking() {
        let mut input = CLANG_BUNDLE_MAGIC.to_vec();
        input.extend_from_slice(&1_u64.to_le_bytes());
        input.extend_from_slice(&[0_u8; 7]);
        let error = inspect_prefix(&input, input.len() as u64).expect_err("truncated bundle");
        assert!(matches!(
            error,
            InspectionError::Validation {
                format: ModuleFormat::HipFatBinary,
                error: ValidationError::TruncatedHeader {
                    field: "bundle descriptor table"
                }
            }
        ));
    }

    #[test]
    fn rejects_truncated_bundle_magic_without_panicking() {
        let input = &CLANG_BUNDLE_MAGIC[..CLANG_BUNDLE_MAGIC.len() - 1];
        let error = inspect_prefix(input, input.len() as u64).expect_err("truncated bundle magic");
        assert!(matches!(
            error,
            InspectionError::Validation {
                error: ValidationError::TruncatedHeader {
                    field: "Clang bundle magic"
                },
                ..
            }
        ));
    }

    #[test]
    fn rejects_truncated_bundle_entry_id_without_panicking() {
        let mut input = CLANG_BUNDLE_MAGIC.to_vec();
        input.extend_from_slice(&1_u64.to_le_bytes());
        input.extend_from_slice(&56_u64.to_le_bytes());
        input.extend_from_slice(&0_u64.to_le_bytes());
        input.extend_from_slice(&1_u64.to_le_bytes());
        let error = inspect_prefix(&input, input.len() as u64).expect_err("truncated entry ID");
        assert!(matches!(
            error,
            InspectionError::Validation {
                error: ValidationError::TruncatedHeader {
                    field: "bundle entry ID"
                },
                ..
            }
        ));
    }

    #[test]
    fn rejects_every_recognizable_truncation_of_a_valid_bundle() {
        let input = clang_bundle(&[("hip-amdgcn-amd-amdhsa--gfx90c", b"payload")]);
        for length in 8..input.len() {
            assert!(
                inspect_prefix(&input[..length], length as u64).is_err(),
                "truncation at byte {length} must fail closed"
            );
        }
    }

    #[test]
    fn rejects_bundle_payload_outside_the_file() {
        let mut input = clang_bundle(&[("hip-amdgcn-amd-amdhsa--gfx90c", b"payload")]);
        let payload_size_offset = CLANG_BUNDLE_MAGIC.len() + 8 + 8;
        input[payload_size_offset..payload_size_offset + 8]
            .copy_from_slice(&u64::MAX.to_le_bytes());
        let error = inspect_prefix(&input, input.len() as u64).expect_err("invalid payload range");
        assert!(matches!(
            error,
            InspectionError::Validation {
                error: ValidationError::IntegerOverflow {
                    field: "bundle payload range"
                },
                ..
            }
        ));
    }

    #[test]
    fn rejects_non_overflowing_bundle_payload_outside_the_file() {
        let mut input = clang_bundle(&[("hip-amdgcn-amd-amdhsa--gfx90c", b"payload")]);
        let payload_offset = CLANG_BUNDLE_MAGIC.len() + 8;
        let outside = u64::try_from(input.len()).expect("small fixture");
        input[payload_offset..payload_offset + 8].copy_from_slice(&outside.to_le_bytes());
        let error = inspect_prefix(&input, outside).expect_err("payload is out of bounds");
        assert!(matches!(
            error,
            InspectionError::Validation {
                error: ValidationError::BundlePayloadOutOfBounds { index: 0, .. },
                ..
            }
        ));
    }

    #[test]
    fn rejects_bundle_payload_overlapping_its_header() {
        let mut input = clang_bundle(&[("hip-amdgcn-amd-amdhsa--gfx90c", b"payload")]);
        let payload_offset = CLANG_BUNDLE_MAGIC.len() + 8;
        input[payload_offset..payload_offset + 8].copy_from_slice(&0_u64.to_le_bytes());
        let error =
            inspect_prefix(&input, input.len() as u64).expect_err("payload overlaps header");
        assert!(matches!(
            error,
            InspectionError::Validation {
                error: ValidationError::BundlePayloadOverlapsHeader { index: 0, .. },
                ..
            }
        ));
    }

    #[test]
    fn rejects_overlapping_bundle_payloads() {
        let first_id = "hip-amdgcn-amd-amdhsa--gfx90c";
        let second_id = "hip-amdgcn-amd-amdhsa--gfx942";
        let mut input = clang_bundle(&[(first_id, b"first"), (second_id, b"second")]);
        let first_offset_field = CLANG_BUNDLE_MAGIC.len() + 8;
        let first_offset = u64::from_le_bytes(
            input[first_offset_field..first_offset_field + 8]
                .try_into()
                .expect("eight-byte offset"),
        );
        let second_offset_field = first_offset_field + 24 + first_id.len();
        input[second_offset_field..second_offset_field + 8]
            .copy_from_slice(&(first_offset + 1).to_le_bytes());
        let error = inspect_prefix(&input, input.len() as u64).expect_err("payloads overlap");
        assert!(matches!(
            error,
            InspectionError::Validation {
                error: ValidationError::BundlePayloadsOverlap {
                    first_index: 0,
                    second_index: 1
                },
                ..
            }
        ));
    }

    #[test]
    fn rejects_bundle_entry_count_over_the_conservative_bound() {
        let mut input = CLANG_BUNDLE_MAGIC.to_vec();
        input.extend_from_slice(&257_u64.to_le_bytes());
        let error = inspect_prefix(&input, input.len() as u64).expect_err("entry count is bounded");
        assert!(matches!(
            error,
            InspectionError::Validation {
                error: ValidationError::InvalidBundleCount {
                    count: 257,
                    maximum: 256
                },
                ..
            }
        ));
    }

    #[test]
    fn recognizes_cuda_fat_binary_magic() {
        assert_eq!(
            inspect_prefix(&[0xb1, 0x43, 0x62, 0x46], 4)
                .expect("valid magic")
                .format,
            ModuleFormat::CudaFatBinary
        );
    }
}
