# SPDX-License-Identifier: CC0-1.0

[CmdletBinding()]
param(
    [string] $OutputDirectory = 'target/oracle-candidates'
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$sourceRoot = Join-Path $root 'target/oracle-maintenance'
$outputRoot = Join-Path $root $OutputDirectory
New-Item -ItemType Directory -Force $sourceRoot, $outputRoot | Out-Null

function Get-VerifiedFile(
    [string] $Url,
    [string] $Destination,
    [string] $Sha256
) {
    Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $Destination
    $actual = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $Sha256) {
        throw "hash mismatch for $Url`: expected $Sha256, found $actual"
    }
}

function Expand-OneArchive([string] $Archive, [string] $Destination) {
    New-Item -ItemType Directory -Force $Destination | Out-Null
    Expand-Archive -LiteralPath $Archive -DestinationPath $Destination -Force
    $directories = @(Get-ChildItem -LiteralPath $Destination -Directory)
    if ($directories.Count -ne 1) {
        throw "$Archive must contain exactly one top-level directory"
    }
    $directories[0].FullName
}

function Assert-Hash([string] $Path, [string] $Sha256) {
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $Sha256) {
        throw "hash mismatch for $Path`: expected $Sha256, found $actual"
    }
}

$cudaUrl = 'https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-13.3.29-archive.zip'
$cudaArchiveHash = '1feb7dd266813ffe8dbc24e115183a5ac35a4795c8d34aca0df85ab616b64d9c'
$cudaHeaderHash = '31df84e16179b6d97db4b3c0bae7697392a370b41983f4a8962f0e5a8069b577'
$cudaTypedefHash = 'd4ab98b677cfb7ab7abdda830d8106b62f10386a11ac7efc0335e6bbd4fe74f4'
$hipGeneralUrl = 'https://github.com/ROCm/HIP/archive/d135154e63ee2e44e7de3a7143dea4bca860df78.zip'
$hipGeneralArchiveHash = '9d4221874dece55578c994ef8b06e8f607f5e033fb9ae701d198078f09f4ff94'
$hipGeneralHeaderHash = '2074d65bbe51ee34773637b0c2a8c8afad4ab3ea2cb776535fbaf69aa107bafe'
$clrGeneralUrl = 'https://github.com/ROCm/clr/archive/396c84e07b78590c7b6e92977539ed109d11ff0a.zip'
$clrGeneralArchiveHash = '7bd8c8a717ae3d92905864b9806077d9873650006b94785ee2f2ea03802838a5'
$hipWindowsUrl = 'https://github.com/ROCm/HIP/archive/04d503ec19aa637e2982220dd81913223c759cf4.zip'
$hipWindowsArchiveHash = 'af75c40c2777151e0dad3b7e960111d1793742ccb7d3eb5a0e1ffd2b181758f1'
$hipWindowsHeaderHash = '16f713e8b5633ae434e59a3a92c0e521f435e5be7bd5952e7818fe9adf087eac'
$clrWindowsUrl = 'https://github.com/ROCm/clr/archive/c5824370effd4609972653f6490f7313f5bc18af.zip'
$clrWindowsArchiveHash = '9b50b1437d98a791fd03bf6a3c2bcbfb0f310fbb7487f6e4369182d206bc81e5'
$hipDocsUrl = 'https://rocm.docs.amd.com/projects/HIP/en/docs-7.14.0/doxygen/html/hip__runtime__api_8h_source.html'
$hipDocsHeaderHash = '33f38708cbffaf5b5f4b51c90ca3748331cab99fd9da3cad137480bb850d74ea'

$cudaArchive = Join-Path $sourceRoot 'cuda.zip'
$hipGeneralArchive = Join-Path $sourceRoot 'hip-general.zip'
$clrGeneralArchive = Join-Path $sourceRoot 'clr-general.zip'
$hipWindowsArchive = Join-Path $sourceRoot 'hip-windows.zip'
$clrWindowsArchive = Join-Path $sourceRoot 'clr-windows.zip'
$hipDocsHtml = Join-Path $sourceRoot 'hip-runtime-api-source.html'
Get-VerifiedFile $cudaUrl $cudaArchive $cudaArchiveHash
Get-VerifiedFile $hipGeneralUrl $hipGeneralArchive $hipGeneralArchiveHash
Get-VerifiedFile $clrGeneralUrl $clrGeneralArchive $clrGeneralArchiveHash
Get-VerifiedFile $hipWindowsUrl $hipWindowsArchive $hipWindowsArchiveHash
Get-VerifiedFile $clrWindowsUrl $clrWindowsArchive $clrWindowsArchiveHash
Invoke-WebRequest -UseBasicParsing -Uri $hipDocsUrl -OutFile $hipDocsHtml

$cudaRoot = Expand-OneArchive $cudaArchive (Join-Path $sourceRoot 'cuda')
$hipGeneralRoot = Expand-OneArchive $hipGeneralArchive (Join-Path $sourceRoot 'hip-general')
$clrGeneralRoot = Expand-OneArchive $clrGeneralArchive (Join-Path $sourceRoot 'clr-general')
$hipWindowsRoot = Expand-OneArchive $hipWindowsArchive (Join-Path $sourceRoot 'hip-windows')
$clrWindowsRoot = Expand-OneArchive $clrWindowsArchive (Join-Path $sourceRoot 'clr-windows')
$cudaInclude = Join-Path $cudaRoot 'include'
$cudaHeader = Join-Path $cudaInclude 'cuda.h'
$cudaTypedefHeader = Join-Path $cudaInclude 'cudaTypedefs.h'
$hipGeneralHeader = Join-Path $hipGeneralRoot 'include/hip/hip_runtime_api.h'
$hipWindowsHeader = Join-Path $hipWindowsRoot 'include/hip/hip_runtime_api.h'
Assert-Hash $cudaHeader $cudaHeaderHash
Assert-Hash $cudaTypedefHeader $cudaTypedefHash
Assert-Hash $hipGeneralHeader $hipGeneralHeaderHash
Assert-Hash $hipWindowsHeader $hipWindowsHeaderHash

$docsInclude = Join-Path $sourceRoot 'hip-docs/include/hip'
New-Item -ItemType Directory -Force $docsInclude | Out-Null
$hipDocsHeader = Join-Path $docsInclude 'hip_runtime_api.h'
& python3 (Join-Path $root '.github/scripts/extract-doxygen-source.py') $hipDocsHtml $hipDocsHeader
if ($LASTEXITCODE -ne 0) { throw "Doxygen source extraction failed with $LASTEXITCODE" }
Assert-Hash $hipDocsHeader $hipDocsHeaderHash

$generatedGeneral = Join-Path $sourceRoot 'generated-general'
$generatedWindows = Join-Path $sourceRoot 'generated-windows'
New-Item -ItemType Directory -Force (Join-Path $generatedGeneral 'hip'), (Join-Path $generatedWindows 'hip') | Out-Null
$utf8NoBom = [Text.UTF8Encoding]::new($false)
[IO.File]::WriteAllText(
    (Join-Path $generatedGeneral 'hip/hip_version.h'),
    "#ifndef HIP_VERSION_H`n#define HIP_VERSION_H`n#define HIP_VERSION_MAJOR 7`n#define HIP_VERSION_MINOR 14`n#define HIP_VERSION_PATCH 60850`n#define HIP_VERSION 71460850`n#endif`n",
    $utf8NoBom
)
[IO.File]::WriteAllText(
    (Join-Path $generatedWindows 'hip/hip_version.h'),
    "#ifndef HIP_VERSION_H`n#define HIP_VERSION_H`n#define HIP_VERSION_MAJOR 7`n#define HIP_VERSION_MINOR 2`n#define HIP_VERSION_PATCH 53210`n#define HIP_VERSION 70253210`n#endif`n",
    $utf8NoBom
)
[IO.File]::WriteAllText(
    (Join-Path $generatedGeneral 'stdlib.h'),
    "#ifndef OCGPU_ORACLE_STDLIB_H`n#define OCGPU_ORACLE_STDLIB_H`n#include <stddef.h>`n#endif`n",
    $utf8NoBom
)

Push-Location $root
try {
    $cudaOutput = Join-Path $outputRoot 'cuda-13.3-13030.json'
    $cudaArguments = @(
        'run', '-p', 'ocgpu-oracle', '--', 'extract-vendor', 'cuda',
        '--header', $cudaHeader,
        '--inventory-id', 'cuda-vendor-13.3-13030',
        '--source-name', 'NVIDIA CUDA Driver API cuda.h',
        '--source-version', 'CUDA 13.3 Update 1 / API 13030',
        '--provenance', 'https://docs.nvidia.com/cuda/archive/13.3.1/cuda-driver-api/index.html',
        '--platforms', 'aarch64-unknown-linux-gnu,x86_64-pc-windows-msvc,x86_64-unknown-linux-gnu',
        '--output', $cudaOutput,
        '--include', $generatedGeneral,
        '--include', $cudaInclude,
        '--artifact', "authoritative-header-archive|$cudaUrl|sha256:$cudaArchiveHash|CUDA 13.3 Update 1 / API 13030|archive",
        '--artifact', "authoritative-driver-header|$cudaUrl|sha256:$cudaHeaderHash|CUDA 13.3 Update 1 / API 13030|include/cuda.h"
    )
    & cargo +1.97.1 @cudaArguments
    if ($LASTEXITCODE -ne 0) { throw "CUDA extraction failed with $LASTEXITCODE" }

    $procArguments = @(
        'run', '-p', 'ocgpu-oracle', '--', 'extract-cuda-proc-typedefs',
        '--header', $cudaTypedefHeader,
        '--source-version', 'CUDA 13.3 Update 1 / API 13030',
        '--provenance', 'https://docs.nvidia.com/cuda/archive/13.3.1/cuda-driver-api/group__CUDA__DRIVER__ENTRY__POINT.html',
        '--output', (Join-Path $outputRoot 'cuda-13.3-13030-proc-address.json'),
        '--include', $cudaInclude,
        '--artifact', "authoritative-header-archive|$cudaUrl|sha256:$cudaArchiveHash|CUDA 13.3 Update 1 / API 13030|archive",
        '--artifact', "authoritative-proc-address-typedef-header|$cudaUrl|sha256:$cudaTypedefHash|CUDA 13.3 Update 1 / API 13030|include/cudaTypedefs.h"
    )
    & cargo +1.97.1 @procArguments
    if ($LASTEXITCODE -ne 0) { throw "CUDA proc typedef extraction failed with $LASTEXITCODE" }

    $hipGeneralInclude = Join-Path $hipGeneralRoot 'include'
    $clrGeneralInclude = Join-Path $clrGeneralRoot 'hipamd/include'
    $generalArguments = @(
        'run', '-p', 'ocgpu-oracle', '--', 'extract-vendor', 'hip',
        '--header', $hipDocsHeader,
        '--inventory-id', 'hip-general-7.14.60850',
        '--source-name', 'AMD HIP Runtime API 7.14 Doxygen declaration',
        '--source-version', 'HIP 7.14.60850 / ROCm 7.14.0',
        '--provenance', $hipDocsUrl,
        '--platforms', 'aarch64-unknown-linux-gnu,x86_64-unknown-linux-gnu',
        '--output', (Join-Path $outputRoot 'hip-general-7.14.60850.json'),
        '--include', (Split-Path -Parent (Split-Path -Parent $hipDocsHeader)),
        '--include', $generatedGeneral,
        '--include', $hipGeneralInclude,
        '--include', $clrGeneralInclude,
        '--semantic-header', $hipGeneralHeader,
        '--semantic-provenance', 'https://github.com/ROCm/HIP/blob/d135154e63ee2e44e7de3a7143dea4bca860df78/include/hip/hip_runtime_api.h',
        '--semantic-include', $generatedGeneral,
        '--semantic-include', $hipGeneralInclude,
        '--semantic-include', $clrGeneralInclude,
        '--artifact', "authoritative-header|$hipDocsUrl|sha256:$hipDocsHeaderHash|docs-7.14.0 / HIP 7.14.60850|include/hip/hip_runtime_api.h",
        '--artifact', "supporting-clr-source|$clrGeneralUrl|sha256:$clrGeneralArchiveHash|396c84e07b78590c7b6e92977539ed109d11ff0a|hipamd/include",
        '--artifact', "supporting-hip-source|$hipGeneralUrl|sha256:$hipGeneralArchiveHash|d135154e63ee2e44e7de3a7143dea4bca860df78|archive",
        '--artifact', "semantic-hip-header|$hipGeneralUrl|sha256:$hipGeneralHeaderHash|d135154e63ee2e44e7de3a7143dea4bca860df78|include/hip/hip_runtime_api.h"
    )
    & cargo +1.97.1 @generalArguments
    if ($LASTEXITCODE -ne 0) { throw "general HIP extraction failed with $LASTEXITCODE" }

    $hipWindowsInclude = Join-Path $hipWindowsRoot 'include'
    $clrWindowsInclude = Join-Path $clrWindowsRoot 'hipamd/include'
    $windowsArguments = @(
        'run', '-p', 'ocgpu-oracle', '--', 'extract-vendor', 'hip',
        '--header', $hipWindowsHeader,
        '--inventory-id', 'hip-windows-7.2.0',
        '--source-name', 'AMD HIP SDK for Windows Runtime API',
        '--source-version', 'HIP SDK for Windows 7.2.0 / HIP 7.2.53210',
        '--provenance', 'https://github.com/ROCm/hip/blob/rocm-7.2.0/include/hip/hip_runtime_api.h',
        '--platforms', 'x86_64-pc-windows-msvc',
        '--output', (Join-Path $outputRoot 'hip-windows-7.2.0.json'),
        '--include', $generatedWindows,
        '--include', $generatedGeneral,
        '--include', $hipWindowsInclude,
        '--include', $clrWindowsInclude,
        '--artifact', "authoritative-hip-source|$hipWindowsUrl|sha256:$hipWindowsArchiveHash|rocm-7.2.0 tag object 04d503ec19aa637e2982220dd81913223c759cf4 / HIP 7.2.53210|archive",
        '--artifact', "authoritative-hip-header|$hipWindowsUrl|sha256:$hipWindowsHeaderHash|rocm-7.2.0 tag object 04d503ec19aa637e2982220dd81913223c759cf4 / HIP 7.2.53210|include/hip/hip_runtime_api.h",
        '--artifact', "supporting-clr-source|$clrWindowsUrl|sha256:$clrWindowsArchiveHash|rocm-7.2.0 commit c5824370effd4609972653f6490f7313f5bc18af|hipamd/include"
    )
    & cargo +1.97.1 @windowsArguments
    if ($LASTEXITCODE -ne 0) { throw "Windows HIP extraction failed with $LASTEXITCODE" }
} finally {
    Pop-Location
}

Write-Host "Wrote review-only official oracle candidates to $outputRoot"
