# SPDX-License-Identifier: CC0-1.0

<#
.SYNOPSIS
Rebuilds the compact HIP 5/6/7 compatibility evidence from pinned sources.

.DESCRIPTION
This maintainer-only command downloads every HIP and CLR archive named by
runtime-profiles.json, verifies archive and reviewed member hashes, extracts
multi-target declarations and enum values, rebuilds
runtime-profile-declarations.json, and compares it with the committed artifact.
Normal builds and ordinary oracle checks remain offline.
#>

[CmdletBinding()]
param(
    [string] $WorkDirectory = 'target/hip-runtime-profile-maintenance',
    [string] $ArchiveDirectory,
    [switch] $Update
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$ledgerPath = Join-Path $root 'oracle/vendor/hip/runtime-profiles.json'
$committedPath = Join-Path $root 'oracle/vendor/hip/runtime-profile-declarations.json'
$ledger = Get-Content -LiteralPath $ledgerPath -Raw | ConvertFrom-Json
if ([int]$ledger.schema_version -ne 1 -or $ledger.inventory_id -ne 'hip-runtime-profiles') {
    throw 'runtime-profiles.json has unexpected metadata'
}

function Resolve-UnderRoot([string] $Path) {
    if ([IO.Path]::IsPathRooted($Path)) { return $Path }
    Join-Path $root $Path
}

function Get-BareHash([string] $Hash) {
    if ($Hash -notmatch '^sha256:[0-9a-f]{64}$') {
        throw "invalid canonical SHA-256 $Hash"
    }
    $Hash.Substring(7)
}

function Assert-Hash([string] $Path, [string] $Expected) {
    $expectedBare = Get-BareHash $Expected
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expectedBare) {
        throw ('hash mismatch for {0}: expected {1}, found {2}' -f $Path, $expectedBare, $actual)
    }
}

$workRoot = Resolve-UnderRoot $WorkDirectory
New-Item -ItemType Directory -Force $workRoot | Out-Null
$runRoot = Join-Path $workRoot ([Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force $runRoot | Out-Null
$expandedByHash = @{}
$archiveByHash = @{}
if (-not [string]::IsNullOrWhiteSpace($ArchiveDirectory)) {
    $archiveRoot = Resolve-UnderRoot $ArchiveDirectory
    foreach ($candidate in Get-ChildItem -LiteralPath $archiveRoot -Recurse -File -Filter '*.zip') {
        $hash = (Get-FileHash -LiteralPath $candidate.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        if (-not $archiveByHash.ContainsKey($hash)) {
            $archiveByHash[$hash] = $candidate.FullName
        }
    }
}

function Get-VerifiedArchiveRoot(
    [string] $Label,
    [string] $Url,
    [string] $ExpectedHash
) {
    $hash = Get-BareHash $ExpectedHash
    if ($expandedByHash.ContainsKey($hash)) {
        return $expandedByHash[$hash]
    }
    if ($archiveByHash.ContainsKey($hash)) {
        $archive = $archiveByHash[$hash]
    } elseif (-not [string]::IsNullOrWhiteSpace($ArchiveDirectory)) {
        throw "archive cache lacks sha256:$hash for $Url"
    } else {
        $archive = Join-Path $runRoot "$Label-$hash.zip"
        Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $archive
    }
    Assert-Hash $archive $ExpectedHash
    $destination = Join-Path $runRoot "$Label-$hash"
    New-Item -ItemType Directory -Force $destination | Out-Null
    Expand-Archive -LiteralPath $archive -DestinationPath $destination
    $topLevels = @(Get-ChildItem -LiteralPath $destination -Directory)
    if ($topLevels.Count -ne 1) {
        throw "$Url must contain exactly one top-level directory"
    }
    $expandedByHash[$hash] = $topLevels[0].FullName
    $topLevels[0].FullName
}

function Get-Release([string] $Id) {
    $matches = @($ledger.reviewed_releases | Where-Object { $_.id -eq $Id })
    if ($matches.Count -ne 1) { throw "expected exactly one reviewed release $Id" }
    $matches[0]
}

$hipRoots = @{}
$clrRoots = @{}
foreach ($release in $ledger.reviewed_releases) {
    $hipRoot = Get-VerifiedArchiveRoot 'hip' $release.hip_archive_url $release.hip_archive_sha256
    $clrRoot = Get-VerifiedArchiveRoot 'clr' $release.clr_archive_url $release.clr_archive_sha256
    Assert-Hash (Join-Path $hipRoot $release.hip_header_path) $release.hip_header_sha256
    Assert-Hash (Join-Path $hipRoot 'VERSION') $release.hip_version_sha256
    Assert-Hash (Join-Path $clrRoot $release.clr_cmake_path) $release.clr_cmake_sha256
    $hipRoots[$release.id] = $hipRoot
    $clrRoots[$release.id] = $clrRoot
}

$utf8NoBom = [Text.UTF8Encoding]::new($false)
$stubInclude = Join-Path $runRoot 'stubs'
New-Item -ItemType Directory -Force $stubInclude | Out-Null
$stringStub = @'
#ifndef OCGPU_PROFILE_STRING_H
#define OCGPU_PROFILE_STRING_H
#include <stddef.h>
#endif
'@
$stdlibStub = @'
#ifndef OCGPU_PROFILE_STDLIB_H
#define OCGPU_PROFILE_STDLIB_H
#include <stddef.h>
#endif
'@
[IO.File]::WriteAllText((Join-Path $stubInclude 'string.h'), $stringStub, $utf8NoBom)
[IO.File]::WriteAllText((Join-Path $stubInclude 'stdlib.h'), $stdlibStub, $utf8NoBom)

function Write-VersionHeader([string] $Directory, [int64] $RuntimeVersion) {
    $major = [Math]::Floor($RuntimeVersion / 10000000)
    $minor = [Math]::Floor(($RuntimeVersion % 10000000) / 100000)
    $patch = $RuntimeVersion % 100000
    $hipDirectory = Join-Path $Directory 'hip'
    New-Item -ItemType Directory -Force $hipDirectory | Out-Null
    $contents = @"
#ifndef HIP_VERSION_H
#define HIP_VERSION_H
#define HIP_VERSION_MAJOR $major
#define HIP_VERSION_MINOR $minor
#define HIP_VERSION_PATCH $patch
#define HIP_VERSION (HIP_VERSION_MAJOR * 10000000 + HIP_VERSION_MINOR * 100000 + HIP_VERSION_PATCH)
#endif
"@
    [IO.File]::WriteAllText((Join-Path $hipDirectory 'hip_version.h'), $contents, $utf8NoBom)
}

$allPlatforms = 'aarch64-unknown-linux-gnu,x86_64-pc-windows-msvc,x86_64-unknown-linux-gnu'
$linuxPlatforms = 'aarch64-unknown-linux-gnu,x86_64-unknown-linux-gnu'
$extractions = @(
    [ordered]@{ ReleaseId = 'hip-5.7.31541'; InventoryId = 'hip-profile-5.7.0-review'; DeclarationRuntimeVersion = 50731921; Platforms = $allPlatforms; HeaderRole = 'authoritative-hip-header'; Output = 'hip-5.7.0.json' },
    [ordered]@{ ReleaseId = 'hip-5.7.31921'; InventoryId = 'hip-profile-5.7.1-review'; Platforms = $allPlatforms; HeaderRole = 'authoritative-hip-header'; Output = 'hip-5.7.1.json' },
    [ordered]@{ ReleaseId = 'hip-6.1.40093'; InventoryId = 'hip-profile-6.1.2-review'; Platforms = $allPlatforms; HeaderRole = 'authoritative-hip-header'; Output = 'hip-6.1.2.json' },
    [ordered]@{ ReleaseId = 'hip-6.2.41134'; InventoryId = 'hip-profile-6.2.4-review'; Platforms = $allPlatforms; HeaderRole = 'authoritative-hip-header'; Output = 'hip-6.2.4.json' },
    [ordered]@{ ReleaseId = 'hip-6.4.43484'; InventoryId = 'hip-profile-6.4.2-review'; Platforms = $allPlatforms; HeaderRole = 'authoritative-hip-header'; Output = 'hip-6.4.2.json' },
    [ordered]@{ ReleaseId = 'hip-7.2.53210'; InventoryId = 'hip-profile-7.2.53210-review'; Platforms = $allPlatforms; HeaderRole = 'authoritative-hip-header'; Output = 'hip-7.2.0.json' },
    [ordered]@{ ReleaseId = 'hip-7.14.60850'; InventoryId = 'hip-profile-7.14.60850-review'; Platforms = $linuxPlatforms; HeaderRole = 'semantic-hip-header'; Output = 'hip-7.14.0.json' }
)
$outputs = @{}

Push-Location $root
try {
    foreach ($spec in $extractions) {
        $release = Get-Release $spec.ReleaseId
        $hipRoot = $hipRoots[$spec.ReleaseId]
        $clrRoot = $clrRoots[$spec.ReleaseId]
        $versionInclude = Join-Path $runRoot "generated-$($spec.InventoryId)"
        $declarationRuntimeVersion = if ($null -ne $spec.DeclarationRuntimeVersion) {
            [int64]$spec.DeclarationRuntimeVersion
        } else {
            [int64]$release.runtime_version
        }
        Write-VersionHeader $versionInclude $declarationRuntimeVersion
        $output = Join-Path $runRoot $spec.Output
        $archiveRevision = if ($spec.ReleaseId -eq 'hip-7.14.60850') {
            $release.hip_commit
        } else {
            "rocm-$($release.rocm_release)"
        }
        $archiveName = [IO.Path]::GetFileNameWithoutExtension([Uri]$release.hip_archive_url)
        $provenance = "https://github.com/ROCm/HIP/blob/$archiveName/$($release.hip_header_path)"
        $major = [Math]::Floor($declarationRuntimeVersion / 10000000)
        $minor = [Math]::Floor(($declarationRuntimeVersion % 10000000) / 100000)
        $patch = $declarationRuntimeVersion % 100000
        $arguments = @(
            'run', '-p', 'ocgpu-oracle', '--', 'extract-vendor', 'hip',
            '--header', (Join-Path $hipRoot $release.hip_header_path),
            '--inventory-id', $spec.InventoryId,
            '--source-name', 'AMD HIP Runtime API',
            '--source-version', "HIP $major.$minor.$patch / ROCm $($release.rocm_release)",
            '--provenance', $provenance,
            '--platforms', $spec.Platforms,
            '--output', $output,
            '--include', $versionInclude,
            '--include', $stubInclude,
            '--include', (Join-Path $hipRoot 'include'),
            '--include', (Join-Path $clrRoot 'hipamd/include'),
            '--artifact', "$($spec.HeaderRole)|$($release.hip_archive_url)|$($release.hip_header_sha256)|$archiveRevision|$($release.hip_header_path)",
            '--artifact', "supporting-clr-source|$($release.clr_archive_url)|$($release.clr_archive_sha256)|$($release.clr_commit)|hipamd/include"
        )
        & cargo @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "$($spec.ReleaseId) declaration extraction failed with $LASTEXITCODE"
        }
        $outputs[$spec.ReleaseId] = $output
    }

    $candidate = Join-Path $runRoot 'runtime-profile-declarations.json'
    $builderArguments = @{
        Hip570Inventory = $outputs['hip-5.7.31541']
        Hip571Inventory = $outputs['hip-5.7.31921']
        Hip612Inventory = $outputs['hip-6.1.40093']
        Hip624Inventory = $outputs['hip-6.2.41134']
        Hip642Inventory = $outputs['hip-6.4.43484']
        Hip720Inventory = $outputs['hip-7.2.53210']
        Hip714Inventory = $outputs['hip-7.14.60850']
        Output = $candidate
    }
    & (Join-Path $root '.github/scripts/build-hip-profile-snapshot.ps1') @builderArguments
    if ($LASTEXITCODE -ne 0) {
        throw "HIP profile snapshot build failed with $LASTEXITCODE"
    }

    if ($Update) {
        Copy-Item -LiteralPath $candidate -Destination $committedPath -Force
        Write-Host "Updated $committedPath from verified sources"
    } else {
        $candidateHash = (Get-FileHash -LiteralPath $candidate -Algorithm SHA256).Hash
        $committedHash = (Get-FileHash -LiteralPath $committedPath -Algorithm SHA256).Hash
        if ($candidateHash -ne $committedHash) {
            throw "runtime-profile-declarations.json is stale; verified candidate remains at $candidate"
        }
        Write-Host 'Verified HIP runtime-profile source hashes and declaration snapshot freshness'
    }
} finally {
    Pop-Location
}

Write-Host "Maintainer evidence retained under $runRoot"
