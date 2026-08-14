# SPDX-License-Identifier: CC0-1.0

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $Hip571Inventory,
    [Parameter(Mandatory = $true)]
    [string] $Hip612Inventory,
    [Parameter(Mandatory = $true)]
    [string] $Hip624Inventory,
    [Parameter(Mandatory = $true)]
    [string] $Hip642Inventory,
    [Parameter(Mandatory = $true)]
    [string] $Hip720Inventory,
    [Parameter(Mandatory = $true)]
    [string] $Hip714Inventory,
    [string] $Output = 'oracle/vendor/hip/runtime-profile-declarations.json'
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$ledger = Get-Content -LiteralPath (Join-Path $root 'oracle/vendor/hip/runtime-profiles.json') -Raw | ConvertFrom-Json
$commonNames = @($ledger.common_functions.name) + @($ledger.common_adapters.name)
$typeNames = @(
    'hipError_t',
    'hipDevice_t',
    'hipDeviceAttribute_t',
    'hipDeviceptr_t',
    'hipCtx_t',
    'hipStream_t',
    'hipEvent_t',
    'hipModule_t',
    'hipFunction_t'
)

function Resolve-Input([string] $Path) {
    if ([IO.Path]::IsPathRooted($Path)) { return $Path }
    Join-Path $root $Path
}

function Assert-Sequence([string] $Label, [object[]] $Actual, [string[]] $Expected) {
    if ($Actual.Count -ne $Expected.Count) {
        throw "$Label has $($Actual.Count) values; expected $($Expected.Count)"
    }
    for ($index = 0; $index -lt $Expected.Count; $index++) {
        if ([string]$Actual[$index] -ne $Expected[$index]) {
            throw "$Label differs at index $index`: expected $($Expected[$index]), found $($Actual[$index])"
        }
    }
}

$allPlatforms = @(
    'aarch64-unknown-linux-gnu',
    'x86_64-pc-windows-msvc',
    'x86_64-unknown-linux-gnu'
)
$linuxPlatforms = @(
    'aarch64-unknown-linux-gnu',
    'x86_64-unknown-linux-gnu'
)
$sources = [ordered]@{
    'hip-5.7.31541' = [ordered]@{ Path = Resolve-Input $Hip571Inventory; InventoryId = 'hip-profile-5.7.1-review'; Platforms = $allPlatforms; HeaderRole = 'authoritative-hip-header' }
    'hip-5.7.31921' = [ordered]@{ Path = Resolve-Input $Hip571Inventory; InventoryId = 'hip-profile-5.7.1-review'; Platforms = $allPlatforms; HeaderRole = 'authoritative-hip-header' }
    'hip-6.1.40093' = [ordered]@{ Path = Resolve-Input $Hip612Inventory; InventoryId = 'hip-profile-6.1.2-review'; Platforms = $allPlatforms; HeaderRole = 'authoritative-hip-header' }
    'hip-6.2.41134' = [ordered]@{ Path = Resolve-Input $Hip624Inventory; InventoryId = 'hip-profile-6.2.4-review'; Platforms = $allPlatforms; HeaderRole = 'authoritative-hip-header' }
    'hip-6.4.43484' = [ordered]@{ Path = Resolve-Input $Hip642Inventory; InventoryId = 'hip-profile-6.4.2-review'; Platforms = $allPlatforms; HeaderRole = 'authoritative-hip-header' }
    'hip-7.2.53210' = [ordered]@{ Path = Resolve-Input $Hip720Inventory; InventoryId = 'hip-profile-7.2.53210-review'; Platforms = $allPlatforms; HeaderRole = 'authoritative-hip-header' }
    'hip-7.14.60850' = [ordered]@{ Path = Resolve-Input $Hip714Inventory; InventoryId = 'hip-profile-7.14.60850-review'; Platforms = $linuxPlatforms; HeaderRole = 'semantic-hip-header' }
}

$snapshots = @()
foreach ($release in $ledger.reviewed_releases) {
    $source = $sources[$release.id]
    if ($null -eq $source) { throw "unexpected reviewed release $($release.id)" }
    $inventory = Get-Content -LiteralPath $source.Path -Raw | ConvertFrom-Json
    if ([int]$inventory.schema_version -ne 1 -or
        $inventory.spdx_license_identifier -ne 'CC0-1.0' -or
        $inventory.inventory_id -ne $source.InventoryId) {
        throw "$($release.id) selected an unexpected source inventory"
    }
    Assert-Sequence "$($release.id) inventory platforms" @($inventory.platforms) $source.Platforms
    $headerArtifacts = @(
        $inventory.source_artifacts | Where-Object {
            $_.role -eq $source.HeaderRole -and
            $_.url -eq $release.hip_archive_url -and
            $_.sha256 -eq $release.hip_header_sha256 -and
            $_.path -eq $release.hip_header_path
        }
    )
    if ($headerArtifacts.Count -ne 1) {
        throw "$($release.id) source inventory does not contain exactly one matching pinned HIP header artifact"
    }
    $headerArtifact = $headerArtifacts[0]
    if ([string]::IsNullOrWhiteSpace($headerArtifact.revision)) {
        throw "$($release.id) pinned HIP header artifact has no revision"
    }
    $functions = @(
        $inventory.entries |
            Where-Object { $_.kind -eq 'function' -and ($_.name -eq 'hipRuntimeGetVersion' -or $commonNames -contains $_.name) } |
            Sort-Object @{ Expression = { if ($_.name -eq 'hipRuntimeGetVersion') { -1 } else { [Array]::IndexOf($commonNames, $_.name) } } } |
            ForEach-Object {
                [ordered]@{
                    name = $_.name
                    normalized_signature = $_.normalized_signature
                    signature_hash = $_.signature_hash
                    platforms = @($_.platforms)
                }
            }
    )
    if ($functions.Count -ne 27) {
        throw "$($release.id) produced $($functions.Count) bootstrap/common functions, expected 27"
    }
    foreach ($function in $functions) {
        Assert-Sequence "$($release.id) function $($function.name) platforms" @($function.platforms) $source.Platforms
    }
    $types = @(
        foreach ($name in $typeNames) {
            $entries = @($inventory.entries | Where-Object { $_.name -eq $name })
            if ($entries.Count -ne 1) { throw "$($release.id) must have exactly one transitive type $name" }
            $entry = $entries[0]
            Assert-Sequence "$($release.id) type $name platforms" @($entry.platforms) $source.Platforms
            [ordered]@{
                name = $entry.name
                kind = $entry.kind
                normalized_signature = $entry.normalized_signature
                signature_hash = $entry.signature_hash
                platforms = @($entry.platforms)
            }
        }
    )
    $attributes = @(
        foreach ($expected in $ledger.device_attributes) {
            $entries = @(
                $inventory.entries | Where-Object {
                    $_.kind -eq 'enum_value' -and $_.name -eq $expected.name
                }
            )
            if ($entries.Count -ne 1) {
                throw "$($release.id) must have exactly one enum declaration for $($expected.name)"
            }
            $entry = $entries[0]
            if ($null -eq $entry.numeric_value) {
                throw "$($release.id) inventory did not evaluate $($expected.name)"
            }
            if ([int64]$entry.numeric_value -ne [int64]$expected.value) {
                throw "$($release.id) $($expected.name) value is $($entry.numeric_value); expected $($expected.value)"
            }
            Assert-Sequence "$($release.id) attribute $($expected.name) platforms" @($entry.platforms) $source.Platforms
            [ordered]@{
                name = $entry.name
                value = [int64]$entry.numeric_value
                normalized_signature = $entry.normalized_signature
                signature_hash = $entry.signature_hash
                platforms = @($entry.platforms)
            }
        }
    )
    $snapshots += [ordered]@{
        release_id = $release.id
        source_inventory_id = $inventory.inventory_id
        source_inventory_platforms = @($inventory.platforms)
        source_header_artifact = [ordered]@{
            role = $headerArtifact.role
            url = $headerArtifact.url
            sha256 = $headerArtifact.sha256
            revision = $headerArtifact.revision
            path = $headerArtifact.path
        }
        target_abi = [ordered]@{
            pointer_width_bits = 64
            size_t_width_bits = 64
            enum_width_bits = 32
            success_value = 0
            null_pointer_sentinel = 'all-bits-zero'
        }
        functions = $functions
        transitive_types = $types
        device_attributes = $attributes
    }
}

$hip570 = $ledger.reviewed_releases | Where-Object { $_.id -eq 'hip-5.7.31541' }
$hip571 = $ledger.reviewed_releases | Where-Object { $_.id -eq 'hip-5.7.31921' }
if ($null -eq $hip570 -or $null -eq $hip571 -or
    $hip570.hip_archive_url -ne $hip571.hip_archive_url -or
    $hip570.hip_header_sha256 -ne $hip571.hip_header_sha256 -or
    $hip570.hip_version_sha256 -ne $hip571.hip_version_sha256 -or
    $hip570.hip_commit -notmatch 'peeled commit 80681169ae20de8f8025b4fad799204c3ae7de50' -or
    $hip571.hip_commit -ne '80681169ae20de8f8025b4fad799204c3ae7de50') {
    throw 'HIP 5.7.0/5.7.1 shared declaration identity is no longer proven'
}

$document = [ordered]@{
    schema_version = 1
    spdx_license_identifier = 'CC0-1.0'
    inventory_id = 'hip-runtime-profile-declarations'
    provenance = 'Normalized factual declarations and evaluated enum values extracted independently from each exact HIP header artifact pinned by runtime-profiles.json; no vendor source text is copied.'
    snapshots = $snapshots
}
$destination = Resolve-Input $Output
$json = $document | ConvertTo-Json -Depth 20
[IO.File]::WriteAllText($destination, "$json`n", [Text.UTF8Encoding]::new($false))
Write-Host "Wrote $destination"
